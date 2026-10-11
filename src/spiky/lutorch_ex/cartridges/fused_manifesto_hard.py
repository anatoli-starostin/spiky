"""FusedManifestoHardLUT — hybrid, GPU-efficient twin of ManifestoHardLUT, for both weight functions.

Same Manifesto math (hard sign-addressed read; two-alternative soft backward through the neighbour weight ``w``,
``weight="rational"`` or ``"softsign"``), but the forward/backward dispatch to whichever implementation is fastest for
the current (op, batch, device), numerically equivalent to the reference (the oracle):

* train and eval -> the hand-written 'cuda' kernels (``csrc/fused_manifesto.cu``: one forward and one backward kernel,
  nothing per table in global memory; both weight functions, the soft-sign temperatures' gradients included) whenever
  the input / configuration is a case they cover. A covered case whose extension is missing raises
  NativeUnavailableError (``_fallback``), unless LUTORCH_EX_NO_CUDA_EXT=1;
* otherwise eval -> the pure compiled gather read; train -> the NATIVE lprojection kernels when available (the
  rational surrogate, or the soft-sign surrogate tail of ``_native_softsign``), else tier-1: embedding_bag plus a
  straight-through surrogate.

``backend`` forces a path ('pure_eval'/'tier1'/'native'/'cuda'/'auto'); 'auto' is the hybrid.
``knobs`` (a :class:`ManifestoCudaKnobs`) sets the 'cuda' kernels' launch configuration.
"""
from __future__ import annotations

import torch

from ._fused_manifesto_cuda import (MODE_HARD, auto_wants_cuda, init_cuda_backend, manifesto_cuda_forward,
                                    require_cuda, shape_ok)
from ._fused_ops import FusedHardSTE, _acc_dtype, _global_cells, fused_hard_read, validate_backend
from ._native_ops import NativeHard, native_available, raise_if_forced_native_unavailable, require_native_or_report
from ._native_softsign import NativeSoftSignHard
from .manifesto_base import ManifestoLUT

_LOW_PREC = (torch.bfloat16, torch.float16)


class FusedManifestoHardLUT(ManifestoLUT):
    #: Every backend _forward_impl dispatches on ('auto' picks one of the others per call).
    _BACKENDS = ("auto", "pure_eval", "tier1", "native", "cuda")

    def __init__(self, spec, *, backend: str = "auto", knobs=None, **kw):
        validate_backend(type(self).__name__, backend, self._BACKENDS)
        super().__init__(spec, **kw)
        self.backend = backend
        init_cuda_backend(self, knobs)

    def _supports_low_precision(self) -> bool:
        return True  # bf16/fp16: fp32 addressing + fp32-accumulated reads

    def _pick(self, x: torch.Tensor) -> str:
        if not self.training:
            # eval: the 'cuda' forward kernel when it can run (fastest at every batch measured, both GPUs); else the
            # compiled gather read. A missing extension is reported loudly; a non-cuda input falls through quietly.
            return "cuda" if auto_wants_cuda(self, x, "pure_eval") else "pure_eval"
        has_native = native_available(x.device)
        if auto_wants_cuda(self, x, "native" if has_native else "tier1"):
            return "cuda"                             # train: one fused forward + one fused backward kernel
        if has_native:
            return "native"                           # train: native backward (fp32/bf16/fp16, both modes)
        # train (CPU / no native): embedding_bag + STE. On a CUDA input this is an involuntary fallback: report its
        # cause: NativeUnavailableError (quiet if disabled on purpose). Already reported above when
        # the 'cuda' kernels were the first choice for this input (one banner per fallback, naming 'tier1').
        if not shape_ok(self, x):
            require_native_or_report(type(self).__name__, x, "auto", "tier1")
        return "tier1"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # The 'cuda' backend is one fused kernel per direction: dispatch it directly, like FusedConfidenceLUT, instead
        # of through ManifestoLUT.forward's torch.compile(_forward_impl), which only graph-breaks at the extension call
        # and costs ~0.1 ms of host time per eval call. Every other backend keeps the base forward unchanged.
        if self.backend == "cuda" or (self.backend == "auto" and self._pick(x) == "cuda"):
            return self._forward_impl(x)
        return super().forward(x)

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        # bf16/fp16 support lives here (not in the pure base). Addressing runs in fp32 so the
        # discrete bit decisions don't flip vs fp32; every read/reduction accumulates in fp32
        # and the output is cast back to the input dtype once, at the end.
        low = x.dtype in _LOW_PREC
        xa = x.float() if low else x
        be = self._pick(x) if self.backend == "auto" else self.backend
        if self.backend == "native":
            raise_if_forced_native_unavailable(type(self).__name__, x)
        if self.backend == "cuda":
            require_cuda(self, x)
        self.last_backend = be                        # provenance for benchmarks (what actually ran)
        if be == "cuda":                              # addressing, read and backward all inside the kernels
            return manifesto_cuda_forward(self, x, MODE_HARD)
        # Native train path uses the compiled addressing (fuses the eager [B,G,tph,nap] materialization);
        # eval/tier1 keep plain _addresses (eval is already compiled whole by the base forward).
        z, u, c, j_star, u_abs_star, c_alt = self._addr(xa) if be == "native" else self._addresses(xa)
        # Table-dropout keep-mask [B,G,tph] (train+grad only; None at eval / rate 0), in the fp32
        # accumulation dtype. Threaded into the native / STE custom-autograd Functions so a dropped
        # table contributes 0 to BOTH value and gradient (the SAME mask is reused in their backward).
        dmask = self._table_dropout_mask(x.shape[0], self.weights.device, _acc_dtype(self.weights.dtype))
        if be == "pure_eval":
            rd = self._read(c)
            grp_out = rd.sum(dim=2, dtype=_acc_dtype(rd.dtype))  # fp32-accum for bf16/fp16 (eval)
        elif be == "native":
            # The grad target z is passed in the INPUT dtype (bf16) so the input grad stays bf16 end to end (one
            # cast), while the fp32 addressing above supplies c/c_alt/us. Single mode reuses the native forward and
            # weight-grad kernels (index-only); only the input grad scatters to one coordinate.
            al, bl, ag, bg, us = self._star(z, u, j_star)
            zc = x[:, self.in_head, :] if low else z
            if self.weight == "rational":
                grp_out = NativeHard.apply(self.weights, zc, c, c_alt, us, ag, bg, self.single, dmask)
            else:
                t_soft, t_select = self._softsign_temps(us.dtype)
                grp_out = NativeSoftSignHard.apply(self.weights, zc, c, c_alt, us, ag, bg,
                                                   self.single, t_soft, t_select, dmask)
        elif self.weight == "rational":  # tier1: embedding_bag + the hand-written rational STE
            al, bl, ag, bg, us = self._star(z, u, j_star)
            grp_out = FusedHardSTE.apply(self.weights, z, c, c_alt, al, bl, self.single, dmask)
        else:  # tier1, softsign: embedding_bag value + the surrogate through w (autograd reaches the temperatures)
            W = self.weights.float() if low else self.weights
            w = self._blend_w(u_abs_star)
            y_c = fused_hard_read(W, c, drop_mask=dmask)
            W2 = W.reshape(-1, W.shape[-1])
            G, tph, K = self.spec.n_groups, self.spec.tph, self.spec.n_cells
            diff = (W2[_global_cells(c_alt, G, tph, K)] - W2[_global_cells(c, G, tph, K)]).detach()
            sw = w.unsqueeze(-1) * diff
            if dmask is not None:
                sw = sw * dmask.unsqueeze(-1)
            surr = sw.sum(dim=2)
            grp_out = y_c + (surr - surr.detach())
        return self._route(grp_out, x).to(x.dtype)
