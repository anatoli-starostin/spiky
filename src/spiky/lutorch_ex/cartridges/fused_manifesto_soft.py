"""FusedManifestoSoftLUT — hybrid, GPU-efficient twin of ManifestoSoftLUT.

Same (1-U)/U two-cell blend, dispatched per (op, batch, device) to the fastest path and
numerically equivalent to the pure ManifestoSoftLUT oracle:

* ``pure``   — base two-cell read + blend (the compiled read; best for small-batch eval);
* ``tier1``  — one F.embedding_bag with per_sample_weights=[1-U, U] (fuses read+scale+sum);
* ``native`` — the native lprojection_forward_smooth + its na1-smooth backward;
* ``cuda``   — hand-written kernels (``csrc/fused_manifesto.cu``): one forward and one backward
  kernel, nothing per table in global memory.

``backend`` forces a path ('pure'/'tier1'/'native'/'cuda'/'auto'); 'auto' is the hybrid. In
training 'auto' takes 'cuda' whenever it can run, at every batch size; otherwise (and in eval)
the batch-size heuristic below applies unchanged. ``knobs`` (a :class:`ManifestoCudaKnobs`) sets
the 'cuda' kernels' launch configuration.
"""
from __future__ import annotations

import torch

from ._fused_manifesto_cuda import (MODE_SOFT, auto_wants_cuda, init_cuda_backend, manifesto_cuda_forward,
                                    require_cuda, shape_ok)
from ._fused_ops import fused_blend_read, _acc_dtype, validate_backend
from ._native_ops import NativeSoft, native_available, raise_if_forced_native_unavailable, require_native_or_report
from .manifesto_base import ManifestoLUT
from .uncertainty import rational_uncertainty

_LOW_PREC = (torch.bfloat16, torch.float16)


class FusedManifestoSoftLUT(ManifestoLUT):
    #: Every backend _forward_impl dispatches on ('auto' picks one of the others per call). The pure path is
    #: called 'pure' here (not 'pure_eval' as in the other twins): auto also uses it for CPU training.
    _BACKENDS = ("auto", "pure", "tier1", "native", "cuda")

    def __init__(self, spec, *, backend: str = "auto", knobs=None, **kw):
        validate_backend(type(self).__name__, backend, self._BACKENDS)
        super().__init__(spec, **kw)
        self.backend = backend
        init_cuda_backend(self, knobs)

    def _supports_low_precision(self) -> bool:
        return True  # bf16/fp16: fp32 addressing + fp32-accumulated reads (native / tier-1)

    def _combine(self, y_hard, y_alt, u_abs_star):  # pragma: no cover - forward is overridden
        raise NotImplementedError

    def _star_global(self, z, u, j_star):
        G, tph, nap, d_in = self.spec.n_groups, self.spec.tph, self.spec.nap, self.spec.d_in
        B = z.shape[0]
        je = j_star.unsqueeze(-1)
        u_signed = u.gather(-1, je).squeeze(-1)
        al = self.anchor_a.unsqueeze(0).expand(B, G, tph, nap).gather(-1, je).squeeze(-1)
        bl = al if self.single else (  # single: b is a placeholder; input grad scatters to a only
            self.anchor_b.unsqueeze(0).expand(B, G, tph, nap).gather(-1, je).squeeze(-1)
        )
        off = torch.arange(G, device=z.device).view(1, G, 1) * d_in
        return al + off, bl + off, u_signed

    def _pure_blend(self, c, c_alt, u_abs_star, drop_mask=None):
        y_hard, y_alt = self._read_pair(c, c_alt)
        u = rational_uncertainty(u_abs_star).unsqueeze(-1)
        blend = y_hard + u * (y_alt - y_hard)
        if drop_mask is not None:                              # table dropout before the tph-sum
            blend = blend * drop_mask.unsqueeze(-1)
        return blend.sum(dim=2, dtype=_acc_dtype(blend.dtype))  # fp32-accum for bf16/fp16

    # H100 measurements: pure read wins soft eval at small/mid batch; the fused embedding_bag
    # (tier1) wins at large batch (both eval and the train step); native wins the train step at
    # small/mid batch. Threshold separates B=128 from B=24576.
    _LARGE_BATCH = 4096

    def _pick(self, x: torch.Tensor) -> str:
        large = x.is_cuda and x.shape[0] >= self._LARGE_BATCH
        if self.training:
            # Train: the fused 'cuda' kernels at every batch size (measured faster than tier1 and native at the
            # canonical h16 d48 tph64 nap8, 32,768 vectors). Without them, the heuristic below as before.
            nxt = "tier1" if large else ("native" if native_available(x.device) else ("tier1" if x.is_cuda else "pure"))
            if auto_wants_cuda(self, x, nxt):
                return "cuda"
        if large:
            return "tier1"                                   # embedding_bag wins at large batch
        if not self.training:
            return "pure"                                    # eval small/mid: compiled read wins
        if native_available(x.device):
            return "native"                                  # train small/mid: native step wins (fp32/bf16/fp16)
        # Here native was the choice and is unavailable: on a CUDA input that is an involuntary fallback (reported
        # once per process, or raised under SPIKY_LUTORCH_REQUIRE_NATIVE=1). The large-batch tier1 above is a
        # deliberate heuristic and stays quiet. Already reported above when the 'cuda' kernels were the first choice
        # for this input (one banner per fallback).
        if not shape_ok(self, x):
            require_native_or_report(type(self).__name__, x, "auto", "tier1")
        return "tier1" if x.is_cuda else "pure"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # The 'cuda' backend is one fused kernel per direction: dispatch it directly, like FusedConfidenceLUT, instead
        # of through ManifestoLUT.forward's torch.compile(_forward_impl), which only graph-breaks at the extension call
        # and costs ~0.1 ms of host time per eval call. Every other backend keeps the base forward unchanged.
        if self.backend == "cuda" or (self.backend == "auto" and self._pick(x) == "cuda"):
            return self._forward_impl(x)
        return super().forward(x)

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        # bf16/fp16 support lives here (not in the pure base). fp32 addressing; fp32-accumulated
        # reads; output cast back to the input dtype once at the end.
        low = x.dtype in _LOW_PREC
        xa = x.float() if low else x
        be = self._pick(x) if self.backend == "auto" else self.backend
        if self.backend == "native":
            raise_if_forced_native_unavailable(type(self).__name__, x)
        if self.backend == "cuda":
            require_cuda(self, x)
        self.last_backend = be                               # provenance for benchmarks (what actually ran)
        if be == "cuda":                                     # addressing, read and backward all inside the kernels
            return manifesto_cuda_forward(self, x, MODE_SOFT)
        # Every TRAIN path uses the compiled addressing (fuses the eager [B,G,tph,nap] materialization),
        # including the large-batch tier-1 (embedding_bag) route this cartridge picks at scale. Eval
        # keeps plain _addresses — it is already compiled whole by the base forward, so gating on
        # self.training (False in eval) avoids a nested compile.
        z, u, c, j_star, u_abs_star, c_alt = self._addr(xa) if self.training else self._addresses(xa)
        # Table-dropout keep-mask [B,G,tph] (train+grad only; None at eval / rate 0), fp32-acc dtype.
        dmask = self._table_dropout_mask(x.shape[0], self.weights.device, _acc_dtype(self.weights.dtype))
        if be == "pure":
            grp_out = self._pure_blend(c, c_alt, u_abs_star, drop_mask=dmask)
        elif be == "native":
            # Single mode reuses the native smooth forward + weight-grad; input grad to a only.
            # Grad target z passed in the input dtype (bf16) -> input grad stays bf16, one cast.
            ag, bg, us = self._star_global(z, u, j_star)
            zc = x[:, self.in_head, :] if low else z
            grp_out = NativeSoft.apply(self.weights, zc, c, c_alt, us, ag, bg, self.single, dmask)
        else:  # tier1
            U = rational_uncertainty(u_abs_star)
            grp_out = fused_blend_read(self.weights, c, c_alt, U, drop_mask=dmask)
        return self._route(grp_out, x).to(x.dtype)
