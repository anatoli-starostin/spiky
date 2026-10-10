"""FusedManifestoHardLUT — hybrid, GPU-efficient twin of ManifestoHardLUT.

Same manifesto math (hard sign-addressed read; two-alternative soft backward), but the
forward/backward dispatch to whichever implementation is fastest for the current
(op, batch, device), never slower than the pure cartridge and numerically equivalent to
it (the oracle):

* eval  -> the pure compiled gather read (fastest on the inference path);
* train -> tier-2 NATIVE lprojection kernels when available, else tier-1 embedding_bag +
  the custom straight-through autograd.Function.

``backend`` forces a path ('pure_eval'/'tier1'/'native'/'auto'); 'auto' is the hybrid.
"""
from __future__ import annotations

import torch

from ._fused_ops import FusedHardSTE, _acc_dtype, validate_backend
from ._native_ops import NativeHard, native_available, raise_if_forced_native_unavailable, require_native_or_report
from .manifesto_base import ManifestoLUT

_LOW_PREC = (torch.bfloat16, torch.float16)


class FusedManifestoHardLUT(ManifestoLUT):
    #: Every backend _forward_impl dispatches on ('auto' picks one of the others per call).
    _BACKENDS = ("auto", "pure_eval", "tier1", "native")

    def __init__(self, spec, *, backend: str = "auto", **kw):
        validate_backend(type(self).__name__, backend, self._BACKENDS)
        super().__init__(spec, **kw)
        self.backend = backend

    def _supports_low_precision(self) -> bool:
        return True  # bf16/fp16: fp32 addressing + fp32-accumulated reads (native / tier-1)

    def _needs_alt(self) -> bool:
        return self.training  # eval reads only c_t

    def _combine(self, y_hard, y_alt, u_abs_star):  # pragma: no cover - forward is overridden
        raise NotImplementedError

    def _star(self, z, u, j_star):
        """Return (a_local, b_local, a_global, b_global, u_signed_star) at the j* bit.

        In single mode b is a placeholder (= a): each bit tests one coordinate vs zero, so
        the input grad scatters to a only (FusedHardSTE honours ``single``; native is not
        used in single mode).
        """
        G, tph, nap, d_in = self.spec.n_groups, self.spec.tph, self.spec.nap, self.spec.d_in
        B = z.shape[0]
        je = j_star.unsqueeze(-1)
        u_signed = u.gather(-1, je).squeeze(-1)
        al = self.anchor_a.unsqueeze(0).expand(B, G, tph, nap).gather(-1, je).squeeze(-1)
        bl = al if self.single else (
            self.anchor_b.unsqueeze(0).expand(B, G, tph, nap).gather(-1, je).squeeze(-1)
        )
        off = torch.arange(G, device=z.device).view(1, G, 1) * d_in
        return al, bl, al + off, bl + off, u_signed

    def _pick(self, x: torch.Tensor) -> str:
        if not self.training:
            return "pure_eval"                        # eval: compiled gather read wins
        if native_available(x.device):
            return "native"                           # train: native backward (fp32/bf16/fp16, both modes)
        # train (CPU / no native): embedding_bag + STE. On a CUDA input this is an involuntary fallback: report its
        # cause loudly (once per process), or raise under SPIKY_LUTORCH_REQUIRE_NATIVE=1.
        require_native_or_report(type(self).__name__, x, "auto", "tier1")
        return "tier1"

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        # bf16/fp16 support lives here (not in the pure base). Addressing runs in fp32 so the
        # discrete bit decisions don't flip vs fp32; every read/reduction accumulates in fp32
        # and the output is cast back to the input dtype once, at the end.
        low = x.dtype in _LOW_PREC
        xa = x.float() if low else x
        be = self._pick(x) if self.backend == "auto" else self.backend
        if self.backend == "native":
            raise_if_forced_native_unavailable(type(self).__name__, x)
        self.last_backend = be                        # provenance for benchmarks (what actually ran)
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
            # Single mode reuses the native forward + weight-grad kernels (index-only); only
            # the input-grad scatters to one coordinate (handled inside NativeHard). The grad
            # target z is passed in the INPUT dtype (bf16) so the input grad stays bf16 end to
            # end (one cast), while the fp32 addressing above supplies c/c_alt/us.
            al, bl, ag, bg, us = self._star(z, u, j_star)
            zc = x[:, self.in_head, :] if low else z
            grp_out = NativeHard.apply(self.weights, zc, c, c_alt, us, ag, bg, self.single, dmask)
        else:  # tier1
            al, bl, ag, bg, us = self._star(z, u, j_star)
            grp_out = FusedHardSTE.apply(self.weights, z, c, c_alt, al, bl, self.single, dmask)
        return self._route(grp_out, x).to(x.dtype)
