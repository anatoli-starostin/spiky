"""FusedManifestoHardLUT — hybrid, GPU-efficient twin of ManifestoHardLUT.

Same manifesto math (hard sign-addressed read; two-alternative soft backward), but the
forward/backward dispatch to whichever implementation is fastest for the current
(op, batch, device), never slower than the pure cartridge and numerically equivalent to
it (the oracle):

* eval  -> the pure compiled gather read (fastest on the inference path);
* train -> tier-2 NATIVE lutorch_cuda kernels when available, else tier-1 embedding_bag +
  the custom straight-through autograd.Function.

``backend`` forces a path ('pure_eval'/'tier1'/'native'/'auto'); 'auto' is the hybrid.
"""
from __future__ import annotations

import torch

from ._fused_ops import FusedHardSTE
from ._native_ops import NativeHard, native_available
from .manifesto_base import ManifestoLUT


class FusedManifestoHardLUT(ManifestoLUT):
    def __init__(self, spec, *, backend: str = "auto", **kw):
        super().__init__(spec, **kw)
        self.backend = backend

    def _needs_alt(self) -> bool:
        return self.training  # eval reads only c_t

    def _combine(self, y_hard, y_alt, u_abs_star):  # pragma: no cover - forward is overridden
        raise NotImplementedError

    def _star(self, z, u, j_star):
        """Return (a_local, b_local, a_global, b_global, u_signed_star) at the j* pair."""
        G, tph, nap, d_in = self.spec.n_groups, self.spec.tph, self.spec.nap, self.spec.d_in
        B = z.shape[0]
        je = j_star.unsqueeze(-1)
        u_signed = u.gather(-1, je).squeeze(-1)
        al = self.anchor_a.unsqueeze(0).expand(B, G, tph, nap).gather(-1, je).squeeze(-1)
        bl = self.anchor_b.unsqueeze(0).expand(B, G, tph, nap).gather(-1, je).squeeze(-1)
        off = torch.arange(G, device=z.device).view(1, G, 1) * d_in
        return al, bl, al + off, bl + off, u_signed

    def _pick(self, x: torch.Tensor) -> str:
        if not self.training:
            return "pure_eval"                        # eval: compiled gather read wins
        if native_available(x.device):
            return "native"                           # train: native backward (CUDA default)
        return "tier1"                                # train (CPU or no native): embedding_bag + STE

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        z, u, c, j_star, u_abs_star, c_alt = self._addresses(x)
        be = self._pick(x) if self.backend == "auto" else self.backend
        if be == "pure_eval":
            grp_out = self._read(c).sum(dim=2)            # pure compiled gather read + sum (eval)
        elif be == "native":
            al, bl, ag, bg, us = self._star(z, u, j_star)
            grp_out = NativeHard.apply(self.weights, z, c, c_alt, us, ag, bg)
        else:  # tier1
            al, bl, ag, bg, us = self._star(z, u, j_star)
            grp_out = FusedHardSTE.apply(self.weights, z, c, c_alt, al, bl)
        return self._route(grp_out, x)
