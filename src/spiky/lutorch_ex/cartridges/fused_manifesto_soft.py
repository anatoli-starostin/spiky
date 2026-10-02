"""FusedManifestoSoftLUT — hybrid, GPU-efficient twin of ManifestoSoftLUT.

Same (1-U)/U two-cell blend, dispatched per (op, batch, device) to the fastest path and
numerically equivalent to the pure ManifestoSoftLUT oracle:

* ``pure``   — base two-cell read + blend (the compiled read; best for small-batch eval);
* ``tier1``  — one F.embedding_bag with per_sample_weights=[1-U, U] (fuses read+scale+sum);
* ``native`` — lutorch_cuda lprojection_forward_smooth + its na1-smooth backward.

``backend`` forces a path ('pure'/'tier1'/'native'/'auto'); 'auto' is the hybrid.
"""
from __future__ import annotations

import torch

from ._fused_ops import fused_blend_read
from ._native_ops import NativeSoft, native_available
from .manifesto_base import ManifestoLUT
from .uncertainty import rational_uncertainty


class FusedManifestoSoftLUT(ManifestoLUT):
    def __init__(self, spec, *, backend: str = "auto", **kw):
        super().__init__(spec, **kw)
        self.backend = backend

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

    def _pure_blend(self, c, c_alt, u_abs_star):
        y_hard, y_alt = self._read_pair(c, c_alt)
        u = rational_uncertainty(u_abs_star).unsqueeze(-1)
        return (y_hard + u * (y_alt - y_hard)).sum(dim=2)

    # H100 measurements: pure read wins soft eval at small/mid batch; the fused embedding_bag
    # (tier1) wins at large batch (both eval and the train step); native wins the train step at
    # small/mid batch. Threshold separates B=128 from B=24576.
    _LARGE_BATCH = 4096

    def _pick(self, x: torch.Tensor) -> str:
        large = x.is_cuda and x.shape[0] >= self._LARGE_BATCH
        if large:
            return "tier1"                                   # embedding_bag wins at large batch
        if not self.training:
            return "pure"                                    # eval small/mid: compiled read wins
        if native_available(x.device):
            return "native"                                  # train small/mid: native step wins
        return "tier1" if x.is_cuda else "pure"

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        z, u, c, j_star, u_abs_star, c_alt = self._addresses(x)
        be = self._pick(x) if self.backend == "auto" else self.backend
        if be == "pure":
            grp_out = self._pure_blend(c, c_alt, u_abs_star)
        elif be == "native":
            # Single mode reuses the native smooth forward + weight-grad; input grad to a only.
            ag, bg, us = self._star_global(z, u, j_star)
            grp_out = NativeSoft.apply(self.weights, z, c, c_alt, us, ag, bg, self.single)
        else:  # tier1
            U = rational_uncertainty(u_abs_star)
            grp_out = fused_blend_read(self.weights, c, c_alt, U)
        return self._route(grp_out, x)
