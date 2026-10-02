"""FusedManifestoHardLUT — the fused, GPU-efficient twin of ManifestoHardLUT.

Identical manifesto math (hard sign-addressed read; two-alternative soft backward with the
rational uncertainty), but the read+sum over a group's ``tph`` tables is fused into one
``embedding_bag`` kernel, and the training straight-through path is a custom
``autograd.Function`` with a hand-written scatter backward (which also sidesteps the
inductor mis-schedule the pure cartridge hit at large batch). Numerically equivalent to
ManifestoHardLUT (the oracle). Shares all geometry/addressing with the ManifestoLUT base.
"""
from __future__ import annotations

import torch

from ._fused_ops import FusedHardSTE, fused_hard_read
from .manifesto_base import ManifestoLUT


class FusedManifestoHardLUT(ManifestoLUT):
    """Fused hard cartridge; see module docstring. Combine is unused (forward is overridden)."""

    def _needs_alt(self) -> bool:
        # Eval reads only c_t (the hard eval shortcut); training needs the alternative.
        return self.training

    def _combine(self, y_hard, y_alt, u_abs_star):  # pragma: no cover - not used (forward overridden)
        raise NotImplementedError

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        z, u, c, j_star, u_abs_star, c_alt = self._addresses(x)
        if not self.training:
            grp_out = fused_hard_read(self.weights, c)          # one embedding_bag, hard sum
        else:
            B, G, tph, nap = x.shape[0], self.spec.n_groups, self.spec.tph, self.spec.nap
            je = j_star.unsqueeze(-1)
            a_star = self.anchor_a.unsqueeze(0).expand(B, G, tph, nap).gather(-1, je).squeeze(-1)
            b_star = self.anchor_b.unsqueeze(0).expand(B, G, tph, nap).gather(-1, je).squeeze(-1)
            grp_out = FusedHardSTE.apply(self.weights, z, c, c_alt, a_star, b_star)
        return self._route(grp_out, x)
