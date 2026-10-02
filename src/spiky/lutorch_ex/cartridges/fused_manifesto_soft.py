"""FusedManifestoSoftLUT — the fused, GPU-efficient twin of ManifestoSoftLUT.

Identical manifesto math (the (1-U)/U two-cell blend as both value and gradient), but the
read+scale+sum over a group's ``tph`` tables and its two cells is fused into one
``embedding_bag`` call with ``per_sample_weights`` = [1-U, U]. Being ordinary autograd,
its gradients match ManifestoSoftLUT exactly. Shares geometry/addressing with the base.
"""
from __future__ import annotations

import torch

from ._fused_ops import fused_blend_read
from .manifesto_base import ManifestoLUT
from .uncertainty import rational_uncertainty


class FusedManifestoSoftLUT(ManifestoLUT):
    """Fused soft cartridge; see module docstring. Combine is unused (forward is overridden)."""

    def _combine(self, y_hard, y_alt, u_abs_star):  # pragma: no cover - not used (forward overridden)
        raise NotImplementedError

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        z, u, c, j_star, u_abs_star, c_alt = self._addresses(x)
        U = rational_uncertainty(u_abs_star)                    # [B, G, tph]
        grp_out = fused_blend_read(self.weights, c, c_alt, U)   # one embedding_bag + per_sample_weights
        return self._route(grp_out, x)
