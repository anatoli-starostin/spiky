"""ManifestoHardLUT — the hard Manifesto reference cartridge (Gen-1 variant 1.1 with ``weight="rational"``, Gen-2
variant 2.3 with ``weight="softsign"``).

"Hard forward, two-alternative soft backward."

All shared structure (anchor pairs, MSB-first addressing, head routing, the two-cell read, the neighbour weight
``w``) lives in :class:`~spiky.lutorch_ex.cartridges.manifesto_base.ManifestoLUT`.

The math
========
Per group ``g`` and table ``t`` the base computes the hard cell ``c_t`` (MSB-first sign-bit address over the table's
``nap`` margins ``u_j = z[a_j] - z[b_j]``), its least-confident-bit-flip neighbour ``c_t'`` (``j* = argmin_j |u_j|``),
the deciding margin ``u_{j*}`` and the neighbour weight ``w_t = w(|u_{j*}|)`` (``weight="rational"``:
``0.5/(1 + |u_{j*}|)``; ``weight="softsign"``: the learned-temperature sigmoid).

Hard forward.
    ``y = sum_t W_t[c_t]``: one row per table, summed over the group's ``tph`` tables (and over groups into the shared
    output in the fan-in case). Eval reads a single cell per table.

Two-alternative soft backward.
    The forward **value** is exactly the hard read, but gradients flow as if the output were the blend
    ``y~ = sum_t [(1 - w_t) W_t[c_t] + w_t W_t[c_t']]``. The index set ``c_t, c_t', j*`` is stop-gradient.

Gradient asymmetry (by design).
    The **weight-table** gradient is hard: only the addressed cell ``c_t`` receives gradient (coefficient 1); the
    neighbour ``c_t'`` gets none. The blend shapes the **input** gradient (and, with ``softsign``, the two
    temperatures' gradients) only. (The soft sibling,
    :class:`~spiky.lutorch_ex.cartridges.manifesto_soft.ManifestoSoftLUT`, lets the blend drive the value and the
    weight gradient too.)

Read path: in training, ``F.embedding_bag`` (the per-table gather and the tph sum in one kernel) for the hard value,
plus the two cells' difference, detached, for the surrogate; backward is plain autograd through ``w``. torch.compile
wraps the eval read on CUDA only (ReferenceForward). bf16 / fp16 run as fp32 maths with one rounding per leaf (ReferenceForward, spec amendment C).
"""
from __future__ import annotations

import torch

from ._fused_ops import fused_hard_read
from .manifesto_base import ManifestoLUT
from .reference_base import ReferenceForward


class ManifestoHardLUT(ReferenceForward, ManifestoLUT):
    """Hard Manifesto reference cartridge; see module docstring for the math."""

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        z, u, c, j_star, u_abs_star, c_alt = self._addresses(x)
        if not self.training:
            grp_out = self._read(c).sum(dim=2)                   # hard eval: one cell per table
        else:
            w = self._blend_w(u_abs_star)
            dmask = self._table_dropout_mask(x.shape[0], self.weights.device, self.weights.dtype)
            y_c = fused_hard_read(self.weights, c, drop_mask=dmask)  # value = sum_t keep_t W[c_t]
            y_hard_pt, y_alt_pt = self._read_pair(c, c_alt)      # the two cells for the surrogate
            diff = (y_alt_pt - y_hard_pt).detach()               # detached: no weight grad from the surrogate
            sw = w.unsqueeze(-1) * diff
            if dmask is not None:                                # dropped tables contribute no surrogate grad
                sw = sw * dmask.unsqueeze(-1)
            surr = sw.sum(dim=2)
            grp_out = y_c + (surr - surr.detach())               # value == y_c; grad via w into the input (and temps)
        return self._route(grp_out, x)
