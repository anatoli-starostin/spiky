"""ManifestoHardLUT — the gen-1 reference cartridge (hard variant 1.1).

"Hard forward, two-alternative soft backward with a rational uncertainty function."
Greenfield re-implementation of the original ``spiky.lutorch.multi_head_lut.MultiHeadLut``
(variant 1.1 in the LUT-ablation table, ``smooth_mode=False``), built to the cartridge
contract with zero imports from the old ``lutorch``.

All shared structure (anchor pairs, MSB-first addressing, head routing, the two-cell
read) lives in :class:`~spiky.lutorch_ex.cartridges.manifesto_base.ManifestoLUT`; this
class supplies only :meth:`_combine`.

The gen-1 math
==============
Per group ``g`` and table ``t`` the base computes the hard cell ``c_t`` (MSB-first
sign-bit address over the table's ``nap`` margins ``u_j = z[a_j] - z[b_j]``), its
least-confident-bit-flip neighbour ``c_t'`` (``j* = argmin_j |u_j|``), and the deciding
margin ``u_{j*}``. This cartridge then combines them:

Hard forward.
    ``y = sum_t W_t[c_t]`` — one row per table, summed over the group's ``tph`` tables
    (and over groups into the shared output in the fan-in case). No score, no blend at eval.

Two-alternative soft backward.
    The forward **value** is exactly the hard read, but gradients flow as if the output
    were the blend ``y~ = sum_t[(1 - U_t) W_t[c_t] + U_t W_t[c_t']]`` with
    ``U_t = U(u_{j*}) = 0.5/(1 + |u_{j*}|)`` (rational uncertainty: ``U(0)=0.5``, ``U->0``
    as ``|u|->inf``, in ``(0, 0.5]``). The index set ``c_t, c_t', j*`` is stop-gradient.

Gradient asymmetry (by design, matching gen-1 ``smooth_mode=False``).
    The **weight-table** gradient is hard: only the addressed cell ``c_t`` receives
    gradient (coefficient 1); the alternative ``c_t'`` gets none. The uncertainty blend
    ``y~`` shapes the **input/addressing** gradient only. (The soft-forward sibling,
    :class:`~spiky.lutorch_ex.cartridges.manifesto_soft.ManifestoSoftLUT`, instead lets
    ``y~`` drive both the value and the weight gradient.)
"""
from __future__ import annotations

import torch

from .manifesto_base import ManifestoLUT
from .uncertainty import rational_uncertainty


class ManifestoHardLUT(ManifestoLUT):
    """Gen-1 reference cartridge (hard variant 1.1); see module docstring for the math."""

    def _combine(
        self, y_hard: torch.Tensor, y_alt: torch.Tensor, u_star: torch.Tensor
    ) -> torch.Tensor:
        if not self.training:
            return y_hard  # hard read, exactly
        # Straight-through composite: value == y_hard, but the input sees the blend's
        # gradient via U(u_star). g is detached so the weight table learns on the hard
        # cell only (the alternative gets no weight gradient).
        g = (y_hard - y_alt).detach()
        u_term = (-rational_uncertainty(u_star)).unsqueeze(-1) * g
        return y_hard + (u_term - u_term.detach())
