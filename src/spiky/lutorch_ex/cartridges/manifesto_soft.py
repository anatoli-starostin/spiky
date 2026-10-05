"""ManifestoSoftLUT — the soft-forward sibling (Gen-1 variant 1.2).

"Soft forward, two-alternative blend with a rational uncertainty function." The soft
counterpart of :class:`~spiky.lutorch_ex.cartridges.manifesto_hard.ManifestoHardLUT`.

All shared structure (anchor pairs, MSB-first addressing, head routing, the two-cell
read) lives in :class:`~spiky.lutorch_ex.cartridges.manifesto_base.ManifestoLUT`; this
class supplies only :meth:`_combine`.

The math
========
Where the hard cartridge returns ``W[c_t]`` and only *back-propagates* through the blend,
this cartridge makes the blend itself the output — at both train and eval:

    y = sum_t [ (1 - U_t) * W_t[c_t] + U_t * W_t[c_t'] ]

with ``c_t`` the MSB-first sign-bit address, ``c_t'`` its least-confident-bit-flip
neighbour (``j* = argmin_j |u_j|``), and the rational uncertainty
``U_t = 0.5 / (1 + |u_{j*}|)`` (``U(0)=0.5`` -> even blend; ``U->0`` as ``|u|->inf`` ->
all weight on ``c_t``): weight ``1 - U`` on the hard cell ``c_t`` and ``U`` on the
alternative ``c_t'``.

Because the blend is an ordinary differentiable expression (no stop-gradient on the
values), gradients flow through ``y~`` into **both** cell rows — ``(1 - U)`` into
``W[c_t]`` and ``U`` into ``W[c_t']`` — and into the input via ``U(u_{j*})``. Only the
discrete indices ``c_t, c_t', j*`` are non-differentiable. The value is continuous across
a bit flip (at ``u=0`` the two cells carry equal weight ``0.5`` and simply swap roles).
"""
from __future__ import annotations

import torch

from .manifesto_base import ManifestoLUT
from .uncertainty import rational_uncertainty


class ManifestoSoftLUT(ManifestoLUT):
    """Soft-forward cartridge (Gen-1 variant 1.2); see module docstring for the math."""

    def _combine(
        self, y_hard: torch.Tensor, y_alt: torch.Tensor, u_abs_star: torch.Tensor
    ) -> torch.Tensor:
        # (1 - U)*y_hard + U*y_alt, written as y_hard + U*(y_alt - y_hard) to save a temporary.
        # Used as both value and gradient (train and eval).
        u = rational_uncertainty(u_abs_star).unsqueeze(-1)  # [B, G, tph, 1]
        return y_hard + u * (y_alt - y_hard)
