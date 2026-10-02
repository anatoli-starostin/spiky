"""SoftSignHardLUT — Gen-2 variant 2.3 (hard forward, 2-alternative soft-surrogate backward).

Unfused, pure-PyTorch reference. Forward is the hard read ``y = sum_t W_t[c_t]`` (eval reads one
cell per table); the backward is the two-cell soft surrogate with the learned-temperature weight
``w`` from :class:`~spiky.lutorch_ex.cartridges.softsign_base.SoftSignLUT`. The weight gradient
reflects the *actual* hard forward — a 1-row scatter at ``c_t`` only — while ``w`` shapes the
input/temperature gradient, exactly as the Gen-1 ``ManifestoHardLUT`` does with ``U``.

fp32/fp64 only (pure path): bf16/fp16 raises — use the fused twin ``FusedSoftSignHardLUT``.
"""
from __future__ import annotations

import torch

from .softsign_base import SoftSignLUT


class SoftSignHardLUT(SoftSignLUT):
    """Gen-2 hard cartridge (variant 2.3); see module docstring."""

    def _needs_alt(self) -> bool:
        # Eval needs only the hard cell c_t (the neighbour shapes the surrogate gradient only).
        return self.training

    def _combine(
        self, y_hard: torch.Tensor, y_alt: torch.Tensor, u_abs_star: torch.Tensor
    ) -> torch.Tensor:
        # Straight-through composite: value == y_hard, but the input/temperatures see the
        # gradient of the two-cell blend y_hard + w*(y_alt - y_hard) via w(|u*|). The cell
        # difference is detached, so the weight table learns on the hard cell c_t only (1-row).
        w = self._blend_w(u_abs_star).unsqueeze(-1)
        surr = w * (y_alt - y_hard).detach()
        return y_hard + (surr - surr.detach())
