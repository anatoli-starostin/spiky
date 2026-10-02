"""SoftSignSmoothLUT — Gen-2 variant 2.4 (hybrid_smooth two-cell forward, 2-alt backward).

Unfused, pure-PyTorch reference. The two-cell blend is the forward value AND the gradient (as in
FastMultiHeadLut ``hybrid_smooth`` with ``backward_topk=1``):

    y = sum_t [ (1 - w_t) W_t[c_t] + w_t W_t[c_t'] ]

with ``w_t`` the learned-temperature weight from
:class:`~spiky.lutorch_ex.cartridges.softsign_base.SoftSignLUT`. Because the blend is an ordinary
differentiable expression (only the indices ``c, c', j*`` are non-differentiable), gradients flow
into both cell rows — ``(1 - w)`` into ``W[c]`` and ``w`` into ``W[c']`` — and into the input and
the two temperatures via ``w``. The input gradient reaches only the deciding pair ``j*``.

fp32/fp64 only (pure path): bf16/fp16 raises — use the fused twin ``FusedSoftSignSmoothLUT``.
"""
from __future__ import annotations

import torch

from .softsign_base import SoftSignLUT


class SoftSignSmoothLUT(SoftSignLUT):
    """Gen-2 smooth cartridge (variant 2.4); see module docstring."""

    def _combine(
        self, y_hard: torch.Tensor, y_alt: torch.Tensor, u_abs_star: torch.Tensor
    ) -> torch.Tensor:
        # (1 - w)*y_hard + w*y_alt = y_hard + w*(y_alt - y_hard); value and gradient both.
        w = self._blend_w(u_abs_star).unsqueeze(-1)  # [B, G, tph, 1]
        return y_hard + w * (y_alt - y_hard)
