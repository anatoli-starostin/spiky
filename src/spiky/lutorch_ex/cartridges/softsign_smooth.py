"""SoftSignSmoothLUT — Gen-2 variant 2.4 (hybrid_smooth two-cell forward, 2-alternative backward).

The two-cell blend is the forward value AND the gradient:

    y = sum_t [ (1 - w_t) W_t[c_t] + w_t W_t[c_t'] ]

with ``w_t`` the learned-temperature weight from
:class:`~spiky.lutorch_ex.cartridges.softsign_base.SoftSignLUT`. Gradients flow into both cell
rows — ``(1 - w)`` into ``W[c]`` and ``w`` into ``W[c']`` — and into the input and the two
temperatures via ``w`` (the input gradient reaches only the deciding pair ``j*``). Only the indices
``c, c', j*`` are non-differentiable.

Read path: training uses one ``F.embedding_bag`` with ``per_sample_weights = [1-w, w]`` (fuses
read + scale + tph sum); eval uses the plain two-cell read + blend. Backward is plain autograd (the
blend, including the temperatures, is differentiable — no custom ``autograd.Function``).
torch.compile wraps the eval read on CUDA only (inherited). fp32/fp64 only — the base ``forward``
raises on bf16/fp16 (bf16 benchmarked and dropped: the embedding_bag read upcasts to fp32 for
accumulation, so it gave no speedup). No native lutorch_cuda kernel (Gen-1 kernels hardcode the
inverse-L1 uncertainty, not this learned-temperature sigmoid).
"""
from __future__ import annotations

import torch

from ._fused_ops import fused_blend_read
from .softsign_base import SoftSignLUT


class SoftSignSmoothLUT(SoftSignLUT):
    """Gen-2 smooth cartridge (variant 2.4); see module docstring."""

    def _combine(self, y_hard, y_alt, u_abs_star):  # pragma: no cover - forward is overridden
        raise NotImplementedError

    def _pure_blend(self, c, c_alt, w):
        y_hard, y_alt = self._read_pair(c, c_alt)
        return (y_hard + w.unsqueeze(-1) * (y_alt - y_hard)).sum(dim=2)

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        z, u, c, j_star, u_abs_star, c_alt = self._addresses(x)
        w = self._blend_w(u_abs_star)
        if self.training:
            grp_out = fused_blend_read(self.weights, c, c_alt, w)
        else:
            grp_out = self._pure_blend(c, c_alt, w)
        return self._route(grp_out, x)
