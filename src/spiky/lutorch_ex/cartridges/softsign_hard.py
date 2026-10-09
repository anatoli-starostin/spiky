"""SoftSignHardLUT — Gen-2 variant 2.3 (hard forward, 2-alternative soft-surrogate backward).

Forward is the hard read ``y = sum_t W_t[c_t]``; the backward is the two-cell soft surrogate with
the learned-temperature weight ``w`` from
:class:`~spiky.lutorch_ex.cartridges.softsign_base.SoftSignLUT`. The weight gradient reflects the
*actual* hard forward — a 1-row scatter at ``c_t`` only — while ``w`` shapes the input/temperature
gradient (exactly as Gen-1 ``ManifestoHardLUT`` does with ``U``).

Read path: ``F.embedding_bag`` (fuses the per-table gather + the tph sum in one kernel) for the
hard value; eval reads a single cell per table. Backward is plain autograd — the weight ``w``
(including the two temperatures) is fully differentiable, so autograd produces the exact
2-alternative backward; no custom ``autograd.Function`` is needed. torch.compile wraps the eval
read on CUDA only (inherited from the base). fp32/fp64 only: like the other non-fused cartridges
it carries no mixed-precision handling, so the base ``forward`` raises on bf16/fp16 (bf16 was
benchmarked and dropped — the embedding_bag read upcasts to fp32 for accumulation, so it gave no
speedup). No native kernel of its own (the native lprojection kernels hardcode the Gen-1
inverse-L1 uncertainty).
"""
from __future__ import annotations

import torch

from ._fused_ops import fused_hard_read
from .softsign_base import SoftSignLUT


class SoftSignHardLUT(SoftSignLUT):
    """Gen-2 hard cartridge (variant 2.3); see module docstring."""

    def _needs_alt(self) -> bool:
        return self.training  # eval reads only c_t

    def _combine(self, y_hard, y_alt, u_abs_star):  # pragma: no cover - forward is overridden
        raise NotImplementedError

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
            grp_out = y_c + (surr - surr.detach())               # value == y_c; grad via w into input/temps
        return self._route(grp_out, x)
