"""SoftSignHardLUT — Gen-2 variant 2.3 (hard forward, 2-alternative soft-surrogate backward).

Forward is the hard read ``y = sum_t W_t[c_t]``; the backward is the two-cell soft surrogate with
the learned-temperature weight ``w`` from
:class:`~spiky.lutorch_ex.cartridges.softsign_base.SoftSignLUT`. The weight gradient reflects the
*actual* hard forward — a 1-row scatter at ``c_t`` only — while ``w`` shapes the input/temperature
gradient (exactly as Gen-1 ``ManifestoHardLUT`` does with ``U``).

Read path: ``F.embedding_bag`` (fuses the per-table gather + the tph sum in one kernel) for the
hard value; eval reads a single cell per table. Backward is plain autograd — the weight ``w``
(including the two temperatures) is fully differentiable, so autograd produces the exact
2-alternative backward; no custom ``autograd.Function`` is needed. fp32 accumulation and bf16/fp16
come from the embedding_bag read running on the fp32-upcast table (``_supports_low_precision`` ->
True; fp32 addressing, output cast once at the end). torch.compile wraps the eval read on CUDA
only (inherited from the base). The Gen-1 native lutorch_cuda kernels are NOT used: they hardcode
the inverse-L1 uncertainty and its derivative, not this learned-temperature sigmoid.
"""
from __future__ import annotations

import torch

from ._fused_ops import fused_hard_read, _acc_dtype
from .softsign_base import SoftSignLUT

_LOW_PREC = (torch.bfloat16, torch.float16)


class SoftSignHardLUT(SoftSignLUT):
    """Gen-2 hard cartridge (variant 2.3); see module docstring."""

    def _supports_low_precision(self) -> bool:
        return True  # bf16/fp16: fp32 addressing + fp32-accumulated embedding_bag read

    def _needs_alt(self) -> bool:
        return self.training  # eval reads only c_t

    def _combine(self, y_hard, y_alt, u_abs_star):  # pragma: no cover - forward is overridden
        raise NotImplementedError

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        low = x.dtype in _LOW_PREC
        # Addressing in fp32 for bf16/fp16 so the bit decisions don't flip; cast out once at end.
        z, u, c, j_star, u_abs_star, c_alt = self._addresses(x.float() if low else x)
        if not self.training:
            rd = self._read(c)                                   # hard eval: one cell per table
            grp_out = rd.sum(dim=2, dtype=_acc_dtype(rd.dtype))
        else:
            w = self._blend_w(u_abs_star)
            y_c = fused_hard_read(self.weights, c)               # value = sum_t W[c_t]; weight grad 1-row
            y_hard_pt, y_alt_pt = self._read_pair(c, c_alt)      # the two cells for the surrogate
            diff = (y_alt_pt - y_hard_pt).detach()               # detached: no weight grad from the surrogate
            surr = (w.unsqueeze(-1) * diff).sum(dim=2, dtype=_acc_dtype(diff.dtype))
            grp_out = y_c + (surr - surr.detach())               # value == y_c; grad via w into input/temps
        return self._route(grp_out, x).to(x.dtype)
