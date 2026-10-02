"""FusedSoftSignHardLUT — GPU-efficient twin of SoftSignHardLUT (Gen-2 variant 2.3).

Same hard forward (``y = sum_t W[c_t]``) + 2-alternative soft-surrogate backward, with the read
fused:

* eval  -> one ``F.embedding_bag`` hard read (compiled by the base ``forward`` on CUDA);
* train -> the embedding_bag hard value plus the straight-through surrogate: value stays the hard
  read (weight grad is a 1-row scatter at ``c_t``), while the detached cell difference times the
  learned-temperature weight ``w`` carries the input/temperature gradient.

fp32 accumulation + bf16/fp16 support come from the embedding_bag read running on the fp32-upcast
table; the surrogate weight ``w`` is differentiable, so autograd gives the exact 2-alt backward.
Like the smooth twin, the Gen-1 ``native`` kernels (fixed inverse-L1 uncertainty) do not apply to
Gen-2, so no native backend is used.
"""
from __future__ import annotations

import torch

from ._fused_ops import fused_hard_read, _acc_dtype
from .softsign_base import SoftSignLUT

_LOW_PREC = (torch.bfloat16, torch.float16)


class FusedSoftSignHardLUT(SoftSignLUT):
    def _supports_low_precision(self) -> bool:
        return True  # bf16/fp16: fp32 addressing + fp32-accumulated embedding_bag read

    def _needs_alt(self) -> bool:
        return self.training  # eval reads only c_t

    def _combine(self, y_hard, y_alt, u_abs_star):  # pragma: no cover - forward is overridden
        raise NotImplementedError

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        low = x.dtype in _LOW_PREC
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
