"""FusedSoftSignSmoothLUT — GPU-efficient twin of SoftSignSmoothLUT (Gen-2 variant 2.4).

Same two-cell learned-temperature blend, numerically equivalent to the unfused oracle, but the
read is fused:

* eval  -> the pure two-cell read + blend (compiled by the base ``forward`` on CUDA);
* train -> one ``F.embedding_bag`` with ``per_sample_weights = [1-w, w]`` (fuses read+scale+sum),
  with fp32 accumulation and bf16/fp16 support.

The Gen-1 ``native`` lutorch_cuda path is NOT reused: those kernels bake in the Gen-1 inverse-L1
uncertainty and its derivative, not this learned-temperature sigmoid, so they cannot back Gen-2.
The blend weight ``w`` is a fully differentiable function of the margins and the two temperatures,
so autograd produces the exact 2-alternative backward (input, temperatures and both cell rows) and
fp32 accumulation comes from the embedding_bag read running on the fp32-upcast table — no custom
autograd.Function is needed. See the module docstring of ``softsign_base``.
"""
from __future__ import annotations

import torch

from ._fused_ops import fused_blend_read, _acc_dtype
from .softsign_base import SoftSignLUT

_LOW_PREC = (torch.bfloat16, torch.float16)


class FusedSoftSignSmoothLUT(SoftSignLUT):
    def _supports_low_precision(self) -> bool:
        return True  # bf16/fp16: fp32 addressing + fp32-accumulated embedding_bag read

    def _combine(self, y_hard, y_alt, u_abs_star):  # pragma: no cover - forward is overridden
        raise NotImplementedError

    def _pure_blend(self, c, c_alt, w):
        y_hard, y_alt = self._read_pair(c, c_alt)
        blend = y_hard + w.unsqueeze(-1) * (y_alt - y_hard)
        return blend.sum(dim=2, dtype=_acc_dtype(blend.dtype))  # fp32-accum for bf16/fp16

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        low = x.dtype in _LOW_PREC
        # Addressing in fp32 for bf16/fp16 so the bit decisions don't flip; cast out once at end.
        z, u, c, j_star, u_abs_star, c_alt = self._addresses(x.float() if low else x)
        w = self._blend_w(u_abs_star)
        if self.training:
            grp_out = fused_blend_read(self.weights, c, c_alt, w)
        else:
            grp_out = self._pure_blend(c, c_alt, w)
        return self._route(grp_out, x).to(x.dtype)
