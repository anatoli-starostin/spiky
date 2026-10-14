"""FusedManifestoSoftLUT — the fused twin of ManifestoSoftLUT, for both weight functions.

Same ``(1-w)`` / ``w`` two-cell blend (``weight="rational"`` or ``"softsign"``) as value and gradient, computed by the
hand-written CUDA kernels (``csrc/fused_manifesto.cu``: one forward and one backward kernel, nothing per table in
global memory, the soft-sign temperatures' gradients included). Those kernels are its only implementation: a
case they do not cover (a CPU input, a float64 input, a float16 or float64 table, nap > 16, d_in beyond int16
anchors, a d_out one CTA cannot cover) raises, naming the case, and so does a missing extension, naming the cause. The
reference class :class:`~spiky.lutorch_ex.cartridges.manifesto_soft.ManifestoSoftLUT` runs everywhere else, with the
same state_dict.

bfloat16 / float16 inputs and a bfloat16 table are supported: addressing in float32, every read and reduction
accumulated in float32, the output rounded to the input dtype once. ``knobs`` (a :class:`ManifestoCudaKnobs`) sets the
launch configuration.
"""
from __future__ import annotations

import torch

from ._fused_manifesto_cuda import MODE_SOFT, init_cuda, manifesto_cuda_forward
from .manifesto_base import ManifestoLUT


class FusedManifestoSoftLUT(ManifestoLUT):
    def __init__(self, spec, *, knobs=None, **kw):
        super().__init__(spec, **kw)
        init_cuda(self, knobs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward_impl(x)

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        return manifesto_cuda_forward(self, x, MODE_SOFT)
