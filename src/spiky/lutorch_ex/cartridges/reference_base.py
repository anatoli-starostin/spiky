"""The reference cartridges' forward (spec rule 4): the maths in plain PyTorch, ``torch.compile``d on CUDA.

A mixin for the reference classes only (ManifestoHardLUT / ManifestoSoftLUT / ConfidenceLUT and its quantised
subclass). The shared base (:class:`~spiky.lutorch_ex.cartridges.manifesto_base.ManifestoLUT`) has no runnable forward
(spec amendment B), so a fused class, which inherits that base and not this mixin, has nowhere to fall back to.

On CUDA the eval forward is always compiled; the train forward only for a class that sets ``_COMPILE_TRAIN`` (the
confidence family, whose score read-out is a chain of small kernels Inductor fuses). Everything else, CPU included,
runs eager. ``LUTORCH_EX_NO_COMPILE=1`` turns compilation off. Each instance builds its compiled forwards lazily.
"""
from __future__ import annotations

import torch

from .manifesto_base import _COMPILE_ENABLED, _LOW_PRECISION


class ReferenceForward:
    """Mixin: ``forward`` = :meth:`_forward_impl`, compiled on CUDA (see the module docstring)."""

    #: Compile the TRAIN forward too (not just eval). False for the Manifesto references (their train step is
    #: memory-bound and eager is fastest); True for the confidence family.
    _COMPILE_TRAIN: bool = False
    #: ``dynamic=`` for the TRAIN compile: True traces one shape-agnostic graph; None (torch.compile's auto) specialises
    #: on the first shape, which fuses the quantised monolith's backward better.
    _COMPILE_TRAIN_DYNAMIC = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Low precision is a fused-cartridge feature for now; the references reject it with a clear error rather than
        # silently running lossy bf16 maths.
        if x.dtype in _LOW_PRECISION or self.weights.dtype in _LOW_PRECISION:
            raise TypeError(
                f"{type(self).__name__} does not support low precision: got input dtype "
                f"{x.dtype} and weight dtype {self.weights.dtype}. bf16/fp16 is supported only by "
                "the fused cartridges (FusedManifestoHardLUT / FusedManifestoSoftLUT / FusedConfidenceLUT); keep this "
                "cartridge (and ProjectionMHL wrapping it) in float32/float64."
            )
        if _COMPILE_ENABLED and x.is_cuda:
            if not self.training:
                if self._compiled is None:
                    self._compiled = torch.compile(self._forward_impl, dynamic=True)
                return self._compiled(x)
            if self._COMPILE_TRAIN:
                if self._compiled_train is None:
                    self._compiled_train = torch.compile(self._forward_impl, dynamic=self._COMPILE_TRAIN_DYNAMIC)
                return self._compiled_train(x)
        return self._forward_impl(x)
