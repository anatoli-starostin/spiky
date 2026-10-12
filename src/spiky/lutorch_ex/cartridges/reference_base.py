"""The reference cartridges' forward (spec rule 4): the maths in plain PyTorch, ``torch.compile``d on CUDA.

A mixin for the reference classes only (ManifestoHardLUT / ManifestoSoftLUT / ConfidenceLUT and its quantised
subclass). The shared base (:class:`~spiky.lutorch_ex.cartridges.manifesto_base.ManifestoLUT`) has no runnable forward
(spec amendment B), so a fused class, which inherits that base and not this mixin, has nowhere to fall back to.

On CUDA the eval forward is always compiled; the train forward only for a class that sets ``_COMPILE_TRAIN`` (the
confidence family, whose score read-out is a chain of small kernels Inductor fuses). Everything else, CPU included,
runs eager. ``LUTORCH_EX_NO_COMPILE=1`` turns compilation off. Each instance builds its compiled forwards lazily.

Low precision (spec amendment C): a bf16 / fp16 input or table runs the same maths in fp32, on fp32 views of the
low-precision parameters and of the input, the same contract as the fused classes. Every read and reduction
accumulates in fp32; each result is rounded exactly once: the output to the input dtype, and, through the backward of
the views, the input gradient to the input dtype and each parameter's gradient to that parameter's dtype.
"""
from __future__ import annotations

import torch
from torch.nn.utils.stateless import _reparametrize_module

from .manifesto_base import _COMPILE_ENABLED, _LOW_PRECISION


class ReferenceForward:
    """Mixin: ``forward`` = :meth:`_forward_impl`, compiled on CUDA (see the module docstring)."""

    #: Compile the TRAIN forward too (not just eval). False for the Manifesto references (their train step is
    #: memory-bound and eager is fastest); True for the confidence family.
    _COMPILE_TRAIN: bool = False
    #: ``dynamic=`` for the TRAIN compile: True traces one shape-agnostic graph; None (torch.compile's auto) specialises
    #: on the first shape, which fuses the quantised monolith's backward better.
    _COMPILE_TRAIN_DYNAMIC = True
    #: Whether bf16 / fp16 run (as fp32 maths, see the module docstring). False only for QuantisedConfidenceLUT,
    #: outside this spec's families: its precision contract belongs to the quantised-cartridges work.
    _ACCEPTS_LOW_PRECISION: bool = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lowp = {n: t for n, t in (*self.named_parameters(), *self.named_buffers()) if t.dtype in _LOW_PRECISION}
        if x.dtype not in _LOW_PRECISION and not lowp:
            return self._dispatch(x)
        if not self._ACCEPTS_LOW_PRECISION:
            raise TypeError(
                f"{type(self).__name__} does not support low precision: got input dtype {x.dtype} and weight dtype "
                f"{self.weights.dtype}; keep it (and ProjectionMHL wrapping it) in float32/float64.")
        # fp32 views: autograd rounds each leaf's fp32-accumulated gradient once, back to the leaf's own dtype.
        with _reparametrize_module(self, {n: t.float() for n, t in lowp.items()}):
            y = self._dispatch(x.float() if x.dtype in _LOW_PRECISION else x)
        return y.to(x.dtype)

    def _dispatch(self, x: torch.Tensor) -> torch.Tensor:
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
