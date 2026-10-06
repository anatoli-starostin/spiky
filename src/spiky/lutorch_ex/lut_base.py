"""MultiHeadLUT — the cartridge contract.

A *cartridge* is a swappable lookup-math strategy. Every cartridge subclasses
:class:`MultiHeadLUT` and honours a single shape contract; nothing outside the cartridge
(notably :class:`~spiky.lutorch_ex.projection.ProjectionMHL`) ever reaches inside it. The
base knows only the :class:`~spiky.lutorch_ex.lut_spec.LUTSpec` and the shape contract.

Shape contract (always 3-D)
---------------------------
``x: [B, h_in, d_in]  ->  y: [B, h_out, d_out]``.

The unifying invariant: **a table always reads a ``d_in``-wide vector and writes a
``d_out``-wide vector.** The head pattern (:class:`LUTSpec`) says how those vectors are
routed, over ``n_groups = max(h_in, h_out)`` table groups; group ``g`` reads input head
``g % h_in`` and writes output head ``g % h_out``:

* ``h_in == h_out == H`` — group ``h`` reads ``x[:, h, :]`` and writes ``y[:, h, :]``.
* ``h_in == 1, h_out == H`` (fan-out) — every group reads the shared input ``x[:, 0, :]``
  and writes its own output head.
* ``h_in == H, h_out == 1`` (fan-in) — each group reads its own input head ``x[:, g, :]``
  and *all* group contributions are summed into the single shared output ``y[:, 0, :]``.

Semantics: what the value and the gradient are is the cartridge's business, and the families
differ. A *hard* cartridge (ManifestoHardLUT, SoftSignHardLUT and their fused twins) returns the
hard lookup value at train and eval, and in training lets a surrogate gradient flow through a
two-cell blend. A *soft* cartridge (ManifestoSoftLUT, SoftSignSmoothLUT, their fused twins) returns
that blend itself as the value, at train and eval. The confidence cartridges (ConfidenceLUT,
QuantisedConfidenceLUT) return a score-weighted read -- one cell, or a two-cell blend -- at train
and eval, and differentiate that value directly.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from .lut_spec import LUTSpec


class MultiHeadLUT(nn.Module, ABC):
    """Abstract base for all cartridges — the contract, nothing else.

    Subclasses implement :meth:`forward` honouring the 3-D shape contract documented at
    the module level. ``__init__`` takes the :class:`LUTSpec` plus arbitrary
    ``**cartridge_kwargs`` consumed by the concrete cartridge.
    """

    def __init__(self, spec: LUTSpec, **cartridge_kwargs):
        super().__init__()
        self.spec = spec

    def _check_input(self, x: torch.Tensor) -> None:
        """Validate the 3-D input shape ``[B, h_in, d_in]`` against the spec."""
        if x.dim() != 3:
            raise ValueError(f"expected a 3-D input [B, h_in, d_in], got shape {tuple(x.shape)}")
        if x.shape[1] != self.spec.h_in or x.shape[2] != self.spec.d_in:
            raise ValueError(
                f"input shape {tuple(x.shape)} != [B, {self.spec.h_in}, {self.spec.d_in}]"
            )

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - abstract
        """Map ``[B, h_in, d_in]`` to ``[B, h_out, d_out]``."""
        raise NotImplementedError
