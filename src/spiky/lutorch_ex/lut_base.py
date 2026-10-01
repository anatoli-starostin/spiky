"""MultiHeadLUT — the cartridge contract.

A *cartridge* is a swappable lookup-math strategy. Every cartridge subclasses
:class:`MultiHeadLUT` and honours a single, mode-agnostic **shape contract**; nothing
outside the cartridge (notably :class:`~spiky.lutorch_ex.projection.ProjectionMHL`) ever
reaches inside it. The base knows only the :class:`~spiky.lutorch_ex.lut_spec.LUTSpec`
and the shape contract — never anything cartridge-specific (addressing, blending,
quantisation, …).

Shape contract
--------------
A cartridge maps ``H`` heads, each turning a ``d_in``-vector into a ``d_out``-vector.
Input rank is preserved on output:

* **per-head (3-D) mode**: ``x: [B, H, d_in]  ->  [B, H, d_out]``. Head ``h`` reads its
  own ``d_in`` slice; heads are kept separate (H preserved).
* **flat (2-D) mode**: ``x: [B, H*d_in]  ->  [B, H*d_out]``. The flat input is viewed
  internally as ``H`` contiguous slices of width ``d_in`` (i.e. reshaped to
  ``[B, H, d_in]``), run through the per-head path, then flattened back.

Semantics: the module returns the **hard** lookup value at eval; in training the value
is still the hard read, but a surrogate gradient flows (how exactly is the cartridge's
business). A cartridge must accept either rank and return the matching rank.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from .lut_spec import LUTSpec


class MultiHeadLUT(nn.Module, ABC):
    """Abstract base for all cartridges — the contract, nothing else.

    Subclasses implement :meth:`forward` honouring the shape contract documented at the
    module level. ``__init__`` takes the :class:`LUTSpec` plus arbitrary
    ``**cartridge_kwargs`` consumed by the concrete cartridge.
    """

    def __init__(self, spec: LUTSpec, **cartridge_kwargs):
        super().__init__()
        self.spec = spec

    def _as_per_head(self, x: torch.Tensor) -> tuple[torch.Tensor, bool]:
        """Normalise an input to per-head ``[B, H, d_in]`` and report whether it was flat.

        Returns ``(x3, was_flat)``. Raises ``ValueError`` on a shape that violates the
        contract. Concrete cartridges can use this to avoid re-implementing the rank
        handling; re-flatten the result with :meth:`_restore_rank`.
        """
        H, d_in = self.spec.n_heads, self.spec.d_in
        if x.dim() == 2:
            B, f = x.shape
            if f != H * d_in:
                raise ValueError(
                    f"flat input width {f} != n_heads*d_in = {H}*{d_in} = {H * d_in}"
                )
            return x.reshape(B, H, d_in), True
        if x.dim() == 3:
            B, h, d = x.shape
            if h != H or d != d_in:
                raise ValueError(
                    f"per-head input shape {tuple(x.shape)} != [B, {H}, {d_in}]"
                )
            return x, False
        raise ValueError(f"expected a 2-D or 3-D input, got shape {tuple(x.shape)}")

    def _restore_rank(self, y3: torch.Tensor, was_flat: bool) -> torch.Tensor:
        """Flatten ``[B, H, d_out]`` back to ``[B, H*d_out]`` iff the input was flat."""
        if was_flat:
            B = y3.shape[0]
            return y3.reshape(B, self.spec.out_features)
        return y3

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - abstract
        """Map ``[B, d_in]``/``[B, H, d_in]`` to ``[B, d_out]``/``[B, H, d_out]``."""
        raise NotImplementedError
