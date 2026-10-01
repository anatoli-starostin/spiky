"""LUTSpec — the immutable geometry shared by every cartridge.

A cartridge (a concrete :class:`~spiky.lutorch_ex.lut_base.MultiHeadLUT`) is fully
described, geometry-wise, by a ``LUTSpec``: how many heads, how many tables per head,
how many anchor pairs (hence cells) per table, and the per-head input / output widths.
Everything cartridge-specific (the lookup math, quantisation, …) lives in the cartridge;
the spec is just the shape skeleton the wrapper and the contract agree on.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LUTSpec:
    """Immutable description of a multi-head LUT's geometry.

    Attributes:
        n_heads: H — number of heads. The total number of lookup tables is ``P = n_heads * tph``.
        tph: tables per head.
        nap: anchor pairs per table. Each table holds ``K = 2 ** nap`` cells, one per
            sign pattern of its ``nap`` margins.
        d_in: per-head input width (how many features one head reads).
        d_out: per-head output width (how many features one head emits).
    """

    n_heads: int
    tph: int
    nap: int
    d_in: int
    d_out: int

    def __post_init__(self) -> None:
        for name in ("n_heads", "tph", "nap", "d_in", "d_out"):
            v = getattr(self, name)
            if not isinstance(v, int) or v < 1:
                raise ValueError(f"LUTSpec.{name} must be a positive int, got {v!r}")

    @property
    def n_tables(self) -> int:
        """Total number of lookup tables, ``P = n_heads * tph``."""
        return self.n_heads * self.tph

    @property
    def n_cells(self) -> int:
        """Cells per table, ``K = 2 ** nap``."""
        return 1 << self.nap

    @property
    def in_features(self) -> int:
        """Flattened per-sample input width in flat (2-D) mode, ``n_heads * d_in``."""
        return self.n_heads * self.d_in

    @property
    def out_features(self) -> int:
        """Flattened per-sample output width in flat (2-D) mode, ``n_heads * d_out``."""
        return self.n_heads * self.d_out
