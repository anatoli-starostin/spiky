"""LUTSpec — the immutable geometry shared by every cartridge.

A cartridge (a concrete :class:`~spiky.lutorch_ex.lut_base.MultiHeadLUT`) is fully
described, geometry-wise, by a ``LUTSpec``. The head structure is given by a *pair*
``(h_in, h_out)`` so that a cartridge can fan in or fan out across heads, not just map
head-for-head:

* ``h_in == h_out``  — per-head on both sides (the plain multi-head case; ``1, 1`` is the
  degenerate single-head case and is allowed).
* ``h_in == 1, h_out != 1``  — **fan-out**: one shared whole input feeds every output head.
* ``h_in != 1, h_out == 1``  — **fan-in**: each input head is read, and all contributions
  are summed into the single shared output.

Any other combination (both ``!= 1`` and unequal) is forbidden — there is no consistent
head-grouping for it. The number of table groups is ``n_groups = max(h_in, h_out)``.

Everything cartridge-specific (the lookup math, quantisation, …) lives in the cartridge;
the spec is just the shape skeleton the wrapper and the contract agree on.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LUTSpec:
    """Immutable description of a multi-head LUT's geometry.

    Attributes:
        h_in: number of input heads. ``1`` means the whole input is shared across groups.
        h_out: number of output heads. ``1`` means all groups sum into one shared output.
        tph: tables per group (``n_tables = max(h_in, h_out) * tph``).
        nap: anchor pairs per table. Each table holds ``K = 2 ** nap`` cells.
        d_in: per-head input width (how many features one table reads).
        d_out: per-head output width (how many features one table writes).
    """

    h_in: int
    h_out: int
    tph: int
    nap: int
    d_in: int
    d_out: int

    def __post_init__(self) -> None:
        for name in ("h_in", "h_out", "tph", "nap", "d_in", "d_out"):
            v = getattr(self, name)
            if not isinstance(v, int) or v < 1:
                raise ValueError(f"LUTSpec.{name} must be a positive int, got {v!r}")
        # Allowed head patterns only: equal, fan-out (h_in==1), or fan-in (h_out==1).
        if not (self.h_in == self.h_out or self.h_in == 1 or self.h_out == 1):
            raise ValueError(
                f"LUTSpec head pattern not allowed: h_in={self.h_in}, h_out={self.h_out}. "
                "Allowed: h_in == h_out (per-head), h_in == 1 (fan-out), or h_out == 1 (fan-in). "
                "Both != 1 and unequal is forbidden (no consistent head grouping)."
            )

    @property
    def n_groups(self) -> int:
        """Number of table groups, ``max(h_in, h_out)``."""
        return max(self.h_in, self.h_out)

    @property
    def n_tables(self) -> int:
        """Total number of lookup tables, ``n_groups * tph``."""
        return self.n_groups * self.tph

    @property
    def n_cells(self) -> int:
        """Cells per table, ``K = 2 ** nap``."""
        return 1 << self.nap

    @property
    def in_features(self) -> int:
        """Flattened input width, ``h_in * d_in`` (what compress must produce)."""
        return self.h_in * self.d_in

    @property
    def out_features(self) -> int:
        """Flattened output width, ``h_out * d_out`` (what decompress consumes)."""
        return self.h_out * self.d_out
