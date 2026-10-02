"""lutorch_ex — greenfield multi-head LUT library.

A clean, cartridge-based re-implementation of the lutorch lookup-table FFN, with zero
imports from the old ``spiky.lutorch``. The pieces:

* :class:`LUTSpec` — the immutable geometry (heads / tables / anchor pairs / widths).
* :class:`MultiHeadLUT` — the cartridge contract (an ABC + the shape contract).
* cartridges — swappable lookup-math strategies, e.g. :class:`ManifestoHardLUT` and its
  soft-forward sibling :class:`ManifestoSoftLUT` (both built on :class:`ManifestoLUT`).
* :class:`ProjectionMHL` — a compress/decompress bottleneck around any cartridge.
"""
from .cartridges import (
    ManifestoHardLUT,
    ManifestoLUT,
    ManifestoSoftLUT,
    rational_uncertainty,
)
from .lut_base import MultiHeadLUT
from .lut_spec import LUTSpec
from .projection import ProjectionMHL, SupportsDecompressBake

__all__ = [
    "LUTSpec",
    "MultiHeadLUT",
    "ManifestoLUT",
    "ManifestoHardLUT",
    "ManifestoSoftLUT",
    "ProjectionMHL",
    "SupportsDecompressBake",
    "rational_uncertainty",
]
