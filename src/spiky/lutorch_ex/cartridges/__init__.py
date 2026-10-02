"""Cartridges — swappable lookup-math strategies behind the MultiHeadLUT contract."""
from .manifesto_base import ManifestoLUT
from .manifesto_hard import ManifestoHardLUT
from .manifesto_soft import ManifestoSoftLUT
from .uncertainty import rational_uncertainty

__all__ = [
    "ManifestoLUT",
    "ManifestoHardLUT",
    "ManifestoSoftLUT",
    "rational_uncertainty",
]
