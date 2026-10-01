"""Cartridges — swappable lookup-math strategies behind the MultiHeadLUT contract."""
from .manifesto_hard import ManifestoHardLUT
from .uncertainty import rational_uncertainty

__all__ = [
    "ManifestoHardLUT",
    "rational_uncertainty",
]
