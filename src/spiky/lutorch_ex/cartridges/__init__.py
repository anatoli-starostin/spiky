"""Cartridges — swappable lookup-math strategies behind the MultiHeadLUT contract."""
from .manifesto_hard import ManifestoHardLUT
from .uncertainty import rational_uncertainty, rational_uncertainty_quadratic

__all__ = [
    "ManifestoHardLUT",
    "rational_uncertainty",
    "rational_uncertainty_quadratic",
]
