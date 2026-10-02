"""Cartridges — swappable lookup-math strategies behind the MultiHeadLUT contract."""
from .fused_manifesto_hard import FusedManifestoHardLUT
from .fused_manifesto_soft import FusedManifestoSoftLUT
from .manifesto_base import ManifestoLUT
from .manifesto_hard import ManifestoHardLUT
from .manifesto_soft import ManifestoSoftLUT
from .uncertainty import rational_uncertainty

__all__ = [
    "ManifestoLUT",
    "ManifestoHardLUT",
    "ManifestoSoftLUT",
    "FusedManifestoHardLUT",
    "FusedManifestoSoftLUT",
    "rational_uncertainty",
]
