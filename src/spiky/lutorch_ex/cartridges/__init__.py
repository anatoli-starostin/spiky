"""Cartridges — swappable lookup-math strategies behind the MultiHeadLUT contract."""
from .confidence import ConfidenceLUT
from .quantised_confidence import QuantisedConfidenceLUT
from .fused_manifesto_hard import FusedManifestoHardLUT
from .fused_manifesto_soft import FusedManifestoSoftLUT
from .fused_softsign import FusedSoftSignHardLUT, FusedSoftSignLUT, FusedSoftSignSmoothLUT
from .manifesto_base import ManifestoLUT
from .manifesto_hard import ManifestoHardLUT
from .manifesto_soft import ManifestoSoftLUT
from .softsign_base import SoftSignLUT
from .softsign_hard import SoftSignHardLUT
from .softsign_smooth import SoftSignSmoothLUT
from .uncertainty import rational_uncertainty

__all__ = [
    "ManifestoLUT",
    "ManifestoHardLUT",
    "ManifestoSoftLUT",
    "FusedManifestoHardLUT",
    "FusedManifestoSoftLUT",
    "SoftSignLUT",
    "SoftSignHardLUT",
    "SoftSignSmoothLUT",
    "FusedSoftSignLUT",
    "FusedSoftSignHardLUT",
    "FusedSoftSignSmoothLUT",
    "ConfidenceLUT",
    "QuantisedConfidenceLUT",
    "rational_uncertainty",
]
