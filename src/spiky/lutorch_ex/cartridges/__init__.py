"""Cartridges — swappable lookup-math strategies behind the MultiHeadLUT contract."""
from .confidence import ConfidenceLUT
from .fused_confidence import FusedConfidenceLUT
from .quantised_confidence import DeployedQuantisedConfidenceLUT, QuantisedConfidenceLUT
from .fused_manifesto_hard import FusedManifestoHardLUT
from .fused_manifesto_soft import FusedManifestoSoftLUT
from .manifesto_base import ManifestoLUT, cell_tv_penalty
from .manifesto_hard import ManifestoHardLUT
from .manifesto_soft import ManifestoSoftLUT
from .uncertainty import rational_uncertainty

__all__ = [
    "ManifestoLUT",
    "ManifestoHardLUT",
    "ManifestoSoftLUT",
    "FusedConfidenceLUT",
    "FusedManifestoHardLUT",
    "FusedManifestoSoftLUT",
    "ConfidenceLUT",
    "QuantisedConfidenceLUT",
    "DeployedQuantisedConfidenceLUT",
    "rational_uncertainty",
    "cell_tv_penalty",
]
