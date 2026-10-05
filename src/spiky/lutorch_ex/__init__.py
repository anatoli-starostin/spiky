"""lutorch_ex — a multi-head lookup-table (LUT) layer library.

A cartridge-based lookup-table feed-forward layer. The pieces:

* :class:`LUTSpec` — the immutable geometry (heads / tables / anchor pairs / widths).
* :class:`MultiHeadLUT` — the cartridge contract (an ABC + the shape contract).
* cartridges — swappable lookup-math strategies, e.g. :class:`ManifestoHardLUT` and its
  soft-forward sibling :class:`ManifestoSoftLUT` (both built on :class:`ManifestoLUT`).
* :class:`ProjectionMHL` — a compress/decompress bottleneck around any cartridge.
"""
from .cartridges import (
    ConfidenceLUT,
    QuantisedConfidenceLUT,
    DeployedQuantisedConfidenceLUT,
    FusedManifestoHardLUT,
    FusedManifestoSoftLUT,
    FusedSoftSignHardLUT,
    FusedSoftSignLUT,
    FusedSoftSignSmoothLUT,
    ManifestoHardLUT,
    ManifestoLUT,
    ManifestoSoftLUT,
    SoftSignHardLUT,
    SoftSignLUT,
    SoftSignSmoothLUT,
    cell_tv_penalty,
    rational_uncertainty,
)
from .lut_base import MultiHeadLUT
from .lut_spec import LUTSpec
from .projection import ProjectionMHL, SupportsDecompressBake
from .deploy import (
    SupportsDeploymentExport,
    export_deployment,
    load_deployment,
)

__all__ = [
    "LUTSpec",
    "MultiHeadLUT",
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
    "cell_tv_penalty",
    "ProjectionMHL",
    "SupportsDecompressBake",
    "rational_uncertainty",
    "export_deployment",
    "load_deployment",
    "DeployedQuantisedConfidenceLUT",
    "SupportsDeploymentExport",
]
