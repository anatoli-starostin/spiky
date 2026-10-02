"""Rational uncertainty function, shared across cartridges.

A cartridge that blends its hard read with an alternative cell (for the backward pass)
weights the alternative by an *uncertainty* ``U(u)`` of the deciding margin ``u``: when
the margin is tiny the sign bit is unreliable, so the neighbour deserves weight; when the
margin is large the bit is certain, so the hard cell owns (almost) all the weight.
"""
from __future__ import annotations

import torch


def rational_uncertainty(margin: torch.Tensor) -> torch.Tensor:
    """``U(u) = 0.5 / (1 + |u|)`` — the rational uncertainty function.

    ``U(0) = 0.5`` (maximal uncertainty -> an even blend of the hard cell and its bit-flip
    neighbour); ``U -> 0`` as ``|u| -> inf`` (confident -> all weight on the hard cell);
    monotonically decreasing in ``|u|`` and bounded in ``(0, 0.5]``.
    """
    return 0.5 / (1.0 + margin.abs())
