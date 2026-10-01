"""Rational uncertainty functions, shared across cartridges.

A cartridge that blends its hard read with an alternative cell (for the backward pass)
weights the alternative by an *uncertainty* ``U(u)`` of the deciding margin ``u``: when
the margin is tiny the sign bit is unreliable, so the neighbour deserves weight; when
the margin is large the bit is certain, so the hard cell owns (almost) all the weight.

Both functions here are **rational** in ``u`` (hence smooth and cheap), map ``0 -> 0.5``
(maximal uncertainty: an even blend) and decay to ``0`` as ``|u| -> inf``, staying in
``(0, 0.5]``. They differ only in how fast they decay.
"""
from __future__ import annotations

import torch


def rational_uncertainty(margin: torch.Tensor) -> torch.Tensor:
    """``U(u) = 0.5 / (1 + |u|)`` — the INVERSE-L1 uncertainty (the gen-1 default).

    ``U(0) = 0.5`` (maximal uncertainty -> even blend of the hard cell and its
    bit-flip neighbour); ``U -> 0`` as ``|u| -> inf`` (confident -> all weight on the
    hard cell); monotonically decreasing in ``|u|`` and bounded in ``(0, 0.5]``.
    """
    return 0.5 / (1.0 + margin.abs())


def rational_uncertainty_quadratic(margin: torch.Tensor) -> torch.Tensor:
    """``U(u) = 0.5 / (1 + u^2)`` — the INVERSE-QUADRATIC variant.

    Same endpoints as :func:`rational_uncertainty` (``U(0)=0.5``, ``U->0`` at infinity,
    range ``(0, 0.5]``) but a sharper, quadratic decay away from the decision boundary.
    """
    return 0.5 / (1.0 + margin * margin)
