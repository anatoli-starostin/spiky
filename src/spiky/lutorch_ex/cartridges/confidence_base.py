"""Shared scaffolding of the confidence family (spec amendment A): the Gen-3 parameters, nothing that runs.

:class:`ConfidenceBase` adds to :class:`~spiky.lutorch_ex.cartridges.manifesto_base.ManifestoLUT` (tables, anchors,
addressing) the confidence score's learned temperatures ``beta`` / ``gamma`` and, at ``read_top_n = 2``, the blend
temperature ``tau``, all stored log-parametrised. The reference
(:class:`~spiky.lutorch_ex.cartridges.confidence.ConfidenceLUT`, and its quantised subclass) and the fused twin
(:class:`~spiky.lutorch_ex.cartridges.fused_confidence.FusedConfidenceLUT`) both build on it; neither inherits the
other's forward (amendment B). The maths is in each of them.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from .manifesto_base import ManifestoLUT


class ConfidenceBase(ManifestoLUT):
    """Gen-3 confidence-scored cartridge state; see the module docstring.

    Args (beyond the shared :class:`ManifestoLUT` ones):
        read_top_n: 1 (single cell, row 3.1) or 2 (two-cell τ-blend, row 3.2 champion).
        beta_init / gamma_init: initial β, γ (stored log-parametrised; champion (β,γ)=(2,1)).
        read_tau_init: initial τ for read_top_n=2 (champion 0.5).
        read_tau_learnable: whether τ is a learned nn.Parameter (default True) or a frozen buffer.
        learnable_score: whether β, γ are learned nn.Parameters (default True) or frozen buffers.
    """

    _WEIGHTED = False            # scored cells: no neighbour-weight function (weight= / soft-sign temperatures)

    def __init__(
        self,
        spec,
        *,
        read_top_n: int = 1,
        beta_init: float = 2.0,
        gamma_init: float = 1.0,
        read_tau_init: float = 0.5,
        read_tau_learnable: bool = True,
        learnable_score: bool = True,
        **kw,
    ):
        super().__init__(spec, **kw)
        if read_top_n not in (1, 2):
            raise ValueError(f"read_top_n must be 1 or 2, got {read_top_n}")
        if not (beta_init > 0 and gamma_init > 0 and read_tau_init > 0):
            raise ValueError(
                f"beta_init, gamma_init, read_tau_init must be > 0, got "
                f"({beta_init}, {gamma_init}, {read_tau_init})"
            )
        self.read_top_n = int(read_top_n)

        # Learned margin-score temperatures, log-parametrised so β, γ stay positive.
        lb, lg = math.log(float(beta_init)), math.log(float(gamma_init))
        if learnable_score:
            self.confidence_log_beta = nn.Parameter(torch.tensor(lb))
            self.confidence_log_gamma = nn.Parameter(torch.tensor(lg))
        else:
            self.register_buffer("confidence_log_beta", torch.tensor(lb))
            self.register_buffer("confidence_log_gamma", torch.tensor(lg))

        # Two-cell blend temperature τ (only for read_top_n == 2).
        if self.read_top_n == 2:
            lt = math.log(float(read_tau_init))
            if read_tau_learnable:
                self.log_read_tau = nn.Parameter(torch.tensor(lt))
            else:
                self.register_buffer("log_read_tau", torch.tensor(lt))
