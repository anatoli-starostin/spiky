"""Shared scaffolding of the confidence family: the Gen-3 parameters, nothing that runs.

:class:`ConfidenceBase` adds to :class:`~spiky.lutorch_ex.cartridges.manifesto_base.ManifestoLUT` (tables, anchors,
addressing) the confidence score's learned temperatures ``beta`` / ``gamma`` and, at ``read_top_n = 2``, the blend
temperature ``tau``, all stored log-parametrised. The reference
(:class:`~spiky.lutorch_ex.cartridges.confidence.ConfidenceLUT`, and its quantised subclass) and the fused twin
(:class:`~spiky.lutorch_ex.cartridges.fused_confidence.FusedConfidenceLUT`) both build on it; neither inherits the
other's forward. The maths is in each of them.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn

from .manifesto_base import ManifestoLUT


class ConfidenceBase(ManifestoLUT):
    """Gen-3 confidence-scored cartridge state; see the module docstring.

    Args (beyond the shared :class:`ManifestoLUT` ones):
        read_top_n: required, no default: 1 (single cell, row 3.1) or 2 (two-cell τ-blend, row 3.2 champion). Every
            construction states the read width.
        beta_init / gamma_init: initial β, γ (stored log-parametrised; champion (β,γ)=(2,1)).
        read_tau_init: initial τ (default 0.5, the champion). Only valid at read_top_n=2.
        read_tau_learnable: whether τ is a learned nn.Parameter (default True) or a frozen buffer. Only valid at
            read_top_n=2.
        learnable_score: whether β, γ are learned nn.Parameters (default True) or frozen buffers.

    τ is the temperature of the two-cell blend σ(−2|u_j*|/τ); the read_top_n=1 read has no blend and no τ. Passing
    read_tau_init or read_tau_learnable with read_top_n=1 raises ValueError rather than being silently ignored.
    """

    _HAS_UNCERTAINTY_FUNC = False    # scored cells: no uncertainty_func= and no soft-sign temperatures

    def __init__(
        self,
        spec,
        *,
        read_top_n: int,
        beta_init: float = 2.0,
        gamma_init: float = 1.0,
        read_tau_init: Optional[float] = None,
        read_tau_learnable: Optional[bool] = None,
        learnable_score: bool = True,
        **kw,
    ):
        super().__init__(spec, **kw)
        if read_top_n not in (1, 2):
            raise ValueError(f"read_top_n must be 1 or 2, got {read_top_n}")
        # None = not given: the τ arguments exist only for the two-cell blend, so giving either at n=1 is a mistake (it
        # used to be accepted and dropped without a word).
        given = {k: v for k, v in (("read_tau_init", read_tau_init), ("read_tau_learnable", read_tau_learnable))
                 if v is not None}
        if read_top_n == 1 and given:
            args = ", ".join(f"{k}={v!r}" for k, v in given.items())
            raise ValueError(
                f"{type(self).__name__}: {args} given with read_top_n=1, where it has no effect. τ is the temperature "
                f"of the two-cell blend σ(-2|u_j*|/τ) between a table's cell and its least-confident neighbour; the "
                f"read_top_n=1 read takes one cell per table, has no blend and therefore no τ (no log_read_tau is "
                f"created). Pass read_top_n=2 if you meant the τ-blend read, or drop "
                f"{'these arguments' if len(given) > 1 else 'the argument'} for the single-cell read.")
        read_tau_init = 0.5 if read_tau_init is None else read_tau_init
        read_tau_learnable = True if read_tau_learnable is None else read_tau_learnable
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
