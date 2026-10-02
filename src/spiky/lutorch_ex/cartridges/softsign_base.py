"""Shared structure for the Gen-2 "soft-sign" cartridge family (FastMHL math, 2-alternative).

The Gen-2 cartridges keep everything the Gen-1 Manifesto family has — MSB-first sign-bit
addressing, the single least-confident Hamming-1 neighbour ``c'`` (``j* = argmin_j |u_j|``),
the two-cell read, head routing — and change only the blend/surrogate *weight*. Where Gen-1
uses the fixed rational uncertainty ``U(u) = 0.5/(1+|u|)``, Gen-2 uses a squashed-margin sigmoid
with two **learned per-layer temperatures** (exactly FastMultiHeadLut's ``hybrid_smooth`` weight):

    rho = |u_{j*}| / (T_soft_score + |u_{j*}|)     (a rational soft-sign of the deciding margin)
    w   = sigmoid(-2 * rho / T_select)  in (0, 1/2]  (the weight on the neighbour cell c')

so the two-cell blend is ``(1 - w) W[c] + w W[c']``, continuous across a bit flip (``w = 1/2`` at
``u = 0``) and tending to the hard read as the margin grows. ``T_soft_score`` and ``T_select`` are
stored log-parametrised and learned per layer (initialised at 0.5, as in FastMHL).

This is the **2-alternative** backward only (``backward_topk = 1``): the surrogate softmax is
restricted to the two cells ``{c, c'}``, whose renormalised weight on ``c'`` is exactly ``w`` — so
the backward blend coincides with the forward blend. The all-K softmax backward
(``backward_topk = 0``), ``n_alternatives > 1``, the WTA primitive and ``exp_outputs`` are
deliberately out of scope.

Two cartridges subclass this base (both use ``F.embedding_bag`` for the fused train read,
eval-only torch.compile on CUDA, a plain-autograd backward and bf16/fp16 support):
* :class:`~spiky.lutorch_ex.cartridges.softsign_hard.SoftSignHardLUT` (variant 2.3): hard forward,
  2-alt soft-surrogate backward.
* :class:`~spiky.lutorch_ex.cartridges.softsign_smooth.SoftSignSmoothLUT` (variant 2.4): the
  two-cell blend as both forward value and gradient.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from .manifesto_base import ManifestoLUT


class SoftSignLUT(ManifestoLUT):
    """Base for the Gen-2 soft-sign cartridges: the learned-temperature blend weight ``w``.

    Adds the two learned per-layer temperatures to the shared Manifesto addressing and provides
    :meth:`_blend_w`. Subclasses override ``_forward_impl`` (embedding_bag read + the hard STE or
    the two-cell blend). Abstract like :class:`ManifestoLUT` — not instantiated directly.
    """

    def __init__(
        self,
        spec,
        *,
        soft_score_temp: float = 0.5,
        select_temp: float = 0.5,
        learnable_temps: bool = True,
        **kw,
    ):
        super().__init__(spec, **kw)
        ls = math.log(float(soft_score_temp))
        lx = math.log(float(select_temp))
        if learnable_temps:
            self.log_soft_score_temp = nn.Parameter(torch.tensor(ls))
            self.log_select_temp = nn.Parameter(torch.tensor(lx))
        else:
            self.register_buffer("log_soft_score_temp", torch.tensor(ls))
            self.register_buffer("log_select_temp", torch.tensor(lx))

    def _blend_w(self, u_abs_star: torch.Tensor) -> torch.Tensor:
        """Weight on the neighbour cell ``c'``: ``w = sigmoid(-2*rho/T_sel)`` in ``(0, 1/2]``.

        ``rho = |u_{j*}| / (T_s + |u_{j*}|)``. Temperatures are taken in the margin's dtype so the
        blend runs in the addressing precision (fp32 for the fused bf16 path, fp32/fp64 otherwise);
        a no-op for a matching-dtype model.
        """
        t_s = self.log_soft_score_temp.to(u_abs_star.dtype).exp()
        t_sel = self.log_select_temp.to(u_abs_star.dtype).exp()
        rho = u_abs_star / (t_s + u_abs_star)
        return torch.sigmoid(-2.0 * rho / t_sel)
