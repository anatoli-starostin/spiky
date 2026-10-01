"""ManifestoHardLUT — the gen-1 reference cartridge.

"Hard forward, two-alternative soft backward with a rational uncertainty function."
This is the greenfield re-implementation of the original
``spiky.lutorch.multi_head_lut.MultiHeadLut`` (variant 1.1 in the LUT-ablation table),
built to the cartridge contract with zero imports from the old ``lutorch``.

The math
========
Per head ``h`` and table ``t`` (there are ``H`` heads, ``tph`` tables each):

Addressing (sign-bit lookup).
    The table owns ``nap`` anchor pairs ``(a_j, b_j)`` of coordinates in its head's
    ``d_in`` input slice. Each pair yields a signed margin ``u_j = z[a_j] - z[b_j]``.
    The ``nap`` sign bits ``[u_j > eps]`` form, **LSB-first** (pair ``j`` carries weight
    ``2**j``), the integer address ``c_t`` into the table's weight rows
    ``W_t in R^{K x d_out}`` with ``K = 2**nap``.

Hard forward.
    ``y = sum_t W_t[c_t]`` — a pure sign-addressed lookup, one row per table, summed
    over the head's ``tph`` tables. No score, no blend at eval.

Two-alternative soft backward.
    The forward **value** is exactly the hard read, but gradients flow as if the output
    were the two-cell blend

        y~ = sum_t [ (1 - U_t) * W_t[c_t] + U_t * W_t[c_t'] ]

    where ``c_t' = c_t`` with the single bit of the **least-confident** pair flipped
    (``j* = argmin_j |u_j|``), and ``U_t = U(u_{j*}) = 0.5 / (1 + |u_{j*}|)`` is the
    rational uncertainty (``uncertainty.rational_uncertainty``). ``c_t``, ``c_t'`` and
    ``j*`` are stop-gradient (the address is a non-differentiable argmax/sign), so only
    the margin ``u_{j*}`` (hence the input ``z``) sees a gradient. The surrogate is C^1
    across a bit flip (at ``u=0`` the two cells carry equal weight ``0.5`` and the hard
    and alt cells simply swap), and only jumps when ``j*`` switches to a different pair.

Fidelity note (see the PR description / report).
    In this hard-forward reference the **weight-table** gradient is the *hard* one: only
    the addressed cell ``c_t`` receives gradient (coefficient 1); the alternative
    ``c_t'`` receives none. The uncertainty blend ``y~`` shapes the **input/addressing**
    gradient only — exactly as the original gen-1 ``smooth_mode=False`` path does (the
    original scatters the weight gradient to the hard cell and routes the blend through
    its anchor-pair backward). Letting ``y~`` drive the weight gradient too (splitting it
    ``(1-U)`` / ``U`` across the two cells) *and* the forward value is the original's
    ``smooth_mode=True`` variant (1.2), which this cartridge does not implement.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from ..lut_base import MultiHeadLUT
from ..lut_spec import LUTSpec
from .uncertainty import rational_uncertainty


class ManifestoHardLUT(MultiHeadLUT):
    """Gen-1 reference cartridge (see module docstring for the math)."""

    def __init__(
        self,
        spec: LUTSpec,
        *,
        seed: int = 0,
        weight_init_std: float = 1e-3,
        cmp_eps: float = 0.0,
        device: Optional[torch.device] = None,
        **unused,
    ):
        super().__init__(spec)
        H, tph, nap, d_in, d_out = (
            spec.n_heads, spec.tph, spec.nap, spec.d_in, spec.d_out,
        )
        if d_in < 2:
            raise ValueError(f"ManifestoHardLUT needs d_in >= 2 to form anchor pairs, got {d_in}")
        self.cmp_eps = float(cmp_eps)

        gen = torch.Generator().manual_seed(seed)

        # Fixed anchor pairs per (head, table): nap distinct (a != b) coordinate pairs,
        # drawn once at init and frozen as buffers (the gen-1 philosophy: the partition
        # geometry is fixed, only the cell contents learn).
        a = torch.empty(H, tph, nap, dtype=torch.long)
        b = torch.empty(H, tph, nap, dtype=torch.long)
        for h in range(H):
            for t in range(tph):
                for j in range(nap):
                    aj = int(torch.randint(d_in, (1,), generator=gen).item())
                    bj = int(torch.randint(d_in, (1,), generator=gen).item())
                    while bj == aj:
                        bj = int(torch.randint(d_in, (1,), generator=gen).item())
                    a[h, t, j], b[h, t, j] = aj, bj
        self.register_buffer("anchor_a", a)
        self.register_buffer("anchor_b", b)
        # LSB-first bit weights: pair j -> 2**j.
        self.register_buffer("powers", (1 << torch.arange(nap, dtype=torch.long)))

        # Learnable cell tables: W[h, t, c, :], c in [0, K).
        w = torch.randn(H, tph, spec.n_cells, d_out, generator=gen) * weight_init_std
        self.weights = nn.Parameter(w)

        if device is not None:
            self.to(device)

    def _read(self, idx: torch.Tensor) -> torch.Tensor:
        """Gather cell rows ``W[h, t, idx[b,h,t]]`` -> ``[B, H, tph, d_out]``."""
        B = idx.shape[0]
        H, tph, d_out, K = (
            self.spec.n_heads, self.spec.tph, self.spec.d_out, self.spec.n_cells,
        )
        idx_e = idx.unsqueeze(-1).unsqueeze(-1).expand(B, H, tph, 1, d_out)
        w_e = self.weights.unsqueeze(0).expand(B, H, tph, K, d_out)
        return w_e.gather(3, idx_e).squeeze(3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, tph, nap, d_in = (
            self.spec.n_heads, self.spec.tph, self.spec.nap, self.spec.d_in,
        )
        z, was_flat = self._as_per_head(x)  # [B, H, d_in]
        B = z.shape[0]

        # Margins u_j = z[a_j] - z[b_j] for every (head, table, pair) -> [B, H, tph, nap].
        idx_a = self.anchor_a.reshape(1, H, tph * nap).expand(B, H, tph * nap)
        idx_b = self.anchor_b.reshape(1, H, tph * nap).expand(B, H, tph * nap)
        z_a = z.gather(2, idx_a).reshape(B, H, tph, nap)
        z_b = z.gather(2, idx_b).reshape(B, H, tph, nap)
        u = z_a - z_b

        # Sign bits -> LSB-first address c_t. Comparison is non-differentiable (stop-grad).
        bits = (u > self.cmp_eps).to(torch.long)
        c = (bits * self.powers).sum(dim=-1)  # [B, H, tph]

        # Least-confident pair j* and its signed margin; alternative address is c with
        # that single bit flipped. j*, c, c_alt are all stop-gradient (integer/argmax).
        j_star = u.abs().argmin(dim=-1)                       # [B, H, tph]
        u_star = u.gather(-1, j_star.unsqueeze(-1)).squeeze(-1)  # [B, H, tph] (differentiable)
        c_alt = c ^ self.powers[j_star]                       # flip bit j* -> [B, H, tph]

        y_hard = self._read(c)  # [B, H, tph, d_out] — value and (hard) weight gradient

        if self.training:
            y_alt = self._read(c_alt)
            # Straight-through composite: value == y_hard exactly, but x sees the blend's
            # gradient via U(u_star). g is detached so the weight table learns on the hard
            # cell only (the alternative gets no weight gradient — gen-1 hard-forward).
            g = (y_hard - y_alt).detach()                     # [B, H, tph, d_out]
            u_term = (-rational_uncertainty(u_star)).unsqueeze(-1) * g
            per_table = y_hard + (u_term - u_term.detach())
        else:
            per_table = y_hard

        head_out = per_table.sum(dim=2)  # sum over tph -> [B, H, d_out]
        return self._restore_rank(head_out, was_flat)
