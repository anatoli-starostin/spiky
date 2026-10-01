"""ManifestoHardLUT — the gen-1 reference cartridge.

"Hard forward, two-alternative soft backward with a rational uncertainty function."
Greenfield re-implementation of the original ``spiky.lutorch.multi_head_lut.MultiHeadLut``
(variant 1.1 in the LUT-ablation table), built to the cartridge contract with zero
imports from the old ``lutorch``.

Head routing (the shape-contract invariant)
===========================================
There are ``G = max(h_in, h_out)`` table groups, each with ``tph`` tables. Group ``g``
reads input head ``g % h_in`` (so a shared input when ``h_in == 1``) and writes output
head ``g % h_out`` (so a single shared output when ``h_out == 1``, into which *all* groups
are summed). This one mapping covers the three allowed patterns:
    h_in==h_out==H : group h reads x[:,h,:], writes y[:,h,:]
    h_in==1,h_out==H : every group reads x[:,0,:], group g writes y[:,g,:]
    h_in==H,h_out==1 : group g reads x[:,g,:], ALL groups sum into y[:,0,:]

The gen-1 math (unchanged by the routing)
=========================================
Per group ``g`` and table ``t``:

Addressing (sign-bit lookup).
    The table owns ``nap`` fixed anchor pairs ``(a_j, b_j)`` of coordinates in its
    ``d_in``-wide input vector; margins ``u_j = z[a_j] - z[b_j]``; the ``nap`` sign bits
    ``[u_j > eps]`` form, **LSB-first** (pair ``j`` carries ``2**j``), the integer address
    ``c_t`` into the table's ``K = 2**nap`` rows ``W_t in R^{K x d_out}``.

Hard forward.
    ``y_group = sum_t W_t[c_t]`` — one row per table, summed over the group's ``tph``
    tables. No score, no blend at eval.

Two-alternative soft backward.
    The forward **value** is exactly the hard read, but gradients flow as if the output
    were ``y~ = sum_t [ (1 - U_t) W_t[c_t] + U_t W_t[c_t'] ]``, where ``c_t' = c_t`` with
    the single bit of the **least-confident** pair flipped (``j* = argmin_j |u_j|``) and
    ``U_t = 0.5 / (1 + |u_{j*}|)`` (rational uncertainty: ``U(0)=0.5``, ``U->0`` as
    ``|u|->inf``, in ``(0, 0.5]``). ``c_t, c_t', j*`` are stop-gradient; the surrogate is
    C^1 across a bit flip and only jumps when ``j*`` switches pairs.

Fidelity note.
    As in the original ``smooth_mode=False`` path, the **weight-table** gradient is hard:
    only the addressed cell ``c_t`` receives gradient (coeff 1); the alternative ``c_t'``
    gets none. The uncertainty blend ``y~`` shapes the **input/addressing** gradient only.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from ..anchors import canonical_full_coverage_pairs
from ..lut_base import MultiHeadLUT
from ..lut_spec import LUTSpec
from .uncertainty import rational_uncertainty


class ManifestoHardLUT(MultiHeadLUT):
    """Gen-1 reference cartridge (see module docstring for the math and head routing)."""

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
        G, tph, nap, d_in, d_out = (
            spec.n_groups, spec.tph, spec.nap, spec.d_in, spec.d_out,
        )
        if d_in < 2:
            raise ValueError(f"ManifestoHardLUT needs d_in >= 2 to form anchor pairs, got {d_in}")
        self.cmp_eps = float(cmp_eps)

        # Fixed anchor pairs per (group, table), drawn once at init and frozen (gen-1: the
        # partition geometry is fixed, only the cells learn). Canonical full-coverage policy:
        # distinct canonical (a < b) pairs per table, covering the whole C(d_in, 2) pool.
        a, b = canonical_full_coverage_pairs(d_in, G, tph, nap, seed=seed)
        self.register_buffer("anchor_a", a)
        self.register_buffer("anchor_b", b)
        # LSB-first bit weights: pair j -> 2**j.
        self.register_buffer("powers", (1 << torch.arange(nap, dtype=torch.long)))

        # Group -> input/output head maps (the routing invariant).
        self.register_buffer("in_head", torch.arange(G, dtype=torch.long) % spec.h_in)
        self.register_buffer("out_head", torch.arange(G, dtype=torch.long) % spec.h_out)

        # Learnable cell tables: W[g, t, c, :], c in [0, K).
        wgen = torch.Generator().manual_seed(seed)
        w = torch.randn(G, tph, spec.n_cells, d_out, generator=wgen) * weight_init_std
        self.weights = nn.Parameter(w)

        if device is not None:
            self.to(device)

    def _read(self, idx: torch.Tensor) -> torch.Tensor:
        """Gather cell rows ``W[g, t, idx[b,g,t]]`` -> ``[B, G, tph, d_out]``."""
        B = idx.shape[0]
        G, tph, d_out, K = (
            self.spec.n_groups, self.spec.tph, self.spec.d_out, self.spec.n_cells,
        )
        idx_e = idx.unsqueeze(-1).unsqueeze(-1).expand(B, G, tph, 1, d_out)
        w_e = self.weights.unsqueeze(0).expand(B, G, tph, K, d_out)
        return w_e.gather(3, idx_e).squeeze(3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._check_input(x)  # [B, h_in, d_in]
        spec = self.spec
        G, tph, nap, d_in, d_out = (
            spec.n_groups, spec.tph, spec.nap, spec.d_in, spec.d_out,
        )
        B = x.shape[0]

        # Route each group to its input head: z[b, g, :] = x[b, g % h_in, :]  -> [B, G, d_in].
        z = x[:, self.in_head, :]

        # Margins u_j = z[a_j] - z[b_j] for every (group, table, pair) -> [B, G, tph, nap].
        idx_a = self.anchor_a.reshape(1, G, tph * nap).expand(B, G, tph * nap)
        idx_b = self.anchor_b.reshape(1, G, tph * nap).expand(B, G, tph * nap)
        z_a = z.gather(2, idx_a).reshape(B, G, tph, nap)
        z_b = z.gather(2, idx_b).reshape(B, G, tph, nap)
        u = z_a - z_b

        # Sign bits -> LSB-first address c_t (non-differentiable, stop-grad).
        bits = (u > self.cmp_eps).to(torch.long)
        c = (bits * self.powers).sum(dim=-1)  # [B, G, tph]

        # Least-confident pair j* and its signed margin; alternative = c with that bit flipped.
        j_star = u.abs().argmin(dim=-1)                          # [B, G, tph]
        u_star = u.gather(-1, j_star.unsqueeze(-1)).squeeze(-1)  # [B, G, tph] (differentiable)
        c_alt = c ^ self.powers[j_star]                          # [B, G, tph]

        y_hard = self._read(c)  # [B, G, tph, d_out] — value and (hard) weight gradient

        if self.training:
            y_alt = self._read(c_alt)
            # Straight-through composite: value == y_hard, but x sees the blend's gradient
            # via U(u_star). g detached -> weight table learns on the hard cell only.
            g = (y_hard - y_alt).detach()
            u_term = (-rational_uncertainty(u_star)).unsqueeze(-1) * g
            per_table = y_hard + (u_term - u_term.detach())
        else:
            per_table = y_hard

        grp_out = per_table.sum(dim=2)  # sum over tph -> [B, G, d_out]

        # Scatter groups to output heads: y[:, g % h_out, :] += grp_out[:, g, :]. When
        # h_out == 1 every group sums into head 0 (fan-in); otherwise it is a bijection.
        y = x.new_zeros(B, spec.h_out, d_out)
        y.index_add_(1, self.out_head, grp_out)
        return y
