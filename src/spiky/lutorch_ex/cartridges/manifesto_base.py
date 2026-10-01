"""Shared structure for the Manifesto cartridge family (hard and soft variants).

Both Manifesto cartridges share everything except the final step — how the addressed
cell ``c_t`` is combined with its least-confident-bit-flip neighbour ``c_t'``:

* :class:`~spiky.lutorch_ex.cartridges.manifesto_hard.ManifestoHardLUT` — hard value with
  a straight-through surrogate gradient (gen-1 variant 1.1, ``smooth_mode=False``).
* :class:`~spiky.lutorch_ex.cartridges.manifesto_soft.ManifestoSoftLUT` — the ``(1-U)``/``U``
  two-cell blend as both value and gradient (gen-1 variant 1.2, ``smooth_mode=True``).

Everything else lives here and is identical between them:

- **Anchor pairs**: canonical full coverage (``anchors.canonical_full_coverage_pairs``),
  frozen index buffers ``anchor_a``/``anchor_b`` of shape ``[G, tph, nap]``.
- **Addressing**: MSB-first sign-bit packing (``addressing.msb_first_powers``). Each table's
  ``nap`` margins ``u_j = z[a_j] - z[b_j]`` give sign bits ``[u_j > eps]`` packed (pair 0 =
  high bit) into the cell index ``c_t``; the neighbour ``c_t' = c_t`` with the bit of the
  least-confident pair ``j* = argmin_j |u_j|`` flipped.
- **Head routing** (the shape-contract invariant): ``G = max(h_in, h_out)`` groups, group
  ``g`` reads input head ``g % h_in`` and writes output head ``g % h_out``, groups summed
  into the output (fan-in sums all groups into head 0).

Subclasses implement only :meth:`_combine`.
"""
from __future__ import annotations

from abc import abstractmethod
from typing import Optional

import torch
import torch.nn as nn

from ..addressing import msb_first_powers
from ..anchors import canonical_full_coverage_pairs
from ..lut_base import MultiHeadLUT
from ..lut_spec import LUTSpec


class ManifestoLUT(MultiHeadLUT):
    """Base for the Manifesto cartridges: addressing + two-cell structure + routing.

    Abstract: subclasses supply :meth:`_combine` to turn the two addressed cells
    (``y_hard`` = ``W[c_t]``, ``y_alt`` = ``W[c_t']``) and the deciding margin ``u_star``
    into the per-table output.
    """

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
            raise ValueError(f"Manifesto cartridges need d_in >= 2 to form anchor pairs, got {d_in}")
        self.cmp_eps = float(cmp_eps)

        # Fixed anchor pairs per (group, table): canonical full-coverage policy (distinct
        # canonical a<b pairs per table, covering the whole C(d_in,2) pool). Frozen buffers.
        a, b = canonical_full_coverage_pairs(d_in, G, tph, nap, seed=seed)
        self.register_buffer("anchor_a", a)
        self.register_buffer("anchor_b", b)
        # MSB-first bit weights: pair 0 -> high bit 2**(nap-1).
        self.register_buffer("powers", msb_first_powers(nap))
        # Group -> input/output head maps (the routing invariant).
        self.register_buffer("in_head", torch.arange(G, dtype=torch.long) % spec.h_in)
        self.register_buffer("out_head", torch.arange(G, dtype=torch.long) % spec.h_out)

        # Learnable cell tables: W[g, t, c, :], c in [0, K).
        wgen = torch.Generator().manual_seed(seed)
        w = torch.randn(G, tph, spec.n_cells, d_out, generator=wgen) * weight_init_std
        self.weights = nn.Parameter(w)

    def _read(self, idx: torch.Tensor) -> torch.Tensor:
        """Gather cell rows ``W[g, t, idx[b,g,t]]`` -> ``[B, G, tph, d_out]``."""
        B = idx.shape[0]
        G, tph, d_out, K = (
            self.spec.n_groups, self.spec.tph, self.spec.d_out, self.spec.n_cells,
        )
        idx_e = idx.unsqueeze(-1).unsqueeze(-1).expand(B, G, tph, 1, d_out)
        w_e = self.weights.unsqueeze(0).expand(B, G, tph, K, d_out)
        return w_e.gather(3, idx_e).squeeze(3)

    @abstractmethod
    def _combine(
        self, y_hard: torch.Tensor, y_alt: torch.Tensor, u_star: torch.Tensor
    ) -> torch.Tensor:  # pragma: no cover - abstract
        """Combine the two addressed cells into the per-table output ``[B, G, tph, d_out]``.

        Args:
            y_hard: ``W[c_t]`` — the hard-addressed cell, ``[B, G, tph, d_out]``.
            y_alt: ``W[c_t']`` — the least-confident-bit-flip neighbour, ``[B, G, tph, d_out]``.
            u_star: the deciding (least-confident) signed margin, ``[B, G, tph]``.
        """
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._check_input(x)  # [B, h_in, d_in]
        spec = self.spec
        G, tph, nap, d_out = spec.n_groups, spec.tph, spec.nap, spec.d_out
        B = x.shape[0]

        # Route each group to its input head: z[b, g, :] = x[b, g % h_in, :]  -> [B, G, d_in].
        z = x[:, self.in_head, :]

        # Margins u_j = z[a_j] - z[b_j] for every (group, table, pair) -> [B, G, tph, nap].
        idx_a = self.anchor_a.reshape(1, G, tph * nap).expand(B, G, tph * nap)
        idx_b = self.anchor_b.reshape(1, G, tph * nap).expand(B, G, tph * nap)
        z_a = z.gather(2, idx_a).reshape(B, G, tph, nap)
        z_b = z.gather(2, idx_b).reshape(B, G, tph, nap)
        u = z_a - z_b

        # Sign bits -> MSB-first address c_t (non-differentiable, stop-grad).
        bits = (u > self.cmp_eps).to(torch.long)
        c = (bits * self.powers).sum(dim=-1)  # [B, G, tph]

        # Least-confident pair j* and its signed margin; neighbour = c with that bit flipped.
        j_star = u.abs().argmin(dim=-1)                          # [B, G, tph]
        u_star = u.gather(-1, j_star.unsqueeze(-1)).squeeze(-1)  # [B, G, tph] (differentiable)
        c_alt = c ^ self.powers[j_star]                          # [B, G, tph]

        y_hard = self._read(c)
        y_alt = self._read(c_alt)
        per_table = self._combine(y_hard, y_alt, u_star)         # [B, G, tph, d_out]

        grp_out = per_table.sum(dim=2)  # sum over tph -> [B, G, d_out]

        # Scatter groups to output heads (fan-in sums into head 0; else a bijection).
        y = x.new_zeros(B, spec.h_out, d_out)
        y.index_add_(1, self.out_head, grp_out)
        return y
