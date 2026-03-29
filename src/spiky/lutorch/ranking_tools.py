"""
Rank-based projections and attention using random upper-triangular index pairs.

Projects head dimensions through pairwise comparisons (soft, hard, or STE) for
use in linear layers or scaled dot-product attention.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _sample_upper_tri_pairs(
    d_head: int,
    M: Optional[int],
    generator: Optional[torch.Generator] = None,
) -> tuple[int, torch.Tensor]:
    assert d_head >= 2, f"d_head must be >= 2, got {d_head}"
    max_pairs = d_head * (d_head - 1) // 2
    if M is None:
        M = max_pairs
    assert M <= max_pairs, (
        f"M={M} exceeds maximum unique pairs {max_pairs} for d_head={d_head}"
    )
    # i, j: (max_pairs,) — row/col indices into the strict upper triangle
    i, j = torch.triu_indices(d_head, d_head, offset=1)
    # perm: (M,) — subset of pair indices
    perm = torch.randperm(max_pairs, generator=generator)[:M]
    # pairs: (2, M) — pairs[0] = i[perm], pairs[1] = j[perm]
    pairs = torch.stack([i[perm], j[perm]])
    return M, pairs


class RankAttention(nn.Module):
    """
    Attention where queries and keys are projected via pairwise rank features
    (``a - b`` soft/hard/STE), then passed to :func:`torch.nn.functional.scaled_dot_product_attention`.
    """

    def __init__(
        self,
        d_head: int,
        M: Optional[int] = None,
        smooth_mode: bool = True,
        generator: Optional[torch.Generator] = None,
    ):
        super().__init__()
        _, pairs = _sample_upper_tri_pairs(d_head, M, generator=generator)
        self.smooth_mode = smooth_mode
        # self.pairs: (2, M) on the module device after .to(...)
        self.register_buffer("pairs", pairs)

    def soft_rank_projection(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, H, T, d_head); self.pairs: (2, M)
        a = x[..., self.pairs[0]]  # (B, H, T, M)
        b = x[..., self.pairs[1]]  # (B, H, T, M)
        d = a - b  # (B, H, T, M)
        # (B, H, T, M), roughly in (-0.5, 0.5)
        return d / (1 + d.abs()) - 0.5

    def hard_rank_projection(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, H, T, d_head)
        a = x[..., self.pairs[0]]  # (B, H, T, M)
        b = x[..., self.pairs[1]]  # (B, H, T, M)
        # (B, H, T, M) in {-0.5, 0.5}
        return (a > b).float() - 0.5

    def ste_rank_projection(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, H, T, d_head) -> (B, H, T, M)
        soft = self.soft_rank_projection(x)
        hard = self.hard_rank_projection(x)
        return (hard - soft).detach() + soft

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        is_causal: bool = True,
        dropout_p: float = 0.0,
    ) -> torch.Tensor:
        # q, k: (B, H, T, d_head); v: (B, H, T, d_head)
        proj = self.soft_rank_projection if self.smooth_mode else self.ste_rank_projection
        rq = proj(q)  # (B, H, T, M)
        rk = proj(k)  # (B, H, T, M)
        # SDPA: (B,H,T,M) x (B,H,M,T) -> attn (B,H,T,T) @ v (B,H,T,d_head)
        return F.scaled_dot_product_attention(
            rq,
            rk,
            v,
            is_causal=is_causal,
            dropout_p=dropout_p if self.training else 0.0,
        )  # (B, H, T, d_head)


class RankProjection(nn.Module):
    """
    Linear layer on top of pairwise rank features of a single head vector.
    """

    def __init__(
        self,
        d_head: int,
        d_out: int,
        M: Optional[int] = None,
        smooth_mode: bool = True,
        generator: Optional[torch.Generator] = None,
    ):
        super().__init__()
        M, pairs = _sample_upper_tri_pairs(d_head, M, generator=generator)
        self.smooth_mode = smooth_mode
        # self.pairs: (2, M); self.linear: R^{M x d_out}
        self.register_buffer("pairs", pairs)
        self.linear = nn.Linear(M, d_out)

    def soft_rank_projection(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., d_head); self.pairs: (2, M) — leading dims ... broadcast with indexing
        d = x[..., self.pairs[0]] - x[..., self.pairs[1]]  # (..., M)
        # (..., M), roughly in (-1, 1)
        return d / (1 + d.abs())

    def hard_rank_projection(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., d_head) -> (..., M) in {-1, 0, 1}
        d = x[..., self.pairs[0]] - x[..., self.pairs[1]]
        return d.sign()

    def ste_rank_projection(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., d_head) -> (..., M)
        soft = self.soft_rank_projection(x)
        hard = self.hard_rank_projection(x)
        return (hard - soft).detach() + soft

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, d_head)
        proj = self.soft_rank_projection if self.smooth_mode else self.ste_rank_projection
        r = proj(x)  # (B, M)
        return self.linear(r)  # (B, d_out)


def add_rank_preserving_noise(x, scale=0.1):
    # x: (..., d)
    sorted_x, _ = x.sort(dim=-1)
    # minimum gap between adjacent elements
    min_gap = (sorted_x[..., 1:] - sorted_x[..., :-1]).min(dim=-1, keepdim=True).values  # (..., 1)
    noise = torch.rand_like(x) * min_gap * scale
    return x + noise
