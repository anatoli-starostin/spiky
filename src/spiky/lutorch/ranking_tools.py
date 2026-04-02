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

    Args:
        d_qk: Dimension of query/key vectors (pairs are sampled from this space).
        d_v: Dimension of value vectors (passed through unchanged). Defaults to d_qk.
        M: Number of rank features (pairs) for q/k projection. Defaults to all unique pairs from d_qk.
    """

    def __init__(
        self,
        d_qk: int,
        d_v: Optional[int] = None,
        M: Optional[int] = None,
        smooth_mode: bool = True,
        input_scale_noise: float = 0.0,
        temperature: float = 0.1,
        generator: Optional[torch.Generator] = None,
    ):
        super().__init__()
        if d_v is None:
            d_v = d_qk
        _, pairs = _sample_upper_tri_pairs(d_qk, M, generator=generator)
        self.smooth_mode = smooth_mode
        self.input_scale_noise = input_scale_noise
        self.temperature = temperature
        # self.pairs: (2, M) on the module device after .to(...)
        self.register_buffer("pairs", pairs)

    def soft_rank_projection(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, H, T, d_qk); self.pairs: (2, M)
        a = x[..., self.pairs[0]]  # (B, H, T, M)
        b = x[..., self.pairs[1]]  # (B, H, T, M)
        d = a - b  # (B, H, T, M)
        # (B, H, T, M), in (-1, 1); lower temperature -> sharper (like softmax convention)
        return d / (self.temperature + d.abs())

    def hard_rank_projection(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, H, T, d_qk)
        a = x[..., self.pairs[0]]  # (B, H, T, M)
        b = x[..., self.pairs[1]]  # (B, H, T, M)
        # (B, H, T, M) in {-1, 1}
        return (a > b).float() * 2.0 - 1.0

    def ste_rank_projection(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, H, T, d_qk) -> (B, H, T, M)
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
        # q, k: (B, H, T, d_qk); v: (B, H, T, d_v)
        if self.training and self.input_scale_noise > 0.0:
            # Independent scale per (batch, head, token); shape [B, H, T, 1]
            scale_q = 1.0 + (torch.rand(*q.shape[:3], 1, device=q.device) * 2 - 1) * self.input_scale_noise
            scale_k = 1.0 + (torch.rand(*k.shape[:3], 1, device=k.device) * 2 - 1) * self.input_scale_noise
            q = q * scale_q
            k = k * scale_k
        proj = self.soft_rank_projection if self.smooth_mode else self.ste_rank_projection
        rq = proj(q)  # (B, H, T, M)
        rk = proj(k)  # (B, H, T, M)
        # SDPA: (B,H,T,M) x (B,H,M,T) -> attn (B,H,T,T) @ v (B,H,T,d_v)
        return F.scaled_dot_product_attention(
            rq,
            rk,
            v,
            is_causal=is_causal,
            dropout_p=dropout_p if self.training else 0.0,
        )  # (B, H, T, d_v)


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
        temperature: float = 0.1,
        generator: Optional[torch.Generator] = None,
    ):
        super().__init__()
        M, pairs = _sample_upper_tri_pairs(d_head, M, generator=generator)
        self.smooth_mode = smooth_mode
        self.temperature = temperature
        # self.pairs: (2, M); self.linear: R^{M x d_out}
        self.register_buffer("pairs", pairs)
        self.linear = nn.Linear(M, d_out)

    def soft_rank_projection(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., d_head); self.pairs: (2, M) — leading dims ... broadcast with indexing
        d = x[..., self.pairs[0]] - x[..., self.pairs[1]]  # (..., M)
        # (..., M), in (-1, 1); lower temperature -> sharper (like softmax convention)
        return d / (self.temperature + d.abs())

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


class PairVoting(nn.Module):
    """
    Each input pair (a, b) has a dedicated learnable output vector.
    The output is the sum of those vectors weighted by soft or hard rank features.

    Hard:  weight_i = sign(x[a_i] - x[b_i])          ∈ {-1, 0, 1}
    Soft:  weight_i = (x[a_i]-x[b_i]) / (t+|...|)    ∈ (-1, 1)

    output = sum_i weight_i * v_i    (no cross-pair mixing)

    Parameters: M * n_outputs  where M = C(input_dim, 2) by default.

    Args:
        input_dim: Input feature dimension.
        n_outputs: Output dimension.
        M:         Number of pairs to monitor. Defaults to all C(input_dim,2).
        smooth_mode: If True use soft features; if False use sign (STE in training).
        temperature: Sharpness of soft features (lower → sharper).
        generator: Optional RNG for reproducible pair sampling when M < C(d,2).
    """

    def __init__(
        self,
        input_dim: int,
        n_outputs: int,
        M: Optional[int] = None,
        smooth_mode: bool = True,
        temperature: float = 0.1,
        generator: Optional[torch.Generator] = None,
    ):
        super().__init__()
        M, pairs = _sample_upper_tri_pairs(input_dim, M, generator=generator)
        self.smooth_mode = smooth_mode
        self.temperature = temperature
        self.register_buffer("pairs", pairs)           # (2, M)
        self.vectors = nn.Parameter(torch.zeros(M, n_outputs))

    def _features(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., input_dim) -> (..., M)
        d = x[..., self.pairs[0]] - x[..., self.pairs[1]]
        if self.smooth_mode:
            return d / (self.temperature + d.abs())
        # Hard / STE: sign forward, soft backward
        soft = d / (self.temperature + d.abs())
        hard = d.sign()
        return (hard - soft).detach() + soft

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, input_dim) -> (B, n_outputs)
        r = self._features(x)          # (B, M)
        return r @ self.vectors        # (B, n_outputs)


class PositionalPermutation(nn.Module):
    """
    Associates a fixed random permutation with each sequence position.
    Applied to q and k before attention as a non-learnable positional encoding.

    Args:
        maxlen: Maximum sequence length (one permutation per position).
        d: Dimension to permute.
        random_seed: Seed for reproducibility.
    """

    def __init__(self, maxlen: int, d: int, random_seed: int = 0):
        super().__init__()
        gen = torch.Generator()
        gen.manual_seed(random_seed)
        perms = torch.stack([torch.randperm(d, generator=gen) for _ in range(maxlen)])  # (T, d)
        self.register_buffer('perms', perms)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, H, T, d)
        B, H, T, d = x.shape
        idx = self.perms[:T].unsqueeze(0).unsqueeze(0).expand(B, H, T, d)  # (B, H, T, d)
        return x.gather(-1, idx)  # (B, H, T, d)


class LearnedSoftPermutations(nn.Module):
    def __init__(self, n_perms, n, temp=1.0, smooth_mode=False):
        super().__init__()
        self.temp = temp
        self.smooth_mode = smooth_mode
        self.scores = nn.Parameter(torch.randn(n_perms, n, n))  # (n_perms, n, n)
        self._cached_P = None

    def get_P(self):
        return (self.scores / self.temp).softmax(dim=-1)         # (n_perms, n, n)

    def forward(self, x):
        # x: (B, n_perms, n)
        if self.training:
            self._cached_P = None
            soft = self.get_P()                                  # (n_perms, n, n)
        else:
            if self._cached_P is None:
                self._cached_P = self.get_P().detach()
            soft = self._cached_P                                # (n_perms, n, n)

        if self.smooth_mode:
            P = soft
        else:
            idx = soft.argmax(dim=-1)                            # (n_perms, n)
            hard = torch.zeros_like(soft).scatter_(-1, idx.unsqueeze(-1), 1.0)  # (n_perms, n, n)
            P = (hard - soft).detach() + soft                    # (n_perms, n, n)

        return torch.einsum('bpi,pij->bpj', x, P)               # (B, n_perms, n)


def add_rank_preserving_noise(x, scale=0.1):
    # x: (..., d)
    sorted_x, _ = x.sort(dim=-1)
    # minimum gap between adjacent elements
    min_gap = (sorted_x[..., 1:] - sorted_x[..., :-1]).min(dim=-1, keepdim=True).values  # (..., 1)
    noise = torch.rand_like(x) * min_gap * scale
    return x + noise
