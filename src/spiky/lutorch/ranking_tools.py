"""
Rank-based projections and attention using random upper-triangular index pairs.

Projects head dimensions through pairwise comparisons (soft, hard, or STE) for
use in linear layers or scaled dot-product attention.
"""
from __future__ import annotations

from typing import Optional

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from spiky.lutorch.wta_lookup import WTALookupFunction
from spiky.lutorch.lut_helpers import UncertaintyMode


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
    (``a - b`` soft/hard/STE), then passed to SDPA.

    Args:
        d_qk: Dimension of query/key vectors (pairs are sampled from this space).
        d_v: Dimension of value vectors (passed through unchanged). Defaults to d_qk.
        M: Number of rank features (pairs). Defaults to all unique pairs from d_qk.
        smooth_mode: If True use soft projection and soft SDPA; if False use STE for both
            q/k projection and SDPA aggregation (hard forward via sdpa_forward_temperature,
            soft backward via sdpa_temperature).
        temperature: Sharpness of rank projection (lower → sharper, in (-1,1)).
        sdpa_temperature: Controls soft SDPA sharpness (lower → sharper). Used as the
            backward temperature in non-smooth mode.
        sdpa_forward_temperature: In non-smooth mode, used for the hard forward SDPA pass
            (STE). Lower → closer to argmax. Ignored in smooth mode.
    """

    def __init__(
        self,
        d_qk: int,
        d_v: Optional[int] = None,
        M: Optional[int] = None,
        smooth_mode: bool = True,
        input_scale_noise: float = 0.0,
        temperature: float = 0.1,
        sdpa_temperature: float = 1.0,
        sdpa_forward_temperature: float = 1.0,
        learnable_attn_scale_init: Optional[float] = None,
        generator: Optional[torch.Generator] = None,
    ):
        super().__init__()
        if d_v is None:
            d_v = d_qk
        _, pairs = _sample_upper_tri_pairs(d_qk, M, generator=generator)
        self.smooth_mode = smooth_mode
        self.input_scale_noise = input_scale_noise
        self.temperature = temperature
        self.sdpa_temperature = sdpa_temperature
        self.sdpa_forward_temperature = sdpa_forward_temperature
        self.register_buffer("pairs", pairs)
        # Optional learnable softmax-sharpness scalar. When set, overrides the
        # default SDPA 1/sqrt(M) scale via scale = attn_scale / sqrt(M).
        # So softmax-input std ≈ attn_scale · std(rq)·std(rk) regardless of M.
        # Useful for bit attention where rq, rk are ±1 and we want to keep them
        # as bits while still controlling attention sharpness.
        if learnable_attn_scale_init is not None:
            self.attn_scale = nn.Parameter(torch.tensor(float(learnable_attn_scale_init)))
        else:
            self.attn_scale = None

    def soft_rank_projection(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, H, T, d_qk); self.pairs: (2, M)
        d = x[..., self.pairs[0]] - x[..., self.pairs[1]]  # (B, H, T, M)
        return d / (self.temperature + d.abs())  # in (-1, 1)

    def hard_rank_projection(self, x: torch.Tensor) -> torch.Tensor:
        a = x[..., self.pairs[0]]
        b = x[..., self.pairs[1]]
        return (a > b).float() * 2.0 - 1.0  # in {-1, 1}

    def ste_rank_projection(self, x: torch.Tensor) -> torch.Tensor:
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
            scale_q = 1.0 + (torch.rand(*q.shape[:3], 1, device=q.device) * 2 - 1) * self.input_scale_noise
            scale_k = 1.0 + (torch.rand(*k.shape[:3], 1, device=k.device) * 2 - 1) * self.input_scale_noise
            q = q * scale_q
            k = k * scale_k
        proj = self.soft_rank_projection if self.smooth_mode else self.ste_rank_projection
        rq = proj(q)  # (B, H, T, M)
        rk = proj(k)  # (B, H, T, M)
        # Apply learnable softmax-sharpness by pre-scaling rq. Multiplying rq
        # by s makes softmax input = s * rq·rk / sqrt(M) (SDPA's default scale).
        # At deployment this can be folded into a post-popcount scalar multiply.
        if self.attn_scale is not None:
            rq = rq * self.attn_scale
        t_soft = self.sdpa_temperature ** 0.5
        if self.smooth_mode:
            return F.scaled_dot_product_attention(
                rq / t_soft,
                rk / t_soft,
                v,
                is_causal=is_causal,
                dropout_p=dropout_p if self.training else 0.0,
            )  # (B, H, T, d_v)
        # Non-smooth mode: STE over SDPA — hard forward, soft backward
        t_hard = self.sdpa_forward_temperature ** 0.5
        soft_out = F.scaled_dot_product_attention(
            rq / t_soft,
            rk / t_soft,
            v,
            is_causal=is_causal,
            dropout_p=dropout_p if self.training else 0.0,
        )
        if t_hard == t_soft:
            return soft_out
        hard_out = F.scaled_dot_product_attention(
            rq / t_hard,
            rk / t_hard,
            v,
            is_causal=is_causal,
        )
        return (hard_out - soft_out).detach() + soft_out


class RankWTAAttention(nn.Module):
    """
    Attention where queries and keys are rank-projected, the score matrix is
    materialised explicitly (with causal masking), and WTALookup is used as a
    differentiable argmax over keys for each query position.

    grad_c_winner and grad_c_alt are zero-valued tensors in the forward pass
    (created by WTALookupFunction as x.sum()*0) that carry gradients back to
    the score matrix. The gradient connections are:
      grad_c_winner ← (grad_output · v_winner).sum(-1)
      grad_c_alt[k] ← (grad_output · v_alt[k]).sum(-1)
    WTALookup backward then computes:
      du[k] = (grad_c_winner - grad_c_alt[k]) * uncertainty_deriv[k]
    and scatters du to the score matrix at winner/alt key positions.

    In eval mode a plain argmax + gather is used (no gradient needed).

    Args:
        d_qk: Dimension of query/key vectors.
        d_v: Dimension of value vectors. Defaults to d_qk.
        M: Number of rank-feature pairs. Defaults to all C(d_qk, 2).
        smooth_mode: If True use soft rank projection; if False use STE.
        temperature: Sharpness of rank projection (lower → sharper).
        n_alternatives: Runner-up keys tracked by WTALookup for gradient.
        uncertainty_mode: Uncertainty function for WTALookup gradient weighting.
        maxlen: Maximum sequence length for pre-computed causal mask.
    """

    def __init__(
        self,
        d_qk: int,
        d_v: Optional[int] = None,
        M: Optional[int] = None,
        smooth_mode: bool = True,
        temperature: float = 0.1,
        n_alternatives: int = 1,
        uncertainty_mode: UncertaintyMode = UncertaintyMode.INVERSE_L1,
        maxlen: int = 4096,
        generator: Optional[torch.Generator] = None,
    ):
        super().__init__()
        if d_v is None:
            d_v = d_qk
        _, pairs = _sample_upper_tri_pairs(d_qk, M, generator=generator)
        self.smooth_mode = smooth_mode
        self.temperature = temperature
        self.n_alternatives = n_alternatives
        self.uncertainty_mode = uncertainty_mode
        self.register_buffer("pairs", pairs)
        # Pre-computed causal mask: True where attention is allowed (lower triangle)
        causal_mask = torch.ones(maxlen, maxlen, dtype=torch.bool).tril()
        self.register_buffer("causal_mask", causal_mask)
        # Cached batch_offset for WTALookupFunction (invalidated when BH*T changes)
        self._cached_batch_offset: Optional[torch.Tensor] = None
        self._cached_bht: int = -1

    def soft_rank_projection(self, x: torch.Tensor) -> torch.Tensor:
        d = x[..., self.pairs[0]] - x[..., self.pairs[1]]
        return d / (self.temperature + d.abs())

    def hard_rank_projection(self, x: torch.Tensor) -> torch.Tensor:
        a = x[..., self.pairs[0]]
        b = x[..., self.pairs[1]]
        return (a > b).float() * 2.0 - 1.0

    def ste_rank_projection(self, x: torch.Tensor) -> torch.Tensor:
        soft = self.soft_rank_projection(x)
        hard = self.hard_rank_projection(x)
        return (hard - soft).detach() + soft

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        is_causal: bool = True,
    ) -> torch.Tensor:
        # q, k: (B, H, T, d_qk); v: (B, H, T, d_v)
        B, H, T, _ = q.shape
        d_v = v.shape[-1]
        BH = B * H
        M = self.pairs.shape[1]
        n_alt = self.n_alternatives

        proj = self.soft_rank_projection if self.smooth_mode else self.ste_rank_projection
        rq = proj(q).reshape(BH, T, M)  # (BH, T, M)
        rk = proj(k).reshape(BH, T, M)  # (BH, T, M)

        # Materialise score matrix
        scores = torch.bmm(rq, rk.transpose(1, 2))  # (BH, T_q, T_k)

        # Apply pre-computed causal mask
        if is_causal:
            scores = scores.masked_fill(~self.causal_mask[:T, :T], float('-inf'))

        v_flat = v.reshape(BH, T, d_v)

        if not self.training:
            # Eval: plain argmax + gather
            winner_idx = scores.argmax(dim=-1)  # (BH, T_q)
            output = v_flat.gather(1, winner_idx.unsqueeze(-1).expand(BH, T, d_v))
            return output.reshape(B, H, T, d_v)

        # Train: differentiable WTALookup
        # batch_offset[bh*T*n_alt + t*n_alt + k] = (bh*T + t) * T (stride into flattened scores)
        bht = BH * T
        if (
            self._cached_batch_offset is None
            or self._cached_bht != bht
            or self._cached_batch_offset.device != scores.device
        ):
            self._cached_batch_offset = (
                torch.arange(bht, device=scores.device, dtype=torch.long)
                .repeat_interleave(n_alt) * T
            ).contiguous()
            self._cached_bht = bht

        uncertainty_mode_int = 0 if self.uncertainty_mode == UncertaintyMode.INVERSE_L1 else 1
        # Clamp -inf (from causal mask) to a finite sentinel before WTALookupFunction.
        # WTALookupFunction uses z = x.sum()*0 as a zero-valued gradient carrier;
        # if x contains -inf, the sum overflows float32 (many -1e38 elements → -inf)
        # and -inf*0 = NaN, poisoning all outputs.
        # -1e9 is well below any valid score (scores bounded by ±M, M≤C(d_qk,2))
        # yet safe: even BH*T*T * 1e9 ~ 8e18 stays within float32 range (~3.4e38).
        # Clamp gradient is 0 at masked positions, so no gradient flows there.
        scores_wta = scores.clamp(min=-1e9)
        winner_idx, alt_idx, alt_deltas, grad_c_winner, grad_c_alt = WTALookupFunction.apply(
            scores_wta, n_alt, uncertainty_mode_int, self._cached_batch_offset
        )
        # winner_idx: (BH, T); alt_idx: (BH, T, n_alt)
        # grad_c_winner: (BH, T) == 0; grad_c_alt: (BH, T, n_alt) == 0 (forward)

        # Gather V at winner positions: forward output
        output = v_flat.gather(1, winner_idx.unsqueeze(-1).expand(BH, T, d_v))  # (BH, T, d_v)

        # Connect grad_c_winner: adds 0 in forward; in backward produces
        #   grad_c_winner_grad = (grad_output · v_winner).sum(-1)
        output = output + output.detach() * grad_c_winner.unsqueeze(-1)

        # Connect grad_c_alt: adds 0 in forward; in backward produces
        #   grad_c_alt_grad[k] = (grad_output · v_alt[k]).sum(-1)
        alt_idx_flat = alt_idx.reshape(BH, T * n_alt)
        v_at_alt = v_flat.gather(1, alt_idx_flat.unsqueeze(-1).expand(BH, T * n_alt, d_v))
        v_at_alt = v_at_alt.reshape(BH, T, n_alt, d_v)
        output = output + (v_at_alt.detach() * grad_c_alt.unsqueeze(-1)).sum(dim=2)

        return output.reshape(B, H, T, d_v)


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
