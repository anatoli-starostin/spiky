"""
Helper functions and shared enums for LUT-based components.
"""
import math
from enum import Enum
from typing import Optional, Tuple

import torch


def get_balanced_anchor_pairs(
    n_tables: int,
    n_anchor_pairs: int,
    input_dim: int,
    device: torch.device,
    random_seed: Optional[int] = None,
    connected_mode: bool = False,
    anchor_candidates: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate anchor pairs with balanced coverage over input dimensions.
    Matches spike_QK MultiHeadLUT anchor init: indices from concatenated randperms
    so each dimension 0..input_dim-1 appears roughly equally often.

    Args:
        connected_mode: If True, flat_b is flat_a circularly shifted by 1 (connected pairs).

    Returns:
        anchor_pairs_a: [n_tables, n_anchor_pairs] int64
        anchor_pairs_b: [n_tables, n_anchor_pairs] int64
    """
    gen = None
    if random_seed is not None:
        gen = torch.Generator(device=device)
        gen.manual_seed(random_seed)

    def get_balanced_indices(total: int, dim: int) -> torch.Tensor:
        num_full_perms = math.ceil(total / dim)
        # Batched randperm: each row is a random permutation of 0..dim-1 (one rand + argsort, no Python loop)
        perm = torch.rand(num_full_perms, dim, device=device, generator=gen).argsort(dim=1)
        return perm.reshape(-1)[:total]

    if anchor_candidates is not None:
        if anchor_candidates.shape[0] != n_tables:
            raise ValueError(
                f"anchor_candidates first dimension ({anchor_candidates.shape[0]}) "
                f"must match n_tables ({n_tables})"
            )
        anchor_candidates = anchor_candidates.to(device=device, dtype=torch.long)
        max_anchors_per_table = anchor_candidates.shape[1]

        anchor_pairs_a = torch.empty(n_tables, n_anchor_pairs, dtype=torch.long, device=device)
        anchor_pairs_b = torch.empty_like(anchor_pairs_a)

        for t in range(n_tables):
            idx_a = get_balanced_indices(n_anchor_pairs, max_anchors_per_table)
            if connected_mode:
                idx_b = torch.roll(idx_a, shifts=-1, dims=0)
            else:
                idx_b = get_balanced_indices(n_anchor_pairs, max_anchors_per_table)

            anchor_pairs_a[t] = anchor_candidates[t, idx_a]
            anchor_pairs_b[t] = anchor_candidates[t, idx_b]
    else:
        total_needed = n_tables * n_anchor_pairs
        flat_a = get_balanced_indices(total_needed, input_dim)
        if connected_mode:
            flat_b = torch.roll(flat_a, shifts=-1, dims=0)
        else:
            flat_b = get_balanced_indices(total_needed, input_dim)

        # Reshape to [n_tables, n_anchor_pairs]
        anchor_pairs_a = flat_a.view(n_tables, n_anchor_pairs)
        anchor_pairs_b = flat_b.view(n_tables, n_anchor_pairs)

    # Ensure a != b everywhere (match spike_QK collision handling), max 10 attempts
    max_collision_attempts = 10
    for _ in range(max_collision_attempts):
        mask = anchor_pairs_a == anchor_pairs_b
        if not mask.any().item():
            break
        n_collide = mask.sum().item()
        anchor_pairs_b = anchor_pairs_b.clone()
        anchor_pairs_b[mask] = torch.randint(
            0, input_dim, (n_collide,), device=device, dtype=torch.long, generator=gen
        )

    return anchor_pairs_a, anchor_pairs_b


class UncertaintyMode(str, Enum):
    INVERSE_L1 = "inverse_l1"
    INVERSE_QUADRATIC = "inverse_quadratic"


def logarithmic_pe_buckets(num_buckets: int, seq_len: int, device: torch.device) -> torch.Tensor:
    """
    Allocate positional encoding buckets with a logarithmic tail.
    Matches spike_QK allocate_PE_buckets.

    For positions < B_half (num_buckets // 2): bucket = position
    For positions >= B_half: bucket = B_half + int(scale * log(pos / B_half))
    """
    pe_buckets = torch.zeros(seq_len, dtype=torch.long, device=device)
    if num_buckets <= 1:
        return pe_buckets

    B_half = num_buckets // 2
    for pos in range(seq_len):
        if pos < B_half:
            pe_buckets[pos] = pos
        else:
            log_term = math.log(pos / B_half)
            log_max_dist = math.log(seq_len / B_half)
            scale = (num_buckets - B_half) / log_max_dist
            log_bucket = B_half + int(scale * log_term)
            pe_buckets[pos] = min(log_bucket, num_buckets - 1)

    return pe_buckets


def rpe_matrix(buckets: torch.Tensor, seq_len: int, device: torch.device) -> torch.Tensor:
    """
    Allocate relative positional encoding matrix.

    RPE[i, j] = buckets[max(0, i - j)]
    """
    indices = torch.arange(seq_len, device=device)
    diff = indices.unsqueeze(1) - indices.unsqueeze(0)  # [seq_len, seq_len]
    diff = diff.clamp(min=0)  # Only non-negative differences
    diff = diff.clamp(max=len(buckets) - 1)  # Clamp to valid bucket range
    return buckets[diff]

