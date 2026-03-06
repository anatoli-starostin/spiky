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
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate anchor pairs with balanced coverage over input dimensions.
    Matches spike_QK MultiHeadLUT anchor init: indices from concatenated randperms
    so each dimension 0..input_dim-1 appears roughly equally often.

    Returns:
        anchor_pairs_a: [n_tables, n_anchor_pairs] int64
        anchor_pairs_b: [n_tables, n_anchor_pairs] int64
    """
    gen = None
    if random_seed is not None:
        gen = torch.Generator(device=device)
        gen.manual_seed(random_seed)

    total_needed = n_tables * n_anchor_pairs

    def get_balanced_indices(total: int, dim: int) -> torch.Tensor:
        num_full_perms = math.ceil(total / dim)
        indices_list = [
            torch.randperm(dim, device=device, generator=gen)
            for _ in range(num_full_perms)
        ]
        return torch.cat(indices_list)[:total]

    flat_a = get_balanced_indices(total_needed, input_dim)
    flat_b = get_balanced_indices(total_needed, input_dim)

    # Reshape to [n_tables, n_anchor_pairs]
    anchor_pairs_a = flat_a.view(n_tables, n_anchor_pairs)
    anchor_pairs_b = flat_b.view(n_tables, n_anchor_pairs)

    # Ensure a != b everywhere (match spike_QK collision handling)
    mask = anchor_pairs_a == anchor_pairs_b
    while mask.any().item():
        n_collide = mask.sum().item()
        anchor_pairs_b = anchor_pairs_b.clone()
        anchor_pairs_b[mask] = torch.randint(
            0, input_dim, (n_collide,), device=device, dtype=torch.long, generator=gen
        )
        mask = anchor_pairs_a == anchor_pairs_b

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

