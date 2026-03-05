"""
Helper functions and shared enums for LUT-based components.
"""
import math
from enum import Enum

import torch


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

