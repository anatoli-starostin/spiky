"""Anchor tree helpers for the NAP=3 → NAP=6 1-step curriculum.

Adapted from exp400/anchor_tree.py but generalised to handle arbitrary
1-level splits (target_nap need NOT be a power of 2, only even).

For target (NAP=N) → leaf (NAP=N/2) split:
- Sample N anchor pairs per target table via CANONICAL_FULL_COVERAGE.
- Leaf table 2i = first N/2 pairs of target table i.
- Leaf table 2i+1 = last N/2 pairs of target table i.

Then `merge_weight_tensor(leaf_weights, nap_prev=N/2)` produces the target
weights via the outer-add formula:
    parent[bits_A << (N/2) | bits_B, :] = child_A[bits_A, :] + child_B[bits_B, :].
"""
from __future__ import annotations
import torch
from spiky.lutorch.tiny_anchor_pairs_lookup import TinyAnchorPairsLookup
from spiky.lutorch.lut_helpers import AnchorSamplingPolicy


def build_two_level_anchors(input_dim: int,
                            n_target_tables: int,
                            target_nap: int,
                            n_heads: int,
                            random_seed: int,
                            device='cuda'):
    """Build target (NAP=target_nap) anchors and derive a 1-level leaf split.

    Args:
        input_dim: input feature dimension for anchor pair sampling.
        n_target_tables: total number of tables at stage 2 (n_heads × tph_target).
        target_nap: even integer NAP for the target architecture.
        n_heads: passed through to TinyAnchorPairsLookup.
        random_seed: anchor sampling seed.
        device: where to build the lookup (anchors are returned on CPU).

    Returns dict {
        'target': {'anchor_a': [n_target_tables, target_nap] int64,
                   'anchor_b': same, 'nap': target_nap, 'n_tables': n_target_tables},
        'leaf':   {'anchor_a': [2*n_target_tables, target_nap//2] int64,
                   'anchor_b': same, 'nap': target_nap//2, 'n_tables': 2*n_target_tables},
    }
    Leaf ordering: leaf table 2i = first half of target table i; 2i+1 = second half.
    """
    if target_nap % 2 != 0:
        raise ValueError(f"target_nap must be even, got {target_nap}")
    leaf_nap = target_nap // 2

    lookup = TinyAnchorPairsLookup(
        input_dim=input_dim,
        n_tables=n_target_tables,
        n_anchor_pairs=target_nap,
        n_heads=n_heads,
        random_seed=random_seed,
        device=device,
        anchor_sampling_policy=AnchorSamplingPolicy.CANONICAL_FULL_COVERAGE,
    )
    target_a = lookup.anchor_pairs_a.to(torch.int64).cpu().clone()
    target_b = lookup.anchor_pairs_b.to(torch.int64).cpu().clone()

    leaf_a = torch.zeros(2 * n_target_tables, leaf_nap, dtype=torch.int64)
    leaf_b = torch.zeros(2 * n_target_tables, leaf_nap, dtype=torch.int64)
    for i in range(n_target_tables):
        leaf_a[2 * i]     = target_a[i, :leaf_nap]
        leaf_b[2 * i]     = target_b[i, :leaf_nap]
        leaf_a[2 * i + 1] = target_a[i, leaf_nap:]
        leaf_b[2 * i + 1] = target_b[i, leaf_nap:]

    return {
        'target': {'anchor_a': target_a, 'anchor_b': target_b,
                   'nap': target_nap, 'n_tables': n_target_tables},
        'leaf':   {'anchor_a': leaf_a, 'anchor_b': leaf_b,
                   'nap': leaf_nap, 'n_tables': 2 * n_target_tables},
    }


def merge_weight_tensor(prev_weights: torch.Tensor, nap_prev: int) -> torch.Tensor:
    """Merge a leaf-stage weight tensor across the table dim, going from
    (n_tables_prev tables, NAP=nap_prev) to (n_tables_prev/2 tables, NAP=2·nap_prev).

    Args:
        prev_weights: [n_tables_prev, 2^nap_prev, n_outputs]
        nap_prev: NAP at the previous (leaf) level

    Returns:
        new_weights: [n_tables_prev/2, 2^(2·nap_prev), n_outputs]
            new_weights[i, a*2^nap_prev + b, :] = prev[2i, a, :] + prev[2i+1, b, :]
    """
    n_tables_prev, rows_prev, n_out = prev_weights.shape
    assert rows_prev == (1 << nap_prev), \
        f"rows {rows_prev} != 2^{nap_prev}={1 << nap_prev}"
    assert n_tables_prev % 2 == 0, f"n_tables_prev={n_tables_prev} not even"

    n_tables_new = n_tables_prev // 2
    A = prev_weights[0::2]   # [n_tables_new, rows_prev, n_out]
    B = prev_weights[1::2]   # [n_tables_new, rows_prev, n_out]

    # Outer-add: new[i, a, b, :] = A[i, a, :] + B[i, b, :]
    merged = A.unsqueeze(2) + B.unsqueeze(1)
    new_weights = merged.reshape(n_tables_new, rows_prev * rows_prev, n_out)
    return new_weights
