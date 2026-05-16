"""Anchor pair tree for NAP curriculum.

Top-down derivation: sample the FINAL NAP=k_target architecture using
CANONICAL_FULL_COVERAGE, then recursively split each table's anchor pairs
into a binary tree.

For target n_tables_target × NAP_target = N tables × NAP pairs:
- Level 0 (leaves, NAP=1): N × NAP tables, each with 1 anchor pair
- Level 1 (NAP=2): N × NAP/2 tables, each with 2 anchor pairs
- Level 2 (NAP=4): N × NAP/4 tables
- ...
- Level log2(NAP) (root, NAP=NAP_target): N tables, each with NAP_target pairs

Each parent table's anchor pairs are split in half between its 2 children.
Sibling tables (2i, 2i+1) have disjoint anchor pairs whose union = parent.

Functions:
- build_anchor_tree(...)
- merge_weights(child_A, child_B, NAP_child)
"""
from __future__ import annotations
import torch
from spiky.lutorch.tiny_anchor_pairs_lookup import TinyAnchorPairsLookup
from spiky.lutorch.lut_helpers import AnchorSamplingPolicy


def build_anchor_tree(input_dim: int,
                      n_target_tables: int,
                      target_nap: int,
                      n_heads: int,
                      random_seed: int,
                      device='cuda'):
    """Sample the final architecture and derive a binary split tree.

    Returns: dict {level: dict('anchor_a': [n_tables_at_level, NAP_at_level],
                                'anchor_b': same shape,
                                'nap': NAP, 'n_tables': N)}
    where level=0 is root (NAP=target_nap) and level=L-1 is leaves (NAP=1).

    Convention:
    - Tables at deeper levels are ordered such that siblings 2i, 2i+1
      come from parent i at the level above.
    - This makes the merge step trivial: just iterate over pairs of
      consecutive children to produce parents.
    """
    # Verify target_nap is a power of 2
    import math
    L = int(math.log2(target_nap))
    if 2 ** L != target_nap:
        raise ValueError(f"target_nap must be power of 2, got {target_nap}")

    # 1) Build the ROOT (NAP=target_nap) via TinyAnchorPairsLookup
    lookup = TinyAnchorPairsLookup(
        input_dim=input_dim,
        n_tables=n_target_tables,
        n_anchor_pairs=target_nap,
        n_heads=n_heads,
        random_seed=random_seed,
        device=device,
        anchor_sampling_policy=AnchorSamplingPolicy.CANONICAL_FULL_COVERAGE,
    )
    # Shape: [n_target_tables, target_nap], int16 in source; we cast to int64
    root_a = lookup.anchor_pairs_a.to(torch.int64).cpu().clone()
    root_b = lookup.anchor_pairs_b.to(torch.int64).cpu().clone()

    tree = {}
    tree[0] = {'anchor_a': root_a, 'anchor_b': root_b,
               'nap': target_nap, 'n_tables': n_target_tables}

    # 2) Recursively split: each parent table's NAP pairs split into 2 halves
    # of NAP/2 pairs each.
    cur_a, cur_b = root_a, root_b
    cur_n_tables, cur_nap = n_target_tables, target_nap
    for level in range(1, L + 1):
        new_nap = cur_nap // 2
        new_n_tables = cur_n_tables * 2
        # Split: child_left.pairs = parent.pairs[:new_nap]
        #        child_right.pairs = parent.pairs[new_nap:]
        new_a = torch.zeros(new_n_tables, new_nap, dtype=torch.int64)
        new_b = torch.zeros(new_n_tables, new_nap, dtype=torch.int64)
        for i in range(cur_n_tables):
            new_a[2 * i]     = cur_a[i, :new_nap]
            new_b[2 * i]     = cur_b[i, :new_nap]
            new_a[2 * i + 1] = cur_a[i, new_nap:]
            new_b[2 * i + 1] = cur_b[i, new_nap:]
        tree[level] = {'anchor_a': new_a, 'anchor_b': new_b,
                       'nap': new_nap, 'n_tables': new_n_tables}
        cur_a, cur_b, cur_n_tables, cur_nap = new_a, new_b, new_n_tables, new_nap

    return tree


def merge_weights(child_A: torch.Tensor,
                  child_B: torch.Tensor,
                  nap_child: int) -> torch.Tensor:
    """Merge two sibling tables (NAP=nap_child each) into one parent (NAP=2·nap_child).

    Args:
        child_A: [2^nap_child, n_outputs] weights of left child
        child_B: [2^nap_child, n_outputs] weights of right child
        nap_child: NAP of each child

    Returns:
        parent: [2^(2·nap_child), n_outputs] weights such that
            parent[row_idx, :] = child_A[bits_A, :] + child_B[bits_B, :]
        where row_idx = (bits_A << nap_child) | bits_B (MSB convention:
        parent's first nap_child anchor pairs come from A).
    """
    n_out = child_A.shape[-1]
    rows_per_child = 1 << nap_child
    rows_parent = rows_per_child * rows_per_child

    # Build via outer broadcasting:
    # parent[i*rows_per_child + j, :] = child_A[i, :] + child_B[j, :]
    parent = (child_A.unsqueeze(1) + child_B.unsqueeze(0))   # [rows_per_child, rows_per_child, n_out]
    parent = parent.reshape(rows_parent, n_out)
    return parent


def merge_weight_tensor(prev_weights: torch.Tensor, nap_prev: int) -> torch.Tensor:
    """Merge an entire weight tensor across the table dim.

    Args:
        prev_weights: [n_tables_prev, 2^nap_prev, n_outputs]
        nap_prev: NAP at the previous level

    Returns:
        new_weights: [n_tables_prev/2, 2^(2·nap_prev), n_outputs]
    """
    n_tables_prev, rows_prev, n_out = prev_weights.shape
    assert rows_prev == (1 << nap_prev), \
        f"rows {rows_prev} != 2^{nap_prev}={1 << nap_prev}"
    assert n_tables_prev % 2 == 0, f"n_tables_prev={n_tables_prev} not even"

    n_tables_new = n_tables_prev // 2

    # child_A[i] = prev_weights[2i],   shape [n_tables_new, rows_prev, n_out]
    # child_B[i] = prev_weights[2i+1]
    A = prev_weights[0::2]   # [n_tables_new, rows_prev, n_out]
    B = prev_weights[1::2]   # [n_tables_new, rows_prev, n_out]

    # Outer-add: new[i, a, b, :] = A[i, a, :] + B[i, b, :]
    # Shape: [n_tables_new, rows_prev, rows_prev, n_out]
    merged = A.unsqueeze(2) + B.unsqueeze(1)
    new_weights = merged.reshape(n_tables_new, rows_prev * rows_prev, n_out)
    return new_weights
