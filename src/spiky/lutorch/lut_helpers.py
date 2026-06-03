"""
Helper functions and shared enums for LUT-based components.

Public surface for the LUTGPT release: a small set of anchor-sampling
policies plus a couple of helpers used by both lutorch and lutgpt.
"""
import math
from enum import Enum
from typing import Optional, Tuple

import torch


class AnchorSamplingPolicy(str, Enum):
    BALANCED = "balanced"
    # default for legacy anchor_pairs_lookup: balanced-randperm coverage,
    # independent a and b draws.
    CONNECTED = "connected"
    # legacy path: flat_b is flat_a circularly shifted by 1, so anchor
    # pairs form a connected graph.
    CANONICAL_DISTINCT = "canonical_distinct"
    # each table draws n_anchor_pairs canonical (a<b) pairs without
    # replacement from C(input_dim, 2) — within-table distinctness, no
    # cross-table coverage guarantee.
    CANONICAL_FULL_COVERAGE = "canonical_full_coverage"
    # canonical-pool tiled-randperm: full coverage of C(input_dim, 2)
    # whenever n_tables * n_anchor_pairs >= P, plus a greedy swap-repair
    # pass to keep within-table distinctness across perm boundaries.


class UncertaintyMode(str, Enum):
    INVERSE_L1 = "inverse_l1"
    INVERSE_QUADRATIC = "inverse_quadratic"


# =============================================================================
# Canonical-pool sampling helpers
# =============================================================================

def _build_canonical_pool(
    input_dim: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (tri_i, tri_j) — the canonical (a<b) pair pool.

    Both tensors are [P] long, on device, with P = C(input_dim, 2).
    """
    tri_i, tri_j = torch.triu_indices(input_dim, input_dim, offset=1)
    return tri_i.to(device).long(), tri_j.to(device).long()


def _repair_intra_table_duplicates(pairs_table: torch.Tensor) -> None:
    """In-place greedy repair of intra-row duplicates.

    For each table with an intra-row duplicate, swap one duplicate slot
    with a slot in another table such that both tables become (or stay)
    duplicate-free after the swap. Only tables straddling a randperm-tile
    boundary can carry duplicates, so the affected row count is small in
    practice. Raises RuntimeError if no valid swap partner exists.
    """
    n_tables, nap = pairs_table.shape
    for t in range(n_tables):
        while True:
            row_list = pairs_table[t].tolist()
            seen = {}
            dup_pos = -1
            for k, v in enumerate(row_list):
                if v in seen:
                    dup_pos = k
                    break
                seen[v] = k
            if dup_pos == -1:
                break
            dup_val = row_list[dup_pos]
            row_set_excl = set(row_list)
            row_set_excl.discard(dup_val)

            swapped = False
            for t2 in range(n_tables):
                if t2 == t:
                    continue
                row2_list = pairs_table[t2].tolist()
                if dup_val in row2_list:
                    continue
                row2_set = set(row2_list)
                for k2, v2 in enumerate(row2_list):
                    if v2 in row_set_excl:
                        continue
                    row2_set_after = (row2_set - {v2}) | {dup_val}
                    if len(row2_set_after) != len(row2_list):
                        continue
                    tmp = pairs_table[t, dup_pos].clone()
                    pairs_table[t, dup_pos] = pairs_table[t2, k2]
                    pairs_table[t2, k2] = tmp
                    swapped = True
                    break
                if swapped:
                    break
            if not swapped:
                raise RuntimeError(
                    f"CANONICAL_FULL_COVERAGE: failed to repair duplicate in "
                    f"table {t} pos {dup_pos} (val={dup_val}); pool too tight "
                    f"(n_tables={n_tables}, n_anchor_pairs={nap})"
                )


def _get_canonical_full_coverage_pairs(
    n_tables: int,
    n_anchor_pairs: int,
    input_dim: int,
    device: torch.device,
    random_seed: Optional[int],
    n_heads: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample n_anchor_pairs distinct canonical pairs per table, with full
    coverage of C(input_dim, 2) whenever n_tables * n_anchor_pairs >= P.

    Algorithm: tile concatenated randperm(P)'s to fill the slots, then
    greedy-repair any duplicates that span perm-tile boundaries.

    Returns (anchor_pairs_a, anchor_pairs_b), both [n_tables, n_anchor_pairs]
    long, with a < b in every entry.
    """
    tri_i, tri_j = _build_canonical_pool(input_dim, device)
    P = tri_i.shape[0]
    if n_anchor_pairs > P:
        raise ValueError(
            f"CANONICAL_FULL_COVERAGE requires n_anchor_pairs <= "
            f"C(input_dim, 2) = {P}; got n_anchor_pairs={n_anchor_pairs}."
        )

    gen = None
    if random_seed is not None:
        gen = torch.Generator(device=device)
        gen.manual_seed(random_seed)

    per_head = n_tables // n_heads
    all_a = torch.empty(n_tables, n_anchor_pairs, dtype=torch.long, device=device)
    all_b = torch.empty(n_tables, n_anchor_pairs, dtype=torch.long, device=device)

    slots_per_head = per_head * n_anchor_pairs
    for h in range(n_heads):
        repeats = (slots_per_head + P - 1) // P
        perm_cat = torch.cat([
            torch.randperm(P, device=device, generator=gen) for _ in range(repeats)
        ])[:slots_per_head]
        pairs_table = perm_cat.view(per_head, n_anchor_pairs).contiguous()
        _repair_intra_table_duplicates(pairs_table)

        base = h * per_head
        all_a[base:base + per_head] = tri_i[pairs_table]
        all_b[base:base + per_head] = tri_j[pairs_table]

    return all_a, all_b


def _get_canonical_distinct_pairs(
    n_tables: int,
    n_anchor_pairs: int,
    input_dim: int,
    device: torch.device,
    random_seed: Optional[int],
    n_heads: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample n_anchor_pairs distinct canonical pairs (a<b) per table.

    Each table's pairs come from a fresh randperm(P)[:n_anchor_pairs], so
    within-table distinctness holds by construction. No cross-table
    coverage guarantee.

    Returns (anchor_pairs_a, anchor_pairs_b), both [n_tables, n_anchor_pairs],
    with a < b in every entry.
    """
    tri_i, tri_j = _build_canonical_pool(input_dim, device)
    P = tri_i.shape[0]
    if n_anchor_pairs > P:
        raise ValueError(
            f"CANONICAL_DISTINCT requires n_anchor_pairs <= "
            f"C(input_dim, 2) = {P}; got n_anchor_pairs={n_anchor_pairs}."
        )

    gen = None
    if random_seed is not None:
        gen = torch.Generator(device=device)
        gen.manual_seed(random_seed)

    per_head = n_tables // n_heads
    all_a = torch.empty(n_tables, n_anchor_pairs, dtype=torch.long, device=device)
    all_b = torch.empty(n_tables, n_anchor_pairs, dtype=torch.long, device=device)

    for h in range(n_heads):
        per_table_indices = torch.stack([
            torch.randperm(P, device=device, generator=gen)[:n_anchor_pairs]
            for _ in range(per_head)
        ])
        base = h * per_head
        all_a[base:base + per_head] = tri_i[per_table_indices]
        all_b[base:base + per_head] = tri_j[per_table_indices]

    return all_a, all_b


# =============================================================================
# Public dispatcher
# =============================================================================

def get_balanced_anchor_pairs(
    n_tables: int,
    n_anchor_pairs: int,
    input_dim: int,
    device: torch.device,
    random_seed: Optional[int] = None,
    connected_mode: bool = False,
    anchor_candidates: Optional[torch.Tensor] = None,
    policy: Optional["AnchorSamplingPolicy"] = None,
    n_heads: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate (anchor_pairs_a, anchor_pairs_b), both [n_tables, n_anchor_pairs].

    Args:
        connected_mode: legacy flag; when True (and policy is None), maps to
            AnchorSamplingPolicy.CONNECTED. Prefer passing `policy` directly.
        policy: which AnchorSamplingPolicy to use. Defaults to BALANCED
            (or CONNECTED if connected_mode=True).
        anchor_candidates: optional [n_tables, M] long tensor restricting the
            BALANCED / CONNECTED draws to a per-table candidate set.
    """
    if policy is None:
        effective_policy = (
            AnchorSamplingPolicy.CONNECTED if connected_mode else AnchorSamplingPolicy.BALANCED
        )
    else:
        effective_policy = policy

    if effective_policy == AnchorSamplingPolicy.CANONICAL_DISTINCT:
        return _get_canonical_distinct_pairs(
            n_tables, n_anchor_pairs, input_dim, device, random_seed, n_heads,
        )

    if effective_policy == AnchorSamplingPolicy.CANONICAL_FULL_COVERAGE:
        return _get_canonical_full_coverage_pairs(
            n_tables, n_anchor_pairs, input_dim, device, random_seed, n_heads,
        )

    if effective_policy not in (AnchorSamplingPolicy.BALANCED, AnchorSamplingPolicy.CONNECTED):
        raise ValueError(
            f"Unsupported AnchorSamplingPolicy: {effective_policy}"
        )

    # BALANCED / CONNECTED
    gen = None
    if random_seed is not None:
        gen = torch.Generator(device=device)
        gen.manual_seed(random_seed)

    def get_balanced_indices(total: int, dim: int) -> torch.Tensor:
        num_full_perms = math.ceil(total / dim)
        perm = torch.rand(num_full_perms, dim, device=device, generator=gen).argsort(dim=1)
        return perm.reshape(-1)[:total]

    using_anchor_candidates = anchor_candidates is not None
    if using_anchor_candidates:
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
            if effective_policy == AnchorSamplingPolicy.CONNECTED:
                idx_b = torch.roll(idx_a, shifts=-1, dims=0)
            else:
                idx_b = get_balanced_indices(n_anchor_pairs, max_anchors_per_table)
            anchor_pairs_a[t] = anchor_candidates[t, idx_a]
            anchor_pairs_b[t] = anchor_candidates[t, idx_b]
    else:
        total_needed = n_tables * n_anchor_pairs
        flat_a = get_balanced_indices(total_needed, input_dim)
        if effective_policy == AnchorSamplingPolicy.CONNECTED:
            flat_b = torch.roll(flat_a, shifts=-1, dims=0)
        else:
            flat_b = get_balanced_indices(total_needed, input_dim)
        anchor_pairs_a = flat_a.view(n_tables, n_anchor_pairs)
        anchor_pairs_b = flat_b.view(n_tables, n_anchor_pairs)

    # Ensure a != b everywhere; at most 10 retries on collisions.
    for _ in range(10):
        mask = anchor_pairs_a == anchor_pairs_b
        if not mask.any().item():
            break
        n_collide = int(mask.sum().item())
        anchor_pairs_b = anchor_pairs_b.clone()
        if using_anchor_candidates:
            table_idx, pair_idx = mask.nonzero(as_tuple=True)
            idx_in_candidates = torch.randint(
                0, anchor_candidates.shape[1], (n_collide,),
                device=device, dtype=torch.long, generator=gen,
            )
            anchor_pairs_b[table_idx, pair_idx] = anchor_candidates[table_idx, idx_in_candidates]
        else:
            anchor_pairs_b[mask] = torch.randint(
                0, input_dim, (n_collide,), device=device, dtype=torch.long, generator=gen,
            )

    return anchor_pairs_a, anchor_pairs_b


# =============================================================================
# Positional-encoding helpers (used by lut_attention)
# =============================================================================

def logarithmic_pe_buckets(num_buckets: int, seq_len: int, device: torch.device) -> torch.Tensor:
    """Allocate positional-encoding buckets with a logarithmic tail.

    For positions < B_half (num_buckets // 2): bucket = position.
    For positions >= B_half: bucket = B_half + int(scale * log(pos / B_half)).
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
    """Relative positional-encoding matrix: RPE[i, j] = buckets[max(0, i - j)]."""
    indices = torch.arange(seq_len, device=device)
    diff = indices.unsqueeze(1) - indices.unsqueeze(0)
    diff = diff.clamp(min=0)
    diff = diff.clamp(max=len(buckets) - 1)
    return buckets[diff]
