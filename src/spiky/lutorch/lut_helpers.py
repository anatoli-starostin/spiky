"""
Helper functions and shared enums for LUT-based components.
"""
import math
from enum import Enum
from typing import Optional, Tuple

import torch


class AnchorSamplingPolicy(str, Enum):
    BALANCED = "balanced"         # default: balanced randperm coverage, independent a/b
    CONNECTED = "connected"       # flat_b is flat_a shifted by 1 (pairs share indices)
    DISCONNECTED = "disconnected" # per table: 2*nap distinct indices, no index reuse
    FULL_COVERAGE = "full_coverage" # all unique pairs tiled across tables, a != b guaranteed
    DISCONNECTED_FULL_COVERAGE = "disconnected_full_coverage"  # DISCONNECTED + greedy resampling to cover all pairs
    HIERARCHICAL = "hierarchical"  # multi-scale fragments; tph is upper bound, actual count auto-computed
    MULTISCALE = "multiscale"      # sliding windows at all integer anchor distances; tph is upper bound
    CONV2D = "conv2d"              # 2D conv-style; input_dim must be perfect square; nap must be 8; tph is upper bound


class SelfExcitementMode(str, Enum):
    LINEAR = "linear"           # y_o = f_o * mean(|f|)
    QUADRATIC = "quadratic"     # y_o = f_o * mean(|f|)^2
    EXPONENTIAL = "exponential" # y_o = f_o * exp(mean(|f|))


def compute_multiscale_n_tables(
    input_dim: int,
    n_anchor_pairs: int,
    tph: Optional[int] = None,
) -> int:
    """
    Compute the actual number of tables used by MULTISCALE anchor sampling.

    For distance d=1,2,3,...: n_tables_d = input_dim - n_anchor_pairs * d.
    Accumulates until n_tables_d <= 0, then caps at tph if provided.
    """
    total = 0
    d = 1
    while True:
        n = input_dim - n_anchor_pairs * d
        if n <= 0:
            break
        total += n
        d += 1
    if tph is not None:
        total = min(total, tph)
    return total


def compute_hierarchical_n_tables(
    input_dim: int,
    n_anchor_pairs: int,
    tph: Optional[int] = None,
) -> int:
    """
    Compute the actual number of tables used by HIERARCHICAL anchor sampling.

    Window always slides by 1. Distance between anchors doubles each level:
    level s (distance = 2^(s-1)): n_tables_s = input_dim - n_anchor_pairs * 2^(s-1).
    Accumulates until n_tables_s <= 0, then caps at tph if provided.
    """
    total = 0
    s = 1
    while True:
        dist = 2 ** (s - 1)
        n = input_dim - n_anchor_pairs * dist
        if n <= 0:
            break
        total += n
        s += 1
    if tph is not None:
        total = min(total, tph)
    return total


def compute_conv2d_n_tables(
    input_dim: int,
    tph: Optional[int] = None,
) -> int:
    """
    Compute the actual number of tables used by CONV2D anchor sampling.

    Treats input_dim as an H×H grid (H = sqrt(input_dim)).
    For dilation d=1,2,3,...: slides a 3×3 dilated kernel (stride=1) over all valid top-left
    positions (rows 0..H-1-2d, cols 0..H-1-2d).
    nap is always 8: 9 grid points sorted by 1D index → 8 sequential (a[i], a[i+1]) pairs.
    Accumulates until no valid positions remain, then caps at tph if provided.
    """
    H = int(math.isqrt(input_dim))
    if H * H != input_dim:
        raise ValueError(f"CONV2D requires input_dim to be a perfect square, got {input_dim}")
    total = 0
    d = 1
    while True:
        max_r = H - 1 - 2 * d
        max_c = H - 1 - 2 * d
        if max_r < 0 or max_c < 0:
            break
        total += (max_r + 1) * (max_c + 1)
        d += 1
    if tph is not None:
        total = min(total, tph)
    return total


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
    shuffle_per_head: bool = True,
    exclusion_sets: Optional[list] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate anchor pairs with balanced coverage over input dimensions.
    Matches spike_QK MultiHeadLUT anchor init: indices from concatenated randperms
    so each dimension 0..input_dim-1 appears roughly equally often.

    Args:
        connected_mode: Deprecated. Use policy=AnchorSamplingPolicy.CONNECTED instead.
        policy: AnchorSamplingPolicy controlling how pairs are sampled. When provided,
                takes precedence over connected_mode.
        exclusion_sets: Optional list of index sets (lists/tuples of ints). A pair (a, b)
                is excluded if both a and b belong to the same set. Only applies to
                FULL_COVERAGE and DISCONNECTED_FULL_COVERAGE policies.

    Returns:
        anchor_pairs_a: [n_tables, n_anchor_pairs] int64
        anchor_pairs_b: [n_tables, n_anchor_pairs] int64
    """
    # Resolve effective policy
    if policy is None:
        effective_policy = AnchorSamplingPolicy.CONNECTED if connected_mode else AnchorSamplingPolicy.BALANCED
    else:
        effective_policy = policy

    gen = None
    if random_seed is not None:
        gen = torch.Generator(device=device)
        gen.manual_seed(random_seed)

    # ── HIERARCHICAL ────────────────────────────────────────────────────────────
    if effective_policy == AnchorSamplingPolicy.HIERARCHICAL:
        all_a = []
        all_b = []
        s = 1
        while True:
            dist = 2 ** (s - 1)
            n_s = input_dim - n_anchor_pairs * dist
            if n_s <= 0:
                break
            for k in range(n_s):
                # Window slides by 1; distance between anchors = dist (doubles each level)
                all_a.append(torch.tensor(
                    [k + i * dist for i in range(n_anchor_pairs)],
                    dtype=torch.long, device=device,
                ))
                all_b.append(torch.tensor(
                    [k + (i + 1) * dist for i in range(n_anchor_pairs)],
                    dtype=torch.long, device=device,
                ))
            s += 1
        base_a = torch.stack(all_a)
        base_b = torch.stack(all_b)
        per_head = n_tables // n_heads
        base_a = base_a[:per_head]
        base_b = base_b[:per_head]
        if not shuffle_per_head:
            return base_a.repeat(n_heads, 1), base_b.repeat(n_heads, 1)
        chunks_a, chunks_b = [], []
        for h in range(n_heads):
            if h == 0:
                chunks_a.append(base_a)
                chunks_b.append(base_b)
            else:
                perm = torch.randperm(input_dim, device=device, generator=gen)
                chunks_a.append(perm[base_a])
                chunks_b.append(perm[base_b])
        return torch.cat(chunks_a, dim=0), torch.cat(chunks_b, dim=0)

    # ── MULTISCALE ──────────────────────────────────────────────────────────────
    if effective_policy == AnchorSamplingPolicy.MULTISCALE:
        all_a = []
        all_b = []
        d = 1
        while True:
            n_d = input_dim - n_anchor_pairs * d
            if n_d <= 0:
                break
            for k in range(n_d):
                all_a.append(torch.tensor(
                    [k + i * d for i in range(n_anchor_pairs)],
                    dtype=torch.long, device=device,
                ))
                all_b.append(torch.tensor(
                    [k + (i + 1) * d for i in range(n_anchor_pairs)],
                    dtype=torch.long, device=device,
                ))
            d += 1
        base_a = torch.stack(all_a)
        base_b = torch.stack(all_b)
        per_head = n_tables // n_heads
        base_a = base_a[:per_head]
        base_b = base_b[:per_head]
        if not shuffle_per_head:
            return base_a.repeat(n_heads, 1), base_b.repeat(n_heads, 1)
        chunks_a, chunks_b = [], []
        for h in range(n_heads):
            if h == 0:
                chunks_a.append(base_a)
                chunks_b.append(base_b)
            else:
                perm = torch.randperm(input_dim, device=device, generator=gen)
                chunks_a.append(perm[base_a])
                chunks_b.append(perm[base_b])
        return torch.cat(chunks_a, dim=0), torch.cat(chunks_b, dim=0)

    # ── CONV2D ──────────────────────────────────────────────────────────────────
    if effective_policy == AnchorSamplingPolicy.CONV2D:
        if n_anchor_pairs != 8:
            raise ValueError(
                f"CONV2D policy requires n_anchor_pairs == 8, got {n_anchor_pairs}"
            )
        H = int(math.isqrt(input_dim))
        if H * H != input_dim:
            raise ValueError(f"CONV2D requires input_dim to be a perfect square, got {input_dim}")
        all_a = []
        all_b = []
        d = 1
        while True:
            max_r = H - 1 - 2 * d
            max_c = H - 1 - 2 * d
            if max_r < 0 or max_c < 0:
                break
            for r in range(max_r + 1):
                for c in range(max_c + 1):
                    pts = sorted(
                        (r + dr * d) * H + (c + dc * d)
                        for dr in range(3) for dc in range(3)
                    )
                    all_a.append(torch.tensor(pts[:8], dtype=torch.long, device=device))
                    all_b.append(torch.tensor(pts[1:], dtype=torch.long, device=device))
            d += 1
        base_a = torch.stack(all_a)
        base_b = torch.stack(all_b)
        per_head = n_tables // n_heads
        base_a = base_a[:per_head]
        base_b = base_b[:per_head]
        if not shuffle_per_head:
            return base_a.repeat(n_heads, 1), base_b.repeat(n_heads, 1)
        chunks_a, chunks_b = [], []
        for h in range(n_heads):
            if h == 0:
                chunks_a.append(base_a)
                chunks_b.append(base_b)
            else:
                perm = torch.randperm(input_dim, device=device, generator=gen)
                chunks_a.append(perm[base_a])
                chunks_b.append(perm[base_b])
        return torch.cat(chunks_a, dim=0), torch.cat(chunks_b, dim=0)

    # ── DISCONNECTED ────────────────────────────────────────────────────────────
    if effective_policy == AnchorSamplingPolicy.DISCONNECTED:
        if input_dim < 2 * n_anchor_pairs:
            raise ValueError(
                f"DISCONNECTED policy requires input_dim ({input_dim}) >= 2*n_anchor_pairs ({2*n_anchor_pairs})"
            )
        anchor_pairs_a = torch.empty(n_tables, n_anchor_pairs, dtype=torch.long, device=device)
        anchor_pairs_b = torch.empty_like(anchor_pairs_a)
        for t in range(n_tables):
            perm = torch.rand(input_dim, device=device, generator=gen).argsort()[:2 * n_anchor_pairs]
            anchor_pairs_a[t] = perm[0::2]
            anchor_pairs_b[t] = perm[1::2]
        return anchor_pairs_a, anchor_pairs_b

    # ── FULL_COVERAGE ───────────────────────────────────────────────────────────
    if effective_policy == AnchorSamplingPolicy.FULL_COVERAGE:
        # Enumerate all unique pairs from upper triangle
        all_i, all_j = torch.triu_indices(input_dim, input_dim, offset=1, device=device)

        # Filter out pairs where both indices belong to the same exclusion set
        if exclusion_sets is not None:
            # Build per-index set membership: set_id[idx] = bitmask of sets it belongs to
            set_mask = torch.zeros(input_dim, dtype=torch.long, device=device)
            for s_idx, s in enumerate(exclusion_sets):
                for idx in s:
                    set_mask[idx] |= (1 << s_idx)
            # A pair is excluded if set_mask[i] & set_mask[j] != 0
            # (i.e., they share at least one common set)
            shared = set_mask[all_i] & set_mask[all_j]
            keep = shared == 0
            all_i = all_i[keep]
            all_j = all_j[keep]

        n_unique = all_i.shape[0]
        if n_unique == 0:
            raise ValueError("exclusion_sets filtered out all pairs — no valid anchor pairs remain")
        total_slots = n_tables * n_anchor_pairs
        # Tile shuffled pairs to fill all slots
        repeats = math.ceil(total_slots / n_unique)
        perm = torch.rand(repeats, n_unique, device=device, generator=gen).argsort(dim=1).reshape(-1)[:total_slots]
        flat_a = all_i[perm]
        flat_b = all_j[perm]
        return flat_a.view(n_tables, n_anchor_pairs), flat_b.view(n_tables, n_anchor_pairs)

    # ── DISCONNECTED_FULL_COVERAGE ──────────────────────────────────────────────
    if effective_policy == AnchorSamplingPolicy.DISCONNECTED_FULL_COVERAGE:
        if input_dim < 2 * n_anchor_pairs:
            raise ValueError(
                f"DISCONNECTED_FULL_COVERAGE policy requires input_dim ({input_dim}) >= 2*n_anchor_pairs ({2*n_anchor_pairs})"
            )
        # Step 1: sample all tables with DISCONNECTED constraint
        anchor_pairs_a = torch.empty(n_tables, n_anchor_pairs, dtype=torch.long, device=device)
        anchor_pairs_b = torch.empty_like(anchor_pairs_a)
        for t in range(n_tables):
            perm = torch.rand(input_dim, device=device, generator=gen).argsort()[:2 * n_anchor_pairs]
            anchor_pairs_a[t] = perm[0::2]
            anchor_pairs_b[t] = perm[1::2]

        # Step 2: compute covered pairs (canonical: (min, max))
        all_unique_i, all_unique_j = torch.triu_indices(input_dim, input_dim, offset=1, device=device)
        n_unique = all_unique_i.shape[0]
        # Encode each pair as i * input_dim + j (canonical: i < j)
        pair_key = lambda a, b: torch.minimum(a, b) * input_dim + torch.maximum(a, b)

        covered = torch.zeros(input_dim * input_dim, dtype=torch.bool, device=device)
        keys_a = anchor_pairs_a.reshape(-1)
        keys_b = anchor_pairs_b.reshape(-1)
        covered[pair_key(keys_a, keys_b)] = True

        all_pair_keys = all_unique_i * input_dim + all_unique_j  # already canonical (i < j)
        missing_mask = ~covered[all_pair_keys]
        if not missing_mask.any():
            return anchor_pairs_a, anchor_pairs_b

        # Step 3: greedy replacement — find redundant tables and replace them
        missing_i = all_unique_i[missing_mask]  # pairs still uncovered
        missing_j = all_unique_j[missing_mask]

        MAX_STALLS = 5  # give up after this many consecutive forced moves without progress
        stall_count = 0

        for _ in range(n_tables * n_anchor_pairs):  # safety bound
            if not missing_mask.any():
                break

            # Recompute per-table coverage counts
            full_covered = torch.zeros(input_dim * input_dim, dtype=torch.int32, device=device)
            flat_keys = pair_key(anchor_pairs_a.reshape(-1), anchor_pairs_b.reshape(-1))
            full_covered.scatter_add_(0, flat_keys, torch.ones_like(flat_keys, dtype=torch.int32))

            # Score each table by how many of its pairs are unique (appear in no other table)
            unique_counts = torch.tensor(
                [(full_covered[pair_key(anchor_pairs_a[t], anchor_pairs_b[t])] == 1).sum().item()
                 for t in range(n_tables)],
                dtype=torch.int32, device=device,
            )
            # Prefer redundant tables (unique_count==0); fall back to minimum-unique table
            if (unique_counts == 0).any():
                t = int(unique_counts.eq(0).nonzero(as_tuple=True)[0][0].item())
                forced = False
            else:
                t = int(unique_counts.argmin().item())
                forced = True

            n_missing_before = int(missing_mask.sum().item())

            # Pick the first still-missing pair
            m_idx = int(missing_mask.nonzero(as_tuple=True)[0][0].item())
            target_a = all_unique_i[m_idx].item()
            target_b = all_unique_j[m_idx].item()

            # Build a new DISCONNECTED table containing (target_a, target_b) as pair 0
            remaining = [x for x in range(input_dim) if x != target_a and x != target_b]
            extra_count = 2 * (n_anchor_pairs - 1)
            perm_remaining = torch.rand(len(remaining), device=device, generator=gen).argsort()[:extra_count]
            extra_indices = torch.tensor(remaining, device=device)[perm_remaining]
            new_a = torch.cat([torch.tensor([target_a], device=device), extra_indices[0::2]])
            new_b = torch.cat([torch.tensor([target_b], device=device), extra_indices[1::2]])

            # Replace table t; recompute coverage from scratch (covered is cumulative)
            anchor_pairs_a[t] = new_a
            anchor_pairs_b[t] = new_b
            covered.fill_(False)
            covered[pair_key(anchor_pairs_a.reshape(-1), anchor_pairs_b.reshape(-1))] = True
            missing_mask = ~covered[all_pair_keys]

            if forced:
                if int(missing_mask.sum().item()) >= n_missing_before:
                    stall_count += 1
                    if stall_count >= MAX_STALLS:
                        break
                else:
                    stall_count = 0  # made progress, reset

        return anchor_pairs_a, anchor_pairs_b

    # ── BALANCED / CONNECTED ────────────────────────────────────────────────────
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

    # Ensure a != b everywhere (match spike_QK collision handling), max 10 attempts
    max_collision_attempts = 10
    for _ in range(max_collision_attempts):
        mask = anchor_pairs_a == anchor_pairs_b
        if not mask.any().item():
            break
        n_collide = mask.sum().item()
        anchor_pairs_b = anchor_pairs_b.clone()
        if using_anchor_candidates:
            table_idx, pair_idx = mask.nonzero(as_tuple=True)
            idx_in_candidates = torch.randint(
                0,
                anchor_candidates.shape[1],
                (n_collide,),
                device=device,
                dtype=torch.long,
                generator=gen,
            )
            anchor_pairs_b[table_idx, pair_idx] = anchor_candidates[table_idx, idx_in_candidates]
        else:
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

