"""Tests for `partition_sets` kwarg: CANONICAL_DISTINCT within-partition sampling.

Covers:
  - lut_helpers.get_balanced_anchor_pairs(..., partition_sets=...)
  - TinyAnchorPairsLookup(partition_sets=...)
  - BitPermutationLUTInput / BitPermutationLUT (partition_sets forwarded)

Semantic: a pair (a, b) is KEPT only if a and b belong to the same partition.
"""
import pytest
import torch

from spiky.lutorch.lut_helpers import AnchorSamplingPolicy, get_balanced_anchor_pairs
from spiky.lutorch.tiny_anchor_pairs_lookup import TinyAnchorPairsLookup


INPUT_DIM = 64
N_HEADS_PART = 4
D_HEAD = INPUT_DIM // N_HEADS_PART  # 16
HEAD_PARTITIONS = [
    list(range(h * D_HEAD, (h + 1) * D_HEAD)) for h in range(N_HEADS_PART)
]


def _partition_id(partition_sets, input_dim):
    part_id = [-1] * input_dim
    for s_idx, s in enumerate(partition_sets):
        for i in s:
            part_id[i] = s_idx
    return part_id


def test_canonical_distinct_partition_all_within_head():
    """Every sampled pair (a, b) has both endpoints in the same head partition."""
    a, b = get_balanced_anchor_pairs(
        n_tables=128,
        n_anchor_pairs=8,
        input_dim=INPUT_DIM,
        device=torch.device("cpu"),
        random_seed=42,
        policy=AnchorSamplingPolicy.CANONICAL_DISTINCT,
        n_heads=1,
        partition_sets=HEAD_PARTITIONS,
    )
    part_id = _partition_id(HEAD_PARTITIONS, INPUT_DIM)
    part_a = torch.tensor([part_id[int(x)] for x in a.flatten()])
    part_b = torch.tensor([part_id[int(x)] for x in b.flatten()])
    assert (part_a == part_b).all().item(), "some pair crosses a head partition"
    assert (a < b).all().item(), "pair not canonical (a < b)"


def test_canonical_distinct_partition_rejects_non_partition():
    """partition_sets must partition [0, input_dim) — missing index raises."""
    bad = [list(range(0, 16)), list(range(16, 32))]  # only covers 0..31 of 64
    with pytest.raises(ValueError, match="must cover every index"):
        get_balanced_anchor_pairs(
            n_tables=4, n_anchor_pairs=4, input_dim=INPUT_DIM,
            device=torch.device("cpu"), random_seed=0,
            policy=AnchorSamplingPolicy.CANONICAL_DISTINCT,
            n_heads=1, partition_sets=bad,
        )


def test_canonical_distinct_partition_rejects_overlap():
    """partition_sets must be disjoint."""
    bad = [list(range(0, 32)), list(range(30, 64))]  # overlap at 30, 31
    with pytest.raises(ValueError, match="disjoint"):
        get_balanced_anchor_pairs(
            n_tables=4, n_anchor_pairs=4, input_dim=INPUT_DIM,
            device=torch.device("cpu"), random_seed=0,
            policy=AnchorSamplingPolicy.CANONICAL_DISTINCT,
            n_heads=1, partition_sets=bad,
        )


def test_canonical_distinct_partition_budget_too_small():
    """n_anchor_pairs must be <= valid within-partition pair count."""
    # Two partitions of size 3; within-partition pairs = 2*C(3,2) = 6
    tiny_parts = [list(range(3)), list(range(3, 6))]
    # 7 > 6 must raise.
    with pytest.raises(ValueError, match="valid within-partition pairs"):
        get_balanced_anchor_pairs(
            n_tables=1, n_anchor_pairs=7, input_dim=6,
            device=torch.device("cpu"), random_seed=0,
            policy=AnchorSamplingPolicy.CANONICAL_DISTINCT,
            n_heads=1, partition_sets=tiny_parts,
        )


def test_canonical_distinct_partition_distinct_per_table():
    """Within-partition pairs sampled per table remain distinct."""
    n_tables, nap = 32, 10
    a, b = get_balanced_anchor_pairs(
        n_tables=n_tables, n_anchor_pairs=nap, input_dim=INPUT_DIM,
        device=torch.device("cpu"), random_seed=7,
        policy=AnchorSamplingPolicy.CANONICAL_DISTINCT,
        n_heads=1, partition_sets=HEAD_PARTITIONS,
    )
    for t in range(n_tables):
        pairs = {(int(a[t, k]), int(b[t, k])) for k in range(nap)}
        assert len(pairs) == nap, f"table {t} has duplicates"


def test_partition_sets_rejected_for_non_canonical_policy():
    with pytest.raises(ValueError, match="only supported with CANONICAL_DISTINCT"):
        get_balanced_anchor_pairs(
            n_tables=4, n_anchor_pairs=4, input_dim=INPUT_DIM,
            device=torch.device("cpu"), random_seed=0,
            policy=AnchorSamplingPolicy.BALANCED,
            n_heads=1, partition_sets=HEAD_PARTITIONS,
        )


def test_tiny_anchor_pairs_lookup_partition_sets():
    """TinyAnchorPairsLookup forwards partition_sets; anchors stay within-partition."""
    lookup = TinyAnchorPairsLookup(
        input_dim=INPUT_DIM,
        n_tables=64,
        n_anchor_pairs=8,
        n_heads=1,
        random_seed=3,
        device=torch.device("cpu"),
        partition_sets=HEAD_PARTITIONS,
    )
    a = lookup.anchor_pairs_a.long()
    b = lookup.anchor_pairs_b.long()
    part_id = torch.tensor(_partition_id(HEAD_PARTITIONS, INPUT_DIM))
    assert (part_id[a] == part_id[b]).all().item(), "cross-partition pair found"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_bit_permutation_lut_partition_sets():
    """BitPermutationLUT forwards partition_sets to its TinyAnchorPairsLookup."""
    from spiky.lutorch.bit_permutation_lut import BitPermutationLUT

    dev = torch.device("cuda")
    lut = BitPermutationLUT(
        n_inputs=INPUT_DIM,
        n_outputs=32,
        n_heads=1,
        input_nap=8,
        output_nap=8,
        tph=32,
        random_seed=1,
        latent_dtype='bf16',
        device=dev,
        partition_sets=HEAD_PARTITIONS,
    )
    a = lut.anchor.anchor_pairs_a.long().cpu()
    b = lut.anchor.anchor_pairs_b.long().cpu()
    part_id = torch.tensor(_partition_id(HEAD_PARTITIONS, INPUT_DIM))
    assert (part_id[a] == part_id[b]).all().item(), "cross-partition pair found"

    # Smoke: forward still runs (no shape mismatch).
    x = torch.randn(4, INPUT_DIM, device=dev)
    out = lut(x)
    assert out.shape == (4, 1, 32 * 31 // 2)


# ─── partition_pair_weights (B1: weighted multinomial sampling) ─────────────


def test_partition_pair_weights_within_partition_constraint():
    """Even with weighting, every sampled pair stays within one partition."""
    parts = HEAD_PARTITIONS  # 4 partitions of 16 dims each
    weights = [4.0, 2.0, 1.0, 0.5]
    a, b = get_balanced_anchor_pairs(
        n_tables=64, n_anchor_pairs=6, input_dim=INPUT_DIM,
        device=torch.device("cpu"), random_seed=0,
        policy=AnchorSamplingPolicy.CANONICAL_FULL_COVERAGE,
        n_heads=1, partition_sets=parts, partition_pair_weights=weights,
    )
    part_id = torch.tensor(_partition_id(parts, INPUT_DIM))
    assert (part_id[a] == part_id[b]).all().item(), "cross-partition pair under weighting"
    assert (a < b).all().item(), "non-canonical pair (a < b violated)"


def test_partition_pair_weights_per_table_distinct():
    """Multinomial-without-replacement keeps within-table distinctness."""
    n_tables, nap = 64, 8
    a, b = get_balanced_anchor_pairs(
        n_tables=n_tables, n_anchor_pairs=nap, input_dim=INPUT_DIM,
        device=torch.device("cpu"), random_seed=1,
        policy=AnchorSamplingPolicy.CANONICAL_FULL_COVERAGE,
        n_heads=1, partition_sets=HEAD_PARTITIONS,
        partition_pair_weights=[3.0, 2.0, 1.0, 1.0],
    )
    for t in range(n_tables):
        pairs = {(int(a[t, k]), int(b[t, k])) for k in range(nap)}
        assert len(pairs) == nap, f"table {t} has duplicate pair under weighted sampling"


def test_partition_pair_weights_empirical_frequency():
    """Per-pair weight ∝ partition weight: empirical freq should match within tolerance."""
    parts = HEAD_PARTITIONS  # 4 equal-size partitions of 16 → equal pool size per partition
    weights = [4.0, 2.0, 1.0, 1.0]
    n_tables, nap = 4096, 6
    a, _ = get_balanced_anchor_pairs(
        n_tables=n_tables, n_anchor_pairs=nap, input_dim=INPUT_DIM,
        device=torch.device("cpu"), random_seed=2025,
        policy=AnchorSamplingPolicy.CANONICAL_FULL_COVERAGE,
        n_heads=1, partition_sets=parts, partition_pair_weights=weights,
    )
    part_id = torch.tensor(_partition_id(parts, INPUT_DIM))
    pair_parts = part_id[a].flatten()
    counts = torch.bincount(pair_parts, minlength=len(parts)).float()
    empirical = counts / counts.sum()
    expected = torch.tensor(weights) / sum(weights)
    # Allow 2 percentage-point tolerance per partition at this sample size.
    assert torch.allclose(empirical, expected, atol=0.02), (
        f"weighted empirical frequencies {empirical.tolist()} "
        f"deviate from expected {expected.tolist()}"
    )


def test_partition_pair_weights_zero_weight_excluded():
    """Zero-weight partition produces no pairs."""
    weights = [1.0, 1.0, 0.0, 0.0]
    a, _ = get_balanced_anchor_pairs(
        n_tables=128, n_anchor_pairs=6, input_dim=INPUT_DIM,
        device=torch.device("cpu"), random_seed=11,
        policy=AnchorSamplingPolicy.CANONICAL_FULL_COVERAGE,
        n_heads=1, partition_sets=HEAD_PARTITIONS,
        partition_pair_weights=weights,
    )
    part_id = torch.tensor(_partition_id(HEAD_PARTITIONS, INPUT_DIM))
    seen_parts = set(part_id[a].flatten().tolist())
    assert seen_parts.issubset({0, 1}), f"zero-weight partitions produced pairs: {seen_parts}"


def test_partition_pair_weights_unequal_pool_sizes():
    """Weights are per-pair: pool sizes don't bias the empirical distribution.
    Partition 0 has 8 dims → C(8,2)=28 pool pairs; partition 1 has 16 → C(16,2)=120.
    Equal weights [1, 1] should still give ≈ 50/50 empirical frequency.
    """
    parts = [list(range(0, 8)), list(range(8, 24))]
    input_dim = 24
    n_tables, nap = 4096, 4
    a, _ = get_balanced_anchor_pairs(
        n_tables=n_tables, n_anchor_pairs=nap, input_dim=input_dim,
        device=torch.device("cpu"), random_seed=99,
        policy=AnchorSamplingPolicy.CANONICAL_FULL_COVERAGE,
        n_heads=1, partition_sets=parts,
        partition_pair_weights=[1.0, 1.0],
    )
    part_id = torch.tensor(_partition_id(parts, input_dim))
    pair_parts = part_id[a].flatten()
    counts = torch.bincount(pair_parts, minlength=2).float()
    # Per-pair uniform over the pool means total mass per partition is
    # proportional to partition pool size: 28 vs 120 → 28/148 vs 120/148.
    expected = torch.tensor([28.0 / 148.0, 120.0 / 148.0])
    empirical = counts / counts.sum()
    assert torch.allclose(empirical, expected, atol=0.02), (
        f"with equal weights the empirical split {empirical.tolist()} "
        f"should match pool-size split {expected.tolist()}"
    )


def test_partition_pair_weights_validation():
    """API errors: length mismatch, negatives, zero sum, missing partition_sets."""
    parts = HEAD_PARTITIONS
    common = dict(
        n_tables=4, n_anchor_pairs=4, input_dim=INPUT_DIM,
        device=torch.device("cpu"), random_seed=0,
        policy=AnchorSamplingPolicy.CANONICAL_FULL_COVERAGE, n_heads=1,
    )

    with pytest.raises(ValueError, match="must match partition_sets"):
        get_balanced_anchor_pairs(
            **common, partition_sets=parts, partition_pair_weights=[1.0, 2.0],  # 2 vs 4
        )
    with pytest.raises(ValueError, match="non-negative"):
        get_balanced_anchor_pairs(
            **common, partition_sets=parts,
            partition_pair_weights=[1.0, -1.0, 1.0, 1.0],
        )
    with pytest.raises(ValueError, match="positive sum"):
        get_balanced_anchor_pairs(
            **common, partition_sets=parts,
            partition_pair_weights=[0.0, 0.0, 0.0, 0.0],
        )
    with pytest.raises(ValueError, match="requires partition_sets"):
        get_balanced_anchor_pairs(
            **common, partition_sets=None,
            partition_pair_weights=[1.0, 1.0, 1.0, 1.0],
        )


def test_partition_pair_weights_rejected_for_canonical_distinct():
    with pytest.raises(ValueError, match="only supported by CANONICAL_FULL_COVERAGE"):
        get_balanced_anchor_pairs(
            n_tables=4, n_anchor_pairs=4, input_dim=INPUT_DIM,
            device=torch.device("cpu"), random_seed=0,
            policy=AnchorSamplingPolicy.CANONICAL_DISTINCT,
            n_heads=1, partition_sets=HEAD_PARTITIONS,
            partition_pair_weights=[1.0, 1.0, 1.0, 1.0],
        )


def test_partition_pair_weights_too_few_positive_pairs():
    """Need >= n_anchor_pairs pool pairs with positive weight."""
    # Two partitions of 3 → C(3,2)=3 pool pairs each. Zero weight on the second.
    # Positive pool = 3 pairs; n_anchor_pairs=5 must raise.
    parts = [list(range(0, 3)), list(range(3, 6))]
    with pytest.raises(ValueError, match="positive weight"):
        get_balanced_anchor_pairs(
            n_tables=2, n_anchor_pairs=5, input_dim=6,
            device=torch.device("cpu"), random_seed=0,
            policy=AnchorSamplingPolicy.CANONICAL_FULL_COVERAGE,
            n_heads=1, partition_sets=parts,
            partition_pair_weights=[1.0, 0.0],
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_bit_permutation_lut_input_pairs_2_heads_of_4():
    """BitPermutationLUT(n_inputs=8, input_nap=3, tph=10) with 2 heads of 4:
    all input anchor pairs stay within a head, cover all 12 within-partition
    pairs (= 2 * C(4,2)), and no table has an intra-table duplicate.
    """
    from spiky.lutorch.bit_permutation_lut import BitPermutationLUT

    dev = torch.device("cuda")
    partition_sets = [[0, 1, 2, 3], [4, 5, 6, 7]]
    lut = BitPermutationLUT(
        n_inputs=8, n_outputs=8, n_heads=1,
        input_nap=3, output_nap=3, tph=10,
        random_seed=42, latent_dtype='bf16',
        device=dev, partition_sets=partition_sets,
    )
    a = lut.anchor.anchor_pairs_a.long().cpu()
    b = lut.anchor.anchor_pairs_b.long().cpu()

    # Canonical (a < b).
    assert (a < b).all().item()

    # No cross-partition pair.
    part_id = torch.tensor(_partition_id(partition_sets, 8))
    assert (part_id[a] == part_id[b]).all().item(), "cross-partition pair found"

    # Full within-partition coverage: 2 * C(4,2) = 12 distinct pairs.
    valid = {(i, j) for g in partition_sets for i in g for j in g if i < j}
    covered = {(int(a[t, k]), int(b[t, k]))
               for t in range(a.shape[0]) for k in range(a.shape[1])}
    assert covered == valid, (
        f"expected to cover all {len(valid)} within-partition pairs, "
        f"got {len(covered)}; missing: {valid - covered}"
    )

    # No intra-table duplicate.
    for t in range(a.shape[0]):
        pairs_t = {(int(a[t, k]), int(b[t, k])) for k in range(a.shape[1])}
        assert len(pairs_t) == a.shape[1], f"table {t} has duplicate pair"
