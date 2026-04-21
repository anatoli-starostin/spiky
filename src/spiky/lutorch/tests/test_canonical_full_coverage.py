"""Tests for AnchorSamplingPolicy.CANONICAL_FULL_COVERAGE.

Coverage property: when n_tables * n_anchor_pairs >= P, every canonical pair
appears at least once. Within-table distinctness: each table's n_anchor_pairs
slots hold distinct canonical pairs.
"""
import math

import pytest
import torch

from spiky.lutorch.lut_helpers import AnchorSamplingPolicy, get_balanced_anchor_pairs


def _run(n_tables, n_anchor_pairs, input_dim, seed=42, n_heads=1, partition_sets=None):
    a, b = get_balanced_anchor_pairs(
        n_tables=n_tables,
        n_anchor_pairs=n_anchor_pairs,
        input_dim=input_dim,
        device=torch.device("cpu"),
        random_seed=seed,
        policy=AnchorSamplingPolicy.CANONICAL_FULL_COVERAGE,
        n_heads=n_heads,
        partition_sets=partition_sets,
    )
    return a, b


def _coverage_and_distinctness(a, b, input_dim, partition_sets=None):
    # All pairs canonical (a < b).
    assert (a < b).all().item(), "non-canonical pair (a >= b)"
    # Intra-table distinctness.
    n_tables, nap = a.shape
    for t in range(n_tables):
        pairs_t = {(int(a[t, k]), int(b[t, k])) for k in range(nap)}
        assert len(pairs_t) == nap, f"duplicate in table {t}: {a[t].tolist()}"

    # Coverage of expected pool.
    if partition_sets is None:
        P_full = input_dim * (input_dim - 1) // 2
        covered = {(int(a_), int(b_)) for a_, b_ in zip(a.flatten(), b.flatten())}
        total = P_full
    else:
        part_id = [-1] * input_dim
        for s_idx, s in enumerate(partition_sets):
            for i in s:
                part_id[i] = s_idx
        expected = {
            (i, j) for i in range(input_dim) for j in range(i + 1, input_dim)
            if part_id[i] == part_id[j]
        }
        total = len(expected)
        covered = {(int(a_), int(b_)) for a_, b_ in zip(a.flatten(), b.flatten())}
        covered &= expected  # shouldn't matter but clean
    return total, covered


def test_full_coverage_user_case():
    """User's specific request: N=32, nap=8, tph=64. All 496 pairs covered."""
    a, b = _run(n_tables=64, n_anchor_pairs=8, input_dim=32)
    total, covered = _coverage_and_distinctness(a, b, input_dim=32)
    assert total == 496
    assert len(covered) == 496, f"expected 496 pairs, got {len(covered)}"


def test_full_coverage_nap_divides_P_exactly():
    """N=16, nap=3, P=120. tph chosen so tph*nap=120 → exact one-pass cover."""
    a, b = _run(n_tables=40, n_anchor_pairs=3, input_dim=16)
    total, covered = _coverage_and_distinctness(a, b, input_dim=16)
    assert len(covered) == total  # 120 / 120 covered


def test_full_coverage_non_divisor_nap():
    """nap does not divide P — repair kicks in. Coverage must still hold."""
    # N=10, P = 45, nap=4. 45 % 4 = 1 (boundary misalignment).
    # n_tables=12 → slots=48 >= 45 — coverage expected.
    a, b = _run(n_tables=12, n_anchor_pairs=4, input_dim=10)
    total, covered = _coverage_and_distinctness(a, b, input_dim=10)
    assert len(covered) == 45


def test_full_coverage_multi_pass():
    """slots >> P: several full tiles."""
    # N=8, P=28, nap=4, n_tables=21 → slots = 84 = 3*P.
    a, b = _run(n_tables=21, n_anchor_pairs=4, input_dim=8)
    total, covered = _coverage_and_distinctness(a, b, input_dim=8)
    assert len(covered) == 28
    # Each pair should appear ~3 times (84/28 = 3 exactly).
    from collections import Counter
    counts = Counter((int(a_), int(b_)) for a_, b_ in zip(a.flatten(), b.flatten()))
    assert all(c == 3 for c in counts.values()), counts.most_common(3)


def test_full_coverage_with_partition_sets():
    """Partition-restricted pool: coverage over within-partition pairs only."""
    # N=16, 4 heads of size 4: 4 * C(4,2) = 24 within-partition pairs.
    head_parts = [list(range(h * 4, (h + 1) * 4)) for h in range(4)]
    # nap=3, n_tables=10 → slots=30 >= 24 valid pairs.
    a, b = _run(
        n_tables=10, n_anchor_pairs=3, input_dim=16,
        partition_sets=head_parts,
    )
    total, covered = _coverage_and_distinctness(
        a, b, input_dim=16, partition_sets=head_parts,
    )
    assert total == 24
    assert len(covered) == 24

    # All pairs stay within partition.
    part_id = torch.tensor([i // 4 for i in range(16)])
    assert (part_id[a] == part_id[b]).all().item()


def test_full_coverage_under_budget_no_coverage_claim():
    """slots < P: still within-table distinct, but coverage incomplete."""
    a, b = _run(n_tables=10, n_anchor_pairs=4, input_dim=32)
    total, covered = _coverage_and_distinctness(a, b, input_dim=32)
    # 40 < 496, so coverage can't be full — just check distinctness held.
    assert len(covered) == 40
    assert total == 496


def test_full_coverage_bit_permutation_lut_output_pairs():
    """BitPermutationLUT's output pair sampling now uses CFC → full coverage
    for the user's exact configuration."""
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    from spiky.lutorch.bit_permutation_lut import BitPermutationLUT

    lut = BitPermutationLUT(
        n_inputs=32, n_outputs=32, n_heads=1,
        input_nap=6, output_nap=8, tph=64,
        random_seed=42, latent_dtype='bf16',
        device=torch.device('cuda'),
    )
    a = lut.idx_a.cpu().long()
    b = lut.idx_b.cpu().long()
    lo = torch.minimum(a, b).flatten()
    hi = torch.maximum(a, b).flatten()
    covered = {(int(l), int(h)) for l, h in zip(lo.tolist(), hi.tolist())}
    assert len(covered) == 32 * 31 // 2, (
        f"expected full 496 coverage, got {len(covered)}"
    )
