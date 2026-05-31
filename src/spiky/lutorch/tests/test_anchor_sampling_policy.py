"""Tests for AnchorSamplingPolicy modes in MultiHeadLut."""
from collections import Counter

import pytest
import torch

from spiky.lutorch.lut_helpers import AnchorSamplingPolicy, UncertaintyMode
from spiky.lutorch.multi_head_lut import MultiHeadLut

INPUT_DIM = 32
NAP = 4
TPH = 256
N_UNIQUE_PAIRS = INPUT_DIM * (INPUT_DIM - 1) // 2  # 496


def _make_lut(policy, tph=TPH, nap=NAP, seed=42):
    return MultiHeadLut(
        input_dim=INPUT_DIM,
        n_heads=1,
        n_outputs=INPUT_DIM,
        n_anchor_pairs=nap,
        tables_per_head=tph,
        smooth_mode=False,
        n_alternatives=1,
        normalize_weights=False,
        calibrate_output=False,
        initial_weights_noise=0.001,
        uncertainty_mode=UncertaintyMode.INVERSE_L1,
        random_seed=seed,
        device=torch.device("cpu"),
        anchor_sampling_policy=policy,
    )


def _pair_stats(lut):
    """Return (n_covered, all_disconnected) for the lut's anchor pairs."""
    a = lut.lookup.anchor_pairs_a  # [n_tables, nap]
    b = lut.lookup.anchor_pairs_b
    n_tables, nap = a.shape

    pairs = Counter(
        (min(a[t, p].item(), b[t, p].item()), max(a[t, p].item(), b[t, p].item()))
        for t in range(n_tables)
        for p in range(nap)
    )
    n_covered = len(pairs)
    all_disconnected = all(
        len(set(a[t].tolist() + b[t].tolist())) == 2 * nap
        for t in range(n_tables)
    )
    return n_covered, all_disconnected


def _make_lut_for_policy(policy):
    """Return a lut appropriate for the given policy.

    - CONV2D needs input_dim=64 (square dim) and nap=8.
    - CONNECTED_TRIPLETS needs nap divisible by 3.
    - CONNECTED_QUADRUPLES needs nap divisible by 6.
    """
    if policy == AnchorSamplingPolicy.CONV2D:
        return MultiHeadLut(
            input_dim=64, n_heads=1, n_outputs=64, n_anchor_pairs=8,
            tables_per_head=9999, smooth_mode=False, n_alternatives=1,
            normalize_weights=False, calibrate_output=False, initial_weights_noise=0.001,
            uncertainty_mode=UncertaintyMode.INVERSE_L1, random_seed=42,
            device=torch.device("cpu"), anchor_sampling_policy=policy,
        )
    if policy == AnchorSamplingPolicy.CONNECTED_TRIPLETS:
        return _make_lut(policy, nap=6)  # smallest multiple of 3 with reasonable coverage
    if policy == AnchorSamplingPolicy.CONNECTED_QUADRUPLES:
        return _make_lut(policy, nap=6)  # smallest multiple of 6
    return _make_lut(policy)


def test_no_collisions_all_policies():
    """All policies must produce a != b for every pair."""
    for policy in AnchorSamplingPolicy:
        lut = _make_lut_for_policy(policy)
        a, b = lut.lookup.anchor_pairs_a, lut.lookup.anchor_pairs_b
        assert (a == b).sum().item() == 0, f"{policy.value}: collision detected"


def test_disconnected_all_indices_distinct():
    """DISCONNECTED: all 2*nap indices within each table must be distinct."""
    lut = _make_lut(AnchorSamplingPolicy.DISCONNECTED)
    a, b = lut.lookup.anchor_pairs_a, lut.lookup.anchor_pairs_b
    n_tables, nap = a.shape
    for t in range(n_tables):
        indices = set(a[t].tolist() + b[t].tolist())
        assert len(indices) == 2 * nap, f"table {t}: expected {2*nap} distinct indices, got {len(indices)}"


def test_full_coverage_covers_all_pairs():
    """FULL_COVERAGE: all N_UNIQUE_PAIRS must appear at least once."""
    lut = _make_lut(AnchorSamplingPolicy.FULL_COVERAGE)
    n_covered, _ = _pair_stats(lut)
    assert n_covered == N_UNIQUE_PAIRS, f"expected {N_UNIQUE_PAIRS} pairs covered, got {n_covered}"


def test_balanced_partial_coverage():
    """BALANCED: should cover most but not necessarily all pairs at tph=256."""
    lut = _make_lut(AnchorSamplingPolicy.BALANCED)
    n_covered, all_disconnected = _pair_stats(lut)
    assert n_covered > 0
    assert n_covered <= N_UNIQUE_PAIRS
    # Not expected to be fully disconnected
    assert not all_disconnected


def test_canonical_distinct_all_canonical():
    """CANONICAL_DISTINCT: every pair has a < b (canonical orientation)."""
    lut = _make_lut(AnchorSamplingPolicy.CANONICAL_DISTINCT)
    a, b = lut.lookup.anchor_pairs_a, lut.lookup.anchor_pairs_b
    assert (a < b).all().item(), "some pair has a >= b in CANONICAL_DISTINCT mode"


def test_canonical_distinct_pairs_distinct_per_table():
    """CANONICAL_DISTINCT: each table has nap distinct canonical pairs."""
    lut = _make_lut(AnchorSamplingPolicy.CANONICAL_DISTINCT)
    a, b = lut.lookup.anchor_pairs_a, lut.lookup.anchor_pairs_b
    n_tables, nap = a.shape
    for t in range(n_tables):
        pairs_in_table = {
            (min(a[t, p].item(), b[t, p].item()), max(a[t, p].item(), b[t, p].item()))
            for p in range(nap)
        }
        assert len(pairs_in_table) == nap, \
            f"table {t}: expected {nap} distinct canonical pairs, got {len(pairs_in_table)}"


def test_canonical_distinct_rejects_too_many_pairs():
    """CANONICAL_DISTINCT: must raise if n_anchor_pairs > C(input_dim, 2)."""
    with pytest.raises(ValueError, match="CANONICAL_DISTINCT"):
        MultiHeadLut(
            input_dim=4, n_heads=1, n_outputs=4,
            n_anchor_pairs=7,  # > C(4,2) = 6
            tables_per_head=8, smooth_mode=False, n_alternatives=1,
            normalize_weights=False, calibrate_output=False, initial_weights_noise=0.001,
            uncertainty_mode=UncertaintyMode.INVERSE_L1, random_seed=42,
            device=torch.device("cpu"),
            anchor_sampling_policy=AnchorSamplingPolicy.CANONICAL_DISTINCT,
        )


def test_disconnected_not_full_coverage():
    """DISCONNECTED does not guarantee full pair coverage."""
    lut = _make_lut(AnchorSamplingPolicy.DISCONNECTED)
    n_covered, all_disconnected = _pair_stats(lut)
    assert all_disconnected
    # Coverage may be less than full (disconnected constraint restricts pairing)
    assert n_covered <= N_UNIQUE_PAIRS


def test_forward_all_policies(device):
    """All policies produce valid forward pass output."""
    x_default = torch.randn(8, INPUT_DIM, device=device)
    x_conv2d = torch.randn(8, 64, device=device)
    for policy in AnchorSamplingPolicy:
        # _make_lut_for_policy handles per-policy quirks (CONV2D dim,
        # CONNECTED_TRIPLETS / CONNECTED_QUADRUPLES nap divisibility).
        lut = _make_lut_for_policy(policy).to(device)
        if policy == AnchorSamplingPolicy.CONV2D:
            out = lut(x_conv2d)
            assert out.shape == (8, 1, 64), f"{policy.value}: unexpected output shape {out.shape}"
        else:
            out = lut(x_default)
            assert out.shape == (8, 1, INPUT_DIM), f"{policy.value}: unexpected output shape {out.shape}"
        assert not torch.isnan(out).any(), f"{policy.value}: NaN in output"


def test_backwards_compat_connected_anchors_mode():
    """connected_anchors_mode=True still works and is equivalent to CONNECTED policy."""
    lut_bool = MultiHeadLut(
        input_dim=INPUT_DIM, n_heads=1, n_outputs=INPUT_DIM, n_anchor_pairs=NAP,
        tables_per_head=16, smooth_mode=False, n_alternatives=1,
        normalize_weights=False, calibrate_output=False, initial_weights_noise=0.001,
        uncertainty_mode=UncertaintyMode.INVERSE_L1, random_seed=42,
        device=torch.device("cpu"), connected_anchors_mode=True,
    )
    lut_policy = MultiHeadLut(
        input_dim=INPUT_DIM, n_heads=1, n_outputs=INPUT_DIM, n_anchor_pairs=NAP,
        tables_per_head=16, smooth_mode=False, n_alternatives=1,
        normalize_weights=False, calibrate_output=False, initial_weights_noise=0.001,
        uncertainty_mode=UncertaintyMode.INVERSE_L1, random_seed=42,
        device=torch.device("cpu"), anchor_sampling_policy=AnchorSamplingPolicy.CONNECTED,
    )
    assert torch.equal(lut_bool.lookup.anchor_pairs_a, lut_policy.lookup.anchor_pairs_a)
    assert torch.equal(lut_bool.lookup.anchor_pairs_b, lut_policy.lookup.anchor_pairs_b)


def test_disconnected_requires_sufficient_input_dim():
    """DISCONNECTED raises if input_dim < 2*nap (nap=17 needs 34 > 32)."""
    with pytest.raises(ValueError, match="DISCONNECTED"):
        MultiHeadLut(
            input_dim=32, n_heads=1, n_outputs=32, n_anchor_pairs=17,
            tables_per_head=4, smooth_mode=False, n_alternatives=1,
            normalize_weights=False, calibrate_output=False, initial_weights_noise=0.001,
            uncertainty_mode=UncertaintyMode.INVERSE_L1, random_seed=42,
            device=torch.device("cpu"), anchor_sampling_policy=AnchorSamplingPolicy.DISCONNECTED,
        )


def test_disconnected_full_coverage_covers_all_pairs():
    """DISCONNECTED_FULL_COVERAGE: all N_UNIQUE_PAIRS must be covered AND all tables disconnected."""
    lut = _make_lut(AnchorSamplingPolicy.DISCONNECTED_FULL_COVERAGE)
    n_covered, all_disconnected = _pair_stats(lut)
    assert n_covered == N_UNIQUE_PAIRS, f"expected {N_UNIQUE_PAIRS} pairs covered, got {n_covered}"
    assert all_disconnected, "expected all tables to be disconnected"


def test_disconnected_full_coverage_no_collisions():
    """DISCONNECTED_FULL_COVERAGE: a != b for every pair in every table."""
    lut = _make_lut(AnchorSamplingPolicy.DISCONNECTED_FULL_COVERAGE)
    a, b = lut.lookup.anchor_pairs_a, lut.lookup.anchor_pairs_b
    assert (a == b).sum().item() == 0


def test_disconnected_full_coverage_requires_sufficient_input_dim():
    """DISCONNECTED_FULL_COVERAGE raises if input_dim < 2*nap."""
    with pytest.raises(ValueError, match="DISCONNECTED_FULL_COVERAGE"):
        MultiHeadLut(
            input_dim=32, n_heads=1, n_outputs=32, n_anchor_pairs=17,
            tables_per_head=4, smooth_mode=False, n_alternatives=1,
            normalize_weights=False, calibrate_output=False, initial_weights_noise=0.001,
            uncertainty_mode=UncertaintyMode.INVERSE_L1, random_seed=42,
            device=torch.device("cpu"),
            anchor_sampling_policy=AnchorSamplingPolicy.DISCONNECTED_FULL_COVERAGE,
        )


# ── HIERARCHICAL tests ──────────────────────────────────────────────────────────

def test_hierarchical_table_count_small():
    """
    compute_hierarchical_n_tables for input_dim=8, nap=4.

    dist=1: 8 - 4*1 = 4
    dist=2: 8 - 4*2 = 0 → stop
    Total = 4.
    """
    from spiky.lutorch.lut_helpers import compute_hierarchical_n_tables
    assert compute_hierarchical_n_tables(8, 4) == 4


def test_hierarchical_table_count_tph_cap():
    """tph=3 caps the 4-table result to 3."""
    from spiky.lutorch.lut_helpers import compute_hierarchical_n_tables
    assert compute_hierarchical_n_tables(8, 4, tph=3) == 3


def test_hierarchical_anchor_pairs_exact():
    """
    Verify structural properties of HIERARCHICAL anchor pairs for input_dim=8, nap=4.

    Canonical layout (before per-head permutation):
      dist=1: 8 - 4*1 = 4 tables
      dist=2: 8 - 4*2 = 0 → stop
    """
    from spiky.lutorch.lut_helpers import get_balanced_anchor_pairs
    a, b = get_balanced_anchor_pairs(
        n_tables=4, n_anchor_pairs=4, input_dim=8,
        device=torch.device("cpu"),
        policy=AnchorSamplingPolicy.HIERARCHICAL,
    )
    assert a.shape == (4, 4)
    assert b.shape == (4, 4)
    assert a.min() >= 0 and a.max() < 8
    assert b.min() >= 0 and b.max() < 8
    assert (a == b).sum().item() == 0


def test_hierarchical_anchor_pairs_exact_dim16():
    """
    Verify structural properties of HIERARCHICAL anchor pairs for input_dim=16, nap=4.

    Canonical layout (before per-head permutation):
      dist=1: 16 - 4*1 = 12 tables
      dist=2: 16 - 4*2 = 8  tables
      dist=4: 16 - 4*4 = 0 → stop
    Total: 20 tables per head.
    """
    from spiky.lutorch.lut_helpers import compute_hierarchical_n_tables, get_balanced_anchor_pairs
    assert compute_hierarchical_n_tables(16, 4) == 20

    a, b = get_balanced_anchor_pairs(
        n_tables=20, n_anchor_pairs=4, input_dim=16,
        device=torch.device("cpu"),
        policy=AnchorSamplingPolicy.HIERARCHICAL,
    )
    assert a.shape == (20, 4)
    assert b.shape == (20, 4)
    assert a.min() >= 0 and a.max() < 16
    assert b.min() >= 0 and b.max() < 16
    assert (a == b).sum().item() == 0


def test_hierarchical_heads_get_different_anchors():
    """Each head must have different anchor pairs (independent permutations)."""
    from spiky.lutorch.lut_helpers import get_balanced_anchor_pairs, compute_hierarchical_n_tables
    n_heads = 4
    per_head = compute_hierarchical_n_tables(16, 4)  # 20
    a, b = get_balanced_anchor_pairs(
        n_tables=n_heads * per_head, n_anchor_pairs=4, input_dim=16,
        device=torch.device("cpu"), random_seed=42,
        policy=AnchorSamplingPolicy.HIERARCHICAL, n_heads=n_heads,
    )
    assert a.shape == (n_heads * per_head, 4)
    head_anchors = [a[h * per_head:(h + 1) * per_head] for h in range(n_heads)]
    # All heads should have different anchor patterns
    for i in range(n_heads):
        for j in range(i + 1, n_heads):
            assert not torch.equal(head_anchors[i], head_anchors[j]), \
                f"heads {i} and {j} have identical anchor pairs"


def test_multiscale_heads_get_different_anchors():
    """Each head must have different anchor pairs (independent permutations)."""
    from spiky.lutorch.lut_helpers import get_balanced_anchor_pairs, compute_multiscale_n_tables
    n_heads = 4
    per_head = compute_multiscale_n_tables(16, 4)
    a, b = get_balanced_anchor_pairs(
        n_tables=n_heads * per_head, n_anchor_pairs=4, input_dim=16,
        device=torch.device("cpu"), random_seed=42,
        policy=AnchorSamplingPolicy.MULTISCALE, n_heads=n_heads,
    )
    assert a.shape == (n_heads * per_head, 4)
    head_anchors = [a[h * per_head:(h + 1) * per_head] for h in range(n_heads)]
    for i in range(n_heads):
        for j in range(i + 1, n_heads):
            assert not torch.equal(head_anchors[i], head_anchors[j]), \
                f"heads {i} and {j} have identical anchor pairs"


def test_hierarchical_tables_per_head_overridden():
    """MultiHeadLut with HIERARCHICAL sets tables_per_head=4 for input_dim=8, nap=4."""
    lut = MultiHeadLut(
        input_dim=8, n_heads=2, n_outputs=4, n_anchor_pairs=4,
        tables_per_head=1000,  # large tph, should be capped to 4 (8-4*1=4, 8-4*2=0→stop)
        smooth_mode=False, n_alternatives=1,
        normalize_weights=False, calibrate_output=False, initial_weights_noise=0.0,
        uncertainty_mode=UncertaintyMode.INVERSE_L1, random_seed=42,
        device=torch.device("cpu"),
        anchor_sampling_policy=AnchorSamplingPolicy.HIERARCHICAL,
    )
    assert lut.tables_per_head == 4


def test_hierarchical_no_out_of_bounds():
    """All anchor indices must be within [0, input_dim)."""
    lut = MultiHeadLut(
        input_dim=8, n_heads=2, n_outputs=4, n_anchor_pairs=4,
        tables_per_head=1000,
        smooth_mode=False, n_alternatives=1,
        normalize_weights=False, calibrate_output=False, initial_weights_noise=0.0,
        uncertainty_mode=UncertaintyMode.INVERSE_L1, random_seed=42,
        device=torch.device("cpu"),
        anchor_sampling_policy=AnchorSamplingPolicy.HIERARCHICAL,
    )
    a, b = lut.lookup.anchor_pairs_a, lut.lookup.anchor_pairs_b
    assert a.min() >= 0 and a.max() < 8
    assert b.min() >= 0 and b.max() < 8


def test_hierarchical_forward():
    """HIERARCHICAL produces valid forward output for input_dim=8, nap=4."""
    lut = MultiHeadLut(
        input_dim=8, n_heads=2, n_outputs=4, n_anchor_pairs=4,
        tables_per_head=1000,
        smooth_mode=False, n_alternatives=1,
        normalize_weights=False, calibrate_output=False, initial_weights_noise=0.0,
        uncertainty_mode=UncertaintyMode.INVERSE_L1, random_seed=42,
        device=torch.device("cpu"),
        anchor_sampling_policy=AnchorSamplingPolicy.HIERARCHICAL,
    )
    x = torch.randn(4, 8)
    out = lut(x)
    assert out.shape == (4, 2, 4)
    assert not torch.isnan(out).any()


def test_full_coverage_exclusion_sets_no_within_set_pairs():
    """With exclusion_sets, no anchor pair should have both indices in the same set."""
    from spiky.lutorch.lut_helpers import get_balanced_anchor_pairs

    E, pos_dim = 32, 16
    input_dim = 2 * E + pos_dim
    sets = [list(range(E)), list(range(E, 2 * E)), list(range(2 * E, input_dim))]

    a, b = get_balanced_anchor_pairs(
        n_tables=64, n_anchor_pairs=5, input_dim=input_dim,
        device=torch.device("cpu"), random_seed=42,
        policy=AnchorSamplingPolicy.FULL_COVERAGE,
        exclusion_sets=sets,
    )
    # Check no pair has both indices in the same set
    for s in sets:
        s_set = set(s)
        for t in range(a.shape[0]):
            for p in range(a.shape[1]):
                ai, bi = a[t, p].item(), b[t, p].item()
                assert not (ai in s_set and bi in s_set), \
                    f"table {t} pair {p}: ({ai}, {bi}) both in set {s[:3]}..."


def test_full_coverage_exclusion_sets_preserves_coverage():
    """Exclusion sets should still tile all valid cross-set pairs."""
    from spiky.lutorch.lut_helpers import get_balanced_anchor_pairs

    input_dim = 16
    sets = [list(range(8)), list(range(8, 16))]
    # Cross-set pairs: 8 * 8 = 64. Within-set: C(8,2)*2 = 56. Total: C(16,2)=120.
    n_cross = 8 * 8  # pairs where one in [0..7], other in [8..15]

    a, b = get_balanced_anchor_pairs(
        n_tables=32, n_anchor_pairs=4, input_dim=input_dim,
        device=torch.device("cpu"), random_seed=42,
        policy=AnchorSamplingPolicy.FULL_COVERAGE,
        exclusion_sets=sets,
    )
    # All generated pairs must be cross-set
    for t in range(a.shape[0]):
        for p in range(a.shape[1]):
            ai, bi = a[t, p].item(), b[t, p].item()
            same_set = (ai < 8 and bi < 8) or (ai >= 8 and bi >= 8)
            assert not same_set, f"({ai}, {bi}) in same set"

    # With enough tables, all cross-set pairs should appear at least once
    seen = set()
    for t in range(a.shape[0]):
        for p in range(a.shape[1]):
            pair = (min(a[t, p].item(), b[t, p].item()), max(a[t, p].item(), b[t, p].item()))
            seen.add(pair)
    assert len(seen) == n_cross, f"Expected {n_cross} unique cross-set pairs, got {len(seen)}"


def test_exclusion_sets_via_multi_head_lut():
    """MultiHeadLut correctly passes exclusion_sets to anchor pair generation."""
    E, pos_dim = 16, 8
    input_dim = 2 * E + pos_dim
    sets = [list(range(E)), list(range(E, 2 * E)), list(range(2 * E, input_dim))]

    lut = MultiHeadLut(
        input_dim=input_dim, n_heads=1, n_outputs=16,
        n_anchor_pairs=4, tables_per_head=32,
        smooth_mode=False, n_alternatives=1,
        normalize_weights=False, calibrate_output=False,
        initial_weights_noise=0.001,
        uncertainty_mode=UncertaintyMode.INVERSE_L1,
        random_seed=42, device=torch.device("cpu"),
        anchor_sampling_policy=AnchorSamplingPolicy.FULL_COVERAGE,
        exclusion_sets=sets,
    )
    a = lut.lookup.anchor_pairs_a
    b = lut.lookup.anchor_pairs_b
    for s in sets:
        s_set = set(s)
        for t in range(a.shape[0]):
            for p in range(a.shape[1]):
                ai, bi = a[t, p].item(), b[t, p].item()
                assert not (ai in s_set and bi in s_set)
