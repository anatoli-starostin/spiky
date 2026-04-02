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


def test_no_collisions_all_policies():
    """All policies must produce a != b for every pair."""
    for policy in AnchorSamplingPolicy:
        lut = _make_lut(policy)
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


def test_disconnected_not_full_coverage():
    """DISCONNECTED does not guarantee full pair coverage."""
    lut = _make_lut(AnchorSamplingPolicy.DISCONNECTED)
    n_covered, all_disconnected = _pair_stats(lut)
    assert all_disconnected
    # Coverage may be less than full (disconnected constraint restricts pairing)
    assert n_covered <= N_UNIQUE_PAIRS


def test_forward_all_policies(device):
    """All policies produce valid forward pass output."""
    x = torch.randn(8, INPUT_DIM, device=device)
    for policy in AnchorSamplingPolicy:
        lut = _make_lut(policy)
        lut = lut.to(device)
        out = lut(x)
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
