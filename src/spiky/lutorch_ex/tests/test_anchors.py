"""Tests for the canonical full-coverage anchor samplers (pairs and singles)."""
import pytest
import torch

from spiky.lutorch_ex.anchors import (
    canonical_full_coverage_pairs,
    canonical_full_coverage_singles,
)


def test_shapes_and_canonical_ordering():
    a, b = canonical_full_coverage_pairs(d_in=6, n_groups=4, tph=3, nap=2, seed=0)
    assert a.shape == (4, 3, 2) and b.shape == (4, 3, 2)
    assert a.dtype == torch.long and b.dtype == torch.long
    assert bool((a < b).all()), "every pair must be canonical (a < b)"
    assert bool((a >= 0).all()) and bool((b < 6).all())


def test_within_table_distinctness():
    # P = C(8,2) = 28; slots-per-table small, several tables.
    a, b = canonical_full_coverage_pairs(d_in=8, n_groups=3, tph=5, nap=4, seed=7)
    G, tph, nap = a.shape
    for g in range(G):
        for t in range(tph):
            pairs = {(int(a[g, t, j]), int(b[g, t, j])) for j in range(nap)}
            assert len(pairs) == nap, f"table (g={g},t={t}) has a duplicate pair"


def test_full_coverage_when_slots_cover_pool():
    # d_in=4 -> P = C(4,2) = 6; tph*nap = 3*2 = 6 == P -> each group covers the whole pool.
    d_in, G, tph, nap = 4, 2, 3, 2
    a, b = canonical_full_coverage_pairs(d_in, G, tph, nap, seed=3)
    all_pairs = {(i, j) for i in range(d_in) for j in range(i + 1, d_in)}
    assert len(all_pairs) == 6
    for g in range(G):
        covered = {(int(a[g, t, j]), int(b[g, t, j])) for t in range(tph) for j in range(nap)}
        assert covered == all_pairs, f"group {g} does not cover the full pool"


def test_coverage_with_tile_repeat_keeps_table_distinct():
    # slots (tph*nap=8) > P (C(4,2)=6): tiling repeats the pool; repair must keep each
    # table duplicate-free even across the tile boundary.
    d_in, G, tph, nap = 4, 2, 4, 2
    a, b = canonical_full_coverage_pairs(d_in, G, tph, nap, seed=1)
    for g in range(G):
        for t in range(tph):
            pairs = {(int(a[g, t, j]), int(b[g, t, j])) for j in range(nap)}
            assert len(pairs) == nap


def test_determinism():
    kw = dict(d_in=10, n_groups=3, tph=4, nap=3)
    a1, b1 = canonical_full_coverage_pairs(**kw, seed=42)
    a2, b2 = canonical_full_coverage_pairs(**kw, seed=42)
    assert torch.equal(a1, a2) and torch.equal(b1, b2), "same seed must reproduce exactly"
    a3, b3 = canonical_full_coverage_pairs(**kw, seed=43)
    assert not (torch.equal(a1, a3) and torch.equal(b1, b3)), "different seed should differ"


def test_groups_differ():
    # Each group is seeded seed+g, so groups should generally get different anchors.
    a, b = canonical_full_coverage_pairs(d_in=12, n_groups=2, tph=4, nap=3, seed=5)
    assert not (torch.equal(a[0], a[1]) and torch.equal(b[0], b[1]))


def test_nap_exceeds_pool_raises():
    # d_in=3 -> P = C(3,2) = 3; nap=4 > 3 must raise.
    with pytest.raises(ValueError):
        canonical_full_coverage_pairs(d_in=3, n_groups=1, tph=1, nap=4, seed=0)
    # nap == P is allowed.
    canonical_full_coverage_pairs(d_in=3, n_groups=1, tph=1, nap=3, seed=0)


# --------------------------------------------------------------------------------------
# Single-anchor canonical coverage (anchor_mode="single"): pool is the d_in coordinates.
# --------------------------------------------------------------------------------------

def test_singles_shapes_and_range():
    a = canonical_full_coverage_singles(d_in=6, n_groups=4, tph=3, nap=2, seed=0)
    assert a.shape == (4, 3, 2) and a.dtype == torch.long
    assert bool((a >= 0).all()) and bool((a < 6).all())


def test_singles_within_table_distinctness():
    a = canonical_full_coverage_singles(d_in=8, n_groups=3, tph=5, nap=4, seed=7)
    G, tph, nap = a.shape
    for g in range(G):
        for t in range(tph):
            coords = {int(a[g, t, j]) for j in range(nap)}
            assert len(coords) == nap, f"table (g={g},t={t}) repeats a coordinate"


def test_singles_full_coverage_is_permutation():
    # tph*nap == d_in -> each group's single anchors are a permutation of range(d_in).
    d_in, G, tph, nap = 6, 3, 2, 3
    a = canonical_full_coverage_singles(d_in, G, tph, nap, seed=3)
    for g in range(G):
        assert sorted(a[g].reshape(-1).tolist()) == list(range(d_in)), f"group {g} not a permutation"


def test_singles_coverage_with_tile_repeat_keeps_table_distinct():
    # slots (tph*nap=8) > d_in (3): tiling repeats the pool; repair keeps each table distinct.
    d_in, G, tph, nap = 3, 2, 4, 2
    a = canonical_full_coverage_singles(d_in, G, tph, nap, seed=1)
    for g in range(G):
        for t in range(tph):
            coords = {int(a[g, t, j]) for j in range(nap)}
            assert len(coords) == nap


def test_singles_determinism():
    kw = dict(d_in=10, n_groups=3, tph=4, nap=3)
    a1 = canonical_full_coverage_singles(**kw, seed=42)
    a2 = canonical_full_coverage_singles(**kw, seed=42)
    assert torch.equal(a1, a2), "same seed must reproduce exactly"
    a3 = canonical_full_coverage_singles(**kw, seed=43)
    assert not torch.equal(a1, a3), "different seed should differ"


def test_singles_groups_differ():
    a = canonical_full_coverage_singles(d_in=12, n_groups=2, tph=4, nap=3, seed=5)
    assert not torch.equal(a[0], a[1])


def test_singles_nap_exceeds_pool_raises():
    # nap > d_in must raise; nap == d_in is allowed.
    with pytest.raises(ValueError):
        canonical_full_coverage_singles(d_in=3, n_groups=1, tph=1, nap=4, seed=0)
    canonical_full_coverage_singles(d_in=3, n_groups=1, tph=1, nap=3, seed=0)
