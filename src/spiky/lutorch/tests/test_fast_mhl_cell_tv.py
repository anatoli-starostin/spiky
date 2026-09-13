"""Tests for FastMultiHeadLut.cell_tv, the Hamming-1 cell TV penalty on Gen-2 tables.

Coverage:
  (i)   value and gradient against a plain-torch reference that enumerates every pair (c, c ^ 2^b) explicitly,
        for the shared-input and the block-diagonal (multi_head_input) weight layouts;
  (ii)  equality with LightMultiHeadLUT.cell_tv on the identical table tensor (the same mathematical object);
  (iii) adjacency through the module's own read path: setting the margin signs of one table selects row c of
        weights[t] with c the MSB-packed address, and flipping ONE margin moves the read to row c ^ 2^(NAP-1-j)
        -- a Hamming-1 neighbour in row index, i.e. exactly the pairs cell_tv penalises.
"""
import pytest
import torch

torch._dynamo.config.suppress_errors = True

from spiky.lutorch.fast_multi_head_lut import FastMultiHeadLut
from spiky.lutorch.light_multi_head_lut import LightMultiHeadLUT
from spiky.lutorch.lut_helpers import AnchorSamplingPolicy


def _fast(nap=4, n_heads=2, tph=3, multi_head_input=False, input_dim=12, seed=0, policy=None):
    return FastMultiHeadLut(
        input_dim=input_dim, n_heads=n_heads, n_outputs=5, n_anchor_pairs=nap, tables_per_head=tph,
        multi_head_input=multi_head_input, weight_dtype=torch.float64, use_bf16=False, random_seed=seed,
        initial_weights_noise=0.5, anchor_sampling_policy=policy)


def _reference_tv(W, nap):
    """Explicit enumeration of the Hamming-1 pairs of every table: sum ||W[:, c] - W[:, c ^ 2^b]||^2 over
    c < c ^ 2^b, divided by n_tables * (number of pairs per table = nap * 2^(nap-1))."""
    n_tables, K, _ = W.shape
    total, n_pairs = W.new_zeros(()), 0
    for c in range(K):
        for b in range(nap):
            c2 = c ^ (1 << b)
            if c2 > c:
                total = total + ((W[:, c] - W[:, c2]) ** 2).sum()
                n_pairs += 1
    assert n_pairs == nap * (1 << (nap - 1))
    return total / (n_tables * n_pairs)


@pytest.mark.parametrize("multi_head_input", [False, True])
@pytest.mark.parametrize("nap", [1, 3, 5])
def test_value_and_gradient_match_explicit_pair_reference(multi_head_input, nap):
    m = _fast(nap=nap, multi_head_input=multi_head_input)
    got = m.cell_tv()
    (g_got,) = torch.autograd.grad(got, m.weights)
    W = m.weights.detach().clone().requires_grad_(True)
    ref = _reference_tv(W, nap)
    (g_ref,) = torch.autograd.grad(ref, W)
    torch.testing.assert_close(got, ref)
    torch.testing.assert_close(g_got, g_ref)


@pytest.mark.parametrize("multi_head_input", [False, True])
def test_equals_light_cell_tv_on_the_same_tables(multi_head_input):
    nap, n_heads, tph = 4, 2, 3
    fast = _fast(nap=nap, n_heads=n_heads, tph=tph, multi_head_input=multi_head_input)
    light = LightMultiHeadLUT(input_dim=12, n_tables=n_heads * tph, output_dim=5, n_anchor_pairs=nap,
                              random_seed=0, n_heads=n_heads, multi_head_input=multi_head_input)
    light = light.double()
    with torch.no_grad():
        light.tables.copy_(fast.weights)
    assert torch.equal(fast.cell_tv(), light.cell_tv())


def test_adjacency_through_the_read_path():
    nap = 4
    m = _fast(nap=nap, n_heads=1, tph=1, input_dim=2 * nap, policy=AnchorSamplingPolicy.CANONICAL_DISJOINT)
    a, b = m.soft_anchor_a_long[0].tolist(), m.soft_anchor_b_long[0].tolist()
    assert len(set(a) | set(b)) == 2 * nap, "test needs coordinate-disjoint anchor pairs in table 0"
    K = 1 << nap
    with torch.no_grad():
        m.weights.zero_()
        m.weights[0, :, 0] = torch.arange(K, dtype=m.weights.dtype)   # row id stored in output dim 0
    m.eval()

    def x_for(bits):                      # bits[j] = sign bit of margin j (u_j = x[a_j] - x[b_j] > 0)
        x = torch.zeros(1, 2 * nap, dtype=torch.float64)
        for j, s in enumerate(bits):
            x[0, a[j]], x[0, b[j]] = (1.0, -1.0) if s else (-1.0, 1.0)
        return x

    def row_read(bits):
        with torch.no_grad():
            return int(m(x_for(bits))[0, 0, 0].item())

    for c in range(K):
        bits = [(c >> (nap - 1 - j)) & 1 for j in range(nap)]          # MSB-first: margin j <-> 2^(nap-1-j)
        assert row_read(bits) == c
        for j in range(nap):
            flipped = list(bits)
            flipped[j] ^= 1
            c2 = row_read(flipped)
            assert c2 == c ^ (1 << (nap - 1 - j))
            assert bin(c ^ c2).count("1") == 1                          # a Hamming-1 row pair
