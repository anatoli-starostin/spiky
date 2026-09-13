"""Tests for MultiHeadLut.cell_tv, the Hamming-1 cell TV penalty on Gen-1 tables (LProjection.weights).

Coverage:
  (i)   value and gradient against a plain-torch reference enumerating every pair (c, c ^ 2^b) explicitly;
  (ii)  equality with FastMultiHeadLut.cell_tv on the identical table tensor (same normalisation, same object);
  (iii) adjacency through the module's own read path: with coordinate-disjoint anchors and row ids stored in the
        table, each sign pattern reads row c (Gen 1 packs LSB-first: margin j <-> 2^j), flipping one margin moves the
        hard read to c ^ 2^j in eval AND train, and the smooth read blends exactly rows c and c ^ 2^j* (j* the
        least-|margin| pair) with U = 0.5 / (1 + |u|) -- all Hamming-1 row pairs, i.e. exactly what cell_tv penalises;
  (iv)  n_buckets > 1 is refused (the bucket index is not an address bit).
"""
import pytest
import torch

torch._dynamo.config.suppress_errors = True

from spiky.lutorch.fast_multi_head_lut import FastMultiHeadLut
from spiky.lutorch.multi_head_lut import MultiHeadLut


def _gen1(nap=4, n_heads=2, tph=3, n_out=5, input_dim=12, smooth=False, n_buckets=1, seed=0):
    m = MultiHeadLut(input_dim=input_dim, n_heads=n_heads, n_outputs=n_out, n_anchor_pairs=nap,
                     tables_per_head=tph, n_buckets=n_buckets, random_seed=seed, smooth_mode=smooth,
                     initial_weights_noise=0.5)
    return m.double()


def _reference_tv(W, nap):
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


@pytest.mark.parametrize("nap", [1, 3, 5])
def test_value_and_gradient_match_explicit_pair_reference(nap):
    m = _gen1(nap=nap)
    got = m.cell_tv()
    (g_got,) = torch.autograd.grad(got, m.projection.weights)
    W = m.projection.weights.detach().clone().requires_grad_(True)
    ref = _reference_tv(W, nap)
    (g_ref,) = torch.autograd.grad(ref, W)
    torch.testing.assert_close(got, ref)
    torch.testing.assert_close(g_got, g_ref)


def test_equals_fast_cell_tv_on_the_same_tables():
    nap, n_heads, tph = 4, 2, 3
    g1 = _gen1(nap=nap, n_heads=n_heads, tph=tph)
    fast = FastMultiHeadLut(input_dim=12, n_heads=n_heads, n_outputs=5, n_anchor_pairs=nap, tables_per_head=tph,
                            weight_dtype=torch.float64, use_bf16=False, random_seed=0)
    with torch.no_grad():
        fast.weights.copy_(g1.projection.weights)
    assert torch.equal(g1.cell_tv(), fast.cell_tv())


_DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])   # CUDA: the native lookup kernels


def _one_table(smooth, device="cpu"):
    nap = 4
    m = _gen1(nap=nap, n_heads=1, tph=1, n_out=1, input_dim=2 * nap, smooth=smooth).to(device)
    with torch.no_grad():
        m.lookup.anchor_pairs_a.copy_(torch.tensor([[0, 2, 4, 6]]))       # coordinate-disjoint pairs (a_j, b_j)
        m.lookup.anchor_pairs_b.copy_(torch.tensor([[1, 3, 5, 7]]))
        m.projection.weights.zero_()
        m.projection.weights[0, :, 0] = torch.arange(1 << nap, dtype=torch.float64)   # row id stored in the table
    return m, nap


def _x(bits, small=None, mag=1.0, device="cpu"):
    """bits[j] = sign bit of margin j; margin j has |u_j| = 2*mag, except `small` which gets |u| = 0.5."""
    x = torch.zeros(1, 2 * len(bits), dtype=torch.float64)
    for j, s in enumerate(bits):
        h = 0.25 if j == small else mag
        x[0, 2 * j], x[0, 2 * j + 1] = (h, -h) if s else (-h, h)
    return x.to(device)


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("mode", ["eval", "train"])
def test_hard_read_adjacency_through_the_read_path(mode, device):
    m, nap = _one_table(smooth=False, device=device)
    m.train(mode == "train")
    K = 1 << nap
    for c in range(K):
        bits = [(c >> j) & 1 for j in range(nap)]                      # LSB-first: margin j <-> 2^j
        assert int(m(_x(bits, device=device))[0, 0, 0].item()) == c
        for j in range(nap):
            flipped = list(bits)
            flipped[j] ^= 1
            c2 = int(m(_x(flipped, device=device))[0, 0, 0].item())
            assert c2 == c ^ (1 << j) and bin(c ^ c2).count("1") == 1


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("mode", ["eval", "train"])
def test_smooth_read_blends_the_hamming1_neighbour(mode, device):
    m, nap = _one_table(smooth=True, device=device)
    m.train(mode == "train")
    u = 0.5                                                           # |margin| of the least-confident pair
    U = 0.5 / (1.0 + u)
    for c in (0, 5, 10, 15):
        bits = [(c >> j) & 1 for j in range(nap)]
        for j in range(nap):
            got = float(m(_x(bits, small=j, device=device))[0, 0, 0].item())
            assert got == pytest.approx((1.0 - U) * c + U * (c ^ (1 << j)), abs=1e-9)


def test_multiple_buckets_are_refused():
    with pytest.raises(NotImplementedError, match="n_buckets"):
        _gen1(n_buckets=2).cell_tv()
