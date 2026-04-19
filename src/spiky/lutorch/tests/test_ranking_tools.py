"""Tests for ranking_tools.py: DominanceToVector, VectorToDominance,
DominanceCanonicalize, and the shared canonical Borda matrix."""
import math

import pytest
import torch
import torch.nn.functional as F

from spiky.lutorch.ranking_tools import (
    DominanceCanonicalize,
    DominanceToVector,
    VectorToDominance,
    _canonical_borda_m,
)


# ---------- _canonical_borda_m ----------

def test_canonical_borda_m_unit_row_norm():
    """Each row of B has (N-1) non-zero ±1/√(N-1) entries → row L2 norm = 1."""
    for N in [2, 4, 8, 13]:
        m = _canonical_borda_m(N)
        assert m.shape == (N, N * (N - 1) // 2)
        assert torch.allclose(m.norm(dim=1), torch.ones(N), atol=1e-6)


def test_canonical_borda_m_winner_minus_loser():
    N = 5
    m = _canonical_borda_m(N)
    tri_i, tri_j = torch.triu_indices(N, N, offset=1)
    scale = 1.0 / math.sqrt(N - 1)
    for p in range(m.shape[1]):
        i, j = int(tri_i[p]), int(tri_j[p])
        assert m[i, p] == pytest.approx(scale)
        assert m[j, p] == pytest.approx(-scale)
        for k in range(N):
            if k != i and k != j:
                assert m[k, p] == 0.0


# ---------- DominanceToVector ----------

def test_dominance_to_vector_shape(device):
    N, P = 6, 15
    d = torch.randn(3, 4, P, device=device)
    out = DominanceToVector(N).to(device)(d)
    assert out.shape == (3, 4, N)
    assert torch.isfinite(out).all()


def test_dominance_to_vector_matches_manual(device):
    """Without affine LN, output == F.layer_norm(B @ d)."""
    N = 7
    P = N * (N - 1) // 2
    d = torch.randn(5, P, device=device)
    out = DominanceToVector(N, elementwise_affine=False).to(device)(d)
    B = _canonical_borda_m(N).to(device)
    expected = F.layer_norm(torch.einsum('bp,kp->bk', d, B), (N,))
    assert torch.allclose(out, expected, atol=1e-5)


# ---------- VectorToDominance ----------

def test_vector_to_dominance_soft_in_open_interval(device):
    N = 8
    x = torch.randn(10, N, device=device)
    d = VectorToDominance(N, smooth_mode=True, temperature=0.1).to(device)(x)
    assert d.shape == (10, N * (N - 1) // 2)
    assert (d.abs() < 1.0).all()


def test_vector_to_dominance_hard_is_strict_binary(device):
    """Non-smooth mode forward is strict ±1 (ties → −1), matching the
    RankAttention convention. Zero outputs would be 'dead features'."""
    N = 5
    x = torch.randn(7, N, device=device)
    d = VectorToDominance(N, smooth_mode=False, temperature=0.1).to(device)(x)
    tri_i, tri_j = torch.triu_indices(N, N, offset=1)
    expected = (x[..., tri_i] > x[..., tri_j]).to(x.dtype) * 2.0 - 1.0
    assert torch.allclose(d, expected, atol=1e-6)
    # Tie (x_a == x_b) maps to -1, not 0.
    x_tied = torch.zeros(2, N, device=device)
    d_tied = VectorToDominance(N, smooth_mode=False).to(device)(x_tied)
    assert (d_tied == -1.0).all()


def test_vector_to_dominance_ste_backward_non_zero(device):
    """STE routes gradients through the soft path: x.grad is non-zero even
    though forward uses sign (which has ~zero gradient a.e.)."""
    N = 5
    x = torch.randn(3, N, device=device, requires_grad=True)
    v2d = VectorToDominance(N, smooth_mode=False, temperature=0.1).to(device)
    v2d(x).sum().backward()
    assert x.grad is not None and (x.grad.abs() > 0).any()


# ---------- DominanceCanonicalize ----------

def test_canonicalize_shape(device):
    N, P = 6, 15
    d = torch.randn(2, 3, P, device=device)
    out = DominanceCanonicalize(N).to(device)(d)
    assert out.shape == d.shape


def test_canonicalize_preserves_consistent_dominance(device):
    """A dominance built from a real ordering is already consistent;
    canonicalize should leave pair signs unchanged (LN preserves ordering
    within a sample, so Borda → LN → sign round-trip is identity)."""
    N = 7
    # Strictly monotonic per sample (guaranteed no ties).
    x = torch.linspace(-1.0, 1.0, N, device=device).unsqueeze(0).expand(4, -1).contiguous()
    x = x + torch.randn(4, N, device=device) * 0.01
    d = VectorToDominance(N, smooth_mode=False, temperature=0.1).to(device)(x)
    canon = DominanceCanonicalize(
        N, smooth_mode=False, temperature=0.1, elementwise_affine=False,
    ).to(device)
    d_out = canon(d)
    assert torch.allclose(d_out, d, atol=1e-5)


def test_canonicalize_backward_flows(device):
    N, P = 6, 15
    d = torch.randn(3, P, device=device, requires_grad=True)
    canon = DominanceCanonicalize(N, smooth_mode=True).to(device)
    canon(d).sum().backward()
    assert d.grad is not None and (d.grad.abs() > 0).any()
