"""Tests for HyperLUT."""
import pytest
import torch
from spiky.lutorch.hyper_lut import HyperLUT


def test_hyper_lut_output_shape():
    """Output shape is [B, n_heads, n_outputs]."""
    lut = HyperLUT(input_dim=32, n_heads=4, n_outputs=8, n_pairs=100, hidden_dim=64)
    x = torch.randn(16, 32)
    out = lut(x)
    assert out.shape == (16, 4, 8)


def test_hyper_lut_single_head():
    """Works with n_heads=1."""
    lut = HyperLUT(input_dim=16, n_heads=1, n_outputs=16, n_pairs=50, hidden_dim=32)
    x = torch.randn(8, 16)
    out = lut(x)
    assert out.shape == (8, 1, 16)


def test_hyper_lut_backward():
    """Gradients flow through HyperLUT."""
    lut = HyperLUT(input_dim=16, n_heads=2, n_outputs=4, n_pairs=30, hidden_dim=16)
    x = torch.randn(8, 16, requires_grad=True)
    out = lut(x)
    out.sum().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert (x.grad != 0).any()


def test_hyper_lut_deterministic():
    """Same seed produces same pairs and outputs."""
    lut1 = HyperLUT(input_dim=16, n_heads=2, n_outputs=4, n_pairs=30, hidden_dim=16, random_seed=42)
    lut2 = HyperLUT(input_dim=16, n_heads=2, n_outputs=4, n_pairs=30, hidden_dim=16, random_seed=42)
    assert torch.equal(lut1.pairs, lut2.pairs)


def test_hyper_lut_hard_forward():
    """In eval mode, features should be {0, 1} (hard comparisons)."""
    lut = HyperLUT(input_dim=8, n_heads=1, n_outputs=4, n_pairs=10, hidden_dim=8)
    x = torch.randn(4, 8)
    features = lut._hard_features(x)
    assert ((features == 0) | (features == 1)).all()


def test_hyper_lut_ste_gradient_nonzero():
    """STE provides nonzero gradients even though forward is hard."""
    lut = HyperLUT(input_dim=8, n_heads=1, n_outputs=4, n_pairs=10, hidden_dim=8)
    x = torch.randn(4, 8, requires_grad=True)
    features = lut._ste_features(x)
    features.sum().backward()
    assert x.grad is not None
    assert (x.grad != 0).any()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_hyper_lut_cuda():
    """Works on CUDA."""
    lut = HyperLUT(input_dim=32, n_heads=4, n_outputs=8, n_pairs=100, hidden_dim=64, device='cuda')
    lut = lut.cuda()
    x = torch.randn(16, 32, device='cuda', requires_grad=True)
    out = lut(x)
    assert out.shape == (16, 4, 8)
    out.sum().backward()
    assert torch.isfinite(x.grad).all()


def test_hyper_lut_n_pairs_limit():
    """Raises error if n_pairs exceeds max."""
    with pytest.raises(ValueError, match="exceeds max"):
        HyperLUT(input_dim=4, n_heads=1, n_outputs=2, n_pairs=100, hidden_dim=8)
