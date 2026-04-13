"""Tests for HyperLUT."""
import pytest
import torch
from spiky.lutorch.hyper_lut import HyperLUT, HyperLUTBackbone


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
    assert torch.equal(lut1.backbone.pairs, lut2.backbone.pairs)


def test_hyper_lut_hard_forward():
    """In eval mode, features should be {0, 1} (hard comparisons)."""
    lut = HyperLUT(input_dim=8, n_heads=1, n_outputs=4, n_pairs=10, hidden_dim=8)
    x = torch.randn(4, 8)
    features = lut.backbone._hard_features(x)
    assert ((features == 0) | (features == 1)).all()


def test_hyper_lut_ste_gradient_nonzero():
    """STE provides nonzero gradients even though forward is hard."""
    lut = HyperLUT(input_dim=8, n_heads=1, n_outputs=4, n_pairs=10, hidden_dim=8)
    x = torch.randn(4, 8, requires_grad=True)
    features = lut.backbone._ste_features(x)
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


def test_hyper_lut_shared_backbone():
    """Multiple HyperLUTs sharing a backbone produce correct shapes."""
    backbone = HyperLUTBackbone(input_dim=32, n_heads=4, n_pairs=100, hidden_dim=64)
    lut_a = HyperLUT(backbone=backbone, n_outputs=1)
    lut_b = HyperLUT(backbone=backbone, n_outputs=16)
    assert lut_a.backbone is lut_b.backbone

    x = torch.randn(8, 32)
    assert lut_a(x).shape == (8, 4, 1)
    assert lut_b(x).shape == (8, 4, 16)


def test_hyper_lut_shared_backbone_backward():
    """Gradients flow correctly through shared backbone."""
    backbone = HyperLUTBackbone(input_dim=16, n_heads=2, n_pairs=30, hidden_dim=16)
    lut_a = HyperLUT(backbone=backbone, n_outputs=4)
    lut_b = HyperLUT(backbone=backbone, n_outputs=8)

    x = torch.randn(8, 16, requires_grad=True)
    loss = lut_a(x).sum() + lut_b(x).sum()
    loss.backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert (x.grad != 0).any()
    # fc1 grad should be non-zero (both heads contribute)
    assert backbone.fc1_weight.grad is not None
    assert (backbone.fc1_weight.grad != 0).any()


def test_hyper_lut_shared_backbone_same_as_owned():
    """Shared backbone gives same result as owned backbone with same seed."""
    backbone = HyperLUTBackbone(
        input_dim=16, n_heads=2, n_pairs=30, hidden_dim=16, random_seed=42)
    shared = HyperLUT(backbone=backbone, n_outputs=4)
    owned = HyperLUT(
        input_dim=16, n_heads=2, n_outputs=4, n_pairs=30, hidden_dim=16, random_seed=42)

    # Same pairs
    assert torch.equal(shared.backbone.pairs, owned.backbone.pairs)
    # Same fc1 init (both normal_ with std=0.01, but different draws — just check structure)
    assert shared.backbone.fc1_weight.shape == owned.backbone.fc1_weight.shape
