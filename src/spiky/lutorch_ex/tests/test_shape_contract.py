"""Shape-contract tests: both ranks, H-preservation, and flat<->per-head equivalence."""
import pytest
import torch

from spiky.lutorch_ex import LUTSpec, ManifestoHardLUT


def _spec():
    return LUTSpec(n_heads=3, tph=2, nap=4, d_in=8, d_out=5)


@pytest.mark.parametrize("training", [True, False])
def test_per_head_3d_mode(training):
    spec = _spec()
    cart = ManifestoHardLUT(spec, seed=1)
    cart.train(training)
    B = 7
    x = torch.randn(B, spec.n_heads, spec.d_in)
    y = cart(x)
    assert y.shape == (B, spec.n_heads, spec.d_out), "per-head output keeps [B, H, d_out]"


@pytest.mark.parametrize("training", [True, False])
def test_flat_2d_mode(training):
    spec = _spec()
    cart = ManifestoHardLUT(spec, seed=1)
    cart.train(training)
    B = 7
    x = torch.randn(B, spec.in_features)  # [B, H*d_in]
    y = cart(x)
    assert y.shape == (B, spec.out_features), "flat output is [B, H*d_out]"


def test_flat_and_per_head_agree():
    """Flat [B, H*d_in] viewed as H slices must equal the per-head [B, H, d_in] path."""
    spec = _spec()
    cart = ManifestoHardLUT(spec, seed=2).eval()
    B = 6
    x3 = torch.randn(B, spec.n_heads, spec.d_in)
    x2 = x3.reshape(B, spec.in_features)
    y3 = cart(x3)
    y2 = cart(x2)
    assert torch.allclose(y2, y3.reshape(B, spec.out_features))


def test_heads_are_independent():
    """H preservation with teeth: perturbing head h's input moves only head h's output."""
    spec = _spec()
    cart = ManifestoHardLUT(spec, seed=3).eval()
    B = 4
    x = torch.randn(B, spec.n_heads, spec.d_in)
    y0 = cart(x)
    x2 = x.clone()
    x2[:, 1, :] += 10.0  # shove head 1 only
    y1 = cart(x2)
    assert torch.allclose(y1[:, 0, :], y0[:, 0, :]), "head 0 unaffected by head 1's input"
    assert torch.allclose(y1[:, 2, :], y0[:, 2, :]), "head 2 unaffected by head 1's input"


def test_bad_shapes_rejected():
    spec = _spec()
    cart = ManifestoHardLUT(spec, seed=4)
    with pytest.raises(ValueError):
        cart(torch.randn(5, spec.in_features + 1))     # wrong flat width
    with pytest.raises(ValueError):
        cart(torch.randn(5, spec.n_heads, spec.d_in + 1))  # wrong per-head width
    with pytest.raises(ValueError):
        cart(torch.randn(5, spec.n_heads, spec.d_in, 1))   # wrong rank
