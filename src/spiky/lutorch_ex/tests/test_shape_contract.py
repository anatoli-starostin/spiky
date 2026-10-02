"""Shape-contract tests: the three H patterns, routing, and spec validation."""
import pytest
import torch

from spiky.lutorch_ex import LUTSpec, ManifestoHardLUT

PATTERNS = {
    "per_head": (3, 3),
    "fan_out": (1, 4),
    "fan_in": (4, 1),
    "single": (1, 1),
}


def _spec(h_in, h_out):
    return LUTSpec(h_in=h_in, h_out=h_out, tph=2, nap=4, d_in=8, d_out=5)


@pytest.mark.parametrize("name", list(PATTERNS))
@pytest.mark.parametrize("training", [True, False])
def test_shape_all_patterns(name, training):
    h_in, h_out = PATTERNS[name]
    spec = _spec(h_in, h_out)
    cart = ManifestoHardLUT(spec, seed=1)
    cart.train(training)
    B = 7
    y = cart(torch.randn(B, spec.h_in, spec.d_in))
    assert y.shape == (B, spec.h_out, spec.d_out)


def test_flat_2d_input_rejected():
    """The old flat [B, H*d_in] sugar is gone — only 3-D is accepted."""
    spec = _spec(3, 3)
    cart = ManifestoHardLUT(spec, seed=1)
    with pytest.raises(ValueError):
        cart(torch.randn(7, spec.h_in * spec.d_in))  # 2-D
    with pytest.raises(ValueError):
        cart(torch.randn(7, spec.h_in, spec.d_in + 1))  # wrong per-head width
    with pytest.raises(ValueError):
        cart(torch.randn(7, spec.h_in + 1, spec.d_in))  # wrong head count


def test_per_head_independence():
    """h_in==h_out: replacing one input head's slice leaves the other heads' outputs
    untouched (per-head isolation). Exact input->output behaviour is covered by the
    reference-based tests in test_manifesto_hard."""
    spec = _spec(3, 3)
    cart = ManifestoHardLUT(spec, seed=4).eval()
    x = torch.randn(4, 3, spec.d_in)
    y0 = cart(x)
    x2 = x.clone()
    x2[:, 1, :] = torch.randn(4, spec.d_in)  # a genuinely different slice for head 1
    y1 = cart(x2)
    assert torch.allclose(y1[:, 0], y0[:, 0])
    assert torch.allclose(y1[:, 2], y0[:, 2])


def test_fan_out_shared_input():
    """h_in==1: one shared input feeds every output head (shape; value in manifesto tests)."""
    spec = _spec(1, 4)
    cart = ManifestoHardLUT(spec, seed=2).eval()
    y = cart(torch.randn(5, 1, spec.d_in))
    assert y.shape == (5, 4, spec.d_out)


def test_fan_in_sums_into_one_head():
    """h_out==1: all input-head groups sum into the single output head."""
    spec = _spec(4, 1)
    cart = ManifestoHardLUT(spec, seed=3).eval()
    x = torch.randn(5, 4, spec.d_in)
    y = cart(x)
    assert y.shape == (5, 1, spec.d_out)


def test_spec_rejects_forbidden_patterns():
    # both != 1 and unequal -> forbidden
    with pytest.raises(ValueError):
        LUTSpec(h_in=2, h_out=3, tph=1, nap=2, d_in=4, d_out=2)
    with pytest.raises(ValueError):
        LUTSpec(h_in=3, h_out=2, tph=1, nap=2, d_in=4, d_out=2)
    # allowed patterns build fine
    for h_in, h_out in [(1, 1), (3, 3), (1, 5), (5, 1)]:
        LUTSpec(h_in=h_in, h_out=h_out, tph=1, nap=2, d_in=4, d_out=2)


def test_spec_derived_fields():
    spec = LUTSpec(h_in=1, h_out=6, tph=4, nap=3, d_in=8, d_out=5)
    assert spec.n_groups == 6               # max(h_in, h_out)
    assert spec.n_tables == 24              # n_groups * tph
    assert spec.n_cells == 8                # 2**nap
    assert spec.in_features == 8            # h_in * d_in
    assert spec.out_features == 30          # h_out * d_out
