"""ManifestoSoftLUT: the (1-U)/U two-cell blend as value + gradient, and coexistence."""
import pytest
import torch

from spiky.lutorch_ex import (
    LUTSpec,
    ManifestoHardLUT,
    ManifestoSoftLUT,
    ProjectionMHL,
)


def _soft_reference(cart: ManifestoSoftLUT, x: torch.Tensor) -> torch.Tensor:
    """Independent (1-U)/U blend reference, MSB-first addressing, routing invariant."""
    spec = cart.spec
    nap = spec.nap
    B = x.shape[0]
    out = torch.zeros(B, spec.h_out, spec.d_out)
    for b in range(B):
        for g in range(spec.n_groups):
            z = x[b, g % spec.h_in, :]
            acc = torch.zeros(spec.d_out)
            for t in range(spec.tph):
                us = [float(z[int(cart.anchor_a[g, t, j])] - z[int(cart.anchor_b[g, t, j])])
                      for j in range(nap)]
                c = sum((1 << (nap - 1 - j)) for j in range(nap) if us[j] > cart.cmp_eps)
                jstar = min(range(nap), key=lambda j: abs(us[j]))
                c_alt = c ^ (1 << (nap - 1 - jstar))
                U = 0.5 / (1.0 + abs(us[jstar]))
                acc = acc + (1 - U) * cart.weights[g, t, c] + U * cart.weights[g, t, c_alt]
            out[b, g % spec.h_out] = out[b, g % spec.h_out] + acc
    return out


@pytest.mark.parametrize("h_in,h_out", [(2, 2), (1, 3), (3, 1), (1, 1)])
def test_soft_forward_matches_blend_reference(h_in, h_out):
    spec = LUTSpec(h_in=h_in, h_out=h_out, tph=3, nap=4, d_in=6, d_out=4)
    cart = ManifestoSoftLUT(spec, seed=10, weight_init_std=1.0).eval()
    x = torch.randn(5, spec.h_in, spec.d_in)
    assert torch.allclose(cart(x), _soft_reference(cart, x), atol=1e-6)


def test_soft_value_same_train_and_eval():
    spec = LUTSpec(h_in=2, h_out=2, tph=2, nap=3, d_in=5, d_out=3)
    cart = ManifestoSoftLUT(spec, seed=11, weight_init_std=1.0)
    x = torch.randn(4, spec.h_in, spec.d_in)
    cart.eval()
    y_eval = cart(x)
    cart.train()
    with torch.no_grad():
        y_train = cart(x)
    assert torch.allclose(y_eval, y_train, atol=1e-6)  # the blend is applied at eval too


def test_soft_weight_grad_hits_both_cells():
    """Contrast with the hard cartridge: the blend sends weight grad to c_t AND c_t'."""
    spec = LUTSpec(h_in=1, h_out=1, tph=1, nap=3, d_in=4, d_out=2)  # K = 8
    cart = ManifestoSoftLUT(spec, seed=14, weight_init_std=1.0).train()
    x = torch.randn(1, 1, spec.d_in)
    cart(x).sum().backward()

    z = x[0, 0]
    us = [float(z[int(cart.anchor_a[0, 0, j])] - z[int(cart.anchor_b[0, 0, j])])
          for j in range(spec.nap)]
    c = sum((1 << (spec.nap - 1 - j)) for j in range(spec.nap) if us[j] > cart.cmp_eps)
    jstar = min(range(spec.nap), key=lambda j: abs(us[j]))
    c_alt = c ^ (1 << (spec.nap - 1 - jstar))
    assert c != c_alt

    gpc = cart.weights.grad[0, 0].abs().sum(dim=-1)  # [K]
    assert gpc[c] > 0 and gpc[c_alt] > 0, "both the hard cell and its neighbour must get grad"
    for k in range(spec.n_cells):
        if k not in (c, c_alt):
            assert gpc[k] == 0, f"cell {k} (neither c_t nor c_t') must have zero grad"


def test_soft_input_gradient_flows():
    spec = LUTSpec(h_in=2, h_out=1, tph=2, nap=3, d_in=5, d_out=3)
    cart = ManifestoSoftLUT(spec, seed=12, weight_init_std=1.0).train()
    x = torch.randn(8, spec.h_in, spec.d_in, requires_grad=True)
    cart(x).pow(2).sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert x.grad.abs().sum() > 0


def test_soft_determinism():
    spec = LUTSpec(h_in=2, h_out=2, tph=2, nap=3, d_in=6, d_out=3)
    x = torch.randn(4, spec.h_in, spec.d_in)
    y1 = ManifestoSoftLUT(spec, seed=7).eval()(x)
    y2 = ManifestoSoftLUT(spec, seed=7).eval()(x)
    assert torch.equal(y1, y2)


@pytest.mark.parametrize("h_in,h_out", [(3, 3), (1, 4), (4, 1), (1, 1)])
def test_soft_shape_contract(h_in, h_out):
    spec = LUTSpec(h_in=h_in, h_out=h_out, tph=2, nap=4, d_in=8, d_out=5)
    cart = ManifestoSoftLUT(spec, seed=1)
    y = cart(torch.randn(7, spec.h_in, spec.d_in))
    assert y.shape == (7, spec.h_out, spec.d_out)


def test_hard_and_soft_coexist():
    """The point of the exercise: two cartridges, same contract, usable together."""
    spec = LUTSpec(h_in=2, h_out=2, tph=2, nap=3, d_in=6, d_out=4)
    hard = ManifestoHardLUT(spec, seed=1).eval()
    soft = ManifestoSoftLUT(spec, seed=1).eval()
    x = torch.randn(3, spec.h_in, spec.d_in)
    yh, ys = hard(x), soft(x)
    assert yh.shape == ys.shape == (3, spec.h_out, spec.d_out)
    # Same anchors/weights (same seed) but soft blends in the neighbour -> values differ.
    assert not torch.allclose(yh, ys)
    # Both drop into the same wrapper.
    xx = torch.randn(3, 16)
    assert ProjectionMHL(hard, 16).eval()(xx).shape == (3, 16)
    assert ProjectionMHL(soft, 16).eval()(xx).shape == (3, 16)
