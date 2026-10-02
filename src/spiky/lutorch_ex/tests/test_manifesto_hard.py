"""ManifestoHardLUT: hard-forward value (all H patterns) + 2-alternative soft backward."""
import pytest
import torch

from spiky.lutorch_ex import LUTSpec, ManifestoHardLUT


def _reference_hard(cart: ManifestoHardLUT, x: torch.Tensor) -> torch.Tensor:
    """Independent re-computation of the hard output under the routing invariant."""
    spec = cart.spec
    B = x.shape[0]
    out = torch.zeros(B, spec.h_out, spec.d_out)
    for b in range(B):
        for g in range(spec.n_groups):
            z = x[b, g % spec.h_in, :]
            acc = torch.zeros(spec.d_out)
            for t in range(spec.tph):
                c = 0
                for j in range(spec.nap):
                    a = int(cart.anchor_a[g, t, j])
                    bb = int(cart.anchor_b[g, t, j])
                    if (z[a] - z[bb]).item() > cart.cmp_eps:
                        c += (1 << (spec.nap - 1 - j))  # MSB-first (pair 0 = high bit)
                acc = acc + cart.weights[g, t, c]
            out[b, g % spec.h_out] = out[b, g % spec.h_out] + acc  # sum groups per out head
    return out


@pytest.mark.parametrize("h_in,h_out", [(2, 2), (1, 3), (3, 1), (1, 1)])
def test_hard_forward_value_matches_reference(h_in, h_out):
    spec = LUTSpec(h_in=h_in, h_out=h_out, tph=3, nap=4, d_in=6, d_out=4)
    cart = ManifestoHardLUT(spec, seed=10, weight_init_std=1.0).eval()
    x = torch.randn(5, spec.h_in, spec.d_in)
    assert torch.allclose(cart(x), _reference_hard(cart, x), atol=1e-6)


def test_train_value_equals_eval_value():
    spec = LUTSpec(h_in=2, h_out=2, tph=2, nap=3, d_in=5, d_out=3)
    cart = ManifestoHardLUT(spec, seed=12, weight_init_std=1.0)
    x = torch.randn(4, spec.h_in, spec.d_in)
    cart.eval()
    y_eval = cart(x)
    cart.train()
    with torch.no_grad():
        y_train = cart(x)
    assert torch.allclose(y_eval, y_train, atol=1e-6)


@pytest.mark.parametrize("h_in,h_out", [(2, 2), (1, 3), (3, 1)])
def test_backward_flows_to_input_and_weights(h_in, h_out):
    spec = LUTSpec(h_in=h_in, h_out=h_out, tph=2, nap=3, d_in=5, d_out=3)
    cart = ManifestoHardLUT(spec, seed=13, weight_init_std=1.0).train()
    x = torch.randn(8, spec.h_in, spec.d_in, requires_grad=True)
    cart(x).pow(2).sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert x.grad.abs().sum() > 0
    assert cart.weights.grad is not None and cart.weights.grad.abs().sum() > 0


def test_weight_gradient_is_hard_only():
    """Gen-1 fidelity: only addressed cells c_t get weight gradient (single group)."""
    spec = LUTSpec(h_in=1, h_out=1, tph=1, nap=3, d_in=4, d_out=2)  # K = 8
    cart = ManifestoHardLUT(spec, seed=14, weight_init_std=1.0).train()
    x = torch.randn(16, 1, spec.d_in)
    cart(x).sum().backward()
    hit = set()
    for b in range(x.shape[0]):
        c = 0
        for j in range(spec.nap):
            a = int(cart.anchor_a[0, 0, j]); bb = int(cart.anchor_b[0, 0, j])
            if (x[b, 0, a] - x[b, 0, bb]).item() > cart.cmp_eps:
                c += (1 << (spec.nap - 1 - j))  # MSB-first (pair 0 = high bit)
        hit.add(c)
    grad_per_cell = cart.weights.grad[0, 0].abs().sum(dim=-1)
    for c in range(spec.n_cells):
        if c not in hit:
            assert grad_per_cell[c] == 0, f"unaddressed cell {c} must have zero weight grad"


def test_gradient_continuous_across_bit_flip():
    spec = LUTSpec(h_in=1, h_out=1, tph=1, nap=1, d_in=2, d_out=1)
    cart = ManifestoHardLUT(spec, seed=15, weight_init_std=1.0).train()

    def grad_u(eps: float) -> float:
        z = torch.tensor([[[eps, 0.0]]], requires_grad=True)
        cart.zero_grad()
        cart(z).sum().backward()
        return float(z.grad[0, 0, 0])

    eps = 1e-3
    g_pos, g_neg = grad_u(eps), grad_u(-eps)
    assert abs(g_pos - g_neg) < 1e-5, f"grad discontinuous across flip: {g_pos} vs {g_neg}"
    assert abs(g_pos) > 0
