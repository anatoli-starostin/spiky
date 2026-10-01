"""ManifestoHardLUT: hard-forward value correctness + 2-alternative soft backward."""
import pytest
import torch

from spiky.lutorch_ex import LUTSpec, ManifestoHardLUT


def _reference_hard(cart: ManifestoHardLUT, z: torch.Tensor) -> torch.Tensor:
    """Independent re-computation of y = sum_t W_t[c_t] from anchors + input."""
    spec = cart.spec
    B = z.shape[0]
    out = torch.zeros(B, spec.n_heads, spec.d_out)
    for b in range(B):
        for h in range(spec.n_heads):
            acc = torch.zeros(spec.d_out)
            for t in range(spec.tph):
                c = 0
                for j in range(spec.nap):
                    a = int(cart.anchor_a[h, t, j])
                    bb = int(cart.anchor_b[h, t, j])
                    if (z[b, h, a] - z[b, h, bb]).item() > cart.cmp_eps:
                        c += (1 << j)  # LSB-first
                acc = acc + cart.weights[h, t, c]
            out[b, h] = acc
    return out


def test_hard_forward_value_matches_reference():
    spec = LUTSpec(n_heads=2, tph=3, nap=4, d_in=6, d_out=4)
    cart = ManifestoHardLUT(spec, seed=10, weight_init_std=1.0).eval()
    z = torch.randn(5, spec.n_heads, spec.d_in)
    got = cart(z)
    ref = _reference_hard(cart, z)
    assert torch.allclose(got, ref, atol=1e-6), "eval output must be the exact hard read"


def test_train_value_equals_eval_value():
    """The surrogate must not change the forward VALUE — train == eval numerically."""
    spec = LUTSpec(n_heads=2, tph=2, nap=3, d_in=5, d_out=3)
    cart = ManifestoHardLUT(spec, seed=11, weight_init_std=1.0)
    z = torch.randn(4, spec.n_heads, spec.d_in)
    cart.eval()
    y_eval = cart(z)
    cart.train()
    with torch.no_grad():
        y_train = cart(z)
    assert torch.allclose(y_eval, y_train, atol=1e-6)


def test_backward_flows_to_input_and_weights():
    spec = LUTSpec(n_heads=2, tph=2, nap=3, d_in=5, d_out=3)
    cart = ManifestoHardLUT(spec, seed=12, weight_init_std=1.0).train()
    z = torch.randn(8, spec.n_heads, spec.d_in, requires_grad=True)
    cart(z).pow(2).sum().backward()
    assert z.grad is not None and torch.isfinite(z.grad).all()
    assert z.grad.abs().sum() > 0, "input must receive a (surrogate) gradient"
    assert cart.weights.grad is not None and cart.weights.grad.abs().sum() > 0


def test_weight_gradient_is_hard_only():
    """Fidelity to gen-1 hard-forward: only addressed cells c_t get weight gradient."""
    spec = LUTSpec(n_heads=1, tph=1, nap=3, d_in=4, d_out=2)  # K = 8 cells
    cart = ManifestoHardLUT(spec, seed=13, weight_init_std=1.0).train()
    z = torch.randn(16, spec.n_heads, spec.d_in)
    cart(z).sum().backward()
    # Recompute which cells were hard-addressed; every other cell must have zero grad.
    hit = set()
    for b in range(z.shape[0]):
        c = 0
        for j in range(spec.nap):
            a = int(cart.anchor_a[0, 0, j]); bb = int(cart.anchor_b[0, 0, j])
            if (z[b, 0, a] - z[b, 0, bb]).item() > cart.cmp_eps:
                c += (1 << j)
        hit.add(c)
    grad_per_cell = cart.weights.grad[0, 0].abs().sum(dim=-1)  # [K]
    for c in range(spec.n_cells):
        if c not in hit:
            assert grad_per_cell[c] == 0, f"unaddressed cell {c} must have zero weight grad"


def test_gradient_continuous_across_bit_flip():
    """As the deciding margin crosses 0, the input gradient is continuous (C^1 surrogate).

    1 head / 1 table / 1 pair on d_in=2 => the only pair is the deciding one, and its
    margin u = z0 - z1 is fully controllable. The grad w.r.t. u at +eps and -eps must
    match (the hard cell and its neighbour swap, and the sign flips cancel).
    """
    spec = LUTSpec(n_heads=1, tph=1, nap=1, d_in=2, d_out=1)
    cart = ManifestoHardLUT(spec, seed=14, weight_init_std=1.0).train()

    def grad_u(eps: float) -> float:
        # Force margin u = z0 - z1 = eps.
        z = torch.tensor([[[eps, 0.0]]], requires_grad=True)
        cart.zero_grad()
        cart(z).sum().backward()
        # d/du = d/dz0 (and = -d/dz1); read it off the z0 slot.
        return float(z.grad[0, 0, 0])

    eps = 1e-3
    g_pos, g_neg = grad_u(eps), grad_u(-eps)
    assert abs(g_pos - g_neg) < 1e-5, f"grad discontinuous across flip: {g_pos} vs {g_neg}"
    assert abs(g_pos) > 0, "the deciding margin should carry a non-zero gradient"
