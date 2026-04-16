"""Tests for PermutationalLut."""
import pytest
import torch
import torch.nn as nn

from spiky.lutorch.permutational_lut import PermutationalLut


@pytest.fixture
def device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# -----------------------------------------------------------------------------
# Construction & validation
# -----------------------------------------------------------------------------

def test_aligned_requires_equal_naps(device):
    with pytest.raises(ValueError, match="input_nap == output_nap"):
        PermutationalLut(
            n_inputs=32, n_outputs=32, input_nap=5, output_nap=6,
            n_heads=2, tph=4, pair_mode='aligned', device=device,
        )


def test_aligned_requires_equal_dims(device):
    with pytest.raises(ValueError, match="n_inputs == n_outputs"):
        PermutationalLut(
            n_inputs=32, n_outputs=16, input_nap=5, output_nap=5,
            n_heads=2, tph=4, pair_mode='aligned', device=device,
        )


def test_invalid_pair_mode(device):
    with pytest.raises(ValueError, match="pair_mode"):
        PermutationalLut(
            n_inputs=8, n_outputs=8, input_nap=3, output_nap=3,
            n_heads=1, tph=2, pair_mode='nonsense', device=device,
        )


def test_invalid_soft_mode(device):
    with pytest.raises(ValueError, match="soft_mode"):
        PermutationalLut(
            n_inputs=8, n_outputs=8, input_nap=3, output_nap=3,
            n_heads=1, tph=2, soft_mode='quux', device=device,
        )


# -----------------------------------------------------------------------------
# Output shape and value range
# -----------------------------------------------------------------------------

@pytest.mark.parametrize("pair_mode", ['aligned', 'scrambled'])
def test_output_shape(device, pair_mode):
    n_in, n_out = (32, 32)
    lut = PermutationalLut(
        n_inputs=n_in, n_outputs=n_out, input_nap=5, output_nap=5,
        n_heads=4, tph=8, pair_mode=pair_mode, device=device,
        random_seed=0, recompute_in_backward=True,
    ).to(device)
    x = torch.randn(7, n_in, device=device)
    out = lut(x)
    assert out.shape == (7, 4, n_out)
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()


def test_scrambled_supports_different_dims(device):
    lut = PermutationalLut(
        n_inputs=32, n_outputs=16, input_nap=5, output_nap=4,
        n_heads=2, tph=8, pair_mode='scrambled', device=device,
        random_seed=0, recompute_in_backward=True,
    ).to(device)
    x = torch.randn(3, 32, device=device)
    out = lut(x)
    assert out.shape == (3, 2, 16)


@pytest.mark.parametrize("soft_mode", ['sigmoid', 'rational', 'ste'])
def test_per_pair_vote_in_range(device, soft_mode):
    """The centred signed vote must lie strictly in [-0.5, +0.5]."""
    lut = PermutationalLut(
        n_inputs=32, n_outputs=32, input_nap=5, output_nap=5,
        n_heads=2, tph=8, pair_mode='aligned', soft_mode=soft_mode,
        temperature=0.1, device=device, random_seed=0, recompute_in_backward=True,
    ).to(device)
    raw = torch.randn(4, 2, 8, 5, device=device) * 10  # extreme values
    d = lut._signed_vote(raw)
    assert d.min().item() >= -0.5 - 1e-6
    assert d.max().item() <= 0.5 + 1e-6


def test_output_centred_around_zero_at_init(device):
    """At init (small noise weights), per-batch per-head output should sum to ~0."""
    torch.manual_seed(0)
    lut = PermutationalLut(
        n_inputs=32, n_outputs=32, input_nap=5, output_nap=5,
        n_heads=4, tph=64, pair_mode='aligned', device=device,
        random_seed=0, initial_weights_noise=0.001, recompute_in_backward=True,
    ).to(device)
    x = torch.randn(8, 32, device=device)
    out = lut(x)  # [B, H, N]
    # Per-(batch, head) row sum: each scatter writes ±d, so sums must be exactly zero
    row_sum = out.sum(dim=-1)
    assert torch.allclose(row_sum, torch.zeros_like(row_sum), atol=1e-5)


# -----------------------------------------------------------------------------
# Gradients
# -----------------------------------------------------------------------------

@pytest.mark.parametrize("pair_mode", ['aligned', 'scrambled'])
@pytest.mark.parametrize("soft_mode", ['sigmoid', 'rational', 'ste'])
def test_gradients_flow(device, pair_mode, soft_mode):
    lut = PermutationalLut(
        n_inputs=32, n_outputs=32, input_nap=5, output_nap=5,
        n_heads=2, tph=8, pair_mode=pair_mode, soft_mode=soft_mode,
        device=device, random_seed=0, initial_weights_noise=0.1,
        recompute_in_backward=True,
    ).to(device)
    lut.train()
    x = torch.randn(4, 32, device=device, requires_grad=True)
    out = lut(x)
    target = torch.randn_like(out)
    loss = ((out - target) ** 2).mean()
    loss.backward()

    # Inner LUT weights must receive non-zero gradients
    w_grad = lut.inner.projection.weights.grad
    assert w_grad is not None
    assert w_grad.abs().sum().item() > 0
    assert not torch.isnan(w_grad).any()

    # Input gradient should also flow (LUT is differentiable through STE)
    assert x.grad is not None
    assert not torch.isnan(x.grad).any()


def test_ste_hard_forward(device):
    """STE mode: forward output should depend only on the SIGN of raw values."""
    lut = PermutationalLut(
        n_inputs=32, n_outputs=32, input_nap=5, output_nap=5,
        n_heads=2, tph=8, pair_mode='aligned', soft_mode='ste',
        temperature=0.1, device=device, random_seed=0, initial_weights_noise=0.1,
        recompute_in_backward=True,
    ).to(device)
    lut.eval()  # disable any dropout / training-only paths
    raw_small = torch.randn(4, 2, 8, 5, device=device) * 0.001
    raw_big = raw_small * 1000.0  # same signs, very different magnitudes
    d_small = lut._signed_vote(raw_small)
    d_big = lut._signed_vote(raw_big)
    # In STE, hard forward is sign(raw)*0.5, so values must match exactly
    assert torch.allclose(d_small, d_big)


# -----------------------------------------------------------------------------
# Determinism / reproducibility
# -----------------------------------------------------------------------------

def test_same_seed_same_pairs(device):
    a = PermutationalLut(
        n_inputs=32, n_outputs=32, input_nap=5, output_nap=5,
        n_heads=2, tph=8, pair_mode='scrambled',
        random_seed=42, device=device,
    ).to(device)
    b = PermutationalLut(
        n_inputs=32, n_outputs=32, input_nap=5, output_nap=5,
        n_heads=2, tph=8, pair_mode='scrambled',
        random_seed=42, device=device,
    ).to(device)
    assert torch.equal(a.idx_a, b.idx_a)
    assert torch.equal(a.idx_b, b.idx_b)


@pytest.mark.parametrize("soft_mode", ['sigmoid', 'rational', 'ste'])
@pytest.mark.parametrize("pair_mode", ['aligned', 'scrambled'])
def test_matmul_matches_scatter_forward(device, soft_mode, pair_mode):
    """Matmul aggregation must produce the same forward output as scatter."""
    kwargs = dict(
        n_inputs=32, n_outputs=32, input_nap=5, output_nap=5,
        n_heads=2, tph=16, pair_mode=pair_mode, soft_mode=soft_mode,
        temperature=0.1, device=device, random_seed=0,
        initial_weights_noise=0.1, recompute_in_backward=True,
    )
    lut_scatter = PermutationalLut(aggregation='scatter', **kwargs).to(device)
    lut_matmul = PermutationalLut(aggregation='matmul', **kwargs).to(device)

    # Sync weights so both paths see the same table entries
    with torch.no_grad():
        lut_matmul.inner.projection.weights.copy_(lut_scatter.inner.projection.weights)

    lut_scatter.eval()
    lut_matmul.eval()
    x = torch.randn(4, 32, device=device)

    out_s = lut_scatter(x)
    out_m = lut_matmul(x)
    assert out_s.shape == out_m.shape
    assert torch.allclose(out_s, out_m, atol=1e-5, rtol=1e-4), \
        f"max abs diff: {(out_s - out_m).abs().max()}"


@pytest.mark.parametrize("soft_mode", ['sigmoid', 'rational', 'ste'])
def test_matmul_gradients(device, soft_mode):
    """Matmul path must produce valid gradients on weights and input."""
    lut = PermutationalLut(
        n_inputs=32, n_outputs=32, input_nap=5, output_nap=5,
        n_heads=2, tph=16, pair_mode='aligned', soft_mode=soft_mode,
        aggregation='matmul', temperature=0.1, device=device,
        random_seed=0, initial_weights_noise=0.1, recompute_in_backward=True,
    ).to(device)
    lut.train()
    x = torch.randn(4, 32, device=device, requires_grad=True)
    out = lut(x)
    target = torch.randn_like(out)
    loss = ((out - target) ** 2).mean()
    loss.backward()
    w_grad = lut.inner.projection.weights.grad
    assert w_grad is not None
    assert w_grad.abs().sum().item() > 0
    assert not torch.isnan(w_grad).any()
    assert x.grad is not None
    assert not torch.isnan(x.grad).any()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA-only test")
@pytest.mark.parametrize("soft_mode", ['sigmoid', 'rational', 'ste'])
def test_native_matches_pytorch_forward(soft_mode):
    """The fused CUDA kernel must match the pure PyTorch fallback in forward."""
    dev = torch.device("cuda:0")
    lut = PermutationalLut(
        n_inputs=32, n_outputs=32, input_nap=5, output_nap=5,
        n_heads=2, tph=16, pair_mode='aligned', soft_mode=soft_mode,
        temperature=0.1, device=dev, random_seed=0, initial_weights_noise=0.1,
        recompute_in_backward=True,
    ).to(dev)
    lut.eval()
    x = torch.randn(4, 32, device=dev)

    # Force pure PyTorch path
    raw = lut.inner(x)
    out_pt = lut._forward_pytorch(raw)

    # Native path
    out_native = lut(x)

    assert out_pt.shape == out_native.shape
    # Floating-point differences from atomicAdd ordering can be tiny but real
    assert torch.allclose(out_pt, out_native, atol=1e-5, rtol=1e-4), \
        f"max abs diff: {(out_pt - out_native).abs().max()}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA-only test")
@pytest.mark.parametrize("soft_mode", ['sigmoid', 'rational', 'ste'])
def test_native_matches_pytorch_backward(soft_mode):
    """The fused CUDA kernel must produce the same weight gradients as PyTorch."""
    dev = torch.device("cuda:0")

    def make():
        torch.manual_seed(0)
        return PermutationalLut(
            n_inputs=32, n_outputs=32, input_nap=5, output_nap=5,
            n_heads=2, tph=16, pair_mode='aligned', soft_mode=soft_mode,
            temperature=0.1, device=dev, random_seed=0, initial_weights_noise=0.1,
            recompute_in_backward=True,
        ).to(dev)

    # Two identical models, run one through native, one through fallback
    lut_native = make()
    lut_pt = make()
    # Sync weights exactly
    with torch.no_grad():
        lut_pt.inner.projection.weights.copy_(lut_native.inner.projection.weights)

    lut_native.train()
    lut_pt.train()
    x = torch.randn(4, 32, device=dev)
    target = torch.randn(4, 2, 32, device=dev)

    # Native
    out_n = lut_native(x)
    loss_n = ((out_n - target) ** 2).mean()
    loss_n.backward()

    # PyTorch fallback (bypass _can_use_native by calling _forward_pytorch directly)
    raw = lut_pt.inner(x)
    out_p = lut_pt._forward_pytorch(raw)
    loss_p = ((out_p - target) ** 2).mean()
    loss_p.backward()

    g_n = lut_native.inner.projection.weights.grad
    g_p = lut_pt.inner.projection.weights.grad
    assert g_n is not None and g_p is not None
    assert torch.allclose(g_n, g_p, atol=1e-5, rtol=1e-4), \
        f"weight grad max abs diff: {(g_n - g_p).abs().max()}"


def test_aligned_uses_inner_anchor_pairs(device):
    """In aligned mode, idx_a / idx_b should match the inner LUT's anchor pairs."""
    lut = PermutationalLut(
        n_inputs=32, n_outputs=32, input_nap=5, output_nap=5,
        n_heads=4, tph=16, pair_mode='aligned',
        random_seed=7, device=device,
    ).to(device)
    expected_a = lut.inner.lookup.anchor_pairs_a.view(4, 16, 5).reshape(4, 16 * 5).long()
    expected_b = lut.inner.lookup.anchor_pairs_b.view(4, 16, 5).reshape(4, 16 * 5).long()
    assert torch.equal(lut.idx_a, expected_a)
    assert torch.equal(lut.idx_b, expected_b)


# -----------------------------------------------------------------------------
# fp8 quantization
# -----------------------------------------------------------------------------

def test_fp8_forward_backward(device):
    """use_fp8=True should produce valid forward/backward, and weights should differ from fp32."""
    lut_fp8 = PermutationalLut(
        n_inputs=32, n_outputs=32, input_nap=6, output_nap=8,
        n_heads=2, tph=16, pair_mode='scrambled', soft_mode='ste',
        use_fp8=True, random_seed=42, device=device,
        initial_weights_noise=0.1,
    )
    lut_fp32 = PermutationalLut(
        n_inputs=32, n_outputs=32, input_nap=6, output_nap=8,
        n_heads=2, tph=16, pair_mode='scrambled', soft_mode='ste',
        use_fp8=False, random_seed=42, device=device,
        initial_weights_noise=0.1,
    )
    x = torch.randn(4, 32, device=device)
    out_fp8 = lut_fp8(x)
    out_fp32 = lut_fp32(x)
    assert out_fp8.shape == out_fp32.shape
    # fp8 quantization changes the raw values, so outputs may differ
    # but should still be finite and close
    assert torch.isfinite(out_fp8).all()
    assert (out_fp8 - out_fp32).abs().max() < 1.0
    # backward should work
    out_fp8.sum().backward()
    assert lut_fp8.inner.projection.weights.grad is not None


def test_fp8_weights_restored_after_forward(device):
    """After forward, the original fp32 weights should be restored."""
    lut = PermutationalLut(
        n_inputs=32, n_outputs=32, input_nap=6, output_nap=8,
        n_heads=2, tph=16, pair_mode='scrambled', soft_mode='rational',
        use_fp8=True, random_seed=42, device=device,
        initial_weights_noise=0.1,
    )
    w_before = lut.inner.projection.weights.data.clone()
    x = torch.randn(4, 32, device=device)
    lut(x)
    w_after = lut.inner.projection.weights.data
    assert torch.equal(w_before, w_after), "fp32 weights should be restored after forward"


# -----------------------------------------------------------------------------
# return_dominance
# -----------------------------------------------------------------------------

def test_return_dominance_shape(device):
    d_v = 8
    P = d_v * (d_v - 1) // 2  # 28
    lut = PermutationalLut(
        n_inputs=32, n_outputs=d_v, input_nap=5, output_nap=4,
        n_heads=4, tph=16, pair_mode='scrambled', soft_mode='ste',
        return_dominance=True, random_seed=42, device=device,
        initial_weights_noise=0.1,
    )
    x = torch.randn(8, 32, device=device)
    out = lut(x)
    assert out.shape == (8, 4, P), f"expected (8, 4, {P}), got {out.shape}"
    assert torch.isfinite(out).all()
    out.sum().backward()
    assert lut.inner.projection.weights.grad is not None


def test_return_dominance_vs_borda(device):
    """Dominance output, when Borda-projected, should match normal Borda output."""
    d_v = 8
    P = d_v * (d_v - 1) // 2
    kwargs = dict(
        n_inputs=32, n_outputs=d_v, input_nap=5, output_nap=4,
        n_heads=4, tph=16, pair_mode='scrambled', soft_mode='rational',
        random_seed=42, device=device, initial_weights_noise=0.1,
    )
    lut_dom = PermutationalLut(return_dominance=True, **kwargs)
    lut_borda = PermutationalLut(return_dominance=False, **kwargs)
    x = torch.randn(8, 32, device=device)
    dom = lut_dom(x)  # [8, 4, P]
    borda = lut_borda(x)  # [8, 4, d_v]
    # Borda-project dominance: einsum('...p,kp->...k', dom, borda_m)
    borda_from_dom = torch.einsum('bhp,kp->bhk', dom, lut_dom.dom_borda_m)
    assert torch.allclose(borda_from_dom, borda, atol=1e-5), \
        f"max diff: {(borda_from_dom - borda).abs().max()}"
