"""Tests for BitAttention (Phase 1: SDPA fallback; Phase 2: CUDA kernel)."""
import math

import pytest
import torch
import torch.nn.functional as F

from spiky.lutorch.bit_attention import (
    BitAttention,
    _get_native,
    _native_has_bit_attn,
)


_KERNEL_MARK = pytest.mark.skipif(
    not (torch.cuda.is_available() and _native_has_bit_attn()),
    reason="bit_attn_flash_forward kernel not available (rebuild required)",
)


def _random_bits(shape, device):
    """Random ±1 float tensor."""
    return (torch.randint(0, 2, shape, device=device, dtype=torch.float32) * 2 - 1)


def test_output_shape(device):
    B, H, T, d, d_v = 2, 4, 16, 24, 12
    q = _random_bits((B, H, T, d), device)
    k = _random_bits((B, H, T, d), device)
    v = torch.randn(B, H, T, d_v, device=device)
    out = BitAttention(d).to(device)(q, k, v, is_causal=True)
    assert out.shape == (B, H, T, d_v)
    assert torch.isfinite(out).all()


def test_matches_sdpa_default_scale(device):
    """Phase 1 must equal F.scaled_dot_product_attention with 1/√d scale."""
    B, H, T, d, d_v = 2, 2, 8, 20, 6
    q = _random_bits((B, H, T, d), device)
    k = _random_bits((B, H, T, d), device)
    v = torch.randn(B, H, T, d_v, device=device)
    out_bit = BitAttention(d).to(device)(q, k, v, is_causal=True)
    out_sdpa = F.scaled_dot_product_attention(
        q, k, v, is_causal=True, scale=1.0 / math.sqrt(d),
    )
    assert torch.allclose(out_bit, out_sdpa, atol=1e-6)


def test_explicit_scale(device):
    """User-supplied scale is honoured."""
    B, H, T, d, d_v = 2, 2, 8, 16, 6
    q = _random_bits((B, H, T, d), device)
    k = _random_bits((B, H, T, d), device)
    v = torch.randn(B, H, T, d_v, device=device)
    ba = BitAttention(d, scale=0.1).to(device)
    out = ba(q, k, v, is_causal=False)
    ref = F.scaled_dot_product_attention(q, k, v, is_causal=False, scale=0.1)
    assert torch.allclose(out, ref, atol=1e-6)


def test_causal_masks_future(device):
    """Causal mask forbids future keys; first position depends only on itself."""
    B, H, T, d, d_v = 1, 1, 4, 8, 4
    q = _random_bits((B, H, T, d), device)
    k = _random_bits((B, H, T, d), device)
    v = torch.randn(B, H, T, d_v, device=device)
    out = BitAttention(d).to(device)(q, k, v, is_causal=True)
    # First query only attends to first key → output[..., 0, :] == v[..., 0, :].
    assert torch.allclose(out[..., 0, :], v[..., 0, :], atol=1e-6)


def test_backward_flows(device):
    B, H, T, d, d_v = 2, 2, 8, 20, 6
    q = _random_bits((B, H, T, d), device).requires_grad_(True)
    k = _random_bits((B, H, T, d), device).requires_grad_(True)
    v = torch.randn(B, H, T, d_v, device=device, requires_grad=True)
    out = BitAttention(d).to(device)(q, k, v, is_causal=True)
    out.sum().backward()
    assert q.grad is not None and (q.grad.abs() > 0).any()
    assert k.grad is not None and (k.grad.abs() > 0).any()
    assert v.grad is not None and (v.grad.abs() > 0).any()


def test_leading_dim_variations(device):
    """Module accepts any leading-dim shape that SDPA accepts — e.g. 3-D (B, T, d)."""
    B, T, d, d_v = 3, 10, 16, 8
    q = _random_bits((B, T, d), device)
    k = _random_bits((B, T, d), device)
    v = torch.randn(B, T, d_v, device=device)
    out = BitAttention(d).to(device)(q, k, v, is_causal=False)
    assert out.shape == (B, T, d_v)


def test_popcount_identity_holds_for_bits():
    """Sanity: the identity q·k = d − 2·popcount(q XOR k) used by the Phase 2
    kernel is correct for ±1 vectors packed with +1→0, −1→1."""
    d = 17
    for _ in range(8):
        q_bit = torch.randint(0, 2, (d,), dtype=torch.int64)
        k_bit = torch.randint(0, 2, (d,), dtype=torch.int64)
        q_pm = (1 - 2 * q_bit).float()   # +1 → 0 → +1 ; −1 ← 1 ← −1
        k_pm = (1 - 2 * k_bit).float()
        popcount = int((q_bit ^ k_bit).sum())
        assert (q_pm * k_pm).sum().item() == pytest.approx(d - 2 * popcount)


# ---------- Phase 2: kernel path ----------

@_KERNEL_MARK
@pytest.mark.parametrize("d,d_v,T,is_causal", [
    (24, 16, 16, False),
    (24, 16, 16, True),
    (276, 120, 128, True),   # exp314's q/k d=276, v d_v=120
    (32, 32, 8, False),
    (33, 17, 12, True),      # d not multiple of 32, d_v not power of 2
])
def test_kernel_matches_sdpa(d, d_v, T, is_causal):
    """Direct kernel invocation vs F.scaled_dot_product_attention. Since the
    popcount identity is exact for ±1 bits, the two must agree up to fp
    rollup noise in the softmax accumulation order."""
    torch.manual_seed(0)
    BH = 3
    q = (torch.randint(0, 2, (BH, T, d), device="cuda", dtype=torch.float32) * 2 - 1)
    k = (torch.randint(0, 2, (BH, T, d), device="cuda", dtype=torch.float32) * 2 - 1)
    v = torch.randn(BH, T, d_v, device="cuda")
    scale = 1.0 / math.sqrt(d)

    native = _get_native()
    out_k = native.bit_attn_flash_forward(q, k, v, scale, is_causal)
    out_ref = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal, scale=scale)

    # Softmax reordering noise grows with T; use a generous tolerance but
    # tighter than "anything finite".
    diff = (out_k - out_ref).abs()
    assert diff.max().item() < 5e-5, f"max diff {diff.max().item():.3e}"
    assert diff.mean().item() < 1e-5, f"mean diff {diff.mean().item():.3e}"


@_KERNEL_MARK
def test_module_uses_kernel_on_cuda():
    """BitAttention should dispatch to the kernel on CUDA (check indirectly by
    confirming numerical match with SDPA under kernel-covered sizes)."""
    torch.manual_seed(1)
    B, H, T, d, d_v = 2, 4, 32, 24, 16
    q = (torch.randint(0, 2, (B, H, T, d), device="cuda", dtype=torch.float32) * 2 - 1)
    k = (torch.randint(0, 2, (B, H, T, d), device="cuda", dtype=torch.float32) * 2 - 1)
    v = torch.randn(B, H, T, d_v, device="cuda")
    ba = BitAttention(d).cuda()
    out = ba(q, k, v, is_causal=True)
    ref = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=1.0 / math.sqrt(d))
    assert (out - ref).abs().max().item() < 5e-5


@_KERNEL_MARK
def test_kernel_grads_match_sdpa():
    """Backward re-runs SDPA on the saved float tensors, so gradients should
    match autograd through F.scaled_dot_product_attention exactly."""
    torch.manual_seed(2)
    B, H, T, d, d_v = 2, 2, 16, 24, 16
    q_data = (torch.randint(0, 2, (B, H, T, d), device="cuda", dtype=torch.float32) * 2 - 1)
    k_data = (torch.randint(0, 2, (B, H, T, d), device="cuda", dtype=torch.float32) * 2 - 1)
    v_data = torch.randn(B, H, T, d_v, device="cuda")

    # BitAttention path.
    q1 = q_data.clone().requires_grad_(True)
    k1 = k_data.clone().requires_grad_(True)
    v1 = v_data.clone().requires_grad_(True)
    out1 = BitAttention(d).cuda()(q1, k1, v1, is_causal=True)
    out1.sum().backward()

    # Reference SDPA path.
    q2 = q_data.clone().requires_grad_(True)
    k2 = k_data.clone().requires_grad_(True)
    v2 = v_data.clone().requires_grad_(True)
    out2 = F.scaled_dot_product_attention(q2, k2, v2, is_causal=True, scale=1.0 / math.sqrt(d))
    out2.sum().backward()

    for a, b, name in [(q1.grad, q2.grad, "q"), (k1.grad, k2.grad, "k"), (v1.grad, v2.grad, "v")]:
        assert torch.allclose(a, b, atol=1e-5), \
            f"{name}.grad differs: max={(a-b).abs().max().item():.3e}"


@_KERNEL_MARK
def test_kernel_envelope_falls_back():
    """Sizes outside the kernel's envelope (d > 512 or d_v > 128) must fall
    back to SDPA transparently — output still correct."""
    torch.manual_seed(3)
    BH, T = 2, 16
    # d_v > 128: fallback path.
    d, d_v = 24, 192
    q = (torch.randint(0, 2, (BH, T, d), device="cuda", dtype=torch.float32) * 2 - 1)
    k = (torch.randint(0, 2, (BH, T, d), device="cuda", dtype=torch.float32) * 2 - 1)
    v = torch.randn(BH, T, d_v, device="cuda")
    ba = BitAttention(d).cuda()
    out = ba(q, k, v, is_causal=True)
    ref = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=1.0 / math.sqrt(d))
    assert torch.allclose(out, ref, atol=1e-5)
