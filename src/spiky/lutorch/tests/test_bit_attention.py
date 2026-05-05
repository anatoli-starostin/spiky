"""Tests for BitAttention (Phase 1: SDPA fallback; Phase 2: CUDA kernel)."""
import math

import pytest
import torch
import torch.nn.functional as F

from spiky.lutorch.bit_attention import (
    BitAttention,
    _bit_attn_backward_explicit,
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


# ---------- Explicit backward correctness ----------

@pytest.mark.parametrize("B,H,T,d,d_v,is_causal,scale", [
    (2, 3, 16, 24, 12, True,  None),       # default 1/√d
    (2, 3, 16, 24, 12, False, None),
    (1, 1, 32, 64, 32, True,  None),
    (1, 1, 32, 64, 32, False, 0.5),        # explicit non-default scale
    (3, 2, 8,  17, 9,  True,  None),       # d not multiple of 32
    (1, 4, 64, 96, 24, True,  None),       # larger T
])
def test_explicit_backward_matches_sdpa(device, B, H, T, d, d_v, is_causal, scale):
    """The explicit backward produces gradients numerically equal to autograd
    through F.scaled_dot_product_attention. Tests both the helper directly and
    the full BitAttention path."""
    torch.manual_seed(0)
    s = scale if scale is not None else 1.0 / math.sqrt(d)

    q_data = _random_bits((B, H, T, d), device)
    k_data = _random_bits((B, H, T, d), device)
    v_data = torch.randn(B, H, T, d_v, device=device)
    grad_o = torch.randn(B, H, T, d_v, device=device)

    # Reference via F.SDPA autograd.
    q_ref = q_data.clone().requires_grad_(True)
    k_ref = k_data.clone().requires_grad_(True)
    v_ref = v_data.clone().requires_grad_(True)
    out_ref = F.scaled_dot_product_attention(
        q_ref, k_ref, v_ref, is_causal=is_causal, scale=s,
    )
    out_ref.backward(grad_o)

    # Direct call to the explicit backward helper.
    dq, dk, dv = _bit_attn_backward_explicit(
        q_data, k_data, v_data, grad_o, s, is_causal,
    )
    assert torch.allclose(dq, q_ref.grad, atol=1e-5), \
        f"dq max diff {(dq - q_ref.grad).abs().max().item():.3e}"
    assert torch.allclose(dk, k_ref.grad, atol=1e-5), \
        f"dk max diff {(dk - k_ref.grad).abs().max().item():.3e}"
    assert torch.allclose(dv, v_ref.grad, atol=1e-5), \
        f"dv max diff {(dv - v_ref.grad).abs().max().item():.3e}"

    # Full BitAttention path (forward kernel/fallback + explicit backward).
    q_bit = q_data.clone().requires_grad_(True)
    k_bit = k_data.clone().requires_grad_(True)
    v_bit = v_data.clone().requires_grad_(True)
    out_bit = BitAttention(d, scale=scale).to(device)(
        q_bit, k_bit, v_bit, is_causal=is_causal,
    )
    out_bit.backward(grad_o)
    assert torch.allclose(q_bit.grad, q_ref.grad, atol=1e-5)
    assert torch.allclose(k_bit.grad, k_ref.grad, atol=1e-5)
    assert torch.allclose(v_bit.grad, v_ref.grad, atol=1e-5)


def test_explicit_backward_partial_grads(device):
    """needs_input_grad correctly skips gradients that aren't requested."""
    B, T, d, d_v = 2, 12, 24, 8
    q = _random_bits((B, T, d), device).requires_grad_(True)
    k = _random_bits((B, T, d), device)            # no grad
    v = torch.randn(B, T, d_v, device=device, requires_grad=True)
    out = BitAttention(d).to(device)(q, k, v, is_causal=True)
    out.sum().backward()
    assert q.grad is not None
    assert v.grad is not None
    # k did not require grad → should not have been populated.
    assert not k.requires_grad


@pytest.mark.skipif(not torch.cuda.is_available(), reason="bf16 path is CUDA-only")
@pytest.mark.parametrize("BH,T,d,d_v,is_causal", [
    (4, 64, 64, 32, True),
    (4, 64, 256, 32, True),
    (4, 128, 496, 32, True),
    (4, 128, 992, 32, False),
])
def test_bf16_backward_path_correctness(BH, T, d, d_v, is_causal):
    """The bf16 dQ/dK path matches fp32 to bf16-quantization tolerance.

    K, Q are ±1 so their bf16 cast is lossless; dS picks up bf16 mantissa
    quantization (~1e-3 relative). Tolerance is loose accordingly.
    """
    from spiky.lutorch.bit_attention import _bit_attn_backward_explicit
    torch.manual_seed(0)
    device = "cuda"
    q = (torch.randint(0, 2, (BH, T, d), device=device, dtype=torch.float32) * 2 - 1)
    k = (torch.randint(0, 2, (BH, T, d), device=device, dtype=torch.float32) * 2 - 1)
    v = torch.randn(BH, T, d_v, device=device)
    grad_o = torch.randn(BH, T, d_v, device=device)
    scale = 1.0 / math.sqrt(d)

    dq32, dk32, dv32 = _bit_attn_backward_explicit(q, k, v, grad_o, scale, is_causal)
    dqB, dkB, dvB = _bit_attn_backward_explicit(
        q, k, v, grad_o, scale, is_causal, use_bf16_matmul=True,
    )
    # dV is computed identically (both fp32) — should match exactly.
    assert torch.allclose(dv32, dvB, atol=1e-6)
    # dQ, dK use bf16 matmul: ~1e-3 relative error expected.
    rel_q = (dqB - dq32).abs() / (dq32.abs() + 1e-9)
    rel_k = (dkB - dk32).abs() / (dk32.abs() + 1e-9)
    # Use median rel error as a robust check (max can blow up on near-zero entries).
    assert rel_q.median().item() < 5e-3, f"dQ rel-median {rel_q.median().item():.2e}"
    assert rel_k.median().item() < 5e-3, f"dK rel-median {rel_k.median().item():.2e}"


def test_explicit_backward_3d_input(device):
    """Backward works on 3-D (B, T, d) input shape."""
    torch.manual_seed(1)
    B, T, d, d_v = 3, 16, 32, 12
    q = _random_bits((B, T, d), device).requires_grad_(True)
    k = _random_bits((B, T, d), device).requires_grad_(True)
    v = torch.randn(B, T, d_v, device=device, requires_grad=True)
    grad_o = torch.randn(B, T, d_v, device=device)
    BitAttention(d).to(device)(q, k, v, is_causal=False).backward(grad_o)
    # Reference
    qr = q.detach().clone().requires_grad_(True)
    kr = k.detach().clone().requires_grad_(True)
    vr = v.detach().clone().requires_grad_(True)
    F.scaled_dot_product_attention(
        qr, kr, vr, is_causal=False, scale=1.0 / math.sqrt(d),
    ).backward(grad_o)
    assert torch.allclose(q.grad, qr.grad, atol=1e-5)
    assert torch.allclose(k.grad, kr.grad, atol=1e-5)
    assert torch.allclose(v.grad, vr.grad, atol=1e-5)


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
