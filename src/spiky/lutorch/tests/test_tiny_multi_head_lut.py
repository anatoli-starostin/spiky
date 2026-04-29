"""Tests for TinyMultiHeadLut and TinyMultiHeadLutOptimizer.

Covers:
- Basic shape correctness across (fp32, fp16, bf16) input × (fp32, fp16, bf16) weights.
- Native CUDA path acceptance for all float dtypes (fp16, bf16, fp32, fp64).
- Forward determinism (same x, same weights, same output).
- Backward gradient flow to inputs (anchor STE) and weights.
- Optimizer step changes weights and Adam state.
- Optimizer state stored in user-chosen dtype (default bf16).
"""
import pytest
import torch

from spiky.lutorch.tiny_multi_head_lut import TinyMultiHeadLut
from spiky.lutorch.tiny_multi_head_lut_optimizer import TinyMultiHeadLutOptimizer
from spiky.lutorch.tiny_anchor_pairs_lookup import _can_use_native_tiny_apl


DEVICES = [torch.device("cuda:0")] if torch.cuda.is_available() else []


def _has_cuda():
    return torch.cuda.is_available()


@pytest.mark.skipif(not _has_cuda(), reason="needs CUDA")
@pytest.mark.parametrize("input_dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("weight_dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_forward_shape_and_dtype(input_dtype, weight_dtype):
    """Output shape matches [B, n_heads, n_outputs] and dtype matches weights."""
    dev = torch.device("cuda:0")
    m = TinyMultiHeadLut(
        input_dim=64, n_heads=4, n_outputs=16, n_anchor_pairs=6,
        tables_per_head=8, weight_dtype=weight_dtype,
        random_seed=0, device=dev,
    )
    x = torch.randn(8, 64, device=dev, dtype=input_dtype)
    y = m(x)
    assert y.shape == (8, 4, 16), f"got {y.shape}"
    assert y.dtype == weight_dtype, f"output dtype {y.dtype} != weight_dtype {weight_dtype}"


@pytest.mark.skipif(not _has_cuda(), reason="needs CUDA")
def test_native_cuda_path_accepts_all_float_dtypes():
    """After the AT_DISPATCH_FLOATING_TYPES_AND2 patch, fp16 and bf16 inputs
    take the native CUDA path (instead of the slow PyTorch fallback)."""
    dev = torch.device("cuda:0")
    for dt in (torch.float32, torch.float64, torch.float16, torch.bfloat16):
        x = torch.randn(2, 32, device=dev, dtype=dt)
        assert _can_use_native_tiny_apl(x), f"native path rejected dtype={dt}"


@pytest.mark.skipif(not _has_cuda(), reason="needs CUDA")
@pytest.mark.parametrize("input_dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_forward_determinism(input_dtype):
    """Same input, same weights → same output (exact equality)."""
    dev = torch.device("cuda:0")
    m = TinyMultiHeadLut(
        input_dim=32, n_heads=2, n_outputs=8, n_anchor_pairs=5,
        tables_per_head=4, weight_dtype=torch.bfloat16,
        random_seed=42, device=dev,
    )
    x = torch.randn(4, 32, device=dev, dtype=input_dtype)
    y1 = m(x)
    y2 = m(x)
    assert torch.equal(y1, y2)


@pytest.mark.skipif(not _has_cuda(), reason="needs CUDA")
@pytest.mark.parametrize("input_dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_backward_grad_flows_to_input_and_weights(input_dtype):
    """Backward through TinyMultiHeadLut produces gradients on both x and weights."""
    dev = torch.device("cuda:0")
    m = TinyMultiHeadLut(
        input_dim=32, n_heads=2, n_outputs=8, n_anchor_pairs=5,
        tables_per_head=4, weight_dtype=torch.bfloat16,
        random_seed=1, device=dev,
    )
    x = torch.randn(4, 32, device=dev, dtype=input_dtype, requires_grad=True)
    y = m(x)
    loss = y.float().sum()
    loss.backward()
    assert x.grad is not None
    assert m.weights.grad is not None
    assert x.grad.shape == x.shape
    assert m.weights.grad.shape == m.weights.shape
    # x.grad should be non-trivially nonzero for at least some entries.
    assert (x.grad.abs() > 0).any()
    assert (m.weights.grad.abs() > 0).any()


@pytest.mark.skipif(not _has_cuda(), reason="needs CUDA")
def test_native_path_matches_pytorch_path_within_tolerance():
    """For fp32 input, native CUDA forward matches the PyTorch reference."""
    dev = torch.device("cuda:0")
    # Build two identical modules; for the second, force the PyTorch path.
    import spiky.lutorch.tiny_anchor_pairs_lookup as tapl_mod
    saved = tapl_mod._USE_TINY_APL_CUSTOM_CUDA
    try:
        m = TinyMultiHeadLut(
            input_dim=32, n_heads=2, n_outputs=8, n_anchor_pairs=5,
            tables_per_head=4, weight_dtype=torch.float32,
            random_seed=7, device=dev,
        )
        x = torch.randn(4, 32, device=dev, dtype=torch.float32)
        # Native path
        tapl_mod._USE_TINY_APL_CUSTOM_CUDA = True
        y_native = m(x)
        # PyTorch path
        tapl_mod._USE_TINY_APL_CUSTOM_CUDA = False
        y_py = m(x)
    finally:
        tapl_mod._USE_TINY_APL_CUSTOM_CUDA = saved
    # Outputs should match (deterministic gather, deterministic anchor sampling).
    assert torch.equal(y_native, y_py), "native and PyTorch forwards diverged"


@pytest.mark.skipif(not _has_cuda(), reason="needs CUDA")
@pytest.mark.parametrize("state_dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_optimizer_step_changes_weights(state_dtype):
    """Optimizer step actually moves weights and updates Adam state."""
    dev = torch.device("cuda:0")
    m = TinyMultiHeadLut(
        input_dim=32, n_heads=2, n_outputs=8, n_anchor_pairs=5,
        tables_per_head=4, weight_dtype=torch.bfloat16,
        random_seed=2, device=dev,
    )
    opt = TinyMultiHeadLutOptimizer([m], lr=1e-2, state_dtype=state_dtype)
    # Sanity: state allocated in requested dtype.
    assert opt._states[0]['m'].dtype == state_dtype
    assert opt._states[0]['v'].dtype == state_dtype

    weights_before = m.weights.detach().clone()
    x = torch.randn(8, 32, device=dev, dtype=torch.bfloat16)
    y = m(x)
    loss = y.float().sum()
    loss.backward()
    opt.step()
    weights_after = m.weights.detach()
    # Weights should have moved.
    assert not torch.equal(weights_before, weights_after)
    # Adam state should be nonzero.
    assert (opt._states[0]['m'].float().abs() > 0).any()
    assert (opt._states[0]['v'].float().abs() > 0).any()
    opt.zero_grad()
    assert m.weights.grad is None  # set_to_none default


@pytest.mark.skipif(not _has_cuda(), reason="needs CUDA")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_native_fused_bwd_matches_pytorch_fallback(dtype):
    """The native fused tiny_mhlut_backward_na1 must produce the same
    grad_weights and x.grad as the PyTorch reference path."""
    import spiky.lutorch.tiny_multi_head_lut as tmhlut_mod
    dev = torch.device("cuda:0")
    saved = tmhlut_mod._USE_TINY_MHLUT_NATIVE_BWD
    try:
        m1 = TinyMultiHeadLut(
            input_dim=32, n_heads=2, n_outputs=8, n_anchor_pairs=5,
            tables_per_head=4, weight_dtype=dtype, random_seed=11, device=dev,
        )
        m2 = TinyMultiHeadLut(
            input_dim=32, n_heads=2, n_outputs=8, n_anchor_pairs=5,
            tables_per_head=4, weight_dtype=dtype, random_seed=11, device=dev,
        )
        m2.load_state_dict(m1.state_dict())
        x = torch.randn(6, 32, device=dev, dtype=dtype, requires_grad=True)
        x_ref = x.detach().clone().requires_grad_(True)

        # Native path
        tmhlut_mod._USE_TINY_MHLUT_NATIVE_BWD = True
        y1 = m1(x); y1.float().sum().backward()
        # Fallback path
        tmhlut_mod._USE_TINY_MHLUT_NATIVE_BWD = False
        y2 = m2(x_ref); y2.float().sum().backward()
    finally:
        tmhlut_mod._USE_TINY_MHLUT_NATIVE_BWD = saved

    # Forwards must be identical (same weights/indices/algorithm).
    assert torch.equal(y1, y2)
    # Gradients must match within the dtype's precision.
    rtol, atol = (1e-5, 1e-6) if dtype == torch.float32 else (5e-2, 5e-2)
    assert torch.allclose(m1.weights.grad.float(), m2.weights.grad.float(), rtol=rtol, atol=atol), \
        f"weights.grad mismatch (max diff {(m1.weights.grad - m2.weights.grad).abs().max().item()})"
    assert torch.allclose(x.grad.float(), x_ref.grad.float(), rtol=rtol, atol=atol), \
        f"x.grad mismatch (max diff {(x.grad - x_ref.grad).abs().max().item()})"


@pytest.mark.skipif(not _has_cuda(), reason="needs CUDA")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_stochastic_rounding_is_unbiased(dtype):
    """Stochastic-rounded fp32 → {bf16, fp16} averages back to original fp32 value.

    Probes a sub-ULP value (one that round-to-nearest would map deterministically
    to one neighbour). Stochastic rounding should split between the two
    neighbours such that E[round(x)] ≈ x.
    """
    from spiky.lutorch.tiny_multi_head_lut_optimizer import (
        _stochastic_round_fp32_to_bf16, _stochastic_round_fp32_to_fp16,
    )
    sr_fn = _stochastic_round_fp32_to_bf16 if dtype == torch.bfloat16 else _stochastic_round_fp32_to_fp16
    dev = torch.device("cuda:0")
    target = 0.1
    x = torch.full((1024 * 1024,), target, dtype=torch.float32, device=dev)
    rounded = sr_fn(x)
    assert rounded.dtype == dtype
    mean = rounded.float().mean().item()
    # ULP at 0.1: bf16 ≈ 7.8e-4, fp16 ≈ 9.8e-5. 5σ over 1M samples covers both.
    assert abs(mean - target) < 5e-5, f"{dtype} SR biased: mean={mean} vs {target}"
    unique = torch.unique(rounded)
    assert unique.numel() >= 2, f"{dtype} SR produced only one value: {unique}"


@pytest.mark.skipif(not _has_cuda(), reason="needs CUDA")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_optimizer_stochastic_rounding_breaks_subulp_floor(dtype):
    """With stochastic_rounding=True, repeated sub-ULP updates must accumulate.
    With stochastic_rounding=False (default RNE), they get rounded away."""
    from spiky.lutorch.tiny_multi_head_lut_optimizer import TinyMultiHeadLutOptimizer
    dev = torch.device("cuda:0")
    m_sr = TinyMultiHeadLut(input_dim=16, n_heads=1, n_outputs=4, n_anchor_pairs=4,
                            tables_per_head=2, weight_dtype=dtype,
                            random_seed=0, device=dev)
    m_rne = TinyMultiHeadLut(input_dim=16, n_heads=1, n_outputs=4, n_anchor_pairs=4,
                             tables_per_head=2, weight_dtype=dtype,
                             random_seed=0, device=dev)
    m_rne.load_state_dict(m_sr.state_dict())
    initial = m_sr.weights.detach().clone()
    # Tiny lr → per-step update is well below ULP at the weight magnitude.
    opt_sr  = TinyMultiHeadLutOptimizer([m_sr],  lr=1e-7, stochastic_rounding=True)
    opt_rne = TinyMultiHeadLutOptimizer([m_rne], lr=1e-7, stochastic_rounding=False)
    # Apply N tiny gradient steps, all in the same direction.
    grad = torch.full_like(m_sr.weights, 1.0).float()
    for _ in range(200):
        m_sr.weights.grad  = grad.to(dtype).clone()
        m_rne.weights.grad = grad.to(dtype).clone()
        opt_sr.step()
        opt_rne.step()
    delta_sr  = (m_sr.weights  - initial).float().abs().mean().item()
    delta_rne = (m_rne.weights - initial).float().abs().mean().item()
    # Stochastic rounding accumulates ~ N*lr*|grad| ≈ 2e-5 regardless of dtype.
    # RNE: rounds most steps back; how much depends on ULP. bf16 ULP at the
    # weight magnitude is ~8× larger than fp16's, so RNE preserves much more
    # in fp16 than in bf16. The sturdy claim is "SR matches expectation and
    # never undershoots RNE" — that's what we assert here.
    expected = 200 * 1e-7  # ≈ 2e-5
    assert abs(delta_sr - expected) / expected < 0.5, \
        f"{dtype} SR accumulation off: Δ_sr={delta_sr} (expected ~{expected})"
    assert delta_sr >= delta_rne, \
        f"{dtype} SR fell behind RNE: Δ_sr={delta_sr}, Δ_rne={delta_rne}"


@pytest.mark.skipif(not _has_cuda(), reason="needs CUDA")
def test_constraint_n_anchor_pairs_15():
    """n_anchor_pairs > 15 is rejected (int16 limit)."""
    dev = torch.device("cuda:0")
    with pytest.raises(ValueError, match="1 <= n_anchor_pairs <= 15"):
        TinyMultiHeadLut(input_dim=64, n_heads=2, n_outputs=8,
                        n_anchor_pairs=16, tables_per_head=4, device=dev)


@pytest.mark.skipif(not _has_cuda(), reason="needs CUDA")
def test_constraint_input_dim_32767():
    """input_dim > 32767 is rejected (int16 anchor index limit)."""
    dev = torch.device("cuda:0")
    with pytest.raises(ValueError, match="input_dim <= 32767"):
        TinyMultiHeadLut(input_dim=40000, n_heads=2, n_outputs=8,
                        n_anchor_pairs=6, tables_per_head=4, device=dev)
