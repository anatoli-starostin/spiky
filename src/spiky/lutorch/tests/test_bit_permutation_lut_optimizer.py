"""Tests for BitPermutationLUTOptimizer."""
import math

import pytest
import torch

from spiky.lutorch.bit_permutation_lut import BitPermutationLUT
from spiky.lutorch.bit_permutation_lut_optimizer import (
    BitPermutationLUTOptimizer,
    _project_grad_out_to_weight_grad,
    _to_fp8_fixed,
    _from_fp8_fixed,
    _to_fp8_per_table,
    _from_fp8_per_table,
)


CUDA_ONLY = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA-only")


@CUDA_ONLY
def test_fp8_fixed_roundtrip_preserves_signs():
    """fp8-fixed quantization preserves signs for typical latent values."""
    torch.manual_seed(0)
    t = torch.randn(8, 64, 16, device="cuda:0") * 0.01  # small values
    q = _from_fp8_fixed(_to_fp8_fixed(t, mx=1.0), mx=1.0)
    assert q.dtype == torch.float32
    assert (q.sign() == t.sign()).all(), "sign flipped by fp8 quantization"


@CUDA_ONLY
def test_fp8_per_table_roundtrip_bounded_error():
    """fp8 per-table quantization: relative error per table is ~fp8 quantum."""
    torch.manual_seed(0)
    t = torch.randn(16, 32, 8, device="cuda:0")
    q_fp8, scale = _to_fp8_per_table(t)
    q = _from_fp8_per_table(q_fp8, scale)
    # Per-table scale sized so each table's amax → 448; fp8_e4m3fn has ~2^-3 precision
    # in the high range, so relative error is ~1/16 in the worst case.
    err = (q - t).abs()
    amax = t.abs().amax(dim=(1, 2), keepdim=True).clamp(min=1e-20)
    rel_err = (err / amax).max().item()
    assert rel_err < 0.2, f"per-table fp8 rel err too large: {rel_err}"


@CUDA_ONLY
@pytest.mark.parametrize(
    "n_inputs,n_outputs,n_heads,input_nap,output_nap,tph,B",
    [
        (16,  8, 2, 4,  5, 4, 3),
        (32, 12, 2, 5,  8, 6, 4),
        (32, 16, 4, 6, 12, 4, 2),
    ],
)
def test_weight_grad_projection_matches_bruteforce(
    n_inputs, n_outputs, n_heads, input_nap, output_nap, tph, B
):
    """Vectorized _project_grad_out_to_weight_grad matches a double-loop reference."""
    dev = torch.device("cuda:0")
    torch.manual_seed(0)
    lut = BitPermutationLUT(
        n_inputs=n_inputs, n_outputs=n_outputs,
        n_heads=n_heads, input_nap=input_nap, output_nap=output_nap, tph=tph,
        random_seed=3, device=dev,
    )
    x = torch.randn(B, n_inputs, device=dev)
    with torch.no_grad():
        li, _, _, _, _ = lut.anchor(x)
    P = n_outputs * (n_outputs - 1) // 2
    grad_out = torch.randn(B, n_heads, P, device=dev)

    wg_vec = _project_grad_out_to_weight_grad(
        grad_out, li, lut.output_idx_per_table,
        lut.n_heads, lut.tph, lut.output_nap, lut.table_dim, lut.scale,
    )

    # Brute-force reference
    N = n_heads * tph
    wg_ref = torch.zeros_like(wg_vec)
    pair = lut.output_idx_per_table.long()
    for b in range(B):
        for n in range(N):
            h = n // tph
            t = n - h * tph
            e = int(li[b, n].item())
            for k in range(output_nap):
                p = int(pair[h, t, k].item())
                wg_ref[n, e, k] += lut.scale * grad_out[b, h, p]

    assert torch.allclose(wg_vec, wg_ref, atol=1e-5, rtol=1e-5), \
        f"max |diff| = {(wg_vec - wg_ref).abs().max().item()}"


@CUDA_ONLY
def test_optimizer_storage_is_fp8():
    dev = torch.device("cuda:0")
    lut = BitPermutationLUT(
        n_inputs=32, n_outputs=12, n_heads=2, input_nap=5, output_nap=8, tph=8,
        random_seed=0, device=dev,
    )
    opt = BitPermutationLUTOptimizer([lut], lr=1e-3)
    s = opt._states[0]
    assert lut.latent_fp8.dtype == torch.float8_e4m3fn
    assert s["m_fp8"].dtype == torch.float8_e4m3fn
    assert s["v_fp8"].dtype == torch.float8_e4m3fn
    # Per-table scales: [N, 1, 1] float32
    N = lut.n_heads * lut.tph
    assert s["m_scale"].shape == (N, 1, 1)
    assert s["m_scale"].dtype == torch.float32
    opt.close()


@CUDA_ONLY
def test_optimizer_initial_bit_weights_match_latent_sign():
    dev = torch.device("cuda:0")
    lut = BitPermutationLUT(
        n_inputs=32, n_outputs=12, n_heads=2, input_nap=5, output_nap=8, tph=8,
        random_seed=0, device=dev,
    )
    opt = BitPermutationLUTOptimizer([lut], lr=1e-3)
    latent = opt.state_as_float(0)["latent"]
    # bit_pack_signs kernel uses `signs > 0 ? 1 : 0` → 0 latent decodes to -1.
    expected_signs = torch.where(latent > 0, 1.0, -1.0)
    actual_signs = lut.get_bit_weights_as_signs()
    assert torch.equal(actual_signs, expected_signs)
    opt.close()


@CUDA_ONLY
def test_optimizer_step_reduces_synthetic_loss():
    """On a small synthetic target (random BitPermLUT), the optimizer should
    reduce the MSE below its starting value within a small number of steps."""
    dev = torch.device("cuda:0")

    # Target BitPermLUT with fixed ±1 weights.
    target_lut = BitPermutationLUT(
        n_inputs=32, n_outputs=16, n_heads=2,
        input_nap=5, output_nap=8, tph=16,
        random_seed=7, device=dev,
    )
    # Force random bits on the teacher so student can't match at init.
    signs = torch.where(
        torch.randn_like(target_lut.get_bit_weights_as_signs()) > 0, 1.0, -1.0,
    )
    target_lut.set_bit_weights_from_signs(signs)
    target_lut.eval()

    # Student BitPermLUT — same anchor layout (same seed) but its own latent
    # init (small uniform -> ~50% sign match to teacher at step 0).
    student = BitPermutationLUT(
        n_inputs=32, n_outputs=16, n_heads=2,
        input_nap=5, output_nap=8, tph=16,
        random_seed=7, device=dev,
    )
    opt = BitPermutationLUTOptimizer([student], lr=1e-2)

    torch.manual_seed(0)
    N_TRAIN = 2048
    x = torch.randn(N_TRAIN, 32, device=dev)
    with torch.no_grad():
        y = target_lut(x)

    bs = 256
    initial_loss = None
    final_loss = None
    for step in range(1, 201):
        idx = torch.randint(0, N_TRAIN, (bs,), device=dev)
        opt.zero_grad()
        x_batch = x[idx].detach().requires_grad_(True)
        out = student(x_batch)
        loss = ((out - y[idx]) ** 2).mean()
        loss.backward()
        opt.step()
        if step == 1:
            initial_loss = loss.item()
        final_loss = loss.item()
    opt.close()

    assert final_loss < initial_loss * 0.9, \
        f"loss didn't decrease: initial={initial_loss:.4f} -> final={final_loss:.4f}"


@CUDA_ONLY
def test_optimizer_bit_weights_change_over_steps():
    dev = torch.device("cuda:0")
    lut = BitPermutationLUT(
        n_inputs=32, n_outputs=16, n_heads=2,
        input_nap=5, output_nap=8, tph=16,
        random_seed=0, device=dev,
    )
    opt = BitPermutationLUTOptimizer([lut], lr=1e-2)
    bw_before = lut.bit_weights.clone()

    x = torch.randn(16, 32, device=dev, requires_grad=True)
    target = torch.randn(16, 2, 16 * 15 // 2, device=dev)
    for _ in range(20):
        opt.zero_grad()
        loss = ((lut(x) - target) ** 2).mean()
        loss.backward()
        opt.step()
    bw_after = lut.bit_weights
    assert not torch.equal(bw_before, bw_after), "bit_weights unchanged after 20 opt steps"
    opt.close()


@CUDA_ONLY
def test_optimizer_close_detaches_hooks():
    dev = torch.device("cuda:0")
    lut = BitPermutationLUT(
        n_inputs=16, n_outputs=8, n_heads=2, input_nap=4, output_nap=5, tph=4,
        random_seed=0, device=dev,
    )
    opt = BitPermutationLUTOptimizer([lut], lr=1e-3)
    assert len(opt._handles) == 1
    opt.close()
    assert len(opt._handles) == 0
    # After close, forward hook should not fire (no grad_out captured).
    x = torch.randn(4, 16, device=dev, requires_grad=True)
    out = lut(x)
    out.sum().backward()
    assert opt._states[0]["grad_out"] is None


@CUDA_ONLY
def test_optimizer_nan_grad_does_not_corrupt_state():
    """Non-finite grad_out entries must not propagate NaN/Inf into fp8 state.
    Element-wise masking (torch.nan_to_num_) replaces bad entries with 0
    on the GPU; other elements update normally.
    """
    dev = torch.device("cuda:0")
    lut = BitPermutationLUT(
        n_inputs=16, n_outputs=8, n_heads=2, input_nap=4, output_nap=5, tph=4,
        random_seed=0, device=dev,
    )
    opt = BitPermutationLUTOptimizer([lut], lr=1e-2)

    x = torch.randn(4, 16, device=dev, requires_grad=True)
    opt.zero_grad()
    out = lut(x)
    # Poison: multiply output by inf -> grad_out becomes non-finite.
    poison = torch.full_like(out, float("inf"))
    (out * poison).sum().backward()
    opt.step()

    st = opt.state_as_float(0)
    assert torch.isfinite(st["latent"]).all(), "latent went non-finite"
    assert torch.isfinite(st["m"]).all(), "m went non-finite"
    assert torch.isfinite(st["v"]).all(), "v went non-finite"
    opt.close()


@CUDA_ONLY
def test_optimizer_hook_captures_lookup_indices():
    """Forward hook re-runs anchor (in no_grad) to capture lookup_indices."""
    dev = torch.device("cuda:0")
    lut = BitPermutationLUT(
        n_inputs=16, n_outputs=8, n_heads=2, input_nap=4, output_nap=5, tph=4,
        random_seed=0, device=dev,
    )
    opt = BitPermutationLUTOptimizer([lut], lr=1e-3)

    x = torch.randn(3, 16, device=dev, requires_grad=True)
    opt.zero_grad()
    lut(x).sum().backward()
    # After forward+backward: hook fired, captured lookup_indices in state.
    li = opt._states[0]["lookup_indices"]
    assert li is not None
    assert li.shape == (3, lut.n_heads * lut.tph)
    assert li.dtype == torch.int16
    opt.step()
    # After step: state is cleared.
    assert opt._states[0]["lookup_indices"] is None
    opt.close()


@CUDA_ONLY
def test_optimizer_state_memory_is_fp8_only():
    """All persistent optimizer state should be fp8 (1 byte/elem) + small f32 scales."""
    dev = torch.device("cuda:0")
    lut = BitPermutationLUT(
        n_inputs=32, n_outputs=12, n_heads=2, input_nap=5, output_nap=8, tph=16,
        random_seed=0, device=dev,
    )
    opt = BitPermutationLUTOptimizer([lut], lr=1e-3)
    s = opt._states[0]
    N = lut.n_heads * lut.tph  # 32
    W = N * lut.table_dim * lut.output_nap  # 32 * 32 * 8 = 8192
    # fp8 tensors: 1 byte/elem. Latent lives on lut, m/v on optimizer state.
    bytes_fp8 = (
        lut.latent_fp8.numel() * lut.latent_fp8.element_size()
        + s["m_fp8"].numel() * s["m_fp8"].element_size()
        + s["v_fp8"].numel() * s["v_fp8"].element_size()
    )
    assert bytes_fp8 == 3 * W, f"expected 3*{W}={3 * W} fp8 bytes, got {bytes_fp8}"
    # Per-table scales: 4 bytes/table.
    bytes_scale = s["m_scale"].numel() * s["m_scale"].element_size() + s["v_scale"].numel() * s["v_scale"].element_size()
    assert bytes_scale == 2 * 4 * N, f"expected 2*4*{N}={8 * N} scale bytes, got {bytes_scale}"
    opt.close()


@CUDA_ONLY
def test_optimizer_tolerates_huge_grad_without_blowup():
    """Large but finite grad shouldn't NaN the fp8 state (clamp + per-table scales
    should keep things in range)."""
    dev = torch.device("cuda:0")
    lut = BitPermutationLUT(
        n_inputs=16, n_outputs=8, n_heads=2, input_nap=4, output_nap=5, tph=4,
        random_seed=0, device=dev,
    )
    opt = BitPermutationLUTOptimizer([lut], lr=1e-2)
    x = torch.randn(4, 16, device=dev, requires_grad=True)
    for _ in range(5):
        opt.zero_grad()
        # Scale output by 1e6 to produce huge gradients downstream.
        loss = (lut(x) * 1e6).pow(2).sum()
        loss.backward()
        opt.step()
    st = opt.state_as_float(0)
    assert torch.isfinite(st["latent"]).all(), "latent went non-finite under huge grads"
    assert torch.isfinite(st["m"]).all()
    assert torch.isfinite(st["v"]).all()
    # Latent must stay within clamp range [-1, 1].
    assert st["latent"].abs().max().item() <= 1.0 + 1e-5
    opt.close()


@CUDA_ONLY
def test_optimizer_lr_schedule():
    dev = torch.device("cuda:0")
    lut = BitPermutationLUT(
        n_inputs=16, n_outputs=8, n_heads=2, input_nap=4, output_nap=5, tph=4,
        random_seed=0, device=dev,
    )

    calls = []
    opt = BitPermutationLUTOptimizer(
        [lut], lr=1e-2,
        lr_schedule_fn=lambda s: (calls.append(s), 0.0)[1],  # lr multiplier = 0
    )
    x = torch.randn(4, 16, device=dev, requires_grad=True)
    target = torch.randn(4, 2, 28, device=dev)
    latent_before = opt.state_as_float(0)["latent"].clone()
    opt.zero_grad()
    ((lut(x) - target) ** 2).mean().backward()
    opt.step()
    latent_after = opt.state_as_float(0)["latent"]
    # With lr multiplier forced to 0, latent should be unchanged (bit-exact through fp8 rt).
    assert torch.allclose(latent_after, latent_before, atol=1e-6)
    assert calls == [1]
    opt.close()
