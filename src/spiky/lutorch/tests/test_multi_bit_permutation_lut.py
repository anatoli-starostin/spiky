"""Tests for MultiBitPermutationLUT and MultiBitPermutationLUTOptimizer.

Covers:
  * Basic construction for all supported bit_widths.
  * Input validation (bad bit_width, bad input_nap, bad n_outputs).
  * Forward shape + finite-ness for K in {2, 4, 8}.
  * Forward numerical correctness: matches a pure-PyTorch reference that
    reads bit_weights, sign-extends K-bit signed values, and scatter-sums.
  * refresh_bit_weights: editing latent then re-packing changes bit_weights
    consistent with the new latent.
  * Partition_sets: no cross-partition input anchor pair when the flag is set.
  * Backward: gradient flows from loss back to the input carrier (proof that
    autograd Function is wired correctly).
  * Optimizer: training loop reduces MSE loss against a random linear target.
  * Optimizer: close() removes hooks (no re-fire on subsequent forwards).
"""
import math
import pytest
import torch
import torch.nn.functional as F

from spiky.lutorch.multi_bit_permutation_lut import MultiBitPermutationLUT
from spiky.lutorch.multi_bit_permutation_lut_optimizer import (
    MultiBitPermutationLUTOptimizer,
)


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
DEVICE = torch.device("cuda:0")


# ---------------------------------------------------------------------------
# Construction / validation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bit_width", [2, 4, 8])
def test_constructs_all_supported_bit_widths(bit_width):
    m = MultiBitPermutationLUT(
        n_inputs=32, n_outputs=16, n_heads=2,
        input_nap=5, output_nap=8, tph=16,
        bit_width=bit_width, device=DEVICE, random_seed=0,
    )
    # Expected storage shapes.
    assert m.latent_bf16.shape == (2 * 16, 1 << 5, 8)
    n_blocks_k = (8 * bit_width + 31) // 32
    assert m.bit_weights.shape == (2 * 16, 1 << 5, n_blocks_k)
    assert m.latent_bf16.dtype == torch.bfloat16
    assert m.bit_weights.dtype == torch.int32


@pytest.mark.parametrize("bad_width", [0, 1, 3, 5, 7, 9, 16])
def test_rejects_unsupported_bit_width(bad_width):
    with pytest.raises(ValueError, match="bit_width"):
        MultiBitPermutationLUT(
            n_inputs=8, n_outputs=4, n_heads=1,
            input_nap=3, output_nap=4, tph=4,
            bit_width=bad_width, device=DEVICE,
        )


def test_rejects_too_large_input_nap():
    with pytest.raises(ValueError, match="input_nap"):
        MultiBitPermutationLUT(
            n_inputs=32, n_outputs=16, n_heads=1,
            input_nap=16, output_nap=8, tph=16,
            bit_width=4, device=DEVICE,
        )


def test_rejects_small_n_outputs():
    with pytest.raises(ValueError, match="n_outputs"):
        MultiBitPermutationLUT(
            n_inputs=32, n_outputs=1, n_heads=1,
            input_nap=3, output_nap=4, tph=8,
            bit_width=4, device=DEVICE,
        )


# ---------------------------------------------------------------------------
# Forward shape, finite-ness
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bit_width", [2, 4, 8])
def test_forward_shape_and_finite(bit_width):
    m = MultiBitPermutationLUT(
        n_inputs=32, n_outputs=16, n_heads=2,
        input_nap=5, output_nap=12, tph=16,
        bit_width=bit_width, device=DEVICE, random_seed=1,
        initial_weights_noise=0.5,
    )
    x = torch.randn(4, 32, device=DEVICE, requires_grad=True)
    y = m(x)
    assert y.shape == (4, 2, 16 * 15 // 2)
    assert y.dtype == torch.float32
    assert torch.isfinite(y).all()


# ---------------------------------------------------------------------------
# Forward correctness: bitwise against a pure-PyTorch reference.
# ---------------------------------------------------------------------------

def _forward_reference(m: MultiBitPermutationLUT, x: torch.Tensor) -> torch.Tensor:
    """Reference that decodes bit_weights as signed K-bit ints, gathers,
    scatter-sums, and applies the midrise output_bias. Matches the CUDA forward."""
    K = m.bit_width
    slots_per_block = 32 // K
    kmask = (1 << K) - 1

    with torch.no_grad():
        li, _, _, _, _ = m.anchor(x)
        li = li.long()
        W = torch.zeros(
            m.n_heads * m.tph, m.table_dim, m.output_nap,
            dtype=torch.int32, device=DEVICE,
        )
        bw = m.bit_weights
        for k in range(m.output_nap):
            bi = k // slots_per_block
            bit_off = (k % slots_per_block) * K
            raw = (bw[:, :, bi] >> bit_off) & kmask
            # Sign-extend K-bit to int32.
            W[:, :, k] = (raw << (32 - K)).to(torch.int32) >> (32 - K)

        B, N = li.shape
        n_ix = torch.arange(N, device=DEVICE).unsqueeze(0).expand(B, N)
        votes = W[n_ix, li]
        pair_idx = m.output_idx_per_table.long()
        pair_flat = pair_idx.reshape(1, m.n_heads, m.tph * m.output_nap).expand(B, -1, -1)
        votes_h = votes.view(B, m.n_heads, m.tph * m.output_nap).to(torch.int32)
        out_int = torch.zeros(
            B, m.n_heads, m.n_pairs, dtype=torch.int32, device=DEVICE,
        )
        out_int.scatter_add_(2, pair_flat, votes_h)
        return out_int.float() * m.scale + m.output_bias


@pytest.mark.parametrize("bit_width", [2, 4, 8])
def test_forward_matches_python_reference(bit_width):
    torch.manual_seed(42)
    m = MultiBitPermutationLUT(
        n_inputs=32, n_outputs=16, n_heads=2,
        input_nap=6, output_nap=12, tph=8,
        bit_width=bit_width, device=DEVICE, random_seed=42,
        initial_weights_noise=0.7,  # ensure varied signed values
    )
    x = torch.randn(4, 32, device=DEVICE)
    y_cuda = m(x)
    y_ref = _forward_reference(m, x)
    # Bitwise equality expected: int32 sum is exact; scale is a single float mul.
    assert torch.allclose(y_cuda, y_ref, atol=0.0), \
        f"max |diff|: {(y_cuda - y_ref).abs().max().item():.3e}"


# ---------------------------------------------------------------------------
# refresh_bit_weights: editing latent and re-packing.
# ---------------------------------------------------------------------------

def test_refresh_bit_weights_reflects_latent_changes():
    # pre_quant_temperature=0 bypasses rational — direct midrise quantize on latent
    # so this test can check concrete bit patterns.
    m = MultiBitPermutationLUT(
        n_inputs=16, n_outputs=8, n_heads=1,
        input_nap=4, output_nap=8, tph=8,
        bit_width=4, device=DEVICE, random_seed=0,
        initial_weights_noise=0.3, pre_quant_temperature=0.0,
    )
    # Zero the latent -> midrise floor(0*8) = 0 => q_signed = 0 => unsigned bits = 0.
    m.latent_bf16.zero_()
    m.refresh_bit_weights()
    assert (m.bit_weights == 0).all()
    # All latents at +1.0 -> floor(1*8) = 8, clamped to max 7 => bits 0b0111.
    m.latent_bf16.fill_(1.0)
    m.refresh_bit_weights()
    assert (m.bit_weights == 0x77777777).all(), (
        f"expected all 0x77777777, got unique={torch.unique(m.bit_weights)}"
    )
    # All latents at -1.0 -> floor(-1*8) = -8 (min) => two's-comp 4-bit = 0b1000.
    m.latent_bf16.fill_(-1.0)
    m.refresh_bit_weights()
    # Each byte: 0x88. Full block: 0x88888888 = -2004318072 as signed int32.
    expected = torch.tensor(-2004318072, dtype=torch.int32, device=DEVICE)  # 0x88888888
    assert (m.bit_weights == expected).all()


# ---------------------------------------------------------------------------
# Partition_sets: no cross-partition input anchor pair.
# ---------------------------------------------------------------------------

def test_partition_sets_restricts_input_anchors():
    H, d_v = 4, 16
    partition_sets = [list(range(h * d_v, (h + 1) * d_v)) for h in range(H)]
    m = MultiBitPermutationLUT(
        n_inputs=H * d_v, n_outputs=32, n_heads=1,
        input_nap=6, output_nap=10, tph=256,
        bit_width=4, device=DEVICE, random_seed=0,
        partition_sets=partition_sets,
    )
    a = m.anchor.anchor_pairs_a.long().cpu().flatten()
    b = m.anchor.anchor_pairs_b.long().cpu().flatten()
    part_of = {}
    for i, P in enumerate(partition_sets):
        for k in P:
            part_of[k] = i
    cross = sum(1 for u, v in zip(a.tolist(), b.tolist()) if part_of[u] != part_of[v])
    assert cross == 0


# ---------------------------------------------------------------------------
# Backward: gradient flows back to input.
# ---------------------------------------------------------------------------

def test_backward_propagates_to_input():
    m = MultiBitPermutationLUT(
        n_inputs=16, n_outputs=8, n_heads=1,
        input_nap=4, output_nap=8, tph=8,
        bit_width=4, device=DEVICE, random_seed=0,
        initial_weights_noise=0.5,
    )
    x = torch.randn(4, 16, device=DEVICE, requires_grad=True)
    y = m(x)
    y.sum().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    # Some gradient should be non-zero (not all cells saturated-out).
    assert x.grad.abs().sum().item() > 0.0


# ---------------------------------------------------------------------------
# Optimizer: training loop reduces loss.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bit_width", [2, 4, 8])
def test_optimizer_reduces_loss(bit_width):
    torch.manual_seed(0)
    m = MultiBitPermutationLUT(
        n_inputs=32, n_outputs=16, n_heads=1,
        input_nap=6, output_nap=12, tph=32,
        bit_width=bit_width, device=DEVICE, random_seed=42,
        initial_weights_noise=0.5,
    )
    opt = MultiBitPermutationLUTOptimizer([m], lr=5e-3)
    try:
        torch.manual_seed(1)
        x = torch.randn(32, 32, device=DEVICE, requires_grad=True)
        n_pairs = 16 * 15 // 2
        target = torch.randn(32, 1, n_pairs, device=DEVICE) * 0.2

        loss0 = None
        for step in range(60):
            y = m(x)
            loss = F.mse_loss(y, target)
            if step == 0:
                loss0 = loss.item()
            opt.zero_grad()
            loss.backward()
            opt.step()
        assert loss.item() < loss0 * 0.95, \
            f"loss did not decrease ({loss0:.4f} -> {loss.item():.4f})"
    finally:
        opt.close()


def test_optimizer_close_removes_hooks():
    m = MultiBitPermutationLUT(
        n_inputs=16, n_outputs=8, n_heads=1,
        input_nap=4, output_nap=8, tph=8,
        bit_width=4, device=DEVICE, random_seed=0,
    )
    opt = MultiBitPermutationLUTOptimizer([m], lr=1e-3)
    # Run once so hooks fire.
    x = torch.randn(4, 16, device=DEVICE, requires_grad=True)
    y = m(x)
    y.sum().backward()
    assert opt._state[0]["grad_out"] is not None
    opt.close()

    # After close, a fresh backward should NOT re-populate grad_out (hook removed).
    for s in opt._state:
        s["grad_out"] = None
        s["lookup_indices"] = None
    x2 = torch.randn(4, 16, device=DEVICE, requires_grad=True)
    y2 = m(x2)
    y2.sum().backward()
    assert opt._state[0]["grad_out"] is None
