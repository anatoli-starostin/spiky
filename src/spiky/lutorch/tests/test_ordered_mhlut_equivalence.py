"""Test that TinyOrderedMultiHeadLut == TinyMultiHeadLut when all pairs are independent.

When anchor_sampling_policy=CANONICAL_FULL_COVERAGE and input_dim is large enough
that all sampled pairs are disjoint, _build_ordering_table returns the identity map
(ordering_table[t, k] = k for all k) and max_orders == 2^NAP.  In that case:

  - reachable_bit_matrix == bit_matrix  (same MSB-first sign convention)
  - TinyOrderedMHLut forward argmax == TinyMHLut sign-pack index
  - Both backward bodies reconstruct the same p_signs from the saved index
  - Weight and input gradients are identical (up to floating-point order)
"""

import torch
import pytest
from spiky.lutorch.tiny_multi_head_lut import (
    TinyMultiHeadLut,
    TinyOrderedMultiHeadLut,
    _soft_bit_matrix_msb,
)
from spiky.lutorch.lut_helpers import AnchorSamplingPolicy

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _make_pair(nap, n_heads, n_outputs, tph, input_dim, seed, *,
               soft_score_temp=0.5, select_temp=0.5, learnable_temps=True):
    """Build a matched (TinyMultiHeadLut, TinyOrderedMultiHeadLut) pair.

    Both modules share the same random_seed → same anchor pairs.
    Weights are initialised identically (same seed → same RNG state) because
    both constructors call torch.Generator(...).manual_seed(seed + 1) before
    sampling from the same [n_tables, K, n_outputs] shape.
    """
    common = dict(
        input_dim=input_dim,
        n_heads=n_heads,
        n_outputs=n_outputs,
        n_anchor_pairs=nap,
        tables_per_head=tph,
        anchor_sampling_policy=AnchorSamplingPolicy.CANONICAL_FULL_COVERAGE,
        random_seed=seed,
        initial_weights_noise=0.01,
        device=DEVICE,
        soft_score_temp=soft_score_temp,
        select_temp=select_temp,
        learnable_temps=learnable_temps,
        weight_dtype=torch.float32,
        use_bf16=False,
    )
    mhlut = TinyMultiHeadLut(
        **common,
        backward_mode="soft",
        argmax_noise_eps=0.0,
        einsum_bf16_forward=False,
    )
    ordered = TinyOrderedMultiHeadLut(**common)
    return mhlut, ordered


# ---------------------------------------------------------------------------
# Structural assertions
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("nap", [4, 6])
def test_max_orders_equals_table_dim(nap):
    mhlut, ordered = _make_pair(nap, n_heads=2, n_outputs=8, tph=4, input_dim=64, seed=0)
    assert ordered.max_orders == mhlut.table_dim, (
        f"Expected max_orders={mhlut.table_dim}, got {ordered.max_orders}"
    )


@pytest.mark.parametrize("nap", [4, 6])
def test_anchor_pairs_match(nap):
    mhlut, ordered = _make_pair(nap, n_heads=2, n_outputs=8, tph=4, input_dim=64, seed=0)
    assert torch.equal(mhlut.soft_anchor_a_long, ordered.soft_anchor_a_long)
    assert torch.equal(mhlut.soft_anchor_b_long, ordered.soft_anchor_b_long)


@pytest.mark.parametrize("nap", [4, 6])
def test_reachable_bit_matrix_equals_bit_matrix(nap):
    mhlut, ordered = _make_pair(nap, n_heads=2, n_outputs=8, tph=4, input_dim=64, seed=0)
    bit_matrix = _soft_bit_matrix_msb(nap, DEVICE, dtype=torch.float32)  # [NAP, K]
    rbm = ordered.reachable_bit_matrix                                     # [NAP, max_orders]
    assert torch.equal(bit_matrix, rbm), (
        f"reachable_bit_matrix differs from bit_matrix at nap={nap}"
    )


# ---------------------------------------------------------------------------
# Forward equivalence
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("nap,n_heads,n_outputs,tph", [
    (4, 2, 8, 4),
    (6, 3, 16, 8),
])
def test_forward_identical(nap, n_heads, n_outputs, tph):
    mhlut, ordered = _make_pair(nap, n_heads, n_outputs, tph, input_dim=128, seed=7)

    x = torch.randn(16, 128, device=DEVICE)
    out_m = mhlut(x)
    out_o = ordered(x)

    assert out_m.shape == out_o.shape, f"Shape mismatch: {out_m.shape} vs {out_o.shape}"
    assert torch.allclose(out_m, out_o, atol=1e-5), (
        f"Forward outputs differ: max |diff| = {(out_m - out_o).abs().max():.2e}"
    )


# ---------------------------------------------------------------------------
# Backward equivalence
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("nap,n_heads,n_outputs,tph", [
    (4, 2, 8, 4),
    (6, 3, 16, 8),
])
def test_backward_weight_grad_identical(nap, n_heads, n_outputs, tph):
    mhlut, ordered = _make_pair(nap, n_heads, n_outputs, tph, input_dim=128, seed=13,
                                 learnable_temps=False)

    x_m = torch.randn(16, 128, device=DEVICE, requires_grad=False)
    x_o = x_m.clone()

    mhlut.weights.grad = None
    ordered.weight_table.grad = None

    mhlut(x_m).sum().backward()
    ordered(x_o).sum().backward()

    assert torch.allclose(mhlut.weights.grad, ordered.weight_table.grad, atol=1e-5), (
        f"Weight grad differs: max |diff| = {(mhlut.weights.grad - ordered.weight_table.grad).abs().max():.2e}"
    )


@pytest.mark.parametrize("nap,n_heads,n_outputs,tph", [
    (4, 2, 8, 4),
    (6, 3, 16, 8),
])
def test_backward_input_grad_identical(nap, n_heads, n_outputs, tph):
    mhlut, ordered = _make_pair(nap, n_heads, n_outputs, tph, input_dim=128, seed=17,
                                 learnable_temps=False)

    x = torch.randn(16, 128, device=DEVICE)
    x_m = x.clone().requires_grad_(True)
    x_o = x.clone().requires_grad_(True)

    grad_out = torch.randn(16, n_heads, n_outputs, device=DEVICE)
    mhlut(x_m).backward(grad_out)
    ordered(x_o).backward(grad_out.clone())

    assert torch.allclose(x_m.grad, x_o.grad, atol=1e-5), (
        f"Input grad differs: max |diff| = {(x_m.grad - x_o.grad).abs().max():.2e}"
    )


@pytest.mark.parametrize("nap", [4, 6])
def test_backward_temp_grads_identical(nap):
    mhlut, ordered = _make_pair(nap, n_heads=2, n_outputs=8, tph=4, input_dim=128, seed=21,
                                 learnable_temps=True)

    x = torch.randn(16, 128, device=DEVICE)
    mhlut(x).sum().backward()
    ordered(x.clone()).sum().backward()

    assert torch.allclose(
        mhlut.log_soft_score_temp.grad,
        ordered.log_soft_score_temp.grad, atol=1e-4,
    ), "log_soft_score_temp grad differs"

    assert torch.allclose(
        mhlut.log_select_temp.grad,
        ordered.log_select_temp.grad, atol=1e-4,
    ), "log_select_temp grad differs"
