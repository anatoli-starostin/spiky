"""Tests for SoftMultiHeadLUT.

Covers:
- bit_matrix structure (columns are +/-1 binary expansions, MSB top).
- Forward shape, dtype, finiteness.
- Selector-temperature buffer mutation via set_select_temp.
- Backward grads flow to weights and to the input (through soft signs and softmax).
- Hard limit: with hard +/-1 signs and tiny T_sel, soft lookup approximates the
  one-hot-row gather of a discrete table — i.e. SoftMultiHeadLUT collapses to
  the same function MultiHeadLut computes when both temperatures vanish.
- Determinism: same x, same weights, same output.
- Gumbel branch produces valid output and grads.
- nap upper-bound guard rejects oversized configs.
"""
import math

import pytest
import torch

from spiky.lutorch.soft_multi_head_lut import SoftMultiHeadLUT, _bit_matrix


def _has_cuda():
    return torch.cuda.is_available()


DEVICES = [torch.device("cuda:0")] if _has_cuda() else [torch.device("cpu")]


# ---------- bit matrix --------------------------------------------------------

@pytest.mark.parametrize("dim", [1, 2, 3, 5, 8])
def test_bit_matrix_structure(dim):
    bm = _bit_matrix(dim)
    n = 1 << dim
    assert bm.shape == (dim, n)
    # All entries are exactly +/-1.
    assert torch.all((bm == 1) | (bm == -1))
    # Column k must be the binary expansion of k (MSB top), mapped to +/-1.
    for k in range(n):
        for i in range(dim):
            bit = (k >> (dim - 1 - i)) & 1
            expected = 1.0 if bit == 1 else -1.0
            assert bm[i, k].item() == expected, f"col {k} bit {i}: {bm[i, k].item()} != {expected}"


# ---------- forward -----------------------------------------------------------

@pytest.mark.parametrize("device", DEVICES)
def test_forward_shape_and_finite(device):
    m = SoftMultiHeadLUT(
        input_dim=64, n_outputs=16, n_anchor_pairs=6,
        n_heads=4, tables_per_head=8, random_seed=0, device=device,
    )
    x = torch.randn(8, 64, device=device)
    y = m(x)
    assert y.shape == (8, 4, 16)
    assert y.dtype == torch.float32
    assert torch.isfinite(y).all()


# ---------- buffer mutation ---------------------------------------------------

def test_set_select_temp_mutates_buffer():
    m = SoftMultiHeadLUT(
        input_dim=16, n_outputs=4, n_anchor_pairs=3,
        n_heads=1, tables_per_head=2, select_temp=1.0, random_seed=0,
    )
    assert float(m.select_temp) == 1.0
    m.set_select_temp(0.05)
    assert float(m.select_temp) == pytest.approx(0.05, rel=0, abs=1e-6)
    # Buffer (not Parameter): it must not require grad.
    assert not m.select_temp.requires_grad


# ---------- backward ----------------------------------------------------------

@pytest.mark.parametrize("device", DEVICES)
def test_backward_flows_to_weights_and_input(device):
    m = SoftMultiHeadLUT(
        input_dim=32, n_outputs=8, n_anchor_pairs=5,
        n_heads=2, tables_per_head=4, random_seed=0, device=device,
    )
    x = torch.randn(4, 32, device=device, requires_grad=True)
    y = m(x)
    y.sum().backward()
    assert m.weights.grad is not None
    assert torch.isfinite(m.weights.grad).all()
    assert m.weights.grad.abs().sum() > 0
    # Soft signs are differentiable -> input grad must exist and be non-trivial.
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert x.grad.abs().sum() > 0


# ---------- determinism -------------------------------------------------------

@pytest.mark.parametrize("device", DEVICES)
def test_forward_deterministic(device):
    m = SoftMultiHeadLUT(
        input_dim=24, n_outputs=6, n_anchor_pairs=4,
        n_heads=2, tables_per_head=3, random_seed=0, device=device,
    )
    x = torch.randn(5, 24, device=device)
    y1 = m(x)
    y2 = m(x)
    assert torch.equal(y1, y2)


# ---------- hard-limit equivalence with discrete table -----------------------

@pytest.mark.parametrize("device", DEVICES)
def test_hard_limit_matches_one_hot_row(device):
    """At T_soft -> 0 and T_sel -> 0, the soft lookup picks the row whose
    +/-1 bit pattern matches sign(x_a - x_b) per pair, then returns that row's
    weight vector. We verify by constructing the expected one-hot row index
    independently and comparing to the soft output.
    """
    torch.manual_seed(7)
    nap = 5
    n_heads = 2
    tph = 4
    n_outputs = 3
    n_inputs = 32

    m = SoftMultiHeadLUT(
        input_dim=n_inputs, n_outputs=n_outputs, n_anchor_pairs=nap,
        n_heads=n_heads, tables_per_head=tph,
        soft_score_temp=1e-6,                # signs essentially hard
        select_temp=1e-3,                    # softmax essentially one-hot
        initial_weights_noise=1.0,           # large weights so signal >> numerical noise
        random_seed=0, device=device,
    )
    B = 6
    # Use inputs with comfortably-non-zero pair gaps so signs are unambiguous.
    x = torch.randn(B, n_inputs, device=device) * 5.0

    y_soft = m(x)

    # Compute the discrete row index for each (b, t):
    #   bit k of the row index = 1  iff  x[a] > x[b]  (matches +1 in bit_matrix col)
    # bit_matrix layout: row 0 is the MSB. Build the integer accordingly.
    a_idx = m.anchor_pairs_a                  # [n_tables, nap]
    b_idx = m.anchor_pairs_b
    x_a = x[:, a_idx]                          # [B, n_tables, nap]
    x_b = x[:, b_idx]
    sign_pos = (x_a > x_b).long()              # [B, n_tables, nap], 1 if +1, else 0
    # Convert nap bits (MSB at position 0) to row index in [0, 2^nap).
    powers = (1 << torch.arange(nap - 1, -1, -1, device=device)).long()  # [nap]
    row_idx = (sign_pos * powers).sum(dim=-1)  # [B, n_tables]

    # Gather the one-hot row from each table's weights.
    n_tables = n_heads * tph
    table_ix = torch.arange(n_tables, device=device).view(1, n_tables).expand(B, n_tables)
    rows = m.weights[table_ix, row_idx]         # [B, n_tables, n_outputs]
    y_hard = rows.view(B, n_heads, tph, n_outputs).sum(dim=2)

    # Soft and hard should agree to within numerical slack of the temperatures.
    assert torch.allclose(y_soft, y_hard, atol=5e-3, rtol=5e-3), (
        f"max abs diff = {(y_soft - y_hard).abs().max().item():.3e}"
    )


# ---------- gumbel branch -----------------------------------------------------

@pytest.mark.parametrize("device", DEVICES)
def test_gumbel_branch_runs_and_grads(device):
    torch.manual_seed(0)
    m = SoftMultiHeadLUT(
        input_dim=24, n_outputs=6, n_anchor_pairs=4,
        n_heads=2, tables_per_head=3, gumbel=True,
        random_seed=0, device=device,
    )
    x = torch.randn(4, 24, device=device, requires_grad=True)
    y = m(x)
    assert y.shape == (4, 2, 6)
    assert torch.isfinite(y).all()
    y.sum().backward()
    assert torch.isfinite(m.weights.grad).all()
    assert torch.isfinite(x.grad).all()


# ---------- hard mode (straight-through) --------------------------------------

@pytest.mark.parametrize("device", DEVICES)
def test_hard_forward_equals_argmax_row_gather(device):
    """hard=True forward must be exactly the one-hot row gather at argmax(sel)."""
    torch.manual_seed(11)
    nap, n_heads, tph, n_outputs, n_inputs = 5, 2, 4, 3, 32
    m = SoftMultiHeadLUT(
        input_dim=n_inputs, n_outputs=n_outputs, n_anchor_pairs=nap,
        n_heads=n_heads, tables_per_head=tph,
        soft_score_temp=0.01, select_temp=1.0,            # T_sel doesn't affect argmax
        hard=True,
        initial_weights_noise=1.0, random_seed=0, device=device,
    )
    B = 6
    x = torch.randn(B, n_inputs, device=device) * 5.0
    y_hard = m(x)

    # Independently compute the argmax row index from sign(x_a - x_b).
    a_idx, b_idx = m.anchor_pairs_a, m.anchor_pairs_b
    rd = x[:, a_idx] - x[:, b_idx]
    sign_pos = (rd > 0).long()                            # [B, n_tables, nap]
    powers = (1 << torch.arange(nap - 1, -1, -1, device=device)).long()
    row_idx = (sign_pos * powers).sum(dim=-1)             # [B, n_tables]

    n_tables = n_heads * tph
    table_ix = torch.arange(n_tables, device=device).view(1, n_tables).expand(B, n_tables)
    rows = m.weights[table_ix, row_idx]                   # [B, n_tables, n_outputs]
    y_expected = rows.view(B, n_heads, tph, n_outputs).sum(dim=2)

    # Hard mode is exact (no temperature smoothing in forward).
    assert torch.allclose(y_hard, y_expected, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("device", DEVICES)
def test_hard_mode_grads_flow_through_soft_path(device):
    """hard=True must still propagate gradient back through ts (soft signs +
    softmax surrogate) so the input and weights both see non-trivial gradient."""
    torch.manual_seed(0)
    m = SoftMultiHeadLUT(
        input_dim=24, n_outputs=6, n_anchor_pairs=4,
        n_heads=2, tables_per_head=3,
        select_temp=0.5, hard=True,
        random_seed=0, device=device,
    )
    x = torch.randn(4, 24, device=device, requires_grad=True)
    y = m(x)
    y.sum().backward()
    assert m.weights.grad is not None
    assert torch.isfinite(m.weights.grad).all()
    assert m.weights.grad.abs().sum() > 0
    # Input grad must flow back through the soft surrogate (rational sign +
    # softmax), not be killed by the hard one-hot in forward.
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert x.grad.abs().sum() > 0


# ---------- learnable temperatures --------------------------------------------

@pytest.mark.parametrize("device", DEVICES)
def test_learnable_temps_register_parameters(device):
    """learnable_temps=True exposes log_soft_score_temp / log_select_temp as Parameters."""
    m = SoftMultiHeadLUT(
        input_dim=24, n_outputs=6, n_anchor_pairs=4,
        n_heads=2, tables_per_head=3,
        soft_score_temp=0.5, select_temp=0.5,
        learnable_temps=True,
        random_seed=0, device=device,
    )
    assert isinstance(m.log_soft_score_temp, torch.nn.Parameter)
    assert isinstance(m.log_select_temp, torch.nn.Parameter)
    # exp(log_T) recovers the init values.
    assert torch.allclose(m.log_soft_score_temp.exp(), torch.tensor(0.5, device=device), atol=1e-6)
    assert torch.allclose(m.log_select_temp.exp(),     torch.tensor(0.5, device=device), atol=1e-6)
    # Fixed-mode buffers/floats should not be present.
    assert not hasattr(m, "soft_score_temp") or not isinstance(getattr(m, "soft_score_temp", None), float) or getattr(m, "soft_score_temp", None) is None


@pytest.mark.parametrize("device", DEVICES)
def test_learnable_temps_receive_gradient(device):
    """Both log_T parameters get non-zero gradient from a normal forward+backward."""
    torch.manual_seed(0)
    m = SoftMultiHeadLUT(
        input_dim=24, n_outputs=6, n_anchor_pairs=4,
        n_heads=2, tables_per_head=3,
        soft_score_temp=0.5, select_temp=0.5,
        learnable_temps=True,
        random_seed=0, device=device,
    )
    x = torch.randn(8, 24, device=device)
    y = m(x)
    y.sum().backward()
    assert m.log_soft_score_temp.grad is not None
    assert m.log_select_temp.grad is not None
    assert torch.isfinite(m.log_soft_score_temp.grad).all()
    assert torch.isfinite(m.log_select_temp.grad).all()
    assert m.log_soft_score_temp.grad.abs().sum() > 0
    assert m.log_select_temp.grad.abs().sum() > 0


@pytest.mark.parametrize("device", DEVICES)
def test_set_select_temp_works_with_learnable_temps(device):
    """set_select_temp must update the underlying log Parameter when learnable."""
    m = SoftMultiHeadLUT(
        input_dim=16, n_outputs=4, n_anchor_pairs=3,
        n_heads=1, tables_per_head=2,
        select_temp=1.0, learnable_temps=True,
        random_seed=0, device=device,
    )
    assert m.log_select_temp.exp().item() == pytest.approx(1.0, abs=1e-6)
    m.set_select_temp(0.05)
    assert m.log_select_temp.exp().item() == pytest.approx(0.05, abs=1e-6)


@pytest.mark.parametrize("device", DEVICES)
def test_hard_mode_output_finite_for_gumbel(device):
    """gumbel=True + hard=True path runs end-to-end."""
    torch.manual_seed(0)
    m = SoftMultiHeadLUT(
        input_dim=24, n_outputs=6, n_anchor_pairs=4,
        n_heads=2, tables_per_head=3,
        gumbel=True, hard=True, select_temp=0.5,
        random_seed=0, device=device,
    )
    x = torch.randn(4, 24, device=device, requires_grad=True)
    y = m(x)
    assert torch.isfinite(y).all()
    y.sum().backward()
    assert torch.isfinite(m.weights.grad).all()
    assert torch.isfinite(x.grad).all()


# ---------- guards ------------------------------------------------------------

def test_rejects_excessive_nap():
    with pytest.raises(ValueError, match="n_anchor_pairs"):
        SoftMultiHeadLUT(
            input_dim=64, n_outputs=4, n_anchor_pairs=13,  # > 12
            n_heads=1, tables_per_head=1, random_seed=0,
        )


def test_rejects_zero_nap():
    with pytest.raises(ValueError, match="n_anchor_pairs"):
        SoftMultiHeadLUT(
            input_dim=64, n_outputs=4, n_anchor_pairs=0,
            n_heads=1, tables_per_head=1, random_seed=0,
        )
