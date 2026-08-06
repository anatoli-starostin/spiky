"""Tests for LIFDetectorsMHL — the LIF-detector drop-in for the HyperplaneMultiHeadLUT front-end.

CPU-only, deterministic (seeded), fast enough for CI. Covers forward shape/finiteness, gradient flow to
every trainable parameter, the straight-through hard-addressing invariant, the off-diagonal pair mask +
positivity of softplus/exp params, shape-generality for non-default dims, address packing, and a tiny
end-to-end distillation smoke test against a frozen HyperplaneMultiHeadLUT teacher.
"""
import pytest
import torch

from lif_detectors_mhl import LIFDetectorsMHL
from spiky.lutorch.hyperplane_multi_head_lut import HyperplaneMultiHeadLUT


def _model(**kw):
    torch.manual_seed(0)
    cfg = dict(input_dim=17, n_heads=1, n_outputs=6, n_anchor_pairs=6, tables_per_head=32)
    cfg.update(kw)
    return LIFDetectorsMHL(**cfg)


def test_forward_shape_and_finite():
    m = _model()
    x = torch.randn(8, 17)
    for mode in ("st", "hard", "soft"):
        y = m(x, mode=mode)
        assert y.shape == (8, 1, 6)
        assert torch.isfinite(y).all(), f"non-finite output in mode {mode}"


def test_gradients_flow_to_every_param():
    m = _model(n_anchor_pairs=4, tables_per_head=4)   # smaller for speed
    x = torch.randn(16, 17)
    target = torch.randn(16, m.n_heads, m.n_outputs)
    loss = torch.nn.functional.mse_loss(m(x, mode="st"), target)
    loss.backward()
    for name, p in m.named_parameters():
        assert p.grad is not None, f"{name}.grad is None"
        assert torch.isfinite(p.grad).all(), f"{name}.grad has non-finite entries"
        assert p.grad.abs().sum() > 0, f"{name}.grad is all zero"


def test_straight_through_invariant():
    """ST forward value == pure hard/argmax lookup (up to float rounding of y_soft+(y_hard-y_soft));
    the pure-soft blend differs materially."""
    m = _model(n_anchor_pairs=5, tables_per_head=4)
    x = torch.randn(32, 17)
    with torch.no_grad():
        y_st = m(x, mode="st")
        y_hard = m(x, mode="hard")
        y_soft = m(x, mode="soft")
    # output-level STE reconstructs y_hard exactly in value up to fp rounding (~1e-7), not the soft blend
    assert torch.allclose(y_st, y_hard, atol=1e-5), "ST forward must equal the hard/argmax lookup"
    assert not torch.allclose(y_st, y_soft, atol=1e-4), "soft blend should differ from hard forward"


def test_st_table_grad_only_selected_row():
    """Decoupled ST: the TABLE gradient follows the HARD forward — only the argmax-selected row per table
    updates. Single sample => exactly one row per table; batch => at most #distinct addresses, never all rows."""
    m = _model(n_anchor_pairs=4, tables_per_head=3)   # 16 rows/table
    # single sample -> exactly one row per table has nonzero table grad, and it's the argmax address
    x1 = torch.randn(1, 17); tgt1 = torch.randn(1, m.n_heads, m.n_outputs)
    m.zero_grad(set_to_none=True)
    torch.nn.functional.mse_loss(m(x1, mode="st"), tgt1).backward()
    rows_with_grad = (m.table.grad.abs().sum(dim=-1) > 0)          # (n_tables, n_rows)
    per_table = rows_with_grad.sum(dim=-1)                         # (n_tables,)
    assert (per_table == 1).all(), f"expected exactly 1 row/table, got {per_table.tolist()}"
    sel = rows_with_grad.float().argmax(dim=-1)                    # (n_tables,)
    assert torch.equal(sel, m.address(x1)[0]), "table-grad row must equal the argmax address"
    # batch -> touched rows per table bounded by distinct addresses, never all 2**nap rows
    m.zero_grad(set_to_none=True)
    xb = torch.randn(8, 17); tgtb = torch.randn(8, m.n_heads, m.n_outputs)
    torch.nn.functional.mse_loss(m(xb, mode="st"), tgtb).backward()
    per_table_b = (m.table.grad.abs().sum(dim=-1) > 0).sum(dim=-1)
    assert (per_table_b <= 8).all() and int(per_table_b.max()) < m.n_rows, \
        f"table grad must touch only selected rows, got max {int(per_table_b.max())} of {m.n_rows}"


def test_st_detector_grad_full_k_softmax():
    """Decoupled ST: the DETECTOR/address gradient follows the full-K softmax, so theta and the detector
    membrane weights receive nonzero gradient across (most) bits — not just the selected cell."""
    m = _model(n_anchor_pairs=4, tables_per_head=3)
    x = torch.randn(16, 17); tgt = torch.randn(16, m.n_heads, m.n_outputs)
    m.zero_grad(set_to_none=True)
    torch.nn.functional.mse_loss(m(x, mode="st"), tgt).backward()
    assert m.theta.grad is not None and m.theta.grad.abs().sum() > 0, "theta got no address gradient"
    frac = (m.theta.grad.abs() > 0).float().mean().item()
    assert frac > 0.5, f"address grad too sparse ({frac:.2f}); expected full-K coverage"
    assert m.w.grad.abs().sum() > 0 and m.d.grad.abs().sum() > 0, "detector membrane params got no gradient"


def test_pair_mask_and_positivity():
    m = _model(n_anchor_pairs=4, tables_per_head=4)
    # effective ordered-pair weights have a zero diagonal (self-pairs masked)
    eff = m.P * m.offdiag
    diag = torch.diagonal(eff, dim1=-2, dim2=-1)
    assert diag.abs().max() == 0.0
    assert torch.diagonal(m.offdiag).abs().max() == 0.0
    # softplus/exp keep taus and temp strictly positive (even after a step that pushes raws negative)
    with torch.no_grad():
        m.tau_s_raw.fill_(-50.0); m.tau_p_raw.fill_(-50.0); m.log_temp_bit.fill_(-50.0)
    assert (m.tau_s > 0).all() and (m.tau_p > 0).all() and (m.temp_bit > 0).all()


def test_address_packing_msb_first():
    m = _model(n_anchor_pairs=6, tables_per_head=2, n_heads=1)
    assert m.pow2.tolist() == [32, 16, 8, 4, 2, 1]   # MSB-first, matches the teacher
    x = torch.randn(10, 17)
    addr = m.address(x)
    assert addr.shape == (10, m.n_tables)
    assert addr.min() >= 0 and addr.max() < m.n_rows


def test_shape_generality_non_default_dims():
    m = LIFDetectorsMHL(input_dim=5, n_heads=2, n_outputs=3, n_anchor_pairs=4, tables_per_head=3)
    assert m.n_tables == 6 and m.n_rows == 16 and m.n_detectors == 6 * 4
    assert m.pow2.tolist() == [8, 4, 2, 1]
    x = torch.randn(7, 5)
    y = m(x, mode="st")
    assert y.shape == (7, 2, 3) and torch.isfinite(y).all()


def test_distillation_smoke_learns():
    """A tiny frozen HyperplaneMultiHeadLUT teacher: end-to-end action MSE must decrease."""
    torch.manual_seed(0)
    teacher = HyperplaneMultiHeadLUT(input_dim=6, n_heads=1, n_outputs=3, n_anchor_pairs=3,
                                     tables_per_head=2, hyperplane_init="random", random_seed=0,
                                     use_bf16=False)
    for p in teacher.parameters():
        p.requires_grad_(False)

    def oracle(x):
        with torch.no_grad():
            return teacher(x)

    student = LIFDetectorsMHL(input_dim=6, n_heads=1, n_outputs=3, n_anchor_pairs=3, tables_per_head=2)
    opt = torch.optim.Adam(student.parameters(), lr=5e-3)
    gen = torch.Generator().manual_seed(1)
    losses = []
    for step in range(120):
        eps = 2.0 + (0.3 - 2.0) * step / 119
        x = torch.randn(128, 6, generator=gen)
        loss = torch.nn.functional.mse_loss(student(x, eps=eps, mode="st"), oracle(x))
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())
    assert all(t == t for t in losses), "NaN in training loss"
    first = sum(losses[:5]) / 5.0
    last = sum(losses[-5:]) / 5.0
    assert last < 0.8 * first, f"distillation did not learn: first {first:.3f} -> last {last:.3f}"
