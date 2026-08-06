"""Tests for PureLIFDetectorsMHL — the time-to-first-spike (TTFS) LIF-detector LUT front-end.

CPU-only, deterministic. Mirrors the LIFDetectorsMHL suite: forward shape/finiteness, straight-through
invariant, table grad = selected row only, detector-path grad = full-K softmax, plus TTFS-specifics
(no-crossing -> t_window, no NaNs, per-LUT tau/T_cross/temp_bit get gradient).
"""
import torch
from pure_lif_detectors_mhl import PureLIFDetectorsMHL


def _model(**kw):
    torch.manual_seed(0)
    cfg = dict(input_dim=17, n_heads=1, n_outputs=6, n_anchor_pairs=6, tables_per_head=32)
    cfg.update(kw)
    return PureLIFDetectorsMHL(**cfg)


def test_forward_shape_and_finite():
    m = _model()
    x = torch.randn(8, 17)
    for mode in ("st", "hard", "soft"):
        y = m(x, mode=mode)
        assert y.shape == (8, 1, 6)
        assert torch.isfinite(y).all(), f"non-finite output in mode {mode}"


def test_straight_through_invariant():
    m = _model(n_anchor_pairs=5, tables_per_head=4)
    x = torch.randn(32, 17)
    with torch.no_grad():
        y_st, y_hard, y_soft = m(x, mode="st"), m(x, mode="hard"), m(x, mode="soft")
    assert torch.allclose(y_st, y_hard, atol=1e-5), "ST forward must equal the hard/argmax lookup"
    assert not torch.allclose(y_st, y_soft, atol=1e-4), "soft blend should differ from hard forward"


def test_st_table_grad_only_selected_row():
    m = _model(n_anchor_pairs=4, tables_per_head=3)   # 16 rows/table
    x1 = torch.randn(1, 17); tgt1 = torch.randn(1, m.n_heads, m.n_outputs)
    m.zero_grad(set_to_none=True)
    torch.nn.functional.mse_loss(m(x1, mode="st"), tgt1).backward()
    rows_with_grad = (m.table.grad.abs().sum(dim=-1) > 0)
    assert (rows_with_grad.sum(dim=-1) == 1).all(), "expected exactly 1 row/table"
    assert torch.equal(rows_with_grad.float().argmax(dim=-1), m.address(x1)[0]), "row != argmax address"
    m.zero_grad(set_to_none=True)
    xb = torch.randn(8, 17); tgtb = torch.randn(8, m.n_heads, m.n_outputs)
    torch.nn.functional.mse_loss(m(xb, mode="st"), tgtb).backward()
    per_table_b = (m.table.grad.abs().sum(dim=-1) > 0).sum(dim=-1)
    assert (per_table_b <= 8).all() and int(per_table_b.max()) < m.n_rows


def test_st_detector_grad_full_k():
    """Detector-path (address) gradient reaches the deadline L and the membrane params (delay, w) across
    most detectors — the full-K softmax address backward, not one isolated cell."""
    m = _model(n_anchor_pairs=4, tables_per_head=3)
    x = torch.randn(16, 17); tgt = torch.randn(16, m.n_heads, m.n_outputs)
    m.zero_grad(set_to_none=True)
    torch.nn.functional.mse_loss(m(x, mode="st"), tgt).backward()
    assert m.L.grad is not None and (m.L.grad.abs() > 0).float().mean().item() > 0.5, "L address-grad too sparse"
    assert m.delay.grad.abs().sum() > 0 and m.w.grad.abs().sum() > 0, "membrane params got no gradient"


def test_ttfs_no_crossing_returns_t_window_and_finite():
    m = _model(n_anchor_pairs=4, tables_per_head=3)
    with torch.no_grad():
        m.w.fill_(-5.0)                                  # strongly negative -> membrane never reaches theta_mem=1
    x = torch.randn(8, 17)
    _, _, t_hard = m._spike_bits(x)
    assert torch.allclose(t_hard, torch.full_like(t_hard, m.t_window)), "no-crossing must give t_hard=t_window"
    assert torch.isfinite(m(x, mode="st")).all() and torch.isfinite(m(x, mode="soft")).all()


def test_per_lut_params_get_gradient():
    m = _model(n_anchor_pairs=4, tables_per_head=3)
    x = torch.randn(16, 17); tgt = torch.randn(16, m.n_heads, m.n_outputs)
    m.zero_grad(set_to_none=True)
    torch.nn.functional.mse_loss(m(x, mode="st"), tgt).backward()
    for name in ("tau_raw", "log_T_cross", "log_temp_bit", "delay", "w", "L", "table"):
        g = getattr(m, name).grad
        assert g is not None and torch.isfinite(g).all() and g.abs().sum() > 0, f"{name} got no gradient"


def test_positivity_and_shape_generality():
    m = PureLIFDetectorsMHL(input_dim=5, n_heads=2, n_outputs=3, n_anchor_pairs=4, tables_per_head=3)
    assert m.n_tables == 6 and m.n_rows == 16 and m.n_detectors == 6 * 4
    assert m.pow2.tolist() == [8, 4, 2, 1]
    assert (m.tau > 0).all() and (m.T_cross > 0).all() and (m.temp_bit > 0).all()
    y = m(torch.randn(7, 5), mode="st")
    assert y.shape == (7, 2, 3) and torch.isfinite(y).all()
