"""Tests for BucketLIFDetectorsMHL — the single-neuron-per-table, time-bucket-addressed LUT front-end.

CPU-only, deterministic. Mirrors the PureLIFDetectorsMHL suite: forward shape/finiteness, straight-through
invariant, table grad = selected bucket only, address-path grad = full partition (boundaries + membrane),
no-crossing -> t_window -> last bucket, and per-LUT params (tau/T_cross/T_bkt/boundaries) get gradient.
"""
import torch
from spiky.lutorch.bucket_lif_detectors_mhl import BucketLIFDetectorsMHL


def _model(**kw):
    torch.manual_seed(0)
    cfg = dict(input_dim=17, n_heads=1, n_outputs=6, tables_per_head=32)
    cfg.update(kw)
    return BucketLIFDetectorsMHL(**cfg)


def test_forward_shape_and_finite():
    m = _model()
    x = torch.randn(8, 17)
    for mode in ("st", "hard", "soft"):
        y = m(x, mode=mode)
        assert y.shape == (8, 1, 6)
        assert torch.isfinite(y).all(), f"non-finite output in mode {mode}"


def test_partition_of_unity():
    m = _model(tables_per_head=4)
    x = torch.randn(8, 17)
    _, t_soft = m._first_spike(x)
    g = m._bucket_soft(t_soft)
    assert g.shape == (8, 4, m.n_buckets)
    assert torch.allclose(g.sum(-1), torch.ones(8, 4), atol=1e-5), "soft buckets must sum to 1"
    assert (g >= -1e-6).all(), "partition weights must be non-negative"


def test_boundaries_strictly_increasing():
    m = _model(tables_per_head=5)
    b = m.boundaries
    assert b.shape == (5, m.n_buckets - 1)
    assert (b[:, 1:] - b[:, :-1] > 0).all(), "boundaries must be strictly increasing per table"


def test_straight_through_invariant():
    m = _model(n_buckets=8, tables_per_head=4)
    x = torch.randn(32, 17)
    with torch.no_grad():
        y_st, y_hard, y_soft = m(x, mode="st"), m(x, mode="hard"), m(x, mode="soft")
    assert torch.allclose(y_st, y_hard, atol=1e-5), "ST forward must equal the hard/argmax lookup"
    assert not torch.allclose(y_st, y_soft, atol=1e-4), "soft blend should differ from hard forward"


def test_st_table_grad_only_selected_bucket():
    m = _model(n_buckets=8, tables_per_head=3)
    x1 = torch.randn(1, 17); tgt1 = torch.randn(1, m.n_heads, m.n_outputs)
    m.zero_grad(set_to_none=True)
    torch.nn.functional.mse_loss(m(x1, mode="st"), tgt1).backward()
    rows_with_grad = (m.table.grad.abs().sum(dim=-1) > 0)
    assert (rows_with_grad.sum(dim=-1) == 1).all(), "expected exactly 1 bucket/table"
    assert torch.equal(rows_with_grad.float().argmax(dim=-1), m.address(x1)[0]), "bucket != hard address"
    m.zero_grad(set_to_none=True)
    xb = torch.randn(8, 17); tgtb = torch.randn(8, m.n_heads, m.n_outputs)
    torch.nn.functional.mse_loss(m(xb, mode="st"), tgtb).backward()
    per_table_b = (m.table.grad.abs().sum(dim=-1) > 0).sum(dim=-1)
    assert (per_table_b <= 8).all() and int(per_table_b.max()) <= m.n_buckets


def test_st_address_grad_full_partition():
    """Address-path gradient reaches the trainable boundaries and the membrane params (delay, w_raw) — the
    full soft-partition backward, not one isolated bucket."""
    m = _model(n_buckets=8, tables_per_head=3)
    x = torch.randn(16, 17); tgt = torch.randn(16, m.n_heads, m.n_outputs)
    m.zero_grad(set_to_none=True)
    torch.nn.functional.mse_loss(m(x, mode="st"), tgt).backward()
    assert m.beta_raw.grad is not None and m.beta_raw.grad.abs().sum() > 0, "boundaries got no gradient"
    assert m.beta_base.grad is not None and m.beta_base.grad.abs().sum() > 0, "beta_base got no gradient"
    assert m.delay.grad.abs().sum() > 0 and m.w_raw.grad.abs().sum() > 0, "membrane params got no gradient"


def test_no_crossing_returns_t_window_and_last_bucket():
    m = _model(n_buckets=8, tables_per_head=3)
    with torch.no_grad():
        m.w_raw.fill_(-30.0)                             # w = w_max*sigmoid(-30) ~ 0 -> membrane never reaches theta_mem=1
    x = torch.randn(8, 17)
    t_hard, _ = m._first_spike(x)
    assert torch.allclose(t_hard, torch.full_like(t_hard, m.t_window)), "no-crossing must give t_hard=t_window"
    assert torch.equal(m.address(x), torch.full_like(m.address(x), m.n_buckets - 1)), "t_window must fold into last bucket"
    assert torch.isfinite(m(x, mode="st")).all() and torch.isfinite(m(x, mode="soft")).all()


def test_per_lut_params_get_gradient():
    m = _model(n_buckets=8, tables_per_head=3)
    x = torch.randn(16, 17); tgt = torch.randn(16, m.n_heads, m.n_outputs)
    m.zero_grad(set_to_none=True)
    torch.nn.functional.mse_loss(m(x, mode="st"), tgt).backward()
    for name in ("tau_raw", "log_T_cross", "log_T_bkt", "beta_raw", "beta_base", "delay", "w_raw", "table"):
        g = getattr(m, name).grad
        assert g is not None and torch.isfinite(g).all() and g.abs().sum() > 0, f"{name} got no gradient"


def test_bounded_excitatory_weights():
    """Effective synaptic weight w = w_max*sigmoid(w_raw) is strictly positive and bounded in (0, w_max)."""
    m = _model(n_buckets=8, tables_per_head=4)
    w = m.w
    assert w.shape == m.w_raw.shape == (m.n_tables, m.input_dim)
    assert (w > 0).all(), "excitatory weights must be strictly positive"
    assert (w < m.w_max).all(), "weights must stay below w_max for finite w_raw"
    assert m.w_max == 2.0
    # sanity: extreme w_raw saturates the sigmoid to the two bounds
    with torch.no_grad():
        m.w_raw.fill_(30.0)
        assert torch.allclose(m.w, torch.full_like(m.w, m.w_max), atol=1e-3)
        m.w_raw.fill_(-30.0)
        assert (m.w < 1e-6).all()


def test_positivity_and_shape_generality():
    m = BucketLIFDetectorsMHL(input_dim=5, n_heads=2, n_outputs=3, tables_per_head=3, n_buckets=8)
    assert m.n_tables == 6 and m.n_buckets == 8 and m.n_rows == 8
    assert (m.tau > 0).all() and (m.T_cross > 0).all() and (m.T_bkt > 0).all()
    assert m.table.shape == (6, 8, 3)
    y = m(torch.randn(7, 5), mode="st")
    assert y.shape == (7, 2, 3) and torch.isfinite(y).all()
