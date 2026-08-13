"""Tests for LIFMultiHeadLUT — the unified LIF multi-head LUT (the library's single LIF-LUT class).

Folds in the behavioral coverage of the retired BucketLIFDetectorsMHL (n_det=1) and ProductBucketLIFMHL
(n_det>1) suites, tested against LIFMultiHeadLUT directly.
"""
import pytest
import torch
import torch.nn.functional as F
from spiky.lutorch.lif_multi_head_lut import LIFMultiHeadLUT, MAX_CELLS


def _m(**kw):
    torch.manual_seed(0)
    cfg = dict(input_dim=17, n_heads=2, n_outputs=6, tables_per_head=4, n_det=1, n_buckets=16)
    cfg.update(kw)
    return LIFMultiHeadLUT(**cfg)


# ---- shared ----
def test_forward_shape_and_finite():
    for nd, M in ((1, 16), (2, 8), (3, 4)):
        m = _m(n_det=nd, n_buckets=M)
        x = torch.randn(8, 17)
        m.train(); y_train = m(x)                                 # ST path
        m.eval();  y_eval = m(x)                                  # efficient hard path
        for tag, out in (("train", y_train), ("eval", y_eval)):
            assert out.shape == (8, 2, 6) and torch.isfinite(out).all(), f"nd={nd} {tag}"


def test_train_forward_equals_eval_forward():
    """ST (train-mode) forward value == the hard winner, so it must equal the eval-mode hard forward."""
    for nd, M in ((1, 8), (2, 8)):
        m = _m(n_det=nd, n_buckets=M)
        x = torch.randn(24, 17)
        with torch.no_grad():
            m.train(); a = m(x)
            m.eval();  b = m(x)
        assert torch.allclose(a, b, atol=1e-5), f"nd={nd}: train-mode ST forward must equal eval-mode hard forward"


def test_tph_summation():
    """The tables_per_head tables are summed within each head -> (B, n_heads, n_outputs)."""
    m = _m(n_heads=3, tables_per_head=5, n_det=1)
    x = torch.randn(4, 17)
    m.eval(); y = m(x)                                            # (4,3,6) eval-mode hard forward
    # equals per-table hard reads reshaped (n_heads, tph) and summed over tph
    t_hard, t_soft = m._first_spike(x)
    rows = m._hard_read(*(m._bucket(t_hard, t_soft)[:1]))         # (B, n_tables, O)
    assert torch.allclose(y, rows.view(4, 3, 5, 6).sum(2), atol=1e-6)


# ---- universal timing (the only path) ----
def test_universal_timing_init_reproduces_fixed_code():
    """At INIT the universal timing arrival clamp(baseline_C - gain*x + delay, 0, t_window) reproduces
    the ORIGINAL fixed slope-3 latency code clamp(t_window*(0.5 - 3x/32), 0, t_window) EXACTLY (delay=0)."""
    m = _m()
    TW = m.t_window                                                     # 32 -> map = 16 - 3x
    # init: per-channel gain = t_window*3/32 (= old alpha=3), scalar baseline C = 0.5*t_window, delay 0
    assert m.gain.shape == (m.input_dim,)
    assert torch.allclose(m.gain, torch.full_like(m.gain, TW * 3.0 / 32.0), atol=1e-6)
    assert float(m.baseline_C.detach()) == pytest.approx(0.5 * TW)
    assert torch.equal(m.delay, torch.zeros_like(m.delay))
    # the retired fixed-path surface is gone
    assert not hasattr(m, "latency") and not hasattr(m, "universal_timing")
    # arrival at init == the old fixed latency formula, per input channel (delay=0 => same for every
    # table/detector); compare the sorted arrival _membrane produces against the reference.
    x = torch.randn(8, 17)
    old_latency = torch.clamp(TW * (0.5 - 3.0 * x / 32.0), 0.0, TW)     # (B, N)
    a_srt, _ = m._membrane(x)                                           # (B, T, D, N) sorted along N
    B, T, D, N = a_srt.shape
    expected = torch.sort(old_latency, dim=-1).values                  # (B, N)
    assert torch.allclose(a_srt, expected.view(B, 1, 1, N).expand(B, T, D, N), atol=1e-5)
    # anchors of the reproduced code: x=0 -> half window; monotonically non-increasing in x
    assert torch.allclose(torch.clamp(TW * (0.5 - 3.0 * torch.zeros(3) / 32.0), 0.0, TW),
                          torch.full((3,), TW / 2), atol=1e-6)


# ---- free signed delays; positivity on the FINAL arrival ----
def test_delay_default_zero_init():
    m = _m()                                              # delay_init_std unset -> default 0.0
    assert torch.equal(m.delay, torch.zeros_like(m.delay))


def test_delay_free_signed_and_arrival_positivity():
    """Delays are FREE and SIGNED (no >=0 clamp): std>0 gives a zero-centered spread that goes negative.
    Positivity/window-bounding is enforced on the FINAL arrival, not on the delay."""
    m = _m(delay_init_std=4.0)
    d = m.delay.detach()
    assert d.std() > 0, "delay_init_std>0 must give nonzero spread"
    assert (d < 0).any(), "signed (zero-centered) init must produce some negative delays"
    # delays feed the arrival UNCLAMPED; positivity still holds on the final arrival for any delay.
    with torch.no_grad():
        m.delay[0, 0, 0] = -100.0 * m.t_window
        m.delay[0, 0, 1] = 100.0 * m.t_window
    a_srt, _ = m._membrane(torch.randn(8, 17))
    assert (a_srt >= 0).all() and (a_srt <= m.t_window).all(), "final arrival must stay in [0, t_window]"
    assert torch.isfinite(m(torch.randn(8, 17))).all()


# ---- n_det=1 (plain bucket) coverage ----
def test_bucket_n_det1_reads_and_shapes():
    m = _m(n_det=1, n_heads=1, tables_per_head=4, n_buckets=8)
    x = torch.randn(8, 17)
    assert m.cells == 8 and m.table.shape == (4, 8, 6)
    assert m.w_raw.shape == (4, 1, 17) and m.boundaries.shape == (4, 1, 7)   # detector axis always present (D=1)
    # hard read is exactly the single-bucket row of the addressed cell
    t_hard, t_soft = m._first_spike(x)
    b_hard, _ = m._bucket(t_hard, t_soft)
    y_hard = m._hard_read(b_hard)
    manual = torch.stack([m.table[t][b_hard[:, t, 0]] for t in range(m.n_tables)], dim=1)
    assert torch.allclose(y_hard, manual, atol=1e-6)
    assert torch.equal(m.address(x), b_hard[:, :, 0])            # joint index == the single bucket digit


def test_bounded_excitatory_weights():
    m = _m(n_det=1, tables_per_head=4)
    w = m.w
    assert w.shape == m.w_raw.shape == (m.n_tables, 1, m.input_dim)
    assert (w > 0).all() and (w < m.w_max).all() and m.w_max == 2.0
    with torch.no_grad():
        m.w_raw.fill_(30.0); assert torch.allclose(m.w, torch.full_like(m.w, m.w_max), atol=1e-3)
        m.w_raw.fill_(-30.0); assert (m.w < 1e-6).all()


def test_boundaries_strictly_increasing():
    m = _m(n_heads=1, tables_per_head=5, n_det=1)
    b = m.boundaries
    assert b.shape == (5, 1, m.n_buckets - 1) and (b[..., 1:] - b[..., :-1] > 0).all()


def test_no_crossing_folds_into_last_bucket():
    m = _m(n_det=1, n_buckets=8, tables_per_head=3)
    with torch.no_grad():
        m.w_raw.fill_(-30.0)
    x = torch.randn(8, 17)
    t_hard, _ = m._first_spike(x)
    assert torch.allclose(t_hard, torch.full_like(t_hard, m.t_window))
    assert torch.equal(m.address(x), torch.full_like(m.address(x), m.n_buckets - 1))
    m.train(); assert torch.isfinite(m(x)).all()
    m.eval();  assert torch.isfinite(m(x)).all()


# ---- n_det>1 (product / mixed-radix) coverage ----
def test_mixed_radix_and_tensor_product_soft():
    m = _m(n_heads=3, tables_per_head=1, n_det=2, n_buckets=8)
    x = torch.randn(16, 17)
    t_hard, t_soft = m._first_spike(x)
    b_hard, p = m._bucket(t_hard, t_soft)
    grid = m.table.reshape(m.n_tables, m.n_buckets, m.n_buckets, m.n_outputs)
    manual = torch.stack([grid[t][b_hard[:, t, 0], b_hard[:, t, 1]] for t in range(m.n_tables)], dim=1)
    assert torch.allclose(m._hard_read(b_hard), manual, atol=1e-5), "mixed-radix hard index"
    P = p[:, :, 0, :].unsqueeze(-1) * p[:, :, 1, :].unsqueeze(-2)
    dense = torch.einsum('btij,tijo->bto', P, grid)
    assert torch.allclose(m._soft_read(p), dense, atol=1e-4), "soft addr readout == dense tensor product (value)"


def test_st_table_grad_only_selected_cell():
    m = _m(n_det=2, n_buckets=8, tables_per_head=1, n_heads=3)
    x = torch.randn(4, 17); tgt = torch.randn(4, 3, 6)
    m.zero_grad(set_to_none=True)
    F.mse_loss(m(x), tgt).backward()
    cells_with_grad = (m.table.grad.abs().sum(dim=-1) > 0)       # (T, cells)
    assert cells_with_grad.any() and int(cells_with_grad.sum()) < m.table[..., 0].numel()


def test_soft_joint_sums_to_one():
    m = _m(n_det=2, n_buckets=8)
    x = torch.randn(8, 17)
    t_hard, t_soft = m._first_spike(x)
    _, p = m._bucket(t_hard, t_soft)
    assert torch.allclose(p.sum(-1), torch.ones_like(p.sum(-1)), atol=1e-4)   # each detector's dist sums to 1


# ---- freeze_temperature ----
def test_freeze_temperature():
    # default: trainable temps at T=1.0
    md = _m()
    assert md.log_T_cross.requires_grad and md.log_T_bkt.requires_grad
    # frozen: value 0.0 (T=1.0) and non-trainable
    mf = _m(freeze_temperature=True)
    assert not mf.log_T_cross.requires_grad and not mf.log_T_bkt.requires_grad
    assert torch.equal(mf.log_T_cross, torch.zeros_like(mf.log_T_cross))
    assert torch.equal(mf.log_T_bkt, torch.zeros_like(mf.log_T_bkt))
    assert torch.allclose(mf.T_cross, torch.ones_like(mf.T_cross)) and torch.allclose(mf.T_bkt, torch.ones_like(mf.T_bkt))
    # after a couple of ST steps: frozen temps stay put; default temps move
    for m in (md, mf):
        opt = torch.optim.Adam([p for p in m.parameters() if p.requires_grad], lr=1e-1)
        for _ in range(2):
            opt.zero_grad(); F.mse_loss(m(torch.randn(16, 17)), torch.randn(16, 2, 6)).backward(); opt.step()
    assert torch.equal(mf.log_T_cross, torch.zeros_like(mf.log_T_cross)), "frozen T_cross must not move"
    assert torch.equal(mf.log_T_bkt, torch.zeros_like(mf.log_T_bkt)), "frozen T_bkt must not move"
    assert not torch.equal(md.log_T_bkt, torch.zeros_like(md.log_T_bkt)), "trainable T_bkt should move"


# ---- params, temperatures, cap, table_init ----
def test_all_params_incl_temperatures_get_gradient():
    m = _m(n_det=2, n_buckets=8, tables_per_head=3)
    x = torch.randn(16, 17); tgt = torch.randn(16, 2, 6)
    m.zero_grad(set_to_none=True)
    F.mse_loss(m(x), tgt).backward()
    for name, pp in m.named_parameters():
        assert pp.grad is not None and torch.isfinite(pp.grad).all() and pp.grad.abs().sum() > 0, f"{name} no grad"
    assert m.log_T_cross.shape == (m.n_tables,) and m.log_T_bkt.shape == (m.n_tables,)   # trainable per-table temps
    # the universal-timing params are registered and receive gradient via the ST address path
    names = dict(m.named_parameters())
    assert "gain" in names and "baseline_C" in names
    assert names["gain"].grad.abs().sum() > 0 and names["baseline_C"].grad.abs().sum() > 0


def test_table_init_and_cell_cap():
    tab = torch.randn(8, 16, 6)
    m = LIFMultiHeadLUT(input_dim=17, n_heads=2, n_outputs=6, tables_per_head=4, n_det=1, n_buckets=16, table_init=tab)
    assert torch.equal(m.table.detach(), tab)
    with pytest.raises(ValueError):
        LIFMultiHeadLUT(input_dim=17, n_heads=1, n_outputs=6, n_det=5, n_buckets=16)   # 16**5 > 65536
    assert MAX_CELLS == 65536
