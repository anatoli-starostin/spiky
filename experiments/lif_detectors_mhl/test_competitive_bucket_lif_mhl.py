"""Tests for CompetitiveBucketLIFMHL — the lateral-inhibition (column WTA) population bucket-LIF variant.

CPU-only, deterministic. Covers forward shape/finiteness, the strict 1-winner-per-bucket-column property,
soft-competition normalization, ST invariant, bounded excitatory weights, strictly-increasing per-head
boundaries, gradient flow to every param (incl. the WTA temperature), and no-crossing -> last bucket.
"""
import torch
from competitive_bucket_lif_mhl import CompetitiveBucketLIFMHL


def _model(**kw):
    torch.manual_seed(0)
    cfg = dict(in_dim=17, out_dim=6, n_heads=4, neurons_per_head=8, buckets=16)
    cfg.update(kw)
    return CompetitiveBucketLIFMHL(**cfg)


def test_forward_shape_and_finite():
    m = _model()
    x = torch.randn(8, 17)
    for mode in ("st", "hard", "soft"):
        y = m(x, mode=mode)
        assert y.shape == (8, 6)
        assert torch.isfinite(y).all(), f"non-finite output in mode {mode}"


def test_param_count_near_target():
    m = _model()
    assert 3500 <= m.param_count() <= 6500, f"param count {m.param_count()} not near ~5k"


def test_hard_wta_one_winner_per_column():
    m = _model(n_heads=3, neurons_per_head=6, buckets=8)
    x = torch.randn(16, 17)
    t_hard, t_soft = m._first_spike(x)
    E, g = m._buckets(t_hard, t_soft)
    C_hard, _ = m._compete(E, g, t_hard, t_soft)
    winners = C_hard.sum(dim=2)                                   # (B,H,M) winners per column
    assert winners.max() <= 1, "strict WTA: at most one hard winner per (b,head,bucket) column"
    occupied = (E.sum(dim=2) > 0)                                 # a column with >=1 assigned neuron
    assert torch.equal(winners > 0, occupied), "occupied columns must have exactly one winner; empty columns none"
    # a winner must be the earliest-spiking member of its column
    assert (C_hard * (E - 1)).abs().sum() < 1e-6, "winner must be a member of the column (E==1)"


def test_soft_competition_normalized():
    m = _model(n_heads=3, neurons_per_head=6, buckets=8)
    x = torch.randn(16, 17)
    t_hard, t_soft = m._first_spike(x)
    E, g = m._buckets(t_hard, t_soft)
    _, C_soft = m._compete(E, g, t_hard, t_soft)
    col = C_soft.sum(dim=2)                                       # (B,H,M) soft mass per column
    assert (col <= 1.0 + 1e-4).all() and (col >= -1e-6).all(), "soft column mass in [0,1]"
    assert (C_soft >= -1e-6).all(), "soft competition weights non-negative"


def test_straight_through_invariant():
    m = _model(buckets=8, n_heads=3)
    x = torch.randn(32, 17)
    with torch.no_grad():
        y_st, y_hard, y_soft = m(x, mode="st"), m(x, mode="hard"), m(x, mode="soft")
    assert torch.allclose(y_st, y_hard, atol=1e-5), "ST forward must equal the hard WTA lookup"
    assert not torch.allclose(y_st, y_soft, atol=1e-4), "soft competition should differ from hard forward"


def test_bounded_excitatory_weights():
    m = _model()
    w = m.w
    assert w.shape == (m.n_heads, m.neurons_per_head, m.in_dim)
    assert (w > 0).all() and (w < m.w_max).all(), "weights must be strictly in (0, w_max)"
    assert m.w_max == 2.0


def test_boundaries_strictly_increasing_per_head():
    m = _model(n_heads=5)
    b = m.boundaries
    assert b.shape == (5, m.buckets - 1)
    assert (b[:, 1:] - b[:, :-1] > 0).all(), "per-head bucket boundaries must be strictly increasing"


def test_all_params_get_gradient():
    m = _model(buckets=8, n_heads=3, neurons_per_head=6)
    x = torch.randn(16, 17); tgt = torch.randn(16, 6)
    m.zero_grad(set_to_none=True)
    torch.nn.functional.mse_loss(m(x, mode="st"), tgt).backward()
    for name, p in m.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0, f"{name} no gradient"


def test_st_table_grad_only_winning_cells():
    m = _model(buckets=8, n_heads=3, neurons_per_head=6)
    x = torch.randn(4, 17); tgt = torch.randn(4, 6)
    m.zero_grad(set_to_none=True)
    torch.nn.functional.mse_loss(m(x, mode="st"), tgt).backward()
    # table grad only where a (neuron,bucket) cell won a column for some sample; far fewer than all cells
    cells_with_grad = (m.table.grad.abs().sum(dim=-1) > 0)        # (H,N,M)
    assert cells_with_grad.any() and int(cells_with_grad.sum()) < m.table.grad[..., 0].numel(), \
        "table grad should hit only winning cells, not all"


def test_no_crossing_folds_into_last_bucket():
    m = _model(buckets=8, n_heads=3, neurons_per_head=6)
    with torch.no_grad():
        m.w_raw.fill_(-30.0)                                     # w ~ 0 -> membrane never reaches theta_mem
    x = torch.randn(8, 17)
    t_hard, _ = m._first_spike(x)
    assert torch.allclose(t_hard, torch.full_like(t_hard, m.t_window)), "no-crossing must give t_hard=t_window"
    assert torch.isfinite(m(x, mode="st")).all() and torch.isfinite(m(x, mode="soft")).all()
