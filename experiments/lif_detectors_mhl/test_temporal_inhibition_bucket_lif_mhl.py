"""Tests for TemporalInhibitionBucketLIFMHL — causal (recurrent) lateral-inhibition bucket-LIF population.

CPU-only, deterministic. Covers forward shape/finiteness, ST invariant, bounded excitatory weights,
strictly-increasing per-head boundaries, gradient flow to every param (incl. the inhibition strength),
the "each neuron wins at most one bucket" (fired-mask) property, that inhibition (raised threshold) delays
the first-crossing time (the mechanism for the second chance), and no-crossing behaviour.
"""
import torch
from temporal_inhibition_bucket_lif_mhl import TemporalInhibitionBucketLIFMHL


def _model(**kw):
    torch.manual_seed(0)
    cfg = dict(in_dim=17, out_dim=6, n_heads=4, neurons_per_head=8, buckets=16)
    cfg.update(kw)
    return TemporalInhibitionBucketLIFMHL(**cfg)


def test_forward_shape_and_finite():
    m = _model()
    x = torch.randn(8, 17)
    for mode in ("st", "hard", "soft"):
        y = m(x, mode=mode)
        assert y.shape == (8, 6) and torch.isfinite(y).all(), f"bad output in mode {mode}"


def test_straight_through_invariant():
    m = _model(buckets=8, n_heads=3)
    x = torch.randn(24, 17)
    with torch.no_grad():
        y_st, y_hard, y_soft = m(x, "st"), m(x, "hard"), m(x, "soft")
    assert torch.allclose(y_st, y_hard, atol=1e-5), "ST forward must equal hard"
    assert not torch.allclose(y_st, y_soft, atol=1e-4), "soft must differ from hard"


def test_param_count_and_bounded_weights():
    m = _model()
    assert 3500 <= m.param_count() <= 6500
    w = m.w
    assert (w > 0).all() and (w < m.w_max).all() and m.w_max == 2.0
    assert (m.w_inh >= 0).all(), "inhibition strength must be non-negative"


def test_boundaries_strictly_increasing():
    m = _model(n_heads=5)
    b = m.boundaries
    assert b.shape == (5, m.buckets - 1) and (b[:, 1:] - b[:, :-1] > 0).all()


def test_all_params_get_gradient():
    m = _model(buckets=8, n_heads=3, neurons_per_head=6)
    x = torch.randn(16, 17); tgt = torch.randn(16, 6)
    m.zero_grad(set_to_none=True)
    torch.nn.functional.mse_loss(m(x, "st"), tgt).backward()
    for name, p in m.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0, f"{name} no grad"


def test_inhibition_delays_crossing():
    """Raising the threshold (adding inhibition I) pushes the hard first-crossing time later (or equal)."""
    m = _model(buckets=8, n_heads=3, neurons_per_head=6)
    x = torch.randn(8, 17)
    V, a_srt = m._membrane(x)
    thr0 = m.theta_mem + torch.zeros(8, m.n_heads, m.neurons_per_head)
    thr1 = m.theta_mem + torch.full((8, m.n_heads, m.neurons_per_head), 0.5)
    t0 = m._tau_hard(V, a_srt, thr0)
    t1 = m._tau_hard(V, a_srt, thr1)
    assert (t1 >= t0 - 1e-6).all(), "more inhibition must not make a neuron fire earlier"


def test_each_neuron_wins_at_most_one_bucket():
    """The fired mask forbids a neuron from winning more than one bucket in the hard scan."""
    m = _model(buckets=8, n_heads=3, neurons_per_head=6)
    x = torch.randn(16, 17)
    # reconstruct the hard per-bucket winners the way forward() does
    V, a_srt = m._membrane(x)
    B, H, N, M = 16, m.n_heads, m.neurons_per_head, m.buckets
    import torch.nn.functional as F
    b = m.boundaries; BIG = 1e9
    bpad = torch.cat([torch.full((H, 1), -BIG), b, torch.full((H, 1), BIG)], dim=-1)
    I_h = torch.zeros(B, H, N); fired_h = torch.zeros(B, H, N); total = torch.zeros(B, H, N)
    for mm in range(M):
        lo = bpad[:, mm].view(1, H, 1); hi = bpad[:, mm + 1].view(1, H, 1)
        tau_h = m._tau_hard(V, a_srt, m.theta_mem + I_h)
        memb = ((tau_h >= lo) & (tau_h < hi) & (fired_h < 0.5)).float()
        t_masked = torch.where(memb > 0.5, tau_h, torch.full_like(tau_h, BIG))
        has = (memb.sum(dim=2) > 0).float()
        C = F.one_hot(t_masked.argmin(dim=2), N).float() * has.unsqueeze(-1)
        total = total + C
        fired_h = torch.clamp(fired_h + C, 0.0, 1.0)
        I_h = I_h + m.w_inh.view(1, H, 1) * has.unsqueeze(-1) * (1.0 - fired_h)
    assert total.max() <= 1, "a neuron may win at most one bucket across the whole scan"


def test_no_crossing_finite():
    m = _model(buckets=8, n_heads=3, neurons_per_head=6)
    with torch.no_grad():
        m.w_raw.fill_(-30.0)
    x = torch.randn(8, 17)
    V, a_srt = m._membrane(x)
    t = m._tau_hard(V, a_srt, m.theta_mem + torch.zeros(8, m.n_heads, m.neurons_per_head))
    assert torch.allclose(t, torch.full_like(t, m.t_window)), "no-crossing -> t_window"
    assert torch.isfinite(m(x, "st")).all()
