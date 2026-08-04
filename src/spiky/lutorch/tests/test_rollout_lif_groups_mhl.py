"""Tests for RolloutLIFGroupsMHL — the time-stepped LIF-rollout spiking net with intra-group inhibition."""
import torch
from spiky.lutorch.rollout_lif_groups_mhl import RolloutLIFGroupsMHL


def _model(**kw):
    torch.manual_seed(0)
    cfg = dict(in_dim=17, out_dim=6, groups=8, neurons_per_group=14, steps=32)
    cfg.update(kw)
    return RolloutLIFGroupsMHL(**cfg)


def test_forward_shape_and_finite():
    m = _model()
    x = torch.randn(8, 17)
    for mode in ("st", "hard", "soft"):
        y = m(x, mode=mode)
        assert y.shape == (8, 6) and torch.isfinite(y).all(), f"bad output in mode {mode}"


def test_straight_through_invariant():
    m = _model(groups=4, neurons_per_group=6, steps=16)
    x = torch.randn(16, 17)
    with torch.no_grad():
        y_st, y_hard, y_soft = m(x, "st"), m(x, "hard"), m(x, "soft")
    assert torch.allclose(y_st, y_hard, atol=1e-5), "ST forward must equal hard first-spike readout"
    assert not torch.allclose(y_st, y_soft, atol=1e-4), "soft must differ from hard"


def test_param_count_bounded_weights_delays():
    m = _model()
    assert 3500 <= m.param_count() <= 6500 and m.P == 112
    assert (m.W > 0).all() and (m.W < m.w_max).all() and m.w_max == 2.0
    assert (m.delays >= 0).all() and (m.w_inh >= 0).all()


def test_all_params_get_gradient():
    m = _model(groups=4, neurons_per_group=6, steps=16)
    x = torch.randn(16, 17); tgt = torch.randn(16, 6)
    m.zero_grad(set_to_none=True)
    torch.nn.functional.mse_loss(m(x, "st"), tgt).backward()
    for name, p in m.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0, f"{name} no grad"


def test_net_fires_and_never_fired_reads_zero():
    m = _model(groups=4, neurons_per_group=6, steps=16)
    x = torch.randn(8, 17)
    SH, _ = m._rollout(x)
    assert SH.sum() > 0, "the net must actually spike at init"
    with torch.no_grad():
        m.w_raw.fill_(-30.0)                                 # W ~ 0 -> no input current -> no spikes
    ph_hard, _ = m._phi(x)
    assert torch.allclose(ph_hard, torch.zeros_like(ph_hard), atol=1e-6), "never-fired neurons -> phi=0 (t=K)"


def test_delay_gradient_flows():
    """The Gaussian pulse kernel must pass gradient to the per-synapse delays."""
    m = _model(groups=4, neurons_per_group=6, steps=16)
    x = torch.randn(8, 17); tgt = torch.randn(8, 6)
    m.zero_grad(set_to_none=True)
    torch.nn.functional.mse_loss(m(x, "st"), tgt).backward()
    assert m.d_raw.grad.abs().sum() > 0, "delays (d_raw) got no gradient from the temporal kernel"
