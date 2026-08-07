"""Smoke tests for LIFLayer — the minimal stackable LIF spiking transform.

output_transform: "rescale" (DEFAULT, inter-layer input-independent affine) | "log" (final readout) |
"linear" (raw first-spike time).
"""
import math
import pytest
import torch
import torch.nn as nn
from spiky.lutorch.lif_layer import LIFLayer

BASE_PARAMS = {"w_raw", "delay", "gain", "baseline_C", "tau_raw", "log_T_cross"}
RESCALE_PARAMS = BASE_PARAMS | {"rescale_scale", "rescale_bias"}


def _linear_clone(seed, in_dim, out_dim, **kw):
    torch.manual_seed(seed)
    return LIFLayer(in_dim, out_dim, output_transform="linear", **kw)


def test_default_is_rescale_and_is_affine_of_raw_times():
    torch.manual_seed(0)
    m = LIFLayer(17, 64)                                    # default output_transform='rescale'
    assert m.output_transform == "rescale"
    assert set(dict(m.named_parameters())) == RESCALE_PARAMS   # extra affine params present
    x = torch.randn(8, 17)
    y = m(x)
    assert y.shape == (8, 64) and torch.isfinite(y).all()
    # output == rescale_scale * raw_time + rescale_bias  (an affine of the raw first-spike times)
    raw = _linear_clone(0, 17, 64)(x)                      # same seed -> identical LIF params -> same raw times
    assert torch.allclose(y, m.rescale_scale * raw + m.rescale_bias, atol=1e-4)


def test_rescale_is_input_independent():
    """The rescale affine is a fixed learnable per-channel scale+bias — the SAME transform for every input
    (NOT per-sample statistics like LayerNorm). Two different batches get the identical scale/bias."""
    torch.manual_seed(0)
    m = LIFLayer(17, 32)
    assert isinstance(m.rescale_scale, nn.Parameter) and isinstance(m.rescale_bias, nn.Parameter)
    scale, bias = m.rescale_scale.detach().clone(), m.rescale_bias.detach().clone()
    lin = _linear_clone(0, 17, 32)
    for x in (torch.randn(8, 17), 100.0 * torch.randn(64, 17) + 5.0):   # wildly different batches
        y = m(x)
        # same affine coefficients applied regardless of the batch:
        assert torch.equal(m.rescale_scale.detach(), scale) and torch.equal(m.rescale_bias.detach(), bias)
        assert torch.allclose(y, scale * lin(x) + bias, atol=1e-4)


def test_rescale_init_standardizes_raw_times():
    m = LIFLayer(17, 16, rescale_init_mean=12.0, rescale_init_std=2.0)
    assert torch.allclose(m.rescale_scale, torch.full_like(m.rescale_scale, 1.0 / 2.0))
    assert torch.allclose(m.rescale_bias, torch.full_like(m.rescale_bias, -12.0 / 2.0))


def test_linear_mode_reproduces_raw_times():
    torch.manual_seed(0)
    m = LIFLayer(17, 64, output_transform="linear")
    y = m(torch.randn(8, 17))
    assert (y >= 0).all() and (y <= m.t_window).all()      # raw first-spike times in [0, t_window]
    assert set(dict(m.named_parameters())) == BASE_PARAMS  # no rescale params in linear mode


def test_log_mode_bounds_and_no_neg_inf():
    torch.manual_seed(0)
    m = LIFLayer(17, 64, output_transform="log")
    assert m.log_eps == pytest.approx(1e-3 * m.t_window)
    lo, hi = math.log(m.log_eps), math.log(m.t_window + m.log_eps)
    for x in (torch.randn(8, 17), 50 * torch.randn(8, 17), torch.zeros(8, 17)):
        y = m(x)
        assert torch.isfinite(y).all()                     # NO -inf even for immediate spikes
        assert (y >= lo - 1e-4).all() and (y <= hi + 1e-4).all()


def test_no_spike_fallback():
    """Never-crossing neuron -> t_window (linear) / log(t_window+eps) (log). Sharp temp, w~0."""
    for mode in ("log", "linear"):
        torch.manual_seed(0)
        m = LIFLayer(17, 32, output_transform=mode)
        with torch.no_grad():
            m.w_raw.fill_(-30.0); m.log_T_cross.fill_(-4.0)
        y = m(torch.randn(16, 17))
        target = math.log(m.t_window + m.log_eps) if mode == "log" else m.t_window
        assert torch.allclose(y, torch.full_like(y, target), atol=1e-3)


def test_single_layer_all_params_get_gradient():
    torch.manual_seed(0)
    m = LIFLayer(17, 64)                                    # rescale default
    y = m(torch.randn(32, 17))
    m.zero_grad(set_to_none=True)
    (y ** 2).mean().backward()
    got = dict(m.named_parameters())
    assert set(got) == RESCALE_PARAMS
    for n, p in got.items():
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0, f"{n} no grad"


def test_freeze_temperature():
    m = LIFLayer(17, 8, freeze_temperature=True)
    assert not m.log_T_cross.requires_grad
    assert LIFLayer(17, 8).log_T_cross.requires_grad


def test_stacked_rescale_hidden_log_final_grad_flow():
    """Actor composition: rescale hidden layers + a single 'log' FINAL readout. Only the last layer is
    'log'; forward is finite and gradient reaches the rescaler affine params and every LIF layer."""
    dims = [17, 32, 32, 6]
    n = len(dims) - 1
    net = nn.ModuleList([
        LIFLayer(dims[i], dims[i + 1], output_transform=("log" if i == n - 1 else "rescale"))
        for i in range(n)])
    assert [l.output_transform for l in net] == ["rescale", "rescale", "log"]   # log ONLY at the final layer
    torch.manual_seed(0)
    h = torch.randn(32, 17)
    for layer in net:
        h = layer(h)
        assert torch.isfinite(h).all()
    assert h.shape == (32, 6)
    for layer in net:
        layer.zero_grad(set_to_none=True)
    (h ** 2).mean().backward()
    for i, layer in enumerate(net):
        for name, p in layer.named_parameters():
            assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0, \
                f"layer{i}.{name} received no gradient"
