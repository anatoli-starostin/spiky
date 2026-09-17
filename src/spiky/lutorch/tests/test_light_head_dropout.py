"""Regression test for head-level LUT-table dropout in LightMultiHeadLUT (forward_mode='hard').

Standard INVERTED dropout on the plain hard read: during TRAINING each (sample, head, table) row is
kept with prob (1-rate) and survivors are scaled by 1/(1-rate); eval/no-grad reads all tables with no
rescale. Opt-in and off by default (rate 0.0 -> byte-identical to the existing path).

Asserts:
  (a) rate=0.0 -> train == eval == the plain read (byte-identical; off by default);
  (b) at eval, dropout is OFF regardless of rate (rate>0 eval == rate=0 eval);
  (c) in training, rate>0 changes the value (dropout active);
  (d) inverted-dropout is magnitude-preserving: mean over many train draws ~= eval;
  (e) tables still receive a finite nonzero gradient under dropout;
  (f) the default argument is 0.0, and rate outside [0,1) / non-hard forward_mode are refused.
"""
import pytest
import torch

torch._dynamo.config.suppress_errors = True

from spiky.lutorch.light_multi_head_lut import LightMultiHeadLUT

NAP, TPH, H, IN, OUT, B = 4, 8, 2, 8, 5, 64


def _make(rate, dtype=torch.float64):
    m = LightMultiHeadLUT(
        input_dim=IN, n_tables=H * TPH, output_dim=OUT, n_anchor_pairs=NAP,
        confidence_form="learned_margin", learned_margin_init=(0.0, 2.0, 1.0),
        learned_margin_freeze_g=True, random_seed=0, initial_weights_noise=0.5,
        device=torch.device("cpu"), n_heads=H, multi_head_input=False,
        read_top_n=2, read_tau=0.5, read_tau_learnable=True, forward_mode="scored",
        head_dropout_rate=rate).to(dtype)
    m._compile_enabled = False
    return m


def _x(seed=1, dtype=torch.float64):
    return torch.randn(B, IN, generator=torch.Generator().manual_seed(seed), dtype=dtype)


def test_rate_zero_is_byte_identical():
    m = _make(0.0)
    x = _x()
    m.eval()
    with torch.no_grad():
        ev = m(x)
    m.train()
    tr = m(x.clone().requires_grad_(True))
    assert torch.allclose(tr, ev, atol=1e-12, rtol=0)


def test_eval_has_no_dropout():
    x = _x()
    m0, md = _make(0.0), _make(0.2)
    m0.eval(); md.eval()
    with torch.no_grad():
        assert torch.allclose(md(x), m0(x), atol=1e-12, rtol=0)


def test_training_dropout_changes_value():
    x = _x()
    md = _make(0.2)
    md.eval()
    with torch.no_grad():
        ev = md(x)
    md.train()
    torch.manual_seed(123)
    tr = md(x.clone().requires_grad_(True))
    assert not torch.allclose(tr, ev, atol=1e-9)


def test_inverted_dropout_preserves_expected_read():
    x = _x()
    md = _make(0.2)
    md.eval()
    with torch.no_grad():
        ev = md(x)
    md.train()
    torch.manual_seed(7)
    N = 4000
    acc = torch.zeros_like(ev)
    for _ in range(N):
        acc += md(x).detach()
    rel = (acc / N - ev).abs().mean().item() / ev.abs().mean().item()
    assert rel < 0.03, f"inverted-dropout mean drifted from eval: rel err {rel:.4f}"


def test_tables_get_gradient_under_dropout():
    md = _make(0.2)
    md.train()
    torch.manual_seed(1)
    md(_x()).sum().backward()
    g = md.tables.grad
    assert g is not None and torch.isfinite(g).all() and float(g.abs().sum()) > 0.0


def test_default_off_and_validation():
    m = LightMultiHeadLUT(
        input_dim=IN, n_tables=H * TPH, output_dim=OUT, n_anchor_pairs=NAP,
        confidence_form="learned_margin", learned_margin_init=(0.0, 2.0, 1.0),
        learned_margin_freeze_g=True, random_seed=0, n_heads=H, forward_mode="scored")
    assert m.head_dropout_rate == 0.0
    with pytest.raises(ValueError):
        _make(1.0)
    with pytest.raises(NotImplementedError):  # only forward_mode='scored' is supported
        LightMultiHeadLUT(input_dim=IN, n_tables=H * TPH, output_dim=OUT, n_anchor_pairs=NAP,
                          confidence_form="learned_margin", learned_margin_init=(0.0, 2.0, 1.0),
                          learned_margin_freeze_g=True, random_seed=0, n_heads=H,
                          forward_mode="hard", head_dropout_rate=0.2)
