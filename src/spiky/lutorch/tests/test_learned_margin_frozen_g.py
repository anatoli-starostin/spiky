"""Tests for learned_margin with the gain FROZEN (learned_margin_freeze_g=True): g is a fixed buffer
(default 0, i.e. gain exactly 1) while log_beta and log_gamma stay learnable -- the two-parameter score
s = (sum_j m_j) * (prod_j sigmoid(beta m_j))^gamma run as exp_g_0248 against exp_g_0247 (three parameters).
"""
import math
import os

os.environ.setdefault("LUT_DISABLE_COMPILE", "1")

import pytest
import torch

from spiky.lutorch.compression_mhl import CompressionMultiHeadLUT
from spiky.lutorch.fast_multi_head_lut import _confidence_score
from spiky.lutorch.light_multi_head_lut import LightMultiHeadLUT

NAP = 8


def _light(form="learned_margin", **kw):
    base = dict(input_dim=48, n_tables=4 * 16, output_dim=48, n_anchor_pairs=NAP, random_seed=1000,
                n_heads=4, multi_head_input=True, read_top_n=1)
    base.update(kw)
    return LightMultiHeadLUT(confidence_form=form, **base)


def test_frozen_g_is_a_buffer_and_only_beta_gamma_are_parameters():
    fz, lm = _light(learned_margin_freeze_g=True), _light()
    names = {k for k, _ in fz.named_parameters()}
    assert "confidence_g" not in names
    assert {"confidence_log_beta", "confidence_log_gamma"} <= names
    assert {k for k, _ in lm.named_parameters()} - names == {"confidence_g"}
    assert "confidence_g" in dict(fz.named_buffers())
    assert set(fz.state_dict()) == set(lm.state_dict())            # same keys: checkpoints interchange
    assert fz.confidence_g.item() == 0.0 and not fz.confidence_g.requires_grad
    assert fz.learned_confidence_values() == {"g": 0.0, "beta": 2.0, "gamma": 1.0}
    assert fz.learned_margin_freeze_g and not lm.learned_margin_freeze_g
    assert "g frozen" in fz.extra_repr()


@pytest.mark.parametrize("n", [1, 2])
def test_frozen_forward_at_init_is_bitwise_margin_and_bitwise_learnable(n):
    mods = [_light(learned_margin_freeze_g=True, read_top_n=n, read_tau=0.3), _light(read_top_n=n, read_tau=0.3),
            _light("margin", read_top_n=n, read_tau=0.3)]
    t = torch.randn_like(mods[0].tables)
    with torch.no_grad():
        for m in mods:
            m.tables.copy_(t)
    z = torch.randn(9, 4, 48)
    with torch.enable_grad():
        ys = [m(z) for m in mods]
    assert torch.equal(ys[0], ys[1]) and torch.equal(ys[0], ys[2])


@pytest.mark.parametrize("n", [1, 2])
def test_frozen_g_gets_no_gradient_and_never_moves_while_beta_gamma_learn(n):
    m = _light(learned_margin_freeze_g=True, read_top_n=n, read_tau=0.3)
    with torch.no_grad():
        m.tables.normal_()
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=1e-2, weight_decay=0.0)
    lb0, lg0 = m.confidence_log_beta.item(), m.confidence_log_gamma.item()
    for _ in range(3):
        opt.zero_grad()
        z = torch.randn(7, 4, 48)
        (m(z) * torch.randn(7, 4, 48)).sum().backward()
        assert m.confidence_g.grad is None
        assert m.confidence_log_beta.grad.abs().item() > 0 and m.confidence_log_gamma.grad.abs().item() > 0
        opt.step()
    assert m.confidence_g.item() == 0.0
    assert m.confidence_log_beta.item() != lb0 and m.confidence_log_gamma.item() != lg0


def test_frozen_nonzero_g_is_held_at_its_init_and_used_in_the_score():
    m = _light(learned_margin_freeze_g=True, learned_margin_init=(0.4, 2.5, 1.3)).double()
    d = torch.randn(3, 4, 16, NAP, dtype=torch.float64)
    mm = d.abs()
    v = m.learned_confidence_values()
    assert abs(v["g"] - 0.4) < 1e-7
    expected = math.exp(v["g"]) * mm.sum(-1) * torch.sigmoid(v["beta"] * mm).prod(-1) ** v["gamma"]
    torch.testing.assert_close(m.confidence_score(d), expected, rtol=1e-12, atol=0)


def test_learnable_checkpoint_loads_into_frozen_module_strictly():
    lm = _light(learned_margin_init=(-0.2, 2.2, 1.5))
    fz = _light(learned_margin_freeze_g=True)
    fz.load_state_dict(lm.state_dict(), strict=True)
    assert abs(fz.confidence_g.item() + 0.2) < 1e-7 and "confidence_g" not in dict(fz.named_parameters())


def test_guards_and_plumbing():
    with pytest.raises(ValueError):
        _light("margin", learned_margin_freeze_g=True)
    kw = dict(input_dim=32, output_dim=32, inner_in_dim=8, inner_out_dim=8, nap=4, tph=4, n_heads=2,
              confidence_form="learned_margin", random_seed=0)
    c = CompressionMultiHeadLUT(lut_impl="light", learned_margin_freeze_g=True, **kw)
    assert c.lut_light.learned_margin_freeze_g and "confidence_g" not in dict(c.lut_light.named_parameters())
    with pytest.raises(ValueError):
        CompressionMultiHeadLUT(lut_impl="fast", forward_confidence=True, learned_margin_freeze_g=True,
                                **dict(kw, confidence_form="margin"))


def test_frozen_default_is_off_and_learnable_module_unchanged():
    lm = _light()
    assert "confidence_g" in dict(lm.named_parameters()) and not lm.learned_margin_freeze_g
    d = torch.randn(2, 4, 16, NAP)
    assert torch.equal(lm.confidence_score(d), _confidence_score(d, "learned_margin", 1.0, None,
                                                                 lm.learned_confidence_params()))
