"""Tests for the "tanh_margin" confidence form: s = (sum_j m_j) * prod_j tanh(a m_j), m = |d|.

The risky piece is the analytic derivative in _confidence_score_and_dscore:
    dscore/dm_j = prod_k tanh(a m_k) + (sum_k m_k) * a * sech^2(a m_j) * prod_{k != j} tanh(a m_k)
The naive form divides prod_k tanh by tanh(a m_j), which is 0/0 at m_j = 0. The implementation
uses the exclusive product prod_{k != j} (prefix x suffix cumulative products), which stays finite.
These tests pin it against autograd at and near zero margins, with several zeros and with
near-equal margins, gradcheck it in float64, and check continuity and the native-kernel fallback.
"""
import os

os.environ.setdefault("LUT_DISABLE_COMPILE", "1")

import pytest
import torch

from spiky.lutorch.fast_multi_head_lut import (
    TANH_MARGIN_A,
    FastMultiHeadLut,
    _confidence_score,
    _confidence_score_and_dscore,
)
from spiky.lutorch.light_multi_head_lut import LightMultiHeadLUT

_HAS_CUDA = torch.cuda.is_available()


def _autograd_dscore_dm(m, gain=1.0):
    """d score / d m by autograd of the explicit formula IN m (no abs).

    _confidence_score_and_dscore returns the derivative w.r.t. m = |d|; autograd through
    _confidence_score's d.abs() would instead give d/dd, which is 0 at d = 0 (abs's kink),
    so it is not a valid reference exactly at a zero margin.
    """
    mm = m.clone().requires_grad_(True)
    s = gain * mm.sum(dim=-1) * torch.tanh(TANH_MARGIN_A * mm).prod(dim=-1)
    (g,) = torch.autograd.grad(s.sum(), mm)
    return g


def test_gradient_wrt_d_matches_dscore_times_sign_including_zero():
    """The chain rule callers apply: dscore/dd = dscore/dm * sgn(d), with sgn(0) = 0."""
    torch.manual_seed(11)
    d = torch.randn(6, 4, 8, dtype=torch.float64)
    d[..., 2] = 0.0
    dd = d.clone().requires_grad_(True)
    (g,) = torch.autograd.grad(_confidence_score(dd, "tanh_margin").sum(), dd)
    _, ds = _confidence_score_and_dscore(d, "tanh_margin")
    torch.testing.assert_close(g, ds * torch.sign(d), rtol=1e-10, atol=1e-14)


def test_default_scale_constant():
    assert TANH_MARGIN_A == 2.0


def test_score_formula():
    torch.manual_seed(0)
    d = torch.randn(7, 5, 8, dtype=torch.float64)
    m = d.abs()
    expected = m.sum(dim=-1) * torch.tanh(TANH_MARGIN_A * m).prod(dim=-1)
    torch.testing.assert_close(_confidence_score(d, "tanh_margin"), expected, rtol=1e-12, atol=0)
    torch.testing.assert_close(_confidence_score(d, "tanh_margin", 1.7), 1.7 * expected,
                               rtol=1e-12, atol=0)


def test_score_vanishes_when_any_margin_is_zero():
    torch.manual_seed(1)
    d = torch.randn(6, 4, 8, dtype=torch.float64)
    d[..., 5] = 0.0
    assert torch.all(_confidence_score(d, "tanh_margin") == 0)


@pytest.mark.parametrize("gain", [1.0, 1.7])
def test_score_and_dscore_score_matches(gain):
    torch.manual_seed(2)
    d = torch.randn(9, 6, 8, dtype=torch.float64)
    s, _ = _confidence_score_and_dscore(d, "tanh_margin", gain)
    torch.testing.assert_close(s, _confidence_score(d, "tanh_margin", gain), rtol=1e-12, atol=0)


@pytest.mark.parametrize("gain", [1.0, 1.7])
def test_dscore_matches_autograd_random(gain):
    torch.manual_seed(3)
    m = torch.rand(11, 7, 8, dtype=torch.float64) * 2.0 + 0.01
    _, ds = _confidence_score_and_dscore(m, "tanh_margin", gain)
    torch.testing.assert_close(ds, _autograd_dscore_dm(m, gain), rtol=1e-10, atol=1e-14)


@pytest.mark.parametrize("zero_value", [0.0, 1e-300, 1e-12, 1e-8])
def test_dscore_at_and_near_zero_margin(zero_value):
    """The singular case of the naive derivative: one margin at or extremely close to zero."""
    torch.manual_seed(4)
    m = torch.rand(8, 5, 8, dtype=torch.float64) + 0.2
    m[..., 3] = zero_value
    s, ds = _confidence_score_and_dscore(m, "tanh_margin")
    assert torch.isfinite(ds).all() and torch.isfinite(s).all()
    torch.testing.assert_close(ds, _autograd_dscore_dm(m), rtol=1e-10, atol=1e-14)
    if zero_value == 0.0:
        # at m_3 = 0: score is 0, and only the zero anchor gets gradient:
        #   ds/dm_3 = sum(m) * a * prod_{k != 3} tanh(a m_k);  ds/dm_j = 0 for j != 3
        t = torch.tanh(TANH_MARGIN_A * m)
        excl = torch.cat([t[..., :3], t[..., 4:]], dim=-1).prod(dim=-1)
        torch.testing.assert_close(ds[..., 3], m.sum(dim=-1) * TANH_MARGIN_A * excl,
                                   rtol=1e-12, atol=0)
        others = torch.cat([ds[..., :3], ds[..., 4:]], dim=-1)
        assert torch.all(others == 0)


def test_dscore_with_two_zero_margins_is_zero_everywhere():
    torch.manual_seed(5)
    m = torch.rand(6, 4, 8, dtype=torch.float64) + 0.2
    m[..., 1] = 0.0
    m[..., 6] = 0.0
    _, ds = _confidence_score_and_dscore(m, "tanh_margin")
    assert torch.all(ds == 0)
    torch.testing.assert_close(ds, _autograd_dscore_dm(m), rtol=0, atol=0)


@pytest.mark.parametrize("gap", [0.0, 1e-12, 1e-9])
def test_dscore_with_near_equal_margins(gap):
    torch.manual_seed(6)
    m = torch.full((5, 4, 8), 0.4, dtype=torch.float64)
    m = m + gap * torch.arange(8, dtype=torch.float64)
    _, ds = _confidence_score_and_dscore(m, "tanh_margin")
    torch.testing.assert_close(ds, _autograd_dscore_dm(m), rtol=1e-10, atol=1e-14)


def test_dscore_matches_finite_differences():
    torch.manual_seed(7)
    m = torch.rand(3, 3, 8, dtype=torch.float64) * 2.0 + 0.05
    _, ds = _confidence_score_and_dscore(m, "tanh_margin")
    num = torch.zeros_like(m)
    flat, nflat, eps = m.reshape(-1), num.reshape(-1), 1e-7
    for i in range(flat.numel()):
        orig = flat[i].item()
        flat[i] = orig + eps
        sp = _confidence_score(m, "tanh_margin")
        flat[i] = orig - eps
        sm = _confidence_score(m, "tanh_margin")
        flat[i] = orig
        b, rem = divmod(i, 3 * 8)
        t, _j = divmod(rem, 8)
        nflat[i] = (sp[b, t] - sm[b, t]) / (2.0 * eps)
    assert (ds - num).abs().max().item() < 1e-6


def test_gradcheck_float64():
    torch.manual_seed(8)
    base = torch.rand(3, 4, 8, dtype=torch.float64) * 2.0 + 0.05
    signs = torch.where(torch.rand_like(base) > 0.5, 1.0, -1.0)
    d = (base * signs).requires_grad_(True)
    assert torch.autograd.gradcheck(lambda x: _confidence_score(x, "tanh_margin", 1.3), (d,))


def test_form_accepted_by_modules():
    LightMultiHeadLUT(input_dim=12, n_tables=4, output_dim=6, n_anchor_pairs=3,
                      confidence_form="tanh_margin", random_seed=0)
    FastMultiHeadLut(input_dim=12, n_heads=2, n_outputs=4, n_anchor_pairs=3, tables_per_head=2,
                     forward_confidence=True, confidence_form="tanh_margin", random_seed=0,
                     use_bf16=False)


@pytest.mark.skipif(not _HAS_CUDA, reason="native scored eval kernel is CUDA-only")
def test_native_scored_eval_refuses_tanh_margin_and_torch_path_matches():
    H, T, D, NAP = 4, 32, 48, 8
    kw = dict(input_dim=D, n_tables=H * T, output_dim=D, n_anchor_pairs=NAP, random_seed=1000,
              n_heads=H, multi_head_input=True, read_top_n=1, device="cuda")
    mt = LightMultiHeadLUT(confidence_form="tanh_margin", **kw)
    g = torch.Generator().manual_seed(9)
    z = torch.randn(32, H, D, generator=g).cuda()
    assert mt._fused_eval(z.reshape(32, -1).contiguous()) is None
    with torch.no_grad():
        y_eval = mt(z)
    torch.testing.assert_close(y_eval, mt._forward_multi_head(z).detach(), rtol=0, atol=0)


def test_light_n1_tanh_margin_continuous_at_boundary():
    H, T, D, NAP = 4, 16, 48, 8
    m = LightMultiHeadLUT(input_dim=D, n_tables=H * T, output_dim=D, n_anchor_pairs=NAP,
                          confidence_form="tanh_margin", random_seed=1000, n_heads=H,
                          multi_head_input=True, read_top_n=1).double()
    with torch.no_grad():
        m.tables.normal_()
    g = torch.Generator().manual_seed(10)
    z = torch.randn(1, H, D, generator=g, dtype=torch.float64)
    h, t, j = 2, 7, 4
    a, b = int(m.anchor_a[h, t, j]), int(m.anchor_b[h, t, j])

    def y_at(u):
        zz = z.clone()
        zz[0, h, a] = zz[0, h, b] + u
        with torch.enable_grad():
            return m(zz).detach()

    eps = 1e-9
    jump = (y_at(eps) - y_at(-eps))[0, h].norm() / y_at(eps)[0, h].norm()
    assert jump.item() < 1e-6
