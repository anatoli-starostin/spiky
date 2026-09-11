"""Tests for the "sharp_margin" confidence form: s = (sum_j m_j) * (prod_j sigmoid(2 m_j))^gamma, m = |d|.

The selectivity-matched DISCONTINUOUS control (doc/research/lut_ablation/notes_sharp_margin.md).
The analytic derivative in _confidence_score_and_dscore is
    dscore/dm_j = P^gamma + (sum_k m_k) * gamma * P^gamma * 2 * sigmoid(-2 m_j),  P = prod_k sigmoid(2 m_k)
(d log P / dm_j = 2 sigmoid(-2 m_j)). These tests pin it against autograd of the explicit formula in
fp64 -- random, zero, near-zero, all-zero and near-equal margins -- plus finite differences and
gradcheck, for both gammas the runs use; check the per-module gamma plumbing and its guards; the
native-kernel fallback; and that the n=1 read-out is DISCONTINUOUS at a boundary (unlike tanh_margin).
"""
import os

os.environ.setdefault("LUT_DISABLE_COMPILE", "1")

import pytest
import torch

from spiky.lutorch.compression_mhl import CompressionMultiHeadLUT
from spiky.lutorch.fast_multi_head_lut import (
    SHARP_MARGIN_GAMMA,
    FastMultiHeadLut,
    _confidence_score,
    _confidence_score_and_dscore,
)
from spiky.lutorch.light_multi_head_lut import LightMultiHeadLUT

_HAS_CUDA = torch.cuda.is_available()
GAMMAS = [1.75, 3.0]          # exp_g_0245, exp_g_0246
NAP = 8


def _explicit(mm, gamma, gain):
    return gain * mm.sum(dim=-1) * torch.sigmoid(2.0 * mm).prod(dim=-1) ** gamma


def _autograd_dscore_dm(m, gamma, gain=1.0):
    """d score / d m by autograd of the explicit formula IN m (no abs, whose derivative at 0 is 0)."""
    mm = m.clone().requires_grad_(True)
    (g,) = torch.autograd.grad(_explicit(mm, gamma, gain).sum(), mm)
    return g


def test_default_gamma_constant():
    assert SHARP_MARGIN_GAMMA == 1.75


@pytest.mark.parametrize("gamma", GAMMAS)
@pytest.mark.parametrize("gain", [1.0, 3.9])
def test_score_formula(gamma, gain):
    torch.manual_seed(0)
    d = torch.randn(7, 5, NAP, dtype=torch.float64)
    torch.testing.assert_close(_confidence_score(d, "sharp_margin", gain, gamma),
                               _explicit(d.abs(), gamma, gain), rtol=1e-12, atol=0)


def test_none_gamma_means_module_constant():
    torch.manual_seed(1)
    d = torch.randn(4, 3, NAP, dtype=torch.float64)
    assert torch.equal(_confidence_score(d, "sharp_margin", 2.0),
                       _confidence_score(d, "sharp_margin", 2.0, SHARP_MARGIN_GAMMA))
    s0, ds0 = _confidence_score_and_dscore(d, "sharp_margin", 2.0)
    s1, ds1 = _confidence_score_and_dscore(d, "sharp_margin", 2.0, SHARP_MARGIN_GAMMA)
    assert torch.equal(s0, s1) and torch.equal(ds0, ds1)


def test_other_forms_ignore_gamma():
    torch.manual_seed(2)
    d = torch.randn(4, 3, NAP, dtype=torch.float64)
    for form in ("bounded", "bounded_norm", "margin", "min_margin", "tanh_margin"):
        assert torch.equal(_confidence_score(d, form, 1.3), _confidence_score(d, form, 1.3, 3.0))
        a, b = _confidence_score_and_dscore(d, form, 1.3), _confidence_score_and_dscore(d, form, 1.3, 3.0)
        assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])


def test_gamma_one_is_margin():
    torch.manual_seed(3)
    d = torch.randn(6, 4, NAP, dtype=torch.float64)
    torch.testing.assert_close(_confidence_score(d, "sharp_margin", 1.0, 1.0),
                               _confidence_score(d, "margin"), rtol=1e-13, atol=0)
    s, ds = _confidence_score_and_dscore(d, "sharp_margin", 1.0, 1.0)
    sm, dsm = _confidence_score_and_dscore(d, "margin")
    torch.testing.assert_close(s, sm, rtol=1e-13, atol=0)
    torch.testing.assert_close(ds, dsm, rtol=1e-13, atol=0)


@pytest.mark.parametrize("gamma", GAMMAS)
def test_score_does_not_vanish_at_a_boundary(gamma):
    """One margin at 0 multiplies the probability factor by sigmoid(0)^gamma = 0.5^gamma, not 0."""
    torch.manual_seed(4)
    m = torch.rand(6, 4, NAP, dtype=torch.float64) + 0.2
    m0 = m.clone()
    m0[..., 5] = 0.0
    s0 = _confidence_score(m0, "sharp_margin", 1.0, gamma)
    assert torch.all(s0 > 0)
    others = torch.cat([m0[..., :5], m0[..., 6:]], dim=-1)
    expected = m0.sum(dim=-1) * 0.5 ** gamma * torch.sigmoid(2.0 * others).prod(dim=-1) ** gamma
    torch.testing.assert_close(s0, expected, rtol=1e-12, atol=0)


@pytest.mark.parametrize("gamma", GAMMAS)
@pytest.mark.parametrize("gain", [1.0, 25.4])
def test_score_and_dscore_score_is_identical(gamma, gain):
    torch.manual_seed(5)
    d = torch.randn(9, 6, NAP, dtype=torch.float64)
    s, _ = _confidence_score_and_dscore(d, "sharp_margin", gain, gamma)
    assert torch.equal(s, _confidence_score(d, "sharp_margin", gain, gamma))


@pytest.mark.parametrize("gamma", GAMMAS)
@pytest.mark.parametrize("gain", [1.0, 3.9])
def test_dscore_matches_autograd_random(gamma, gain):
    torch.manual_seed(6)
    m = torch.rand(11, 7, NAP, dtype=torch.float64) * 2.0 + 0.01
    _, ds = _confidence_score_and_dscore(m, "sharp_margin", gain, gamma)
    torch.testing.assert_close(ds, _autograd_dscore_dm(m, gamma, gain), rtol=1e-10, atol=1e-14)


@pytest.mark.parametrize("gamma", GAMMAS)
def test_dscore_matches_the_closed_form_rule(gamma):
    """dscore/dm_j = P^gamma + (sum m) * gamma * P^gamma * 2 * sigmoid(-2 m_j), written out literally."""
    torch.manual_seed(7)
    m = torch.rand(5, 4, NAP, dtype=torch.float64) * 1.5
    pg = torch.sigmoid(2.0 * m).prod(dim=-1, keepdim=True) ** gamma
    rule = pg + m.sum(dim=-1, keepdim=True) * gamma * pg * 2.0 * torch.sigmoid(-2.0 * m)
    _, ds = _confidence_score_and_dscore(m, "sharp_margin", 1.0, gamma)
    torch.testing.assert_close(ds, rule, rtol=1e-12, atol=0)


@pytest.mark.parametrize("gamma", GAMMAS)
@pytest.mark.parametrize("zero_value", [0.0, 1e-300, 1e-12, 1e-8])
def test_dscore_at_and_near_zero_margin(gamma, zero_value):
    torch.manual_seed(8)
    m = torch.rand(8, 5, NAP, dtype=torch.float64) + 0.2
    m[..., 3] = zero_value
    s, ds = _confidence_score_and_dscore(m, "sharp_margin", 1.0, gamma)
    assert torch.isfinite(ds).all() and torch.isfinite(s).all()
    torch.testing.assert_close(ds, _autograd_dscore_dm(m, gamma), rtol=1e-10, atol=1e-14)


@pytest.mark.parametrize("gamma", GAMMAS)
def test_dscore_with_several_zero_margins(gamma):
    torch.manual_seed(9)
    m = torch.rand(6, 4, NAP, dtype=torch.float64) + 0.2
    m[..., 1] = 0.0
    m[..., 6] = 0.0
    _, ds = _confidence_score_and_dscore(m, "sharp_margin", 1.0, gamma)
    assert torch.all(ds > 0)
    torch.testing.assert_close(ds, _autograd_dscore_dm(m, gamma), rtol=1e-10, atol=1e-14)


@pytest.mark.parametrize("gamma", GAMMAS)
def test_all_margins_zero_corner(gamma):
    """The all-zero corner: score 0 (sum m = 0) and dscore/dm_j = P^gamma = 0.5^(NAP*gamma) for every j."""
    m = torch.zeros(2, 3, NAP, dtype=torch.float64)
    s, ds = _confidence_score_and_dscore(m, "sharp_margin", 1.0, gamma)
    assert torch.all(s == 0)
    torch.testing.assert_close(ds, torch.full_like(m, 0.5 ** (NAP * gamma)), rtol=1e-12, atol=0)
    torch.testing.assert_close(ds, _autograd_dscore_dm(m, gamma), rtol=1e-10, atol=0)


@pytest.mark.parametrize("gamma", GAMMAS)
@pytest.mark.parametrize("gap", [0.0, 1e-12, 1e-9])
def test_dscore_with_near_equal_margins(gamma, gap):
    m = torch.full((5, 4, NAP), 0.4, dtype=torch.float64) + gap * torch.arange(NAP, dtype=torch.float64)
    _, ds = _confidence_score_and_dscore(m, "sharp_margin", 1.0, gamma)
    torch.testing.assert_close(ds, _autograd_dscore_dm(m, gamma), rtol=1e-10, atol=1e-14)


@pytest.mark.parametrize("gamma", GAMMAS)
def test_gradient_wrt_d_matches_dscore_times_sign_including_zero(gamma):
    torch.manual_seed(10)
    d = torch.randn(6, 4, NAP, dtype=torch.float64)
    d[..., 2] = 0.0
    dd = d.clone().requires_grad_(True)
    (g,) = torch.autograd.grad(_confidence_score(dd, "sharp_margin", 1.0, gamma).sum(), dd)
    _, ds = _confidence_score_and_dscore(d, "sharp_margin", 1.0, gamma)
    torch.testing.assert_close(g, ds * torch.sign(d), rtol=1e-10, atol=1e-14)


@pytest.mark.parametrize("gamma", GAMMAS)
def test_dscore_matches_finite_differences(gamma):
    torch.manual_seed(11)
    m = torch.rand(3, 3, NAP, dtype=torch.float64) * 2.0 + 0.05
    _, ds = _confidence_score_and_dscore(m, "sharp_margin", 1.0, gamma)
    num = torch.zeros_like(m)
    flat, nflat, eps = m.reshape(-1), num.reshape(-1), 1e-7
    for i in range(flat.numel()):
        orig = flat[i].item()
        flat[i] = orig + eps
        sp = _confidence_score(m, "sharp_margin", 1.0, gamma)
        flat[i] = orig - eps
        sm = _confidence_score(m, "sharp_margin", 1.0, gamma)
        flat[i] = orig
        b, rem = divmod(i, 3 * NAP)
        t, _j = divmod(rem, NAP)
        nflat[i] = (sp[b, t] - sm[b, t]) / (2.0 * eps)
    assert (ds - num).abs().max().item() < 1e-6


@pytest.mark.parametrize("gamma", GAMMAS)
def test_gradcheck_float64(gamma):
    torch.manual_seed(12)
    base = torch.rand(3, 4, NAP, dtype=torch.float64) * 2.0 + 0.05
    signs = torch.where(torch.rand_like(base) > 0.5, 1.0, -1.0)
    d = (base * signs).requires_grad_(True)
    assert torch.autograd.gradcheck(lambda x: _confidence_score(x, "sharp_margin", 1.3, gamma), (d,))


def test_form_accepted_by_modules():
    LightMultiHeadLUT(input_dim=12, n_tables=4, output_dim=6, n_anchor_pairs=3,
                      confidence_form="sharp_margin", random_seed=0)
    FastMultiHeadLut(input_dim=12, n_heads=2, n_outputs=4, n_anchor_pairs=3, tables_per_head=2,
                     forward_confidence=True, confidence_form="sharp_margin", random_seed=0,
                     use_bf16=False)


def test_light_gamma_plumbing_and_guards():
    kw = dict(input_dim=12, n_tables=4, output_dim=6, n_anchor_pairs=3, random_seed=0)
    assert LightMultiHeadLUT(confidence_form="sharp_margin", **kw).sharp_margin_gamma == SHARP_MARGIN_GAMMA
    m3 = LightMultiHeadLUT(confidence_form="sharp_margin", sharp_margin_gamma=3.0, **kw)
    assert m3.sharp_margin_gamma == 3.0 and "sharp_margin_gamma=3.0" in m3.extra_repr()
    assert m3._score_form_id == 5
    assert LightMultiHeadLUT(confidence_form="margin", **kw).sharp_margin_gamma is None
    with pytest.raises(ValueError):
        LightMultiHeadLUT(confidence_form="margin", sharp_margin_gamma=3.0, **kw)
    for bad in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            LightMultiHeadLUT(confidence_form="sharp_margin", sharp_margin_gamma=bad, **kw)


def test_compression_mhl_passes_gamma_on_light_and_refuses_elsewhere():
    kw = dict(input_dim=32, output_dim=32, inner_in_dim=8, inner_out_dim=8, nap=4, tph=4, n_heads=2,
              confidence_form="sharp_margin", random_seed=0)
    c = CompressionMultiHeadLUT(lut_impl="light", sharp_margin_gamma=3.0, **kw)
    assert c.lut_light.sharp_margin_gamma == 3.0
    assert CompressionMultiHeadLUT(lut_impl="light", **kw).lut_light.sharp_margin_gamma == SHARP_MARGIN_GAMMA
    with pytest.raises(ValueError):
        CompressionMultiHeadLUT(lut_impl="fast", sharp_margin_gamma=3.0, forward_confidence=True, **kw)


@pytest.mark.parametrize("gamma", GAMMAS)
def test_light_forward_uses_its_own_gamma(gamma):
    H, T, D = 4, 16, 48
    m = LightMultiHeadLUT(input_dim=D, n_tables=H * T, output_dim=D, n_anchor_pairs=NAP,
                          confidence_form="sharp_margin", sharp_margin_gamma=gamma, confidence_gain=2.0,
                          random_seed=1000, n_heads=H, multi_head_input=True, read_top_n=1).double()
    with torch.no_grad():
        m.tables.normal_()
    g = torch.Generator().manual_seed(13)
    z = torch.randn(3, H, D, generator=g, dtype=torch.float64)
    a = m.anchor_a.reshape(1, H, T * NAP).expand(3, H, T * NAP)
    b = m.anchor_b.reshape(1, H, T * NAP).expand(3, H, T * NAP)
    d = (torch.gather(z, 2, a) - torch.gather(z, 2, b)).view(3, H, T, NAP)
    cells = ((d > 0).long() * 2 ** torch.arange(NAP - 1, -1, -1)).sum(-1)
    W = m.tables.detach().reshape(H, T, 1 << NAP, D)
    rows = W[torch.arange(H)[None, :, None], torch.arange(T)[None, None, :], cells]      # [3, H, T, D]
    s = _explicit(d.abs(), gamma, 2.0)
    expected = (s.unsqueeze(-1) * rows).sum(2)
    torch.testing.assert_close(m(z), expected, rtol=1e-12, atol=1e-12)


@pytest.mark.skipif(not _HAS_CUDA, reason="native scored eval kernel is CUDA-only")
@pytest.mark.parametrize("gamma", GAMMAS)
def test_native_scored_eval_refuses_sharp_margin_and_torch_path_matches(gamma):
    H, T, D = 4, 32, 48
    kw = dict(input_dim=D, n_tables=H * T, output_dim=D, n_anchor_pairs=NAP, random_seed=1000,
              n_heads=H, multi_head_input=True, read_top_n=1, device="cuda")
    ms = LightMultiHeadLUT(confidence_form="sharp_margin", sharp_margin_gamma=gamma, **kw)
    mm = LightMultiHeadLUT(confidence_form="margin", **kw)
    g = torch.Generator().manual_seed(14)
    z = torch.randn(32, H, D, generator=g).cuda()
    assert mm._fused_eval(z.reshape(32, -1).contiguous()) is not None     # the kernel exists here
    assert ms._fused_eval(z.reshape(32, -1).contiguous()) is None         # ...and is refused
    with torch.no_grad():
        y_eval = ms(z)
    torch.testing.assert_close(y_eval, ms._forward_multi_head(z).detach(), rtol=0, atol=0)


@pytest.mark.parametrize("gamma", GAMMAS)
def test_light_n1_sharp_margin_discontinuous_at_boundary(gamma):
    H, T, D = 4, 16, 48
    kw = dict(input_dim=D, n_tables=H * T, output_dim=D, n_anchor_pairs=NAP, random_seed=1000,
              n_heads=H, multi_head_input=True, read_top_n=1)

    def jump(form, **extra):
        m = LightMultiHeadLUT(confidence_form=form, **extra, **kw).double()
        with torch.no_grad():
            m.tables.normal_(generator=torch.Generator().manual_seed(15))
        z = torch.randn(1, H, D, generator=torch.Generator().manual_seed(16), dtype=torch.float64)
        h, t, j = 2, 7, 4
        a, b = int(m.anchor_a[h, t, j]), int(m.anchor_b[h, t, j])

        def y_at(u):
            zz = z.clone()
            zz[0, h, a] = zz[0, h, b] + u
            with torch.enable_grad():
                return m(zz).detach()

        return ((y_at(1e-9) - y_at(-1e-9))[0, h].norm() / y_at(1e-9)[0, h].norm()).item()

    assert jump("sharp_margin", sharp_margin_gamma=gamma) > 1e-3
    assert jump("tanh_margin") < 1e-6
