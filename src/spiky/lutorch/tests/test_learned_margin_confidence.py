"""Tests for the "learned_margin" confidence form (doc/research/lut_ablation/notes_learned_margin.md):

    s = exp(g) * (sum_j m_j) * (prod_j sigmoid(beta m_j))^gamma,   beta = exp(log_beta), gamma = exp(log_gamma)

with (g, log_beta, log_gamma) three LEARNABLE scalars per LightMultiHeadLUT. Computed as
(sum m) * exp(g + gamma * sum_j logsigmoid(beta m_j)). Closed-form gradients the tests pin autograd to:
    ds/dm_j       = exp(g) P^gamma + s * gamma * beta * sigmoid(-beta m_j)
    ds/dg         = s
    ds/dlog_beta  = s * gamma * sum_j (beta m_j) sigmoid(-beta m_j)
    ds/dlog_gamma = s * gamma * log P
Covered: bit-identity with margin at init, nesting of sharp_margin, all four gradients (autograd vs closed
form, gradcheck, finite differences) at random / zero / near-zero / near-equal / all-zero margins, fp32
precision of the log-space power, the autograd-only route (analytic sibling refuses; the LightMHL training
path never calls it), parameter registration and guards, gradients reaching the parameters, plumbing,
native-kernel fallback and discontinuity.
"""
import math
import os

os.environ.setdefault("LUT_DISABLE_COMPILE", "1")

import pytest
import torch

import spiky.lutorch.fast_multi_head_lut as FM
from spiky.lutorch.bh4_multi_head_lut import BH4MultiHeadLUT
from spiky.lutorch.compression_mhl import CompressionMultiHeadLUT
from spiky.lutorch.fast_multi_head_lut import (
    LEARNED_MARGIN_INIT,
    FastMultiHeadLut,
    _confidence_score,
    _confidence_score_and_dscore,
)
from spiky.lutorch.light_multi_head_lut import LightMultiHeadLUT

_HAS_CUDA = torch.cuda.is_available()
NAP = 8
F64 = torch.float64


def _params(g=0.0, beta=2.0, gamma=1.0, dtype=F64, device="cpu", grad=False):
    return tuple(torch.tensor(v, dtype=dtype, device=device, requires_grad=grad)
                 for v in (g, math.log(beta), math.log(gamma)))


def _explicit(m, g, beta, gamma):
    return math.exp(g) * m.sum(-1) * torch.sigmoid(beta * m).prod(-1) ** gamma


def _closed_form(m, g, beta, gamma):
    """(s, ds/dm, ds/dg, ds/dlog_beta, ds/dlog_gamma) for m >= 0, fp64."""
    logP = torch.nn.functional.logsigmoid(beta * m).sum(-1)
    pg = math.exp(g) * torch.exp(gamma * logP)
    s = m.sum(-1) * pg
    ds_dm = pg.unsqueeze(-1) + (s * gamma * beta).unsqueeze(-1) * torch.sigmoid(-beta * m)
    ds_dlb = s * gamma * (beta * m * torch.sigmoid(-beta * m)).sum(-1)
    return s, ds_dm, s, ds_dlb, s * gamma * logP


def _margins(case, seed=0):
    torch.manual_seed(seed)
    m = torch.rand(5, 4, NAP, dtype=F64) * 1.5 + 0.05
    if case == "zero":
        m[..., 3] = 0.0
    elif case == "near_zero_1e-300":
        m[..., 3] = 1e-300
    elif case == "near_zero_1e-12":
        m[..., 3] = 1e-12
    elif case == "two_zero":
        m[..., 1] = 0.0
        m[..., 6] = 0.0
    elif case == "near_equal":
        m = torch.full((5, 4, NAP), 0.4, dtype=F64) + 1e-12 * torch.arange(NAP, dtype=F64)
    elif case == "all_zero":
        m = torch.zeros(5, 4, NAP, dtype=F64)
    return m


CASES = ["random", "zero", "near_zero_1e-300", "near_zero_1e-12", "two_zero", "near_equal", "all_zero"]
SETTINGS = [(0.0, 2.0, 1.0), (0.7, 3.1, 1.75), (-0.4, 0.8, 0.6), (math.log(25.4), 2.0, 3.0)]


def test_init_constant_is_margin():
    assert LEARNED_MARGIN_INIT == (0.0, 2.0, 1.0)


@pytest.mark.parametrize("dtype", [torch.float32, F64])
@pytest.mark.parametrize("device", ["cpu"] + (["cuda"] if _HAS_CUDA else []))
def test_init_is_bitwise_identical_to_margin(dtype, device):
    torch.manual_seed(1)
    d = torch.randn(64, 4, 128, NAP, dtype=dtype, device=device)
    d[0, 0, 0, :] = 0.0
    d[1, 1, 1, 2] = 0.0
    p = _params(dtype=dtype, device=device)
    assert p[1].exp().item() == 2.0 and p[2].exp().item() == 1.0
    assert torch.equal(_confidence_score(d, "learned_margin", 1.0, None, p), _confidence_score(d, "margin"))


@pytest.mark.parametrize("g,beta,gamma", SETTINGS)
def test_score_formula(g, beta, gamma):
    m = _margins("random")
    torch.testing.assert_close(_confidence_score(m, "learned_margin", 1.0, None, _params(g, beta, gamma)),
                               _explicit(m, g, beta, gamma), rtol=1e-12, atol=0)


@pytest.mark.parametrize("gamma,gain", [(1.75, 3.9), (3.0, 25.4)])
def test_nests_sharp_margin(gamma, gain):
    d = torch.randn(6, 5, NAP, dtype=F64)
    torch.testing.assert_close(
        _confidence_score(d, "learned_margin", 1.0, None, _params(math.log(gain), 2.0, gamma)),
        _confidence_score(d, "sharp_margin", gain, gamma), rtol=1e-12, atol=0)


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("g,beta,gamma", SETTINGS)
def test_all_gradients_match_closed_form(case, g, beta, gamma):
    m = _margins(case)
    p = _params(g, beta, gamma, grad=True)
    mm = m.clone().requires_grad_(True)          # m >= 0 fed directly: abs is the identity here except at 0
    s = _confidence_score(mm, "learned_margin", 1.0, None, p)
    gm, gg, glb, glg = torch.autograd.grad(s.sum(), (mm, *p))
    cs, cdm, cdg, cdlb, cdlg = _closed_form(m, g, beta, gamma)
    torch.testing.assert_close(s.detach(), cs, rtol=1e-12, atol=0)
    for got in (gm, gg, glb, glg):
        assert torch.isfinite(got).all()
    torch.testing.assert_close(gg, cdg.sum(), rtol=1e-11, atol=1e-300)
    torch.testing.assert_close(glb, cdlb.sum(), rtol=1e-11, atol=1e-300)
    torch.testing.assert_close(glg, cdlg.sum(), rtol=1e-11, atol=1e-300)
    # d/dm through |m|: equal to the closed form wherever m > 0; abs' subgradient 0 where m == 0
    torch.testing.assert_close(gm, cdm * (m > 0), rtol=1e-11, atol=1e-300)


def test_explicit_formula_gradient_at_zero_margins_is_finite_and_closed_form():
    """Without abs (the one-sided derivative at a boundary) the formula's dm gradient is the closed form."""
    for case in ("zero", "two_zero", "all_zero"):
        m = _margins(case).requires_grad_(True)
        (gm,) = torch.autograd.grad(_explicit(m, 0.3, 2.5, 1.5).sum(), m)
        torch.testing.assert_close(gm, _closed_form(m.detach(), 0.3, 2.5, 1.5)[1], rtol=1e-11, atol=1e-300)


@pytest.mark.parametrize("g,beta,gamma", SETTINGS)
def test_all_margins_zero_is_exactly_zero_with_finite_gradients(g, beta, gamma):
    d = torch.zeros(3, 2, NAP, dtype=F64, requires_grad=True)
    p = _params(g, beta, gamma, grad=True)
    s = _confidence_score(d, "learned_margin", 1.0, None, p)
    assert torch.all(s == 0)
    grads = torch.autograd.grad(s.sum(), (d, *p))
    assert all(torch.isfinite(x).all() for x in grads)
    assert all(torch.all(x == 0) for x in grads[1:])      # s = 0 -> every parameter gradient is 0


@pytest.mark.parametrize("case", ["random", "zero", "near_zero_1e-12", "near_equal", "all_zero"])
@pytest.mark.parametrize("g,beta,gamma", SETTINGS[1:])
def test_gradcheck_wrt_parameters(case, g, beta, gamma):
    m = _margins(case, seed=3)[:2, :2]
    p = _params(g, beta, gamma, grad=True)
    assert torch.autograd.gradcheck(lambda a, b, c: _confidence_score(m, "learned_margin", 1.0, None, (a, b, c)), p)


@pytest.mark.parametrize("g,beta,gamma", SETTINGS)
def test_gradcheck_wrt_margins_and_parameters_jointly(g, beta, gamma):
    torch.manual_seed(4)
    base = torch.rand(2, 3, NAP, dtype=F64) * 2.0 + 0.05
    d = (base * torch.where(torch.rand_like(base) > 0.5, 1.0, -1.0)).requires_grad_(True)
    p = _params(g, beta, gamma, grad=True)
    assert torch.autograd.gradcheck(
        lambda x, a, b, c: _confidence_score(x, "learned_margin", 1.0, None, (a, b, c)), (d, *p))


def test_parameter_gradients_match_finite_differences():
    m = _margins("random", seed=5)
    base = [0.2, math.log(1.7), math.log(2.2)]
    p = [torch.tensor(v, dtype=F64, requires_grad=True) for v in base]
    grads = torch.autograd.grad(_confidence_score(m, "learned_margin", 1.0, None, tuple(p)).sum(), p)
    for k in range(3):
        hi, lo = list(base), list(base)
        hi[k] += 1e-6
        lo[k] -= 1e-6
        f = lambda v: _confidence_score(m, "learned_margin", 1.0, None,
                                        tuple(torch.tensor(x, dtype=F64) for x in v)).sum().item()
        assert abs((f(hi) - f(lo)) / 2e-6 - grads[k].item()) < 1e-6 * max(1.0, abs(grads[k].item()))


@pytest.mark.parametrize("beta", [0.5, 2.0, 8.0])
@pytest.mark.parametrize("gamma", [0.5, 1.75, 3.0, 6.0])
def test_float32_log_space_power_is_accurate(beta, gamma):
    torch.manual_seed(6)
    m = torch.rand(200, 16, NAP, dtype=F64) * 1.2
    ref = _confidence_score(m, "learned_margin", 1.0, None, _params(0.1, beta, gamma))
    got = _confidence_score(m.float(), "learned_margin", 1.0, None, _params(0.1, beta, gamma, dtype=torch.float32))
    keep = ref > 1e-30
    rel = ((got.double() - ref).abs() / ref)[keep]
    assert torch.isfinite(got).all() and rel.max().item() < 5e-5


def test_analytic_sibling_refuses_and_score_requires_params_and_others_ignore_them():
    d = torch.randn(3, 2, NAP, dtype=F64)
    with pytest.raises(NotImplementedError):
        _confidence_score_and_dscore(d, "learned_margin", 1.0, None, _params())
    with pytest.raises(ValueError):
        _confidence_score(d, "learned_margin")
    p = _params(0.5, 3.0, 2.0)
    for form in ("bounded", "bounded_norm", "margin", "min_margin", "tanh_margin", "sharp_margin"):
        assert torch.equal(_confidence_score(d, form, 1.3), _confidence_score(d, form, 1.3, None, p))


def _light(form="learned_margin", **kw):
    base = dict(input_dim=48, n_tables=4 * 16, output_dim=48, n_anchor_pairs=NAP, random_seed=1000,
                n_heads=4, multi_head_input=True, read_top_n=1)
    base.update(kw)
    return LightMultiHeadLUT(confidence_form=form, **base)


def test_light_registers_exactly_three_scalar_parameters_per_module():
    lm, mg = _light(), _light("margin")
    extra = {k: v for k, v in lm.named_parameters()} .keys() - {k for k, _ in mg.named_parameters()}
    assert extra == {"confidence_g", "confidence_log_beta", "confidence_log_gamma"}
    for k in extra:
        p = dict(lm.named_parameters())[k]
        assert p.dim() == 0 and p.requires_grad and p.dtype == torch.float32
    assert lm.confidence_log_beta.exp().item() == 2.0 and lm.confidence_log_gamma.exp().item() == 1.0
    assert lm.confidence_g.item() == 0.0
    assert lm.learned_confidence_values() == {"g": 0.0, "beta": 2.0, "gamma": 1.0}
    assert mg.learned_confidence_values() is None and mg.learned_confidence_params() is None
    assert set(mg.state_dict()) == set(_light("margin").state_dict())
    assert lm._score_form_id == 6
    v = _light(learned_margin_init=(0.5, 3.0, 1.75)).learned_confidence_values()
    assert abs(v["g"] - 0.5) < 1e-7 and abs(v["beta"] - 3.0) < 1e-6 and abs(v["gamma"] - 1.75) < 1e-6


def test_light_guards():
    with pytest.raises(ValueError):
        _light("margin", learned_margin_init=(0.0, 2.0, 1.0))
    with pytest.raises(ValueError):
        _light(confidence_gain=2.0)                           # exp(g) is the gain
    for bad in ((0.0, 0.0, 1.0), (0.0, 2.0, -1.0), (float("nan"), 2.0, 1.0), (0.0, float("inf"), 1.0)):
        with pytest.raises(ValueError):
            _light(learned_margin_init=bad)


@pytest.mark.parametrize("n", [1, 2])
def test_light_forward_at_init_is_bitwise_margin_fp32(n):
    lm, mg = _light(read_top_n=n, read_tau=0.3), _light("margin", read_top_n=n, read_tau=0.3)
    with torch.no_grad():
        t = torch.randn_like(mg.tables)
        lm.tables.copy_(t)
        mg.tables.copy_(t)
    z = torch.randn(9, 4, 48)
    with torch.enable_grad():
        assert torch.equal(lm(z), mg(z))


@pytest.mark.parametrize("n", [1, 2])
def test_light_parameters_receive_nonzero_gradients(n):
    m = _light(read_top_n=n, read_tau=0.3).double()
    with torch.no_grad():
        m.tables.normal_()
    z = torch.randn(7, 4, 48, dtype=F64)
    (m(z) * torch.randn(7, 4, 48, dtype=F64)).sum().backward()
    for p in (m.confidence_g, m.confidence_log_beta, m.confidence_log_gamma):
        assert p.grad is not None and torch.isfinite(p.grad) and p.grad.abs().item() > 0


@pytest.mark.parametrize("n", [1, 2])
def test_light_training_path_never_calls_the_analytic_sibling(n, monkeypatch):
    def boom(*a, **k):
        raise AssertionError("analytic _confidence_score_and_dscore was called on the LightMHL path")
    monkeypatch.setattr(FM, "_confidence_score_and_dscore", boom)
    monkeypatch.setattr(FM, "_confidence_grad_x", boom)
    m = _light(read_top_n=n, read_tau=0.3)
    z = torch.randn(5, 4, 48, requires_grad=True)
    m(z).sum().backward()
    assert z.grad is not None and m.confidence_log_gamma.grad is not None


def test_light_forward_matches_manual_formula_with_moved_parameters():
    H, T, D = 4, 16, 48
    m = _light(learned_margin_init=(0.3, 2.7, 1.6)).double()
    with torch.no_grad():
        m.tables.normal_()
    z = torch.randn(3, H, D, dtype=F64)
    a = m.anchor_a.reshape(1, H, T * NAP).expand(3, H, T * NAP)
    b = m.anchor_b.reshape(1, H, T * NAP).expand(3, H, T * NAP)
    d = (torch.gather(z, 2, a) - torch.gather(z, 2, b)).view(3, H, T, NAP)
    cells = ((d > 0).long() * 2 ** torch.arange(NAP - 1, -1, -1)).sum(-1)
    rows = m.tables.detach().reshape(H, T, 1 << NAP, D)[torch.arange(H)[None, :, None],
                                                        torch.arange(T)[None, None, :], cells]
    v = m.learned_confidence_values()
    s = _explicit(d.abs(), v["g"], v["beta"], v["gamma"])
    with torch.enable_grad():
        torch.testing.assert_close(m(z), (s.unsqueeze(-1) * rows).sum(2), rtol=1e-12, atol=1e-12)


def test_compression_mhl_plumbing_and_refusals():
    kw = dict(input_dim=32, output_dim=32, inner_in_dim=8, inner_out_dim=8, nap=4, tph=4, n_heads=2,
              confidence_form="learned_margin", random_seed=0)
    c = CompressionMultiHeadLUT(lut_impl="light", learned_margin_init=(0.1, 2.5, 1.5), **kw)
    v = c.lut_light.learned_confidence_values()
    assert abs(v["beta"] - 2.5) < 1e-6 and abs(v["gamma"] - 1.5) < 1e-6
    with pytest.raises(ValueError):
        CompressionMultiHeadLUT(lut_impl="fast", forward_confidence=True, **kw)


def test_fast_and_bh4_refuse_the_form():
    with pytest.raises(ValueError):
        FastMultiHeadLut(input_dim=12, n_heads=2, n_outputs=4, n_anchor_pairs=3, tables_per_head=2,
                         forward_confidence=True, confidence_form="learned_margin", use_bf16=False)
    with pytest.raises(ValueError):
        BH4MultiHeadLUT(input_dim=16, n_heads=2, tables_per_head=2, n_anchor_pairs=3, output_dim=4,
                        confidence_form="learned_margin")


@pytest.mark.skipif(not _HAS_CUDA, reason="native scored eval kernel is CUDA-only")
def test_native_scored_eval_refuses_learned_margin_and_torch_path_matches():
    kw = dict(device="cuda", read_top_n=1)
    ml, mm = _light(learned_margin_init=(0.2, 2.5, 1.4), **kw), _light("margin", **kw)
    z = torch.randn(32, 4, 48, device="cuda")
    assert mm._fused_eval(z.reshape(32, -1).contiguous()) is not None
    assert ml._fused_eval(z.reshape(32, -1).contiguous()) is None
    with torch.no_grad():
        y_eval = ml(z)
    torch.testing.assert_close(y_eval, ml._forward_multi_head(z).detach(), rtol=0, atol=0)


@pytest.mark.skipif(not _HAS_CUDA, reason="CUDA")
def test_cuda_fp32_forward_at_init_is_bitwise_margin_on_the_torch_path():
    lm, mg = _light(device="cuda"), _light("margin", device="cuda")
    with torch.no_grad():
        t = torch.randn_like(mg.tables)
        lm.tables.copy_(t)
        mg.tables.copy_(t)
    z = torch.randn(16, 4, 48, device="cuda")
    with torch.enable_grad():
        assert torch.equal(lm(z), mg(z))


def test_light_n1_learned_margin_is_discontinuous_at_boundary():
    m = _light(learned_margin_init=(0.0, 2.0, 1.75)).double()
    with torch.no_grad():
        m.tables.normal_(generator=torch.Generator().manual_seed(15))
    z = torch.randn(1, 4, 48, generator=torch.Generator().manual_seed(16), dtype=F64)
    h, t, j = 2, 7, 4
    a, b = int(m.anchor_a[h, t, j]), int(m.anchor_b[h, t, j])

    def y_at(u):
        zz = z.clone()
        zz[0, h, a] = zz[0, h, b] + u
        with torch.enable_grad():
            return m(zz).detach()

    assert ((y_at(1e-9) - y_at(-1e-9))[0, h].norm() / y_at(1e-9)[0, h].norm()).item() > 1e-3
