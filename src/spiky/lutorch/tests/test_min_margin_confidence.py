"""Tests for the "min_margin" confidence form: s = (min_j m_j) * prod_j sigmoid(2 m_j), m = |d|.

The one genuinely new piece of math is the analytic derivative in _confidence_score_and_dscore:
    dscore/dm_j = P * 1[j = j*] + 2 * score * sigmoid(-2 m_j),   P = prod_j sigmoid(2 m_j)
with j* the index torch.min returns (its first minimum). These tests pin that against autograd
(including exact and near ties), gradcheck it in float64, check that the native scored eval
kernel refuses the form so no-grad eval takes the torch path, and check continuity at a boundary.
"""
import os

os.environ.setdefault("LUT_DISABLE_COMPILE", "1")

import pytest
import torch

from spiky.lutorch.fast_multi_head_lut import (
    FastMultiHeadLut,
    _confidence_score,
    _confidence_score_and_dscore,
)
from spiky.lutorch.light_multi_head_lut import LightMultiHeadLUT

_HAS_CUDA = torch.cuda.is_available()


def _autograd_dscore_dm(m, gain):
    """d score / d m from autograd, feeding positive m as d so |d| = d."""
    d = m.clone().requires_grad_(True)
    s = _confidence_score(d, "min_margin", gain)
    (g,) = torch.autograd.grad(s.sum(), d)
    return g


def test_score_formula():
    torch.manual_seed(0)
    d = torch.randn(7, 5, 8, dtype=torch.float64)
    m = d.abs()
    expected = m.min(dim=-1).values * torch.sigmoid(2.0 * m).prod(dim=-1)
    torch.testing.assert_close(_confidence_score(d, "min_margin"), expected, rtol=1e-12, atol=0)
    torch.testing.assert_close(_confidence_score(d, "min_margin", 2.5), 2.5 * expected,
                               rtol=1e-12, atol=0)


def test_score_vanishes_when_any_margin_is_zero():
    torch.manual_seed(1)
    d = torch.randn(6, 4, 8, dtype=torch.float64)
    d[..., 3] = 0.0
    assert torch.all(_confidence_score(d, "min_margin") == 0)


@pytest.mark.parametrize("gain", [1.0, 2.5])
def test_score_and_dscore_score_matches(gain):
    torch.manual_seed(2)
    d = torch.randn(9, 6, 8, dtype=torch.float64)
    s, _ = _confidence_score_and_dscore(d, "min_margin", gain)
    torch.testing.assert_close(s, _confidence_score(d, "min_margin", gain), rtol=0, atol=0)


@pytest.mark.parametrize("gain", [1.0, 2.5])
def test_dscore_matches_autograd_random(gain):
    torch.manual_seed(3)
    m = torch.rand(11, 7, 8, dtype=torch.float64) * 2.0 + 0.05
    _, ds = _confidence_score_and_dscore(m, "min_margin", gain)
    torch.testing.assert_close(ds, _autograd_dscore_dm(m, gain), rtol=1e-12, atol=1e-15)


@pytest.mark.parametrize("gap", [0.0, 1e-12, 1e-9, 1e-6])
def test_dscore_matches_autograd_ties_and_near_ties(gap):
    """Exact ties (gap 0) and near ties: the P term lands on exactly one anchor, the one
    torch.min returns, identically in the analytic helper and in autograd."""
    torch.manual_seed(4)
    m = torch.rand(10, 6, 8, dtype=torch.float64) + 0.5
    m[..., 2] = 0.1
    m[..., 5] = 0.1 + gap
    _, ds = _confidence_score_and_dscore(m, "min_margin")
    torch.testing.assert_close(ds, _autograd_dscore_dm(m, 1.0), rtol=1e-12, atol=1e-15)
    prob = torch.sigmoid(2.0 * m).prod(dim=-1)
    score = m.min(dim=-1).values * prob
    base = 2.0 * score.unsqueeze(-1) * torch.sigmoid(-2.0 * m)
    extra = ds - base                                   # the one-hot P term (plus ~1e-18 round-off)
    assert torch.all((extra.abs() > 1e-12).sum(dim=-1) == 1)
    assert torch.all(extra[..., 2] > 0)                 # index 2: the (first) minimum
    torch.testing.assert_close(extra.sum(dim=-1), prob, rtol=1e-12, atol=0)


def test_dscore_matches_finite_differences_with_separated_minimum():
    torch.manual_seed(5)
    m = torch.rand(4, 3, 8, dtype=torch.float64) * 2.0 + 0.2
    m[..., 1] = 0.05                                    # min separated by >= 0.15
    _, ds = _confidence_score_and_dscore(m, "min_margin")
    num = torch.zeros_like(m)
    flat, nflat, eps = m.reshape(-1), num.reshape(-1), 1e-7
    for i in range(flat.numel()):
        orig = flat[i].item()
        flat[i] = orig + eps
        sp = _confidence_score(m, "min_margin")
        flat[i] = orig - eps
        sm = _confidence_score(m, "min_margin")
        flat[i] = orig
        b, rem = divmod(i, 3 * 8)
        t, _j = divmod(rem, 8)
        nflat[i] = (sp[b, t] - sm[b, t]) / (2.0 * eps)
    assert (ds - num).abs().max().item() < 1e-6


def test_gradcheck_float64_away_from_ties():
    torch.manual_seed(6)
    base = torch.rand(3, 4, 8, dtype=torch.float64) * 2.0 + 0.3
    base[..., 0] = 0.1                                  # separated minimum
    signs = torch.where(torch.rand_like(base) > 0.5, 1.0, -1.0)
    d = (base * signs).requires_grad_(True)
    assert torch.autograd.gradcheck(lambda x: _confidence_score(x, "min_margin", 1.7), (d,))


def test_forms_accepted_by_all_modules():
    LightMultiHeadLUT(input_dim=12, n_tables=4, output_dim=6, n_anchor_pairs=3,
                      confidence_form="min_margin", random_seed=0)
    FastMultiHeadLut(input_dim=12, n_heads=2, n_outputs=4, n_anchor_pairs=3, tables_per_head=2,
                     forward_confidence=True, confidence_form="min_margin", random_seed=0,
                     use_bf16=False)
    with pytest.raises(ValueError):
        LightMultiHeadLUT(input_dim=12, n_tables=4, output_dim=6, n_anchor_pairs=3,
                          confidence_form="not_a_form", random_seed=0)


@pytest.mark.skipif(not _HAS_CUDA, reason="native scored eval kernel is CUDA-only")
@pytest.mark.parametrize("read_top_n", [1])
def test_native_scored_eval_refuses_min_margin_and_torch_path_matches(read_top_n):
    H, T, D, NAP = 4, 32, 48, 8
    kw = dict(input_dim=D, n_tables=H * T, output_dim=D, n_anchor_pairs=NAP, random_seed=1000,
              n_heads=H, multi_head_input=True, read_top_n=read_top_n, device="cuda")
    mm = LightMultiHeadLUT(confidence_form="min_margin", **kw)
    mg = LightMultiHeadLUT(confidence_form="margin", **kw)
    g = torch.Generator().manual_seed(8)
    z = torch.randn(32, H, D, generator=g).cuda()
    flat = z.reshape(32, -1).contiguous()
    if mg._native_msb_scored is not None:
        assert mg._fused_eval(flat) is not None        # the existing forms still go native
    assert mm._fused_eval(flat) is None                # min_margin never does
    with torch.no_grad():
        y_eval = mm(z)                                  # no-grad eval entry point
    y_torch = mm._forward_multi_head(z)                 # the training (autograd) path
    torch.testing.assert_close(y_eval, y_torch.detach(), rtol=0, atol=0)


def test_light_n1_min_margin_continuous_at_boundary():
    H, T, D, NAP = 4, 16, 48, 8
    m = LightMultiHeadLUT(input_dim=D, n_tables=H * T, output_dim=D, n_anchor_pairs=NAP,
                          confidence_form="min_margin", random_seed=1000, n_heads=H,
                          multi_head_input=True, read_top_n=1).double()
    with torch.no_grad():
        m.tables.normal_()
    g = torch.Generator().manual_seed(10)
    z = torch.randn(1, H, D, generator=g, dtype=torch.float64)
    h, t, j = 1, 5, 3
    a, b = int(m.anchor_a[h, t, j]), int(m.anchor_b[h, t, j])

    def y_at(u):
        zz = z.clone()
        zz[0, h, a] = zz[0, h, b] + u
        with torch.enable_grad():
            return m(zz).detach()

    eps = 1e-9
    jump = (y_at(eps) - y_at(-eps))[0, h].norm() / y_at(eps)[0, h].norm()
    assert jump.item() < 1e-6
