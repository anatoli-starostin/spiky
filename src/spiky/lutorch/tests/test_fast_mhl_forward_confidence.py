"""Tests for FastMultiHeadLut's opt-in `forward_confidence` score gate.

The gate multiplies each gathered table row by a smooth per-(token, table)
scalar `score` derived from the routing-margin magnitudes |d| (LookupFFN's
score trick). The hard sign address is unchanged; only magnitude is gated, and
the gate is applied identically in train and eval. Its gradient flows
analytically through score -> |d| -> d -> x and is ADDED to the existing
directional surrogate (the surrogate is left intact).

Coverage:
  (i)   float64 finite-difference gradcheck of the closed-form dscore/dm for
        BOTH confidence_form variants, and of the hard-path analytic grad_x
        (the score->d term, isolated as gated-minus-ungated) and grad_W
        (score-scaled) against numeric finite differences.
  (i')  "bounded_norm" specifically: closed-form identity score == bounded**(1/NAP),
        NAP-independence of its scale, and no exp(2m) overflow at large margins.
  (ii)  flag-OFF bit-identity: forward_confidence=False forward AND backward
        are exactly equal to the no-flag default.
  (iii) train() == eval() for the hard path under the gate (hard address).
  (iv)  both hybrid_smooth forward paths (bmm for n_outputs>=128, gather for
        n_outputs<128) agree on output AND on the saved (main/alt/u) that drive
        the shared backward, under the gate.
  (v)   dtype smoke: fp32 CPU end-to-end; bf16 autocast on CUDA (skipped w/o GPU).
  plus: config guards (exp_outputs incompatibility, bad confidence_form).

These run on CPU (nucstar has no GPU); float64 + use_bf16=False throughout the
numeric checks. torch.compile falls back to eager on CPU where needed.
"""
import math

import pytest
import torch

# Compiled bodies fall back to eager on CPU/float64 instead of erroring.
torch._dynamo.config.suppress_errors = True

from spiky.lutorch.fast_multi_head_lut import (
    FastMultiHeadLut,
    _confidence_score,
    _confidence_score_and_dscore,
    _hybrid_smooth_fwd_bmm,
    _hybrid_smooth_fwd_gather,
    _soft_lut_fwd_body,
)

_CPU = torch.device("cpu")
_HAS_CUDA = torch.cuda.is_available()
_FORMS = ["bounded", "margin", "bounded_norm"]


def _make(*, forward_mode="hard", n_outputs=4, forward_confidence=False,
          confidence_form="bounded", confidence_gain=1.0, input_dim=16, n_heads=2,
          n_anchor_pairs=3, tables_per_head=2, weight_dtype=torch.float64,
          seed=0, device=_CPU):
    return FastMultiHeadLut(
        input_dim=input_dim, n_heads=n_heads, n_outputs=n_outputs,
        n_anchor_pairs=n_anchor_pairs, tables_per_head=tables_per_head,
        forward_mode=forward_mode, weight_dtype=weight_dtype, use_bf16=False,
        forward_confidence=forward_confidence, confidence_form=confidence_form,
        confidence_gain=confidence_gain, random_seed=seed, device=device,
    )


def _num_grad(f, t, eps=1e-6):
    """Central finite-difference gradient of scalar f() w.r.t. tensor t (in-place)."""
    g = torch.zeros_like(t)
    flat = t.reshape(-1)
    gflat = g.reshape(-1)
    for i in range(flat.numel()):
        orig = flat[i].item()
        flat[i] = orig + eps
        fp = float(f())
        flat[i] = orig - eps
        fm = float(f())
        flat[i] = orig
        gflat[i] = (fp - fm) / (2.0 * eps)
    return g


# =============================================================================
# (i) closed-form dscore/dm vs numeric
# =============================================================================

@pytest.mark.parametrize("form", _FORMS)
def test_dscore_dm_matches_finite_diff(form):
    """dscore/dm_j from _confidence_score_and_dscore matches a numeric derivative.

    Perturbs m = |d| directly (via d>0 so |d|=d), checks d(score)/d(m_j).
    """
    torch.manual_seed(0)
    # Positive d so m = |d| = d and perturbing d perturbs m one-for-one.
    d = torch.rand(5, 4, 3, dtype=torch.float64) * 2.0 + 0.1  # [B, n_tables, NAP], >0
    _score, dscore_dm = _confidence_score_and_dscore(d, form)

    num = torch.zeros_like(d)
    flat = d.reshape(-1)
    nflat = num.reshape(-1)
    eps = 1e-7
    for i in range(flat.numel()):
        orig = flat[i].item()
        flat[i] = orig + eps
        sp = _confidence_score(d, form)
        flat[i] = orig - eps
        sm = _confidence_score(d, form)
        flat[i] = orig
        # only the table owning element i changes; map i -> (b, t, j)
        b, rem = divmod(i, 4 * 3)
        t, j = divmod(rem, 3)
        nflat[i] = (sp[b, t] - sm[b, t]) / (2.0 * eps)

    err = (dscore_dm - num).abs().max().item()
    assert err < 1e-5, f"{form}: dscore/dm max abs err {err:.2e}"


# =============================================================================
# (i') "bounded_norm" closed-form identities
# =============================================================================
# bounded_norm exists because "bounded" is a product over NAP factors, so its
# attenuation compounds with the anchor count (~0.054 at NAP=8 on real
# activations, which divides the LUT table gradient by ~19x -- see #112).
# The geometric mean is the same ordering with a NAP-independent scale.

@pytest.mark.parametrize("nap", [1, 2, 3, 8])
def test_bounded_norm_equals_bounded_to_the_one_over_nap(nap):
    """score_norm == score_bounded ** (1/NAP), exactly (this IS the definition)."""
    torch.manual_seed(0)
    d = torch.randn(5, 4, nap, dtype=torch.float64) * 1.5
    s_b = _confidence_score(d, "bounded")
    s_n = _confidence_score(d, "bounded_norm")
    err = (s_n - s_b.pow(1.0 / nap)).abs().max().item()
    assert err < 1e-12, f"NAP={nap}: geomean identity violated by {err:.2e}"
    assert (s_n > 0).all() and (s_n <= 1.0).all(), "score must stay in (0, 1]"


def test_bounded_norm_scale_is_nap_independent():
    """With every margin equal to m, score_norm == sigmoid(2m) for ANY NAP.

    This is the whole point of the form: "bounded" gives sigmoid(2m)**NAP, which
    at m=0.38 (our measured median margin) falls from 0.68 at NAP=1 to 0.054 at
    NAP=8, while the geometric mean stays at 0.68.
    """
    m = 0.381                                   # measured median |d| at anchor sizing
    ref = float(torch.sigmoid(torch.tensor(2.0 * m, dtype=torch.float64)))
    for nap in (1, 2, 4, 8, 12):
        d = torch.full((3, 2, nap), m, dtype=torch.float64)
        s_n = _confidence_score(d, "bounded_norm")
        s_b = _confidence_score(d, "bounded")
        assert abs(s_n.max().item() - ref) < 1e-12, \
            f"NAP={nap}: normalised score moved with NAP ({s_n.max().item():.6f} vs {ref:.6f})"
        assert abs(s_b.max().item() - ref ** nap) < 1e-12
    # and at NAP=8 the un-normalised form really is the ~15-20x attenuation measured
    # in #112 (0.0467 at the median margin; mean 0.0542 over the real distribution)
    assert 0.04 < ref ** 8 < 0.06 and 0.67 < ref < 0.69


def test_bounded_norm_dscore_is_the_normalised_derivative_not_a_rescale():
    """dscore_norm/dm is d(score_norm)/dm, NOT d(score_bounded)/dm divided by NAP.

    Both share the factor sigmoid(-2m_j) (they are exp of the same logsigmoids),
    so the ratio must be exactly score_norm / (NAP * score_bounded) -- which at our
    margins is ~1.6, nowhere near the naive rescale's 1/NAP = 0.125.
    """
    torch.manual_seed(1)
    nap = 8
    d = torch.randn(6, 3, nap, dtype=torch.float64) * 1.5
    s_b, db = _confidence_score_and_dscore(d, "bounded")
    s_n, dn = _confidence_score_and_dscore(d, "bounded_norm")

    expected = (s_n / (nap * s_b)).unsqueeze(-1) * db
    err = (dn - expected).abs().max().item()
    assert err < 1e-12, f"bounded_norm derivative is not the exact normalised one: {err:.2e}"

    # It must NOT be the old derivative merely divided by NAP.
    naive = db / nap
    assert (dn - naive).abs().max().item() > 1e-3, \
        "bounded_norm dscore looks like the bounded derivative rescaled"


def test_bounded_norm_is_a_monotone_transform_of_bounded():
    """x**(1/NAP) is increasing, so the two forms rank (token, table) pairs alike."""
    torch.manual_seed(2)
    d = torch.randn(200, 1, 8, dtype=torch.float64) * 1.2
    s_b = _confidence_score(d, "bounded").reshape(-1)
    s_n = _confidence_score(d, "bounded_norm").reshape(-1)
    assert torch.equal(s_b.argsort(), s_n.argsort()), "ordering differs from bounded"


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_bounded_norm_numerically_stable_at_extremes(dtype):
    """No exp(2m) is ever built: huge margins stay finite, zero margins give 0.5."""
    big = torch.full((2, 2, 8), 1e4, dtype=dtype)
    s_big = _confidence_score(big, "bounded_norm")
    assert torch.isfinite(s_big).all(), "overflowed at large margins"
    assert torch.allclose(s_big, torch.ones_like(s_big)), "large margins should saturate to 1"

    zero = torch.zeros(2, 2, 8, dtype=dtype)
    s_zero = _confidence_score(zero, "bounded_norm")
    assert torch.allclose(s_zero, torch.full_like(s_zero, 0.5), atol=1e-7), \
        "zero margins should give sigmoid(0) = 0.5, not 0.5**NAP"
    _, dz = _confidence_score_and_dscore(zero, "bounded_norm")
    assert torch.isfinite(dz).all()


# =============================================================================
# (i) hard-path analytic grad_x (score term) and grad_W vs numeric
# =============================================================================

@pytest.mark.parametrize("form", _FORMS)
def test_hard_confidence_grads_match_numeric(form):
    """Hard mode: analytic grad_W (score-scaled) and the score->x term match FD.

    grad_x has two parts: the directional surrogate + the new score term. Because
    the surrogate uses the UNSCALED grad_pt, it is bit-identical between the gated
    and ungated backward, so (grad_x_gated - grad_x_ungated) isolates exactly the
    score->d->x term. With margins kept well off the decision boundary the hard
    address is locally constant, so a finite difference of the true (gated) forward
    w.r.t. x sees only that same score path, and w.r.t. W is the exact 1-row grad.
    """
    m = _make(forward_mode="hard", forward_confidence=True, confidence_form=form, seed=2)
    torch.manual_seed(7)
    x = torch.randn(6, m.input_dim, dtype=torch.float64) * 5.0   # large |margins|
    g = torch.randn(6, m.n_heads, m.n_outputs, dtype=torch.float64)

    # analytic (gated)
    xg = x.clone().requires_grad_(True)
    (m(xg) * g).sum().backward()
    gx_on, gW_on = xg.grad.clone(), m.weights.grad.clone()

    # analytic (ungated): same everything, surrogate part only
    m.weights.grad = None
    m.forward_confidence = False
    xu = x.clone().requires_grad_(True)
    (m(xu) * g).sum().backward()
    gx_off = xu.grad.clone()
    m.forward_confidence = True
    score_term = gx_on - gx_off

    # numeric on the true gated forward (eval / no_grad -> hard body, address fixed)
    def L():
        with torch.no_grad():
            return (m(x) * g).sum()

    num_gx = _num_grad(L, x)
    num_gW = _num_grad(L, m.weights.data)

    ex = (score_term - num_gx).abs().max().item()
    ew = (gW_on - num_gW).abs().max().item()
    assert ex < 1e-4, f"{form}: score->x term vs numeric max abs err {ex:.2e}"
    assert ew < 1e-4, f"{form}: grad_W vs numeric max abs err {ew:.2e}"


@pytest.mark.parametrize("form", _FORMS)
def test_hybrid_confidence_weight_grad_matches_numeric(form):
    """Hybrid: grad_W (2-row, score-scaled) matches FD (W does not move u/address)."""
    m = _make(forward_mode="hybrid_smooth", forward_confidence=True,
              confidence_form=form, seed=4)
    torch.manual_seed(11)
    x = torch.randn(6, m.input_dim, dtype=torch.float64) * 5.0
    g = torch.randn(6, m.n_heads, m.n_outputs, dtype=torch.float64)

    xg = x.clone().requires_grad_(True)
    (m(xg) * g).sum().backward()
    gW_on = m.weights.grad.clone()

    def L():
        with torch.no_grad():
            return (m(x) * g).sum()

    num_gW = _num_grad(L, m.weights.data)
    ew = (gW_on - num_gW).abs().max().item()
    assert ew < 1e-4, f"{form}: hybrid grad_W vs numeric max abs err {ew:.2e}"


# =============================================================================
# (ii) flag-off bit-identity
# =============================================================================

@pytest.mark.parametrize("forward_mode", ["hard", "hybrid_smooth"])
def test_flag_off_bit_identical(forward_mode):
    """forward_confidence=False reproduces the no-flag default forward AND backward."""
    kw = dict(forward_mode=forward_mode, seed=5)
    m_default = _make(**kw)                              # flag absent -> defaults False
    m_off = _make(forward_confidence=False, **kw)        # explicitly False
    torch.manual_seed(1)
    x = torch.randn(5, m_off.input_dim, dtype=torch.float64) * 3.0
    g = torch.randn(5, m_off.n_heads, m_off.n_outputs, dtype=torch.float64)

    xd = x.clone().requires_grad_(True)
    yd = m_default(xd); yd.backward(g)
    xo = x.clone().requires_grad_(True)
    yo = m_off(xo); yo.backward(g)

    assert torch.equal(yd, yo), "flag-off forward differs from default"
    assert torch.equal(xd.grad, xo.grad), "flag-off grad_x differs from default"
    assert torch.equal(m_default.weights.grad, m_off.weights.grad), "flag-off grad_W differs"

    # And the gate must actually change something when ON (sanity).
    m_on = _make(forward_confidence=True, **kw)
    with torch.no_grad():
        assert not torch.allclose(m_off(x), m_on(x)), "gate ON produced identical output"


def test_flag_off_matches_raw_body():
    """The hard eval path with the flag off equals the raw ungated _soft_lut_fwd_body."""
    m = _make(forward_mode="hard", forward_confidence=False, seed=6)
    x = torch.randn(4, m.input_dim, dtype=torch.float64) * 3.0
    with torch.no_grad():
        y = m(x)
        ref, _ = _soft_lut_fwd_body(
            x, m.weights, m.soft_anchor_a_long, m.soft_anchor_b_long,
            m.soft_powers, m.n_heads, m.tables_per_head, m.table_dim,
        )
    assert torch.equal(y, ref)


# =============================================================================
# (iii) train == eval under the gate (hard path)
# =============================================================================

@pytest.mark.parametrize("form", _FORMS)
def test_train_equals_eval_hard(form):
    """Hard + gate: the autograd (train) forward equals the no_grad (eval) forward."""
    m = _make(forward_mode="hard", forward_confidence=True, confidence_form=form, seed=8)
    x = torch.randn(7, m.input_dim, dtype=torch.float64) * 3.0
    y_train = m(x.clone().requires_grad_(True))
    with torch.no_grad():
        y_eval = m(x)
    assert torch.allclose(y_train, y_eval, atol=1e-12, rtol=0), \
        f"{form}: train vs eval max abs diff {(y_train - y_eval).abs().max().item():.2e}"


# =============================================================================
# (iv) hybrid bmm-path and gather-path agree under the gate
# =============================================================================

@pytest.mark.parametrize("form", _FORMS)
def test_hybrid_bmm_gather_agree_under_gate(form):
    """The two hybrid_smooth forward kernels compute the same gated result.

    The module picks bmm for n_outputs>=128 and gather otherwise; here both are
    called on identical inputs so the gate's application (fold into S vs multiply
    blended) is checked for consistency. The saved (main_index, alt_index, u) also
    match, so the shared backward is identical between the two paths.
    """
    torch.manual_seed(3)
    B, n_heads, tph, n_anchor_pairs, n_outputs = 6, 2, 2, 3, 5
    n_tables = n_heads * tph
    table_dim = 1 << n_anchor_pairs
    input_dim = 16
    x = torch.randn(B, input_dim, dtype=torch.float64) * 4.0
    weights = torch.randn(n_tables, table_dim, n_outputs, dtype=torch.float64) * 0.1
    a = torch.randint(0, input_dim, (n_tables, n_anchor_pairs))
    b = (a + 1 + torch.randint(0, input_dim - 1, (n_tables, n_anchor_pairs))) % input_dim
    powers = (2 ** torch.arange(n_anchor_pairs - 1, -1, -1)).to(torch.int64)
    T_soft = torch.tensor(0.5, dtype=torch.float64)
    T_sel = torch.tensor(0.5, dtype=torch.float64)

    out_g, mi_g, ai_g, u_g = _hybrid_smooth_fwd_gather(
        x, weights, a, b, powers, T_soft, T_sel, n_heads, tph, table_dim,
        forward_confidence=True, confidence_form=form)
    out_b, mi_b, ai_b, u_b = _hybrid_smooth_fwd_bmm(
        x, weights, a, b, powers, T_soft, T_sel, n_heads, tph, table_dim,
        weights.dtype, forward_confidence=True, confidence_form=form)

    assert torch.equal(mi_g, mi_b) and torch.equal(ai_g, ai_b)
    assert torch.allclose(u_g, u_b, atol=1e-12, rtol=0)
    err = (out_g - out_b).abs().max().item()
    assert err < 1e-10, f"{form}: bmm vs gather gated output max abs err {err:.2e}"


# =============================================================================
# (v) dtype smoke
# =============================================================================

@pytest.mark.parametrize("forward_mode", ["hard", "hybrid_smooth"])
@pytest.mark.parametrize("form", _FORMS)
def test_fp32_cpu_dtype_smoke(forward_mode, form):
    """fp32 CPU end-to-end under the gate: output stays fp32, grads populated."""
    m = FastMultiHeadLut(
        input_dim=16, n_heads=2, n_outputs=4, n_anchor_pairs=3, tables_per_head=2,
        forward_mode=forward_mode, weight_dtype=torch.float32, use_bf16=False,
        forward_confidence=True, confidence_form=form, random_seed=0, device=_CPU)
    x = torch.randn(5, 16, dtype=torch.float32, requires_grad=True)
    y = m(x)
    assert y.shape == (5, 2, 4) and y.dtype == torch.float32
    y.sum().backward()
    assert x.grad is not None and m.weights.grad is not None
    assert m.weights.grad.dtype == torch.float32


@pytest.mark.skipif(not _HAS_CUDA, reason="CUDA required for bf16 autocast smoke")
@pytest.mark.parametrize("forward_mode", ["hard", "hybrid_smooth"])
def test_bf16_autocast_cuda_smoke(forward_mode):
    """bf16 autocast on CUDA under the gate: runs, output dtype == weight dtype."""
    dev = torch.device("cuda:0")
    m = FastMultiHeadLut(
        input_dim=32, n_heads=4, n_outputs=8, n_anchor_pairs=4, tables_per_head=4,
        forward_mode=forward_mode, weight_dtype=torch.float32, use_bf16=True,
        forward_confidence=True, confidence_form="bounded", random_seed=0, device=dev)
    x = torch.randn(9, 32, device=dev, dtype=torch.float32, requires_grad=True)
    y = m(x)
    assert y.shape == (9, 4, 8) and y.dtype == torch.float32
    y.sum().backward()
    assert x.grad is not None and m.weights.grad is not None
    assert m.weights.grad.dtype == torch.float32


# =============================================================================
# config guards
# =============================================================================

def test_guard_exp_outputs_incompatible():
    with pytest.raises(ValueError, match="exp_outputs"):
        FastMultiHeadLut(input_dim=8, n_heads=1, n_outputs=4, n_anchor_pairs=2,
                         forward_confidence=True, exp_outputs=True, use_bf16=False,
                         device=_CPU)


# =============================================================================
# confidence_gain: a constant multiplier that separates SCALE from SELECTIVITY
# =============================================================================

@pytest.mark.parametrize("forward_mode", ["hard", "hybrid_smooth"])
@pytest.mark.parametrize("form", _FORMS)
def test_gain_one_is_bit_identical(forward_mode, form):
    """The default gain=1.0 must not perturb ANY existing number, forward or backward."""
    kw = dict(forward_mode=forward_mode, forward_confidence=True,
              confidence_form=form, seed=5)
    m_def = _make(**kw)                       # gain absent -> defaults to 1.0
    m_one = _make(confidence_gain=1.0, **kw)  # explicitly 1.0
    torch.manual_seed(1)
    x = torch.randn(5, m_one.input_dim, dtype=torch.float64) * 3.0
    g = torch.randn(5, m_one.n_heads, m_one.n_outputs, dtype=torch.float64)

    xd = x.clone().requires_grad_(True)
    m_def(xd).backward(g)
    xo = x.clone().requires_grad_(True)
    m_one(xo).backward(g)

    assert torch.equal(m_def.weights.grad, m_one.weights.grad)
    assert torch.equal(xd.grad, xo.grad)


@pytest.mark.parametrize("form", _FORMS)
def test_gain_scales_the_output_exactly(form):
    """The gate multiplies every gathered row, so gain c scales the WHOLE output by c.

    That exactness is the point: it makes the gain absorbable into the linear decompress
    downstream, so it changes the optimisation problem (how big the FFN's contribution is
    per unit of decompress norm) without changing what the model can express.
    """
    c = 7.5
    base = _make(forward_confidence=True, confidence_form=form, seed=3)
    gained = _make(forward_confidence=True, confidence_form=form, seed=3,
                   confidence_gain=c)
    torch.manual_seed(2)
    x = torch.randn(6, base.input_dim, dtype=torch.float64) * 3.0
    with torch.no_grad():
        y0, y1 = base(x), gained(x)
    err = (y1 - c * y0).abs().max().item()
    assert err < 1e-12, f"{form}: gain did not scale the output exactly ({err:.2e})"


@pytest.mark.parametrize("form", _FORMS)
def test_gain_dscore_matches_finite_diff(form):
    """With a gain the analytic dscore/dm is still the exact derivative (not just scaled)."""
    c = 3.25
    torch.manual_seed(0)
    d = torch.rand(5, 4, 3, dtype=torch.float64) * 2.0 + 0.1
    _s, dscore_dm = _confidence_score_and_dscore(d, form, c)

    num = torch.zeros_like(d)
    flat, nflat = d.reshape(-1), num.reshape(-1)
    eps = 1e-7
    for i in range(flat.numel()):
        orig = flat[i].item()
        flat[i] = orig + eps
        sp = _confidence_score(d, form, c)
        flat[i] = orig - eps
        sm = _confidence_score(d, form, c)
        flat[i] = orig
        b, rem = divmod(i, 4 * 3)
        t, j = divmod(rem, 3)
        nflat[i] = (sp[b, t] - sm[b, t]) / (2.0 * eps)
    err = (dscore_dm - num).abs().max().item()
    assert err < 1e-4, f"{form}: gained dscore/dm max abs err {err:.2e}"


def test_gain_grads_scale_linearly():
    """Table/decompress-side gradients scale with the gain; this is what the arm A
    diagnosis said was 19x too small under "bounded"."""
    c = 4.0
    torch.manual_seed(0)
    base = _make(forward_confidence=True, confidence_form="bounded_norm", seed=3)
    gained = _make(forward_confidence=True, confidence_form="bounded_norm", seed=3,
                   confidence_gain=c)
    x = torch.randn(6, base.input_dim, dtype=torch.float64) * 3.0
    g = torch.randn(6, base.n_heads, base.n_outputs, dtype=torch.float64)
    base(x.clone().requires_grad_(True)).backward(g)
    gained(x.clone().requires_grad_(True)).backward(g)
    ratio = (gained.weights.grad.norm() / base.weights.grad.norm()).item()
    assert abs(ratio - c) < 1e-9, f"grad_W ratio {ratio} != gain {c}"


def test_guard_bad_confidence_gain():
    for bad in (0.0, -1.0):
        with pytest.raises(ValueError, match="confidence_gain"):
            FastMultiHeadLut(input_dim=8, n_heads=1, n_outputs=4, n_anchor_pairs=2,
                             forward_confidence=True, confidence_gain=bad, device=_CPU)


def test_guard_bad_confidence_form():
    with pytest.raises(ValueError, match="confidence_form"):
        FastMultiHeadLut(input_dim=8, n_heads=1, n_outputs=4, n_anchor_pairs=2,
                         confidence_form="nope", device=_CPU)
