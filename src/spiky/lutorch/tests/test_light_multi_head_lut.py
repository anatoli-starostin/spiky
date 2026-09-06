"""Tests for LightMultiHeadLUT -- the pure-autograd LUT control layer.

LightMultiHeadLUT uses our anchor-pair routing geometry with LookupFFN's
backward: one forward, the hard sign address fully detached (no STE), and the
only gradient to x flowing through the differentiable confidence score. These
tests pin exactly that behaviour and contrast it with a surrogate model:

  (1) forward runs; output shape [B, output_dim]; both confidence forms.
  (2) train() == eval() (allclose) -- the gate is applied identically.
  (3) pure-autograd correctness: x.grad and tables.grad match central finite
      differences (float64, margins off-boundary), for both forms.
  (4) torch.autograd.gradcheck passes -- the whole forward is honestly
      differentiable (this is the point of the pure-autograd design; the
      surrogate FastMultiHeadLut would fail an equivalent gradcheck).
  (5) NO STE: detaching the code/index path changes nothing -- x.grad equals the
      grad of an otherwise-identical forward whose index is NOT detached (the
      sign/pack is non-differentiable either way), proving zero STE contribution.
  (6) ZERO directional routing gradient: with very large margins the bounded
      score saturates to 1 and its slope vanishes, so x.grad -> 0. There is no
      surrogate term keeping the routing direction in the graph.
  (7) weight gradient reaches ONLY the selected rows (full LookupFFN row grad).
  (8) fp32 CPU and torch.compile smoke.
  plus config guards.

Runs on CPU (nucstar has no GPU); float64 for the numeric checks.
"""
import pytest
import torch

torch._dynamo.config.suppress_errors = True  # compile falls back to eager on CPU

from spiky.lutorch.light_multi_head_lut import LightMultiHeadLUT
from spiky.lutorch.fast_multi_head_lut import _confidence_score

_CPU = torch.device("cpu")
_FORMS = ["bounded", "margin"]


def _make(*, input_dim=16, n_tables=6, output_dim=4, n_anchor_pairs=3,
          confidence_form="bounded", weight_dtype=torch.float64, seed=0):
    m = LightMultiHeadLUT(
        input_dim=input_dim, n_tables=n_tables, output_dim=output_dim,
        n_anchor_pairs=n_anchor_pairs, confidence_form=confidence_form,
        random_seed=seed, device=_CPU)
    return m.to(weight_dtype) if weight_dtype != torch.float32 else m


def _num_grad(f, t, eps=1e-6):
    """Central finite-difference gradient of scalar f() w.r.t. tensor t (in place)."""
    g = torch.zeros_like(t)
    flat, gflat = t.reshape(-1), g.reshape(-1)
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
# (1) forward / shape / forms
# =============================================================================

@pytest.mark.parametrize("form", _FORMS)
def test_forward_shape(form):
    m = _make(confidence_form=form)
    x = torch.randn(7, m.input_dim, dtype=torch.float64)
    y = m(x)
    assert y.shape == (7, m.output_dim)
    assert y.dtype == torch.float64


# =============================================================================
# (2) train == eval
# =============================================================================

@pytest.mark.parametrize("form", _FORMS)
def test_train_equals_eval(form):
    m = _make(confidence_form=form, seed=1)
    x = torch.randn(6, m.input_dim, dtype=torch.float64) * 3.0
    y_train = m(x.clone().requires_grad_(True))
    with torch.no_grad():
        y_eval = m(x)
    assert torch.allclose(y_train, y_eval, atol=1e-12, rtol=0)


# =============================================================================
# (3) pure-autograd grads match finite differences
# =============================================================================

@pytest.mark.parametrize("form", _FORMS)
def test_grads_match_finite_diff(form):
    m = _make(confidence_form=form, seed=2)
    torch.manual_seed(5)
    x = torch.randn(5, m.input_dim, dtype=torch.float64) * 5.0   # off decision boundaries

    xg = x.clone().requires_grad_(True)
    g = torch.randn(5, m.output_dim, dtype=torch.float64)
    (m(xg) * g).sum().backward()
    gx, gW = xg.grad.clone(), m.tables.grad.clone()

    def L():
        with torch.no_grad():
            return (m(x) * g).sum()

    ex = (gx - _num_grad(L, x)).abs().max().item()
    ew = (gW - _num_grad(L, m.tables.data)).abs().max().item()
    assert ex < 1e-5, f"{form}: x.grad vs numeric {ex:.2e}"
    assert ew < 1e-5, f"{form}: tables.grad vs numeric {ew:.2e}"


@pytest.mark.parametrize("form", _FORMS)
def test_gradcheck(form):
    """The full forward is honestly differentiable -> gradcheck passes."""
    m = _make(input_dim=10, n_tables=4, output_dim=3, n_anchor_pairs=2,
              confidence_form=form, seed=3)
    torch.manual_seed(9)
    # Large, well-separated inputs so a 1e-6 probe never flips the hard address.
    x = (torch.randn(3, m.input_dim, dtype=torch.float64) * 6.0).requires_grad_(True)
    assert torch.autograd.gradcheck(m, (x,), eps=1e-6, atol=1e-4, rtol=1e-3)


# =============================================================================
# (5) NO STE: detaching the index changes nothing
# =============================================================================

@pytest.mark.parametrize("form", _FORMS)
def test_no_ste_detaching_index_changes_nothing(form):
    """x.grad equals the grad of an identical forward whose index is NOT detached.

    LightMultiHeadLUT detaches the sign/pack; a reference that packs the SAME code
    from the non-detached margins must give a bit-identical input gradient, because
    the sign/pack carries no gradient either way. Any nonzero difference would be an
    STE contribution -- there is none.
    """
    m = _make(confidence_form=form, seed=4)
    x0 = torch.randn(5, m.input_dim, dtype=torch.float64) * 4.0

    # module (index detached internally)
    xa = x0.clone().requires_grad_(True)
    g = torch.randn(5, m.output_dim, dtype=torch.float64)
    (m(xa) * g).sum().backward()
    gx_module = xa.grad.clone()

    # Reference forward: index packed from the NON-detached margins. It uses the SAME
    # fused reduction the module uses (F.embedding_bag), so the ONLY difference between
    # the two is the detach -- which is what this test is about. Comparing against the
    # unfused `(rows * score).sum(1)` form instead would fail on summation order at the
    # last ulp and say nothing about STE leakage.
    xb = x0.clone().requires_grad_(True)
    d = xb[:, m.anchor_a] - xb[:, m.anchor_b]
    index = ((d > 0).to(torch.int64) * m.powers.view(1, 1, -1)).sum(dim=-1)  # non-detached input, still non-diff
    flat = m.tables.reshape(m.n_tables * m.table_size, m.output_dim)
    flat_idx = (index + m.table_offset.view(1, -1)).reshape(-1)
    score = _confidence_score(d, form)
    offsets = torch.arange(5, device=x0.device, dtype=torch.long) * m.n_tables
    out_ref = torch.nn.functional.embedding_bag(
        flat_idx, flat, offsets=offsets, mode="sum",
        per_sample_weights=score.reshape(-1).to(flat.dtype))
    m.tables.grad = None
    (out_ref * g).sum().backward()
    gx_ref = xb.grad.clone()

    assert torch.equal(gx_module, gx_ref), \
        f"{form}: detaching index changed x.grad (max diff {(gx_module - gx_ref).abs().max().item():.2e}) -> STE leak"


# =============================================================================
# (6) zero directional routing gradient
# =============================================================================

def test_zero_directional_routing_gradient():
    """The row/routing path contributes EXACTLY zero gradient to x.

    Decompose the input gradient into its two possible paths and check each,
    deterministically (no magnitude thresholds):
      - rows-only (score detached): x reaches the output only through the integer
        index, which is non-differentiable -> x.grad is None / exactly zero. This
        IS the "no directional routing gradient" statement.
      - score-only (rows detached): reproduces the module's full x.grad, i.e. the
        entire input gradient is the score path and nothing else.
    A surrogate model would instead have a nonzero rows/routing contribution.
    """
    m = _make(confidence_form="bounded", seed=6)
    x0 = torch.randn(5, m.input_dim, dtype=torch.float64) * 4.0
    g = torch.randn(5, m.output_dim, dtype=torch.float64)

    xa = x0.clone().requires_grad_(True)
    gx_module = torch.autograd.grad((m(xa) * g).sum(), xa)[0]

    def _rows_and_score(xx):
        d = xx[:, m.anchor_a] - xx[:, m.anchor_b]
        index = ((d.detach() > 0).to(torch.int64) * m.powers.view(1, 1, -1)).sum(dim=-1)
        flat = m.tables.reshape(m.n_tables * m.table_size, m.output_dim)
        rows = flat[(index + m.table_offset.view(1, -1)).reshape(-1)].view(5, m.n_tables, m.output_dim)
        return rows, _confidence_score(d, "bounded")

    # rows-only: score detached -> x only feeds the non-diff index -> zero grad
    xb = x0.clone().requires_grad_(True)
    rows_b, score_b = _rows_and_score(xb)
    gx_rows_only = torch.autograd.grad(((rows_b * score_b.detach().unsqueeze(-1)).sum(1) * g).sum(),
                                       xb, allow_unused=True)[0]
    assert gx_rows_only is None or gx_rows_only.abs().max().item() == 0.0, \
        "routing/row path leaked a nonzero input gradient"

    # score-only: rows detached -> reproduces the module's entire x.grad
    xc = x0.clone().requires_grad_(True)
    rows_c, score_c = _rows_and_score(xc)
    gx_score_only = torch.autograd.grad(((rows_c.detach() * score_c.unsqueeze(-1)).sum(1) * g).sum(), xc)[0]
    assert torch.allclose(gx_module, gx_score_only, atol=1e-12, rtol=0), \
        f"module x.grad is not exactly the score path (max diff {(gx_module - gx_score_only).abs().max().item():.2e})"


# =============================================================================
# (7) weight gradient reaches only the selected rows
# =============================================================================

def test_weight_grad_only_selected_rows():
    m = _make(confidence_form="bounded", seed=7)
    x = torch.randn(4, m.input_dim, dtype=torch.float64) * 4.0
    m(x).sum().backward()

    # rows actually addressed this batch
    d = x[:, m.anchor_a] - x[:, m.anchor_b]
    index = ((d.detach() > 0).to(torch.int64) * m.powers.view(1, 1, -1)).sum(dim=-1)  # [B, n_tables]
    touched = torch.zeros(m.n_tables, m.table_size, dtype=torch.bool)
    for t in range(m.n_tables):
        touched[t, index[:, t].unique()] = True

    row_has_grad = m.tables.grad.abs().sum(dim=-1) > 0   # [n_tables, table_size]
    # every gradful row must be a selected row (no grad leaks to un-addressed rows)
    assert torch.equal(row_has_grad & ~touched, torch.zeros_like(row_has_grad)), \
        "gradient leaked to rows that were never addressed"
    # and at least the selected rows got gradient (sanity)
    assert row_has_grad.any()


# =============================================================================
# (8) dtype / compile smoke
# =============================================================================

@pytest.mark.parametrize("form", _FORMS)
def test_fp32_and_compile_smoke(form):
    m = _make(confidence_form=form, weight_dtype=torch.float32, seed=0)
    x = torch.randn(5, m.input_dim, dtype=torch.float32, requires_grad=True)
    y = m(x)
    assert y.shape == (5, m.output_dim) and y.dtype == torch.float32
    y.sum().backward()
    assert x.grad is not None and m.tables.grad is not None

    cm = torch.compile(m)                     # must not raise (falls back to eager on CPU)
    y2 = cm(torch.randn(5, m.input_dim, dtype=torch.float32))
    assert y2.shape == (5, m.output_dim)


# =============================================================================
# config guards
# =============================================================================

def test_guard_bad_confidence_form():
    with pytest.raises(ValueError, match="confidence_form"):
        LightMultiHeadLUT(input_dim=8, n_tables=2, output_dim=4, n_anchor_pairs=2,
                          confidence_form="nope", device=_CPU)


def test_guard_bad_n_anchor_pairs():
    with pytest.raises(ValueError, match="n_anchor_pairs"):
        LightMultiHeadLUT(input_dim=8, n_tables=2, output_dim=4, n_anchor_pairs=0,
                          device=_CPU)
