"""Tests for LightMultiHeadLUT(forward_mode="hard") -- ablation rows 3.3 (n=1) and 3.4 (n=2).

Forward: the plain hard read sum_t W_t[c_t], no score. Training output is `W[c] + f - sg(f)` with f the scored read
(3.1's / 3.2's function) computed on DETACHED tables: zero in value, so the output is exactly the hard read, while the
input, beta/gamma and tau receive f's gradient and the tables receive only the hard read's (unscaled, 1 row).
Since b5b0f2c3 (the Gen-3 hard-mode divergence fix) f scores with the BOUNDED score s / (sg(mean_bag s) + 1e-6), the
mean taken over the tables that reduce into each bag, so the injected gradient cannot grow with |d|. The references
below are therefore the scored module with that per-bag mean frozen at the evaluation point (== its detach).

Coverage:
  (i)   forward value == a plain-torch hard read, exactly, in train and in eval, both input layouts, n=1 and n=2;
  (ii)  eval (no_grad) never computes the confidence score and never uses the native SCORED kernel;
  (iii) input / confidence-scalar / tau gradients == the bounded-score scored module's (same state), and differ from
        the unbounded scored module's; the table gradient == the autograd gradient of the plain hard read (unscaled,
        one row per table);
  (iv)  float64: torch.autograd.gradcheck of the tables, and central differences of the bounded-score scored function
        (index and per-bag mean fixed) for the input gradient;
  (v)   guards (unknown mode; non-constant cell modes) and default/state_dict compatibility.
"""
import pytest
import torch

torch._dynamo.config.suppress_errors = True

from spiky.lutorch.light_multi_head_lut import LightMultiHeadLUT

NAP, TPH, H, IN, OUT, B = 4, 3, 2, 8, 5, 6
_DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def _make(forward_mode="scored", mhi=False, n=1, form="margin", device="cpu", dtype=torch.float64):
    kw = {}
    if form == "learned_margin":
        kw.update(learned_margin_init=(0.0, 2.0, 1.0), learned_margin_freeze_g=True)
    m = LightMultiHeadLUT(
        input_dim=IN, n_tables=H * TPH, output_dim=OUT, n_anchor_pairs=NAP, confidence_form=form,
        random_seed=0, initial_weights_noise=0.5, device=torch.device(device), n_heads=H, multi_head_input=mhi,
        read_top_n=n, read_tau=0.5, read_tau_learnable=(n > 1), forward_mode=forward_mode, **kw)
    return m.to(dtype)


def _x(m, device="cpu", dtype=torch.float64, seed=0):
    g = torch.Generator().manual_seed(seed)
    shape = (B, H, IN) if m.multi_head_input else (B, IN)
    return torch.randn(*shape, generator=g, dtype=torch.float64).to(device=device, dtype=dtype)


def _margins(m, x):
    if m.multi_head_input:
        a = m.anchor_a.reshape(1, H, TPH * NAP).expand(x.shape[0], H, TPH * NAP)
        b = m.anchor_b.reshape(1, H, TPH * NAP).expand(x.shape[0], H, TPH * NAP)
        return (torch.gather(x, 2, a) - torch.gather(x, 2, b)).view(x.shape[0], H, TPH, NAP)
    return x[:, m.anchor_a] - x[:, m.anchor_b]


def _plain(m, x, W=None):
    """Plain hard read: detached sign address (MSB-first), one row per table, summed per head / overall."""
    W = m.tables if W is None else W
    d = _margins(m, x).detach()
    idx = ((d > 0).long() * m.powers).sum(-1)
    K = 1 << NAP
    if m.multi_head_input:
        Wv = W.view(H, TPH, K, OUT)
        rows = Wv[torch.arange(H, device=x.device).view(1, H, 1), torch.arange(TPH, device=x.device).view(1, 1, TPH), idx]
        return rows.sum(2)
    rows = W[torch.arange(H * TPH, device=x.device).view(1, -1), idx]
    return rows.sum(1)


def _bound_score_at(soft, x):
    """Make `soft` score like the hard forward's surrogate (b5b0f2c3): s / (mean_bag s + 1e-6), the per-bag mean over the
    tables (last axis of the score) frozen at x -- the constant the hard path's detach sees, so gradients at x agree and
    central differences around x measure the same function."""
    score = soft.confidence_score
    with torch.no_grad():
        mean0 = score(_margins(soft, x)).mean(dim=-1, keepdim=True) + 1e-6
    soft.confidence_score = lambda d: score(d) / mean0
    return soft


def _grads(m, x, g):
    x = x.clone().requires_grad_(True)
    m.zero_grad(set_to_none=True)
    (m(x) * g).sum().backward()
    named = {k: p.grad for k, p in m.named_parameters()}
    return x.grad, named


CASES = [(mhi, n, form) for mhi in (False, True) for n in (1, 2) for form in ("margin", "learned_margin")]


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("mhi,n,form", CASES)
def test_forward_value_is_the_plain_hard_read(device, mhi, n, form):
    dt = torch.float64 if device == "cpu" else torch.float32
    m = _make("hard", mhi, n, form, device, dt)
    x = _x(m, device, dt).requires_grad_(True)
    out_train = m(x)
    with torch.no_grad():
        out_eval = m(x)
    # training output = plain + (f - sg(f)) must equal the eval output (plain alone) EXACTLY: the term is zero
    assert torch.equal(out_train, out_eval)
    tol = dict(rtol=0.0, atol=1e-12 if dt == torch.float64 else 1e-5)
    torch.testing.assert_close(out_train.detach(), _plain(m, x).detach(), **tol)


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("mhi,n", [(False, 1), (False, 2), (True, 1), (True, 2)])
def test_eval_path_is_unscored(device, mhi, n):
    dt = torch.float64 if device == "cpu" else torch.float32
    m = _make("hard", mhi, n, "margin", device, dt)

    def boom(*a, **k):
        raise AssertionError("a hard forward must not score at eval")

    m.confidence_score = boom
    m._fused_eval = boom
    x = _x(m, device, dt)
    with torch.no_grad():
        tol = dict(rtol=0.0, atol=1e-12 if dt == torch.float64 else 1e-5)
        torch.testing.assert_close(m(x), _plain(m, x), **tol)


@pytest.mark.parametrize("mhi,n,form", CASES)
def test_input_and_scalar_grads_equal_scored_path_and_table_grad_is_plain(mhi, n, form):
    hard, soft, unbounded = _make("hard", mhi, n, form), _make("scored", mhi, n, form), _make("scored", mhi, n, form)
    soft.load_state_dict(hard.state_dict())
    unbounded.load_state_dict(hard.state_dict())
    x = _x(hard)
    _bound_score_at(soft, x)
    g = torch.randn(hard(x).shape, generator=torch.Generator().manual_seed(1), dtype=torch.float64)
    hx, hp = _grads(hard, x, g)
    sx, sp = _grads(soft, x, g)
    ux, _ = _grads(unbounded, x, g)
    torch.testing.assert_close(hx, sx)
    assert not torch.allclose(hx, ux)                              # the bound is really in the hard backward
    for k in hp:
        if k == "tables":
            continue
        torch.testing.assert_close(hp[k], sp[k], msg=k)
    W = hard.tables.detach().clone().requires_grad_(True)
    (ref,) = torch.autograd.grad((_plain(hard, x, W) * g).sum(), (W,))
    torch.testing.assert_close(hp["tables"], ref)
    assert not torch.allclose(hp["tables"], sp["tables"])          # the scored path's table grad is score-scaled


@pytest.mark.parametrize("mhi,n", [(False, 1), (True, 2)])
def test_gradcheck_float64(mhi, n):
    m = _make("hard", mhi, n, "learned_margin")
    x = _x(m)
    W0 = m.tables.detach().clone().requires_grad_(True)
    assert torch.autograd.gradcheck(lambda W: torch.func.functional_call(m, {"tables": W}, (x,)), (W0,))
    # input gradient == central differences of the bounded-score scored function (tables constant, index and per-bag
    # mean fixed for small eps)
    soft = _make("scored", mhi, n, "learned_margin")
    soft.load_state_dict(m.state_dict())
    _bound_score_at(soft, x)
    g = torch.randn(m(x).shape, generator=torch.Generator().manual_seed(2), dtype=torch.float64)
    hx, _ = _grads(m, x, g)
    eps, num = 1e-6, torch.zeros_like(x)
    flat = x.view(-1)
    with torch.no_grad():
        for i in range(flat.numel()):
            orig = flat[i].item()
            flat[i] = orig + eps
            fp = (soft(x) * g).sum().item()
            flat[i] = orig - eps
            fm = (soft(x) * g).sum().item()
            flat[i] = orig
            num.view(-1)[i] = (fp - fm) / (2 * eps)
    torch.testing.assert_close(hx, num, rtol=1e-5, atol=1e-7)


def test_guards_and_compatibility():
    with pytest.raises(ValueError, match="forward_mode"):
        _make("soft")
    with pytest.raises(ValueError, match="hard"):
        LightMultiHeadLUT(input_dim=IN, n_tables=IN, output_dim=IN, n_anchor_pairs=NAP, cell_mode="gated_affine",
                          forward_mode="hard")
    assert LightMultiHeadLUT(input_dim=IN, n_tables=4, output_dim=OUT, n_anchor_pairs=NAP).forward_mode == "scored"
    assert list(_make("hard").state_dict()) == list(_make("scored").state_dict())
