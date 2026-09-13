"""Tests for FastMultiHeadLut(forward_mode="hard", backward_topk > 0).

Forward: the hard read, unchanged. Backward: the x and log-temperature gradients come from the
full-K soft surrogate RESTRICTED to the kept R = 1 + backward_topk rows ({chosen row} + its
backward_topk least-|d| 1-bit-flip neighbours) and renormalised over them; the weight gradient is
the hard forward's own 1-row gradient (score-scaled under forward_confidence, whose score->x term is
added as on the full-K hard path). backward_topk=1 is row 2.3 of the LUT ablation table.

Coverage:
  (i)   config guards: hard + topk accepted; bf16 weight storage refused (at init, and at call time
        after a runtime flip hybrid_smooth -> hard).
  (ii)  forward unchanged: output == hard / backward_topk=0, train and eval.
  (iii) x and log-temperature grads == autograd of a plain-torch restricted-softmax reference
        (address and kept set held fixed, table weights constant), k in {1, 2, NAP}, with and
        without forward_confidence, single- and multi-head input.
  (iv)  weight grad == autograd of the hard forward (and == hard / backward_topk=0's).
  (v)   float64 central finite differences: of the reference surrogate for x and both
        log-temperatures, of the hard forward for the weights.
  (vi)  k=1: x and log-temperature grads == hybrid_smooth / backward_topk=1's (the same 2-cell
        softmax, whose weight on the alternative is hybrid_smooth's blend weight).
  (vii) runtime flip hybrid_smooth(backward_topk=1) -> hard trains with the sparse backward.
"""
import pytest
import torch

# Compiled bodies fall back to eager on CPU/float64 instead of erroring.
torch._dynamo.config.suppress_errors = True

from spiky.lutorch.fast_multi_head_lut import FastMultiHeadLut, _confidence_score

_DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
_NAP = 4
_H, _TPH, _IN, _OUT, _B = 2, 3, 12, 5, 4


def _make(*, forward_mode="hard", backward_topk=1, forward_confidence=False,
          confidence_form="margin", multi_head_input=False, weight_dtype=torch.float64,
          device="cpu"):
    return FastMultiHeadLut(
        input_dim=_IN, n_heads=_H, n_outputs=_OUT, n_anchor_pairs=_NAP, tables_per_head=_TPH,
        forward_mode=forward_mode, backward_topk=backward_topk, weight_dtype=weight_dtype,
        use_bf16=False, forward_confidence=forward_confidence, confidence_form=confidence_form,
        multi_head_input=multi_head_input, random_seed=3, initial_weights_noise=0.5,
        device=torch.device(device),
    )


def _inputs(m, device, seed=0):
    gen = torch.Generator().manual_seed(seed)
    in_dim = _IN * _H if m.multi_head_input else _IN
    x = torch.randn(_B, in_dim, generator=gen, dtype=torch.float64).to(device)
    g = torch.randn(_B, _H, _OUT, generator=gen, dtype=torch.float64).to(device)
    return x, g


def _module_grads(m, x, g):
    x = x.clone().requires_grad_(True)
    m.zero_grad(set_to_none=True)
    out = m(x)
    (out * g).sum().backward()
    return (out.detach(), x.grad, m.weights.grad, m.log_soft_score_temp.grad,
            m.log_select_temp.grad)


# --- plain-torch references ---------------------------------------------------------------------

def _margins(m, x):
    return x[:, m.soft_anchor_a_long] - x[:, m.soft_anchor_b_long]            # [B, nt, NAP]


def _routing(m, x, k):
    """Hard address and the kept row set, both from detached margins (held fixed)."""
    d = _margins(m, x).detach()
    powers = m.soft_powers.view(1, 1, -1)
    index = ((d > 0).to(torch.int64) * powers).sum(-1)                        # [B, nt]
    if k >= _NAP:
        pos = torch.arange(_NAP, device=d.device).view(1, 1, -1).expand(*index.shape, -1)
    else:
        pos = d.abs().topk(k, dim=-1, largest=False).indices                  # [B, nt, k]
    alt = index.unsqueeze(-1) ^ m.soft_powers[pos]
    return index, torch.cat([index.unsqueeze(-1), alt], dim=-1)               # [B, nt, R]


def _signs(idx):
    shifts = torch.arange(_NAP - 1, -1, -1, device=idx.device)
    return ((idx.unsqueeze(-1) >> shifts) & 1).to(torch.float64) * 2.0 - 1.0


def _surrogate(m, x, log_Ts, log_Tsel, index, kept):
    """Softmax of the full-K surrogate over the kept rows only, weights constant."""
    d = _margins(m, x)
    Ts, Tsel = log_Ts.exp(), log_Tsel.exp()
    p = _signs(index) * d.abs() / (Ts + d.abs())                              # [B, nt, NAP]
    ts = (p.unsqueeze(-2) * _signs(kept)).sum(-1)                             # [B, nt, R]
    sel = torch.softmax(ts / Tsel, dim=-1)
    nt = m.weights.shape[0]
    rows = m.weights.detach()[torch.arange(nt, device=x.device).view(1, nt, 1), kept]
    y = (sel.unsqueeze(-1) * rows).sum(-2)                                    # [B, nt, out]
    return y.view(x.shape[0], _H, _TPH, _OUT).sum(2)


def _hard(m, x, W, index, score=None):
    nt = W.shape[0]
    rows = W[torch.arange(nt, device=x.device).view(1, nt), index]           # [B, nt, out]
    if score is not None:
        rows = rows * score.unsqueeze(-1)
    return rows.view(x.shape[0], _H, _TPH, _OUT).sum(2)


def _reference_grads(m, x, g, k):
    index, kept = _routing(m, x, k)
    xr = x.clone().requires_grad_(True)
    lts = m.log_soft_score_temp.detach().clone().requires_grad_(True)
    ltl = m.log_select_temp.detach().clone().requires_grad_(True)
    loss = (_surrogate(m, xr, lts, ltl, index, kept) * g).sum()
    if m.forward_confidence:
        # the score multiplier's own input gradient, added to the surrogate's
        score = _confidence_score(_margins(m, xr), m.confidence_form, m.confidence_gain)
        loss = loss + (_hard(m, xr, m.weights.detach(), index, score) * g).sum()
    gx, gts, gtl = torch.autograd.grad(loss, (xr, lts, ltl))
    W = m.weights.detach().clone().requires_grad_(True)
    score = (_confidence_score(_margins(m, x), m.confidence_form, m.confidence_gain)
             if m.forward_confidence else None)
    (gw,) = torch.autograd.grad((_hard(m, x, W, index, score) * g).sum(), (W,))
    return gx, gw, gts, gtl


# --- (i) config guards --------------------------------------------------------------------------

def test_hard_with_topk_is_accepted_and_range_still_checked():
    for k in (1, 2, _NAP):
        assert _make(backward_topk=k).backward_topk == k
    with pytest.raises(ValueError, match="backward_topk must be in"):
        _make(backward_topk=_NAP + 1)


def test_hard_topk_refuses_bf16_storage_but_hybrid_topk_still_builds():
    with pytest.raises(ValueError, match=r"fp32 \(or float64\) weight"):
        _make(backward_topk=1, weight_dtype=torch.bfloat16)
    _make(forward_mode="hybrid_smooth", backward_topk=1, weight_dtype=torch.bfloat16)
    _make(forward_mode="hard", backward_topk=0, weight_dtype=torch.bfloat16)


def test_runtime_flip_to_hard_refuses_bf16_storage():
    m = _make(forward_mode="hybrid_smooth", backward_topk=1, weight_dtype=torch.bfloat16)
    m.forward_mode = "hard"
    x = torch.randn(_B, _IN, dtype=torch.float32)
    with pytest.raises(ValueError, match=r"fp32 \(or float64\) weight"):
        m(x)


# --- (ii) forward unchanged ---------------------------------------------------------------------

@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("forward_confidence", [False, True])
def test_forward_output_matches_topk0(device, forward_confidence):
    m0 = _make(backward_topk=0, forward_confidence=forward_confidence, device=device)
    m1 = _make(backward_topk=2, forward_confidence=forward_confidence, device=device)
    m1.load_state_dict(m0.state_dict())
    x, _ = _inputs(m0, device)
    assert torch.equal(m0(x), m1(x))
    with torch.no_grad():
        assert torch.equal(m0(x), m1(x))


# --- (iii) + (iv) against the plain-torch reference ---------------------------------------------

@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("k", [1, 2, _NAP])
@pytest.mark.parametrize("gate", [None, "margin", "bounded_norm"])
@pytest.mark.parametrize("multi_head_input", [False, True])
def test_grads_match_restricted_softmax_reference(device, k, gate, multi_head_input):
    m = _make(backward_topk=k, forward_confidence=gate is not None, confidence_form=gate or "margin",
              multi_head_input=multi_head_input, device=device)
    x, g = _inputs(m, device, seed=k)
    _, gx, gw, gts, gtl = _module_grads(m, x, g)
    rx, rw, rts, rtl = _reference_grads(m, x, g, k)
    torch.testing.assert_close(gx, rx)
    torch.testing.assert_close(gw, rw)
    torch.testing.assert_close(gts, rts)
    torch.testing.assert_close(gtl, rtl)


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("forward_confidence", [False, True])
def test_weight_grad_equals_full_k_hard_path(device, forward_confidence):
    m0 = _make(backward_topk=0, forward_confidence=forward_confidence, device=device)
    m1 = _make(backward_topk=1, forward_confidence=forward_confidence, device=device)
    m1.load_state_dict(m0.state_dict())
    x, g = _inputs(m0, device, seed=5)
    torch.testing.assert_close(_module_grads(m1, x, g)[2], _module_grads(m0, x, g)[2])


# --- (v) finite differences ---------------------------------------------------------------------

def _central_diff(f, t, eps=1e-6):
    grad = torch.zeros_like(t)
    flat, gflat = t.view(-1), grad.view(-1)
    for i in range(flat.numel()):
        orig = flat[i].item()
        flat[i] = orig + eps
        fp = float(f())
        flat[i] = orig - eps
        fm = float(f())
        flat[i] = orig
        gflat[i] = (fp - fm) / (2 * eps)
    return grad


@pytest.mark.parametrize("k", [1, 3])
def test_finite_differences(k):
    m = _make(backward_topk=k)
    x, g = _inputs(m, "cpu", seed=11)
    _, gx, gw, gts, gtl = _module_grads(m, x, g)
    index, kept = _routing(m, x, k)                    # held fixed while perturbing
    # the log-temperatures are stored in fp32: step them in float64, or eps=1e-6 is lost to rounding
    lts = m.log_soft_score_temp.detach().to(torch.float64)
    ltl = m.log_select_temp.detach().to(torch.float64)
    xv = x.clone()

    def surrogate_loss():
        return (_surrogate(m, xv, lts, ltl, index, kept) * g).sum()

    tol = dict(rtol=1e-5, atol=1e-7)
    torch.testing.assert_close(gx, _central_diff(surrogate_loss, xv), **tol)
    torch.testing.assert_close(gts.double(), _central_diff(surrogate_loss, lts), **tol)
    torch.testing.assert_close(gtl.double(), _central_diff(surrogate_loss, ltl), **tol)
    W = m.weights.detach().clone()
    torch.testing.assert_close(gw, _central_diff(lambda: (_hard(m, x, W, index) * g).sum(), W), **tol)


# --- (vi) k=1 is hybrid_smooth's 2-cell softmax --------------------------------------------------

@pytest.mark.parametrize("device", _DEVICES)
def test_topk1_input_and_temperature_grads_equal_hybrid_smooth_topk1(device):
    mh = _make(forward_mode="hard", backward_topk=1, device=device)
    ms = _make(forward_mode="hybrid_smooth", backward_topk=1, device=device)
    ms.load_state_dict(mh.state_dict())
    x, g = _inputs(mh, device, seed=2)
    _, hx, _, hts, htl = _module_grads(mh, x, g)
    _, sx, _, sts, stl = _module_grads(ms, x, g)
    torch.testing.assert_close(hx, sx)
    torch.testing.assert_close(hts, sts)
    torch.testing.assert_close(htl, stl)


# --- (vii) runtime flip -------------------------------------------------------------------------

@pytest.mark.parametrize("device", _DEVICES)
def test_runtime_flip_hybrid_topk_to_hard_uses_sparse_backward(device):
    flipped = _make(forward_mode="hybrid_smooth", backward_topk=1, device=device)
    fresh = _make(forward_mode="hard", backward_topk=1, device=device)
    fresh.load_state_dict(flipped.state_dict())
    flipped.forward_mode = "hard"
    x, g = _inputs(fresh, device, seed=4)
    for a, b in zip(_module_grads(flipped, x, g), _module_grads(fresh, x, g)):
        assert torch.equal(a, b)
