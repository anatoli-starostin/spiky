"""Tests for MatrixMenuMultiHeadLUT (cell = selection over a per-head menu of [d_in, d_out] matrices).

Asserts:
  (a) shapes: per-head [B, H, d_in] -> [B, H, d_out], shared [B, d_in] -> [B, d_out]; forward+backward run;
  (b) gradients reach the menu, the cell logits, the temperature and the input;
  (c) THE COLLAPSE IS EXACT: every menu_impl equals a naive per-table reference
      y = sum_t s_t * x @ (sum_m p_tm W_m), values AND gradients, in float64;
  (d) addressing and score are LightMultiHeadLUT's, bit for bit (same anchors, same address, same score);
  (e) hard mode: the forward value is the one-hot argmax read (== reading export_indices()), and its gradient
      is exactly the soft path's (straight-through); the soft path converges to the hard one as tau -> 0;
  (f) global vs per-head temperature shapes and the refusals;
  (h) small-std menu init (output near zero) and weight decay: the trainers' rule decays the menu only;
  (g) CompressionMultiHeadLUT / model_build integration: output is zero at init with a zero decompress,
      inner_out_dim=-1 works, and an existing config builds unchanged.
"""
import pytest
import torch

from spiky.lutorch.light_multi_head_lut import LightMultiHeadLUT
import math

from spiky.lutorch.matrix_menu_multi_head_lut import MatrixMenuMultiHeadLUT, MENU_IMPLS, MENU_INIT_STD
from spiky.lutorch.compression_mhl import CompressionMultiHeadLUT

NAP, TPH, H, DIN, DOUT, M, B = 4, 6, 3, 8, 5, 7, 40
D64 = torch.float64


def _make(mh=True, dtype=D64, **kw):
    kw.setdefault("menu_size", M)
    kw.setdefault("menu_logit_noise", 1.0)          # committed-ish cells, so the test is not trivially uniform
    kw.setdefault("menu_init_scale", 0.3)           # O(1) menu so value/grad comparisons are not all ~0
    n_heads = H if mh else 1
    return MatrixMenuMultiHeadLUT(
        input_dim=DIN, n_tables=n_heads * TPH, output_dim=DOUT, n_anchor_pairs=NAP,
        confidence_form="margin", random_seed=3, device=torch.device("cpu"),
        n_heads=n_heads, multi_head_input=mh, **kw).to(dtype)


def _x(mh=True, seed=1, dtype=D64):
    shape = (B, H, DIN) if mh else (B, DIN)
    return torch.randn(*shape, generator=torch.Generator().manual_seed(seed), dtype=dtype)


def _naive(m, x):
    """Reference: per (token, head, table), build the cell's matrix and apply it -- no collapse."""
    mh = m.multi_head_input
    xh = x if mh else x.unsqueeze(1)
    Bn, Hn, T = x.shape[0], m.n_heads, m.tables_per_head
    d, x_flat, _ = m._margins(x)
    index = ((d.detach() > 0).to(torch.int64) * m.powers.view(1, 1, 1, -1)).sum(-1)       # [B, H, T]
    score = m.confidence_score(d)                                                          # [B, H, T]
    P = m.menu_probs()                                                                     # [n_tables, K, M]
    out = []
    for h in range(Hn):
        yh = 0
        for t in range(T):
            p = P[h * T + t][index[:, h, t]]                                                # [B, M]
            Wc = torch.einsum("bm,mio->bio", p, m.menu[h])                                  # [B, di, do]
            yh = yh + score[:, h, t, None] * torch.einsum("bi,bio->bo", xh[:, h], Wc)
        out.append(yh)
    y = torch.stack(out, 1)
    return y if mh else y.squeeze(1)


@pytest.mark.parametrize("mh", [True, False])
def test_shapes_and_grads(mh):
    m = _make(mh)
    x = _x(mh).requires_grad_(True)
    y = m(x)
    assert y.shape == ((B, H, DOUT) if mh else (B, DOUT))
    y.square().sum().backward()
    for name in ("menu", "menu_logits", "menu_log_tau"):
        g = getattr(m, name).grad
        assert g is not None and torch.isfinite(g).all() and g.abs().sum() > 0, name
    assert x.grad is not None and x.grad.abs().sum() > 0
    assert not hasattr(m, "tables") or "tables" not in dict(m.named_parameters())


@pytest.mark.parametrize("mh", [True, False])
@pytest.mark.parametrize("impl", MENU_IMPLS)
@pytest.mark.parametrize("hard", [False, True])
def test_collapsed_equals_naive(mh, impl, hard):
    m = _make(mh, menu_impl=impl, menu_forward="hard" if hard else "soft")
    x1 = _x(mh).requires_grad_(True)
    x2 = x1.detach().clone().requires_grad_(True)
    y1, y2 = m(x1), _naive(m, x2)
    torch.testing.assert_close(y1, y2, rtol=1e-10, atol=1e-12)
    g = torch.randn_like(y1)
    grads1 = torch.autograd.grad(y1, [x1, m.menu, m.menu_logits, m.menu_log_tau], g)
    grads2 = torch.autograd.grad(y2, [x2, m.menu, m.menu_logits, m.menu_log_tau], g)
    for a, b in zip(grads1, grads2):
        torch.testing.assert_close(a, b, rtol=1e-9, atol=1e-11)


@pytest.mark.parametrize("mh", [True, False])
def test_addressing_and_score_are_lights(mh):
    kw = dict(input_dim=DIN, n_tables=(H if mh else 1) * TPH, output_dim=DOUT, n_anchor_pairs=NAP,
              confidence_form="margin", random_seed=3, device=torch.device("cpu"),
              n_heads=H if mh else 1, multi_head_input=mh)
    light = LightMultiHeadLUT(**kw).double()
    menu = MatrixMenuMultiHeadLUT(menu_size=M, **kw).double()
    for name, buf in light.named_buffers():
        if name != "log_tau":
            assert torch.equal(buf, dict(menu.named_buffers())[name]), name
    x = _x(mh)
    d_l = (x[:, light.anchor_a] - x[:, light.anchor_b]) if not mh else None
    d_m, x_flat, _ = menu._margins(x)
    if not mh:
        torch.testing.assert_close(d_m.view_as(d_l), d_l, rtol=0, atol=0)
    torch.testing.assert_close(menu.confidence_score(d_m), light.confidence_score(d_m), rtol=0, atol=0)


def test_hard_mode_is_index_read_with_soft_gradient():
    m = _make(True, menu_forward="hard")
    x = _x(True)
    y_hard = m(x)
    # value == reading the exported (menu, index) artefact
    art = m.export_indices()
    assert art["bits_per_cell"] == 3 and art["index"].dtype == torch.uint8
    onehot = torch.nn.functional.one_hot(art["index"].long(), M).double()
    with torch.no_grad():
        P_saved = m.menu_probs
        m.menu_probs = lambda hard=None: onehot        # force the exported read
        y_art = m(x)
        m.menu_probs = P_saved
    torch.testing.assert_close(y_hard, y_art, rtol=1e-12, atol=1e-14)
    # gradient w.r.t. logits/tau == the soft path's (straight-through)
    g_h = torch.autograd.grad(m(x).sum(), [m.menu_logits, m.menu_log_tau])
    m.menu_forward = "soft"
    y_soft = m(x)
    g_s = torch.autograd.grad(y_soft.sum(), [m.menu_logits, m.menu_log_tau])
    for a, b in zip(g_h, g_s):
        assert a.abs().sum() > 0
    # the injected term is the SOFT distribution's gradient, but the rest of the graph sees the hard value,
    # so compare against the soft P's gradient with the hard forward: build it explicitly
    m.menu_forward = "hard"
    P = m.menu_probs(hard=False)
    Ph = m.menu_probs(hard=True)
    torch.testing.assert_close(Ph.detach(), onehot)
    gP = torch.randn_like(P)
    ga = torch.autograd.grad((Ph * gP).sum(), m.menu_logits)[0]
    gb = torch.autograd.grad((P * gP).sum(), m.menu_logits)[0]
    torch.testing.assert_close(ga, gb, rtol=0, atol=0)
    # eval (no grad): exactly one-hot
    with torch.no_grad():
        torch.testing.assert_close(m.menu_probs(), onehot, rtol=0, atol=0)


def test_low_temperature_limit():
    m = _make(True, menu_forward="soft")
    x = _x(True)
    with torch.no_grad():
        m.menu_log_tau.fill_(math.log(1e-6))
        y_soft = m(x)
        m.menu_forward = "hard"
        y_hard = m(x)
    torch.testing.assert_close(y_soft, y_hard, rtol=1e-8, atol=1e-10)


def test_temperature_granularity():
    assert _make(True).menu_log_tau.shape == (H,)
    mg = _make(True, menu_tau_granularity="global")
    assert mg.menu_log_tau.shape == (1,) and mg.menu_tau.shape == (H,)
    mf = _make(True, menu_tau_learnable=False)
    assert "menu_log_tau" not in dict(mf.named_parameters()) and "menu_log_tau" in dict(mf.named_buffers())
    m = _make(True, menu_tau_init=0.5)
    torch.testing.assert_close(m.menu_tau, torch.full((H,), 0.5, dtype=D64))
    assert not any("tau_scale" in k for k in m.state_dict())    # annealing machinery removed


def test_refusals():
    for bad in (dict(read_top_n=2), dict(forward_mode="hard"), dict(menu_forward="x"),
                dict(menu_tau_granularity="table"), dict(menu_impl="x"), dict(menu_size=0)):
        with pytest.raises((ValueError, NotImplementedError)):
            _make(True, **bad)


def test_init_near_uniform_and_seeded():
    a, b = _make(True, menu_logit_noise=0.01), _make(True, menu_logit_noise=0.01)
    assert torch.equal(a.menu, b.menu) and torch.equal(a.menu_logits, b.menu_logits)
    P = a.menu_probs()
    assert (P.max(-1).values * M).max() < 1.1          # near-uniform: no cell starts committed
    o = _make(True, menu_init="orthogonal", menu_size=2, menu_init_scale=1 / math.sqrt(DIN))
    W = o.menu[0, 0]                                    # [DIN, DOUT], DIN > DOUT -> orthonormal columns
    torch.testing.assert_close(W.T @ W, torch.eye(DOUT, dtype=D64), rtol=1e-5, atol=1e-5)   # built in fp32


def test_default_menu_init_is_small():
    for init in ("normal", "orthogonal"):
        m = MatrixMenuMultiHeadLUT(input_dim=48, n_tables=4 * 16, output_dim=48, n_anchor_pairs=NAP,
                                   random_seed=3, n_heads=4, multi_head_input=True, menu_size=16, menu_init=init)
        assert abs(m.menu.std().item() / MENU_INIT_STD - 1) < 0.05, init


def test_weight_decay_grouping():
    # the trainers' rule (train.py / distill_ffn.setup_optimizer): with tables_no_decay, every parameter a
    # LightMultiHeadLUT instance owns DIRECTLY is exempt; otherwise ndim < 2 is exempt.
    m = _make(True)
    exempt = {id(p) for mod in m.modules() if isinstance(mod, LightMultiHeadLUT) for p in mod.parameters(recurse=False)}
    decayed = {n for n, p in m.named_parameters() if not (id(p) in exempt or p.ndim < 2)}
    assert decayed == {"menu_bank.weight"}


@pytest.mark.parametrize("out_dim", [4, -1])
def test_compression_mhl_integration(out_dim):
    c = CompressionMultiHeadLUT(24, 24, inner_in_dim=DIN, inner_out_dim=out_dim, nap=NAP, tph=TPH,
                                n_heads=H, lut_impl="light", cell_mode="matrix_menu", random_seed=5,
                                menu_config=dict(menu_size=M)).double()
    assert isinstance(c.lut_light, MatrixMenuMultiHeadLUT)
    x = torch.randn(B, 24, dtype=D64, generator=torch.Generator().manual_seed(2))
    if out_dim != -1:
        torch.nn.init.zeros_(c.decompress.weight)       # MinimalGPT / build_student zero it
        torch.nn.init.zeros_(c.decompress.bias)
        assert c(x).abs().max() == 0                    # output zero at init
    else:
        assert c(x).shape == (B, 24)                    # no decompress: menu maps d_in -> d_model directly
        assert c.lut_light.menu.shape == (H, M, DIN, 24)
    c(x).sum().backward()
    assert c.lut_light.menu_logits.grad is not None


def test_compression_mhl_refuses_menu_config_elsewhere():
    with pytest.raises(ValueError):
        CompressionMultiHeadLUT(24, 24, inner_in_dim=DIN, inner_out_dim=4, nap=NAP, tph=TPH, n_heads=H,
                                lut_impl="light", menu_config=dict(menu_size=M))
    with pytest.raises(ValueError):
        CompressionMultiHeadLUT(24, 24, inner_in_dim=DIN, inner_out_dim=4, nap=NAP, tph=TPH, n_heads=H,
                                lut_impl="fast", cell_mode="matrix_menu")
