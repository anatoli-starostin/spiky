"""Gen-3 QuantisedConfidenceLUT (p2_int8 quant-aware training).

The STE forward value is a staircase (round/clamp power-of-two), so it is not directly
gradcheckable; the correct target is the SMOOTH SURROGATE that carries the STE's gradient
(`_forward_surrogate`). We therefore (a) fp64-gradcheck the surrogate wrt input, β, γ (and τ for
n=2), and (b) assert the real STE forward's gradient equals the surrogate's — so the gradcheck'd
surrogate is exactly the quant-aware backward. Plus: quant-aware run matrix, fake-quant eval ==
int8 shift-add read (bit-exact), train == eval value, bf16 raises, both anchor modes, both
read_top_n.
"""
import pytest
import torch
import torch.nn.functional as F

from spiky.lutorch_ex import LUTSpec, QuantisedConfidenceLUT
from spiky.lutorch_ex.cartridges._fused_ops import _global_cells

PATTERNS = [(4, 4), (1, 4), (4, 1)]
DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def _normrel(a, b):
    return (a.double() - b.double()).norm().item() / max(b.double().norm().item(), 1e-12)


def _param_leaves(m):
    names = ["confidence_log_beta", "confidence_log_gamma"]
    if m.read_top_n == 2:
        names.append("log_read_tau")
    return names


@pytest.mark.parametrize("read_top_n", [1, 2])
@pytest.mark.parametrize("anchor_mode", ["pairs", "single"])
def test_surrogate_gradcheck_fp64(read_top_n, anchor_mode):
    """fp64 gradcheck of the smooth STE surrogate wrt input + β + γ (+ τ for n=2)."""
    torch.set_default_dtype(torch.float64)
    try:
        spec = LUTSpec(h_in=2, h_out=2, tph=3, nap=3, d_in=5, d_out=4, anchor_mode=anchor_mode)
        m = QuantisedConfidenceLUT(spec, seed=1, weight_init_std=1.0, read_top_n=read_top_n).train()
        x = torch.randn(6, 2, 5, dtype=torch.float64, requires_grad=True)
        coef = m.frozen_ratio(x.detach())      # STE stop-grad ratio frozen at the base point
        # (a) input gradcheck of the smooth frozen-ratio surrogate (staircase STE value is not
        #     directly gradcheckable; this surrogate has the STE's exact gradient).
        assert torch.autograd.gradcheck(lambda xx: m._forward_surrogate(xx, coef).pow(2).sum(), (x,),
                                        eps=1e-6, atol=1e-5)
        # (b) param gradcheck (β, γ, and τ for n=2): a free surrogate in the log-params with the
        #     stop-grad addressing and fake-quant tables FROZEN at the base point. Recomputes the
        #     exact score / blend, so its gradient is the quant-aware backward wrt β/γ/τ.
        xd = x.detach()
        with torch.no_grad():
            z, u, c, j, ua, c_alt = m._addresses(xd)
        G, tph, K, d_out = m.weights.shape
        B = xd.shape[0]
        W2 = m._fake_quant_tables().detach()
        gc = _global_cells(c, G, tph, K)
        gca = _global_cells(c_alt, G, tph, K)
        marg = u.abs().detach()
        mv = ua.detach().unsqueeze(-1)

        def sur(log_beta, log_gamma, *log_tau):
            s = marg.sum(-1) * torch.exp(log_gamma.exp() * F.logsigmoid(log_beta.exp() * marg).sum(-1))
            if read_top_n == 2:
                x2 = 2.0 * mv / log_tau[0].exp()
                ex = s.unsqueeze(-1) * torch.cat([torch.sigmoid(x2), torch.sigmoid(-x2)], dim=-1)
                psw2 = ex * coef
                idx = torch.cat([gc, gca], dim=2).reshape(B * G, 2 * tph)
                psw = torch.cat([psw2[..., 0], psw2[..., 1]], dim=2).reshape(B * G, 2 * tph)
                grp = F.embedding_bag(idx, W2, per_sample_weights=psw, mode="sum").reshape(B, G, d_out)
            else:
                psw = (s * coef).reshape(B * G, tph)
                grp = F.embedding_bag(gc.reshape(B * G, tph), W2, per_sample_weights=psw,
                                      mode="sum").reshape(B, G, d_out)
            return m._route(grp, xd).pow(2).sum()

        names = _param_leaves(m)
        leaves = [getattr(m, n).detach().clone().requires_grad_(True) for n in names]
        assert torch.autograd.gradcheck(sur, tuple(leaves), eps=1e-6, atol=1e-5)
    finally:
        torch.set_default_dtype(torch.float32)


@pytest.mark.parametrize("read_top_n", [1, 2])
@pytest.mark.parametrize("anchor_mode", ["pairs", "single"])
def test_ste_grad_equals_surrogate_grad(read_top_n, anchor_mode):
    """The real STE forward's gradient (input, β, γ, τ) equals the smooth surrogate's — so the
    gradcheck'd surrogate IS the quant-aware backward."""
    spec = LUTSpec(h_in=3, h_out=3, tph=4, nap=4, d_in=6, d_out=5, anchor_mode=anchor_mode)
    m = QuantisedConfidenceLUT(spec, seed=2, weight_init_std=1.0, read_top_n=read_top_n).train()
    names = _param_leaves(m)
    params = [getattr(m, n) for n in names]
    torch.manual_seed(7)
    x0 = torch.randn(32, 3, 6)
    go = torch.randn(32, 3, 5)
    coef = m.frozen_ratio(x0)              # frozen at the base point

    def grads(fwd):
        x = x0.clone().requires_grad_(True)
        return torch.autograd.grad((fwd(x) * go).sum(), [x, *params], retain_graph=False)

    g_ste = grads(m)                                     # m.__call__ -> STE forward
    g_sur = grads(lambda xx: m._forward_surrogate(xx, coef))
    for a, b, nm in zip(g_ste, g_sur, ["input", *names]):
        assert _normrel(a, b) < 1e-5, f"STE vs surrogate grad mismatch for {nm}"


@pytest.mark.parametrize("read_top_n", [1, 2])
@pytest.mark.parametrize("anchor_mode", ["pairs", "single"])
@pytest.mark.parametrize("h_in,h_out", PATTERNS)
@pytest.mark.parametrize("device", DEVICES)
def test_quant_forward_backward_runs(read_top_n, anchor_mode, h_in, h_out, device):
    spec = LUTSpec(h_in=h_in, h_out=h_out, tph=6, nap=5, d_in=12, d_out=8, anchor_mode=anchor_mode)
    m = QuantisedConfidenceLUT(spec, seed=0, weight_init_std=1.0, read_top_n=read_top_n).to(device)
    torch.manual_seed(5)
    x = torch.randn(64, h_in, 12, device=device, requires_grad=True)
    m.train()
    y = m(x)
    assert y.shape == (64, h_out, 8) and y.dtype == x.dtype
    tg = [x, m.weights, m.confidence_log_beta, m.confidence_log_gamma]
    if read_top_n == 2:
        tg.append(m.log_read_tau)
    grads = torch.autograd.grad(y.pow(2).sum(), tg)
    assert all(torch.isfinite(g).all() and g.abs().sum() > 0 for g in grads)


@pytest.mark.parametrize("read_top_n", [1, 2])
@pytest.mark.parametrize("anchor_mode", ["pairs", "single"])
@pytest.mark.parametrize("device", DEVICES)
def test_fakequant_eval_matches_int8_read(read_top_n, anchor_mode, device):
    """The fake-quant eval value equals the int8 shift-add integer read (note §6), bit-exact."""
    spec = LUTSpec(h_in=3, h_out=3, tph=4, nap=4, d_in=6, d_out=5, anchor_mode=anchor_mode)
    m = QuantisedConfidenceLUT(spec, seed=2, weight_init_std=1.0, read_top_n=read_top_n).to(device).eval()
    x = torch.randn(48, 3, 6, device=device)
    with torch.no_grad():
        yq = m(x)
        yi = m.forward_int(x)
    assert _normrel(yq, yi) < 1e-6


@pytest.mark.parametrize("read_top_n", [1, 2])
@pytest.mark.parametrize("anchor_mode", ["pairs", "single"])
def test_train_eval_value_match(read_top_n, anchor_mode):
    spec = LUTSpec(h_in=3, h_out=3, tph=4, nap=4, d_in=6, d_out=5, anchor_mode=anchor_mode)
    m = QuantisedConfidenceLUT(spec, seed=2, weight_init_std=1.0, read_top_n=read_top_n)
    x = torch.randn(32, 3, 6)
    with torch.no_grad():
        m.train(); yt = m(x)
        m.eval(); ye = m(x)
    assert _normrel(yt, ye) < 1e-6


@pytest.mark.parametrize("read_top_n", [1, 2])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_rejects_low_precision(read_top_n, dtype):
    spec = LUTSpec(h_in=2, h_out=2, tph=4, nap=3, d_in=6, d_out=5)
    m = QuantisedConfidenceLUT(spec, seed=0, weight_init_std=1.0, read_top_n=read_top_n).to(dtype)
    with pytest.raises(TypeError, match="does not support low precision"):
        m(torch.randn(8, 2, 6, dtype=dtype))


def test_quant_config_and_wiring():
    spec = LUTSpec(h_in=2, h_out=2, tph=3, nap=3, d_in=5, d_out=4)
    m = QuantisedConfidenceLUT(spec, seed=0, weight_init_std=1.0, read_top_n=2)
    assert m._quant["bits"] == 8 and m._quant["mode"] == "p2_int8"
    names = {n for n, _ in m.named_parameters()}
    assert {"confidence_log_beta", "confidence_log_gamma", "log_read_tau"} <= names
    assert not hasattr(m, "confidence_g")  # g dropped entirely
    with pytest.raises(ValueError):
        QuantisedConfidenceLUT(spec, seed=0, weight_init_std=1.0, quant_mode="bogus")
