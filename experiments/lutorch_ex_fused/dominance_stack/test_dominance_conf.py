"""DominanceConfFast (Dominance on ConfidenceLUT n=1) vs an independent plain-PyTorch float64 reference
(explicit loops over heads/tables, explicit dominance matrix D, score formula written out), value + all gradients
(tables, beta, gamma, input); plus consistency with the library's score and a few structural properties."""
import os, sys
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from spiky.lutorch_ex import LUTSpec, FusedConfidenceLUT, FusedManifestoHardLUT
from spiky.lutorch_ex.cartridges.confidence import ConfidenceLUT
from dominance_conf import DominanceConfFast
from stack import DominanceStackLUT


def naive(m, x, log_beta, log_gamma, W):
    """Independent float64 reference. x [B,h_in,d_in] (requires grad). Pure indexing, no library code."""
    sp = m.spec
    B, G, T, nap, nop, d = x.shape[0], sp.n_groups, sp.tph, sp.nap, m.nop, sp.d_out
    beta, gamma = log_beta.exp(), log_gamma.exp()
    pw = [2 ** (nap - 1 - j) for j in range(nap)]                    # MSB-first
    y = x.new_zeros(B, G, d)
    S = x.new_zeros(B, G, T)
    for g in range(G):
        z = x[:, g % sp.h_in, :]
        for t in range(T):
            u = torch.stack([z[:, int(m.anchor_a[g, t, j])] - z[:, int(m.anchor_b[g, t, j])] for j in range(nap)], -1)
            mag = u.abs()
            s = mag.sum(-1) * torch.exp(gamma * torch.log(torch.sigmoid(beta * mag)).sum(-1))
            S[:, g, t] = s
            c = sum((u[:, j] > 0).long() * pw[j] for j in range(nap))
            row = W[g, t][c] * s[:, None]                              # [B, nop]
            for k in range(nop):                                       # Borda via the explicit pair votes
                a, b = int(m.out_a[g, t, k]), int(m.out_b[g, t, k])
                y[:, g, a] = y[:, g, a] + row[:, k]
                y[:, g, b] = y[:, g, b] - row[:, k]
    return y, S


def loss_of(y):
    return (y * torch.linspace(-1, 1, y.numel(), device=y.device, dtype=y.dtype).reshape(y.shape)).sum()


@pytest.mark.parametrize("cfg", [
    dict(h=3, tph=5, nap=4, d=12, B=40),
    dict(h=2, tph=7, nap=3, d=8, B=33, beta=0.7, gamma=1.8),
    dict(h=4, tph=6, nap=8, d=16, B=25, beta=3.0, gamma=0.5),
])
@pytest.mark.parametrize("dev", ["cpu", "cuda"])
def test_vs_naive_float64(cfg, dev):
    if dev == "cuda" and not torch.cuda.is_available():
        pytest.skip("no CUDA")
    d, h = cfg["d"], cfg["h"]
    sp = LUTSpec(h_in=h, h_out=h, tph=cfg["tph"], nap=cfg["nap"], d_in=d, d_out=d, anchor_mode="pairs")
    m = DominanceConfFast(sp, seed=5, beta_init=cfg.get("beta", 2.0), gamma_init=cfg.get("gamma", 1.0)).to(dev)
    m.weights.data.normal_(0, 0.5, generator=torch.Generator().manual_seed(2)) if dev == "cpu" else \
        m.weights.data.copy_(torch.randn(m.weights.shape, generator=torch.Generator().manual_seed(2)).to(dev) * 0.5)
    x = torch.randn(cfg["B"], h, d, device=dev)
    # CUDA fp32 path (Triton gather + hand-written read/Borda backward) and the module's own plain-torch path
    x1 = x.clone().requires_grad_(True)
    m.zero_grad(); y1 = m(x1); loss_of(y1).backward()
    got = (y1.detach(), m.weights.grad.clone(), m.confidence_log_beta.grad.clone(),
           m.confidence_log_gamma.grad.clone(), x1.grad.clone())
    # float64 independent reference
    W = m.weights.detach().double().clone().requires_grad_(True)
    lb = m.confidence_log_beta.detach().double().clone().requires_grad_(True)
    lg = m.confidence_log_gamma.detach().double().clone().requires_grad_(True)
    xr = x.double().clone().requires_grad_(True)
    yr, _ = naive(m, xr, lb, lg, W)
    loss_of(yr).backward()
    want = (yr.detach(), W.grad, lb.grad, lg.grad, xr.grad)
    for name, a, b in zip(("y", "dW", "dlogbeta", "dloggamma", "dx"), got, want):
        b = b.to(a.dtype)
        tol = 2e-4 * max(1.0, b.abs().max().item())
        assert (a - b).abs().max() < tol, (name, (a - b).abs().max().item(), tol)
    # the module's plain-torch path equals the fast path
    x2 = x.clone().requires_grad_(True)
    m.zero_grad(); y2 = m.forward_reference(x2); loss_of(y2).backward()
    assert (y2.detach() - got[0]).abs().max() < 1e-5 * max(1.0, got[0].abs().max().item())
    assert (x2.grad - got[4]).abs().max() < 1e-4 * max(1.0, got[4].abs().max().item())
    assert (m.weights.grad - got[1]).abs().max() < 1e-4 * max(1.0, got[1].abs().max().item())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_experiment_geometry_fast_vs_reference_path():
    sp = LUTSpec(h_in=8, h_out=8, tph=64, nap=8, d_in=48, d_out=48, anchor_mode="pairs")
    m = DominanceConfFast(sp, seed=11).cuda()
    m.weights.data.normal_(0, 0.5, generator=torch.Generator(device="cuda").manual_seed(3))
    x = torch.randn(3000, 8, 48, device="cuda") * 0.5
    r = []
    for f in (m, m.forward_reference):
        xx = x.clone().requires_grad_(True); m.zero_grad()
        y = f(xx); loss_of(y).backward()
        r.append((y.detach(), m.weights.grad.clone(), xx.grad.clone(), m.confidence_log_beta.grad.clone(),
                  m.confidence_log_gamma.grad.clone()))
    for a, b in zip(*r):
        assert (a - b).abs().max() < 2e-4 * max(1.0, b.abs().max().item())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_score_equals_library_fused_confidence_n1():
    """Same anchors/margins -> same score as FusedConfidenceLUT n=1: with W[g,t,c] a one-hot-in-nothing identity probe,
    FusedConfidenceLUT's output for a table whose row is all ones is the sum of scores over tables."""
    sp = LUTSpec(h_in=2, h_out=2, tph=9, nap=4, d_in=8, d_out=8, anchor_mode="pairs")
    lib = FusedConfidenceLUT(sp, read_top_n=1, seed=3, beta_init=1.7, gamma_init=0.8, table_dropout_rate=0.0).cuda()
    dom = DominanceConfFast(sp, seed=3, beta_init=1.7, gamma_init=0.8).cuda()
    assert torch.equal(lib.anchor_a, dom.anchor_a) and torch.equal(lib.anchor_b, dom.anchor_b)
    lib.weights.data.fill_(1.0)                                  # y_lib[b,g,:] = sum_t s[b,g,t]
    x = torch.randn(50, 2, 8, device="cuda")
    s_lib = lib(x)[..., 0]
    u, _ = dom._margins_and_cells(x)
    s_dom = dom._score(u).sum(-1)
    assert torch.allclose(s_lib, s_dom, rtol=1e-5, atol=1e-5)
    assert torch.allclose(dom._score(u), ConfidenceLUT._score(lib, u), rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_zero_sum_and_stack_with_conf_layer1():
    sp = LUTSpec(h_in=8, h_out=8, tph=16, nap=8, d_in=48, d_out=48, anchor_mode="pairs")
    for second in (FusedManifestoHardLUT(sp, weight="rational", seed=7, weight_init_std=0.3),
                   FusedConfidenceLUT(sp, read_top_n=1, seed=7, weight_init_std=0.3)):
        st = DominanceStackLUT(sp, second, dom_seed=5, post_norm="ln", dom_conf=True).cuda()
        assert isinstance(st.dom, DominanceConfFast)
        x = torch.randn(64, 8, 48, device="cuda", requires_grad=True)
        h = st.dom(x)
        assert h.sum(-1).abs().max() < 1e-4                      # Borda outputs are zero-sum per head
        y = st(x); y.sum().backward()
        assert y.shape == (64, 8, 48) and torch.isfinite(x.grad).all()
        assert st.dom.confidence_log_beta.grad is not None and st.second.weights.grad is not None
        st.eval(); st(x); assert st.mean_hidden_std() > 0 and st.mean_score() > 0
