"""Gen-2 soft-sign cartridges: SoftSignHardLUT and SoftSignSmoothLUT.

Two cartridges (the collapsed Gen-2 family): both read with ``F.embedding_bag`` on the train path,
plain-autograd backward, bf16/fp16 support. Covered here: fp64 gradcheck of the 2-alternative soft
backward (incl. the two learned temperatures) against reference finite differences; the hard
cartridge's value == the hard read and its output-Jacobian == the smooth cartridge's; the bf16
path vs an fp32 reference; every head pattern / anchor mode runs with finite grads; temperatures
are learnable (and freezable).
"""
import pytest
import torch

from spiky.lutorch_ex import (
    LUTSpec,
    ManifestoHardLUT,
    SoftSignHardLUT,
    SoftSignSmoothLUT,
)

CARTRIDGES = [SoftSignHardLUT, SoftSignSmoothLUT]
PATTERNS = [(4, 4), (1, 4), (4, 1)]
DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def _normrel(a, b):
    return (a.float() - b.float()).norm().item() / max(b.float().norm().item(), 1e-12)


@pytest.mark.parametrize("anchor_mode", ["pairs", "single"])
def test_smooth_backward_gradcheck_fp64(anchor_mode):
    """The smooth cartridge is value == gradient; fp64 gradcheck wrt input, weights, and both
    learned temperatures verifies the 2-alternative backward analytically (reference math)."""
    torch.set_default_dtype(torch.float64)
    try:
        spec = LUTSpec(h_in=2, h_out=2, tph=3, nap=3, d_in=5, d_out=4, anchor_mode=anchor_mode)
        m = SoftSignSmoothLUT(spec, seed=1, weight_init_std=1.0).train()
        x = torch.randn(6, 2, 5, dtype=torch.float64, requires_grad=True)
        assert torch.autograd.gradcheck(lambda xx: m(xx).pow(2).sum(), (x,), eps=1e-6, atol=1e-5)
        w0 = m.weights.detach().clone().requires_grad_(True)
        a0 = m.log_soft_score_temp.detach().clone().requires_grad_(True)
        b0 = m.log_select_temp.detach().clone().requires_grad_(True)
        xd = x.detach()

        def f(w, a, b):
            return torch.func.functional_call(
                m, {"weights": w, "log_soft_score_temp": a, "log_select_temp": b}, (xd,)
            ).pow(2).sum()

        assert torch.autograd.gradcheck(f, (w0, a0, b0), eps=1e-6, atol=1e-5)
    finally:
        torch.set_default_dtype(torch.float32)


@pytest.mark.parametrize("anchor_mode", ["pairs", "single"])
def test_hard_value_is_hard_read(anchor_mode):
    """SoftSignHard's forward value is the plain hard read sum_t W[c_t] — identical to the Gen-1
    ManifestoHard eval (same addressing)."""
    spec = LUTSpec(h_in=3, h_out=3, tph=4, nap=3, d_in=6, d_out=5, anchor_mode=anchor_mode)
    ss = SoftSignHardLUT(spec, seed=2, weight_init_std=1.0).eval()
    mh = ManifestoHardLUT(spec, seed=2, weight_init_std=1.0).eval()
    x = torch.randn(32, 3, 6)
    with torch.no_grad():
        assert torch.allclose(ss(x), mh(x), atol=1e-6)


@pytest.mark.parametrize("anchor_mode", ["pairs", "single"])
def test_hard_output_jacobian_matches_smooth(anchor_mode):
    """The hard straight-through output-Jacobian (VJP with a fixed grad_output) equals the smooth
    cartridge's — both route the input/temperature gradient through the same weight w — while the
    hard weight gradient is a 1-row scatter (touches fewer cells) vs the smooth 2-row."""
    spec = LUTSpec(h_in=2, h_out=2, tph=4, nap=3, d_in=6, d_out=5, anchor_mode=anchor_mode)
    h = SoftSignHardLUT(spec, seed=3, weight_init_std=1.0).train()
    s = SoftSignSmoothLUT(spec, seed=3, weight_init_std=1.0).train()
    s.load_state_dict(h.state_dict())
    x = torch.randn(16, 2, 6)
    xh = x.clone().requires_grad_(True); xs = x.clone().requires_grad_(True)
    yh, ys = h(xh), s(xs)
    go = torch.randn_like(yh)
    gxh, gth = torch.autograd.grad(yh, (xh, h.log_select_temp), go, retain_graph=True)
    gxs, gts = torch.autograd.grad(ys, (xs, s.log_select_temp), go, retain_graph=True)
    assert _normrel(gxh, gxs) < 1e-5, "hard/smooth input-Jacobian differ"
    assert abs(gth.item() - gts.item()) < 1e-5 * (abs(gts.item()) + 1e-6)
    gwh, = torch.autograd.grad(yh, h.weights, go, retain_graph=True)
    gws, = torch.autograd.grad(ys, s.weights, go)
    assert int((gwh != 0).sum()) < int((gws != 0).sum()), "hard weight grad should touch fewer cells"


@pytest.mark.parametrize("Cls", CARTRIDGES)
@pytest.mark.parametrize("anchor_mode", ["pairs", "single"])
@pytest.mark.parametrize("h_in,h_out", PATTERNS)
@pytest.mark.parametrize("device", DEVICES)
def test_forward_backward_runs(Cls, anchor_mode, h_in, h_out, device):
    spec = LUTSpec(h_in=h_in, h_out=h_out, tph=6, nap=5, d_in=12, d_out=8, anchor_mode=anchor_mode)
    m = Cls(spec, seed=0, weight_init_std=1.0).to(device)
    torch.manual_seed(5)
    x = torch.randn(64, h_in, 12, device=device, requires_grad=True)
    m.train()
    y = m(x)
    assert y.shape == (64, h_out, 8)
    gx, gw, gt = torch.autograd.grad(y.pow(2).sum(), (x, m.weights, m.log_select_temp))
    assert torch.isfinite(gx).all() and torch.isfinite(gw).all() and torch.isfinite(gt).all()
    assert gx.abs().sum() > 0 and gw.abs().sum() > 0 and gt.abs() > 0
    m.eval()
    with torch.no_grad():
        assert m(x).shape == (64, h_out, 8)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("Cls", CARTRIDGES)
@pytest.mark.parametrize("anchor_mode", ["pairs", "single"])
def test_bf16_matches_fp32_reference(Cls, anchor_mode):
    spec = LUTSpec(h_in=4, h_out=4, tph=6, nap=5, d_in=12, d_out=8, anchor_mode=anchor_mode)
    ref = Cls(spec, seed=0, weight_init_std=1.0).cuda()
    bf = Cls(spec, seed=0, weight_init_std=1.0).cuda().to(torch.bfloat16)
    x = torch.randn(128, 4, 12, device="cuda").to(torch.bfloat16)
    xr = x.float().clone().requires_grad_(True); xb = x.clone().requires_grad_(True)
    ref.train(); bf.train()
    yr, yb = ref(xr), bf(xb)
    assert yb.dtype == torch.bfloat16
    assert _normrel(yb, yr) < 5e-2
    go = torch.randn_like(yr)
    gxr, gwr = torch.autograd.grad(yr, (xr, ref.weights), go)
    gxb, gwb = torch.autograd.grad(yb, (xb, bf.weights), go.to(torch.bfloat16))
    assert _normrel(gxb, gxr) < 5e-2 and _normrel(gwb, gwr) < 5e-2


def test_temps_are_learnable_parameters():
    spec = LUTSpec(h_in=2, h_out=2, tph=3, nap=3, d_in=5, d_out=4)
    m = SoftSignSmoothLUT(spec, seed=0, weight_init_std=1.0)
    names = {n for n, _ in m.named_parameters()}
    assert "log_soft_score_temp" in names and "log_select_temp" in names
    frozen = SoftSignSmoothLUT(spec, seed=0, weight_init_std=1.0, learnable_temps=False)
    fnames = {n for n, _ in frozen.named_parameters()}
    assert "log_soft_score_temp" not in fnames and "log_select_temp" not in fnames
