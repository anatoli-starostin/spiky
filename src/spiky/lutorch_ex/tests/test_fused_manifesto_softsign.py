"""The fused Manifesto classes with weight="softsign" must equal the reference classes with weight="softsign" (the
oracle).

Covers forward value (eval + train) and gradients wrt input, weights AND both learned temperatures, for both
cartridges and both anchor modes, in fp32 on CUDA (the fused classes run only on their CUDA kernels; spec rule 3).
bf16 is checked against the fused fp32 run fed the same bf16-rounded input.
"""
import pytest
import torch

from spiky.lutorch_ex import FusedManifestoHardLUT, FusedManifestoSoftLUT, LUTSpec, ManifestoHardLUT, ManifestoSoftLUT
from spiky.lutorch_ex.tests._requires_extension import needs_manifesto_ext  # noqa: F401 (fixture)
from spiky.lutorch_ex.tests._weights import softsign

PAIRS = [
    (softsign(ManifestoHardLUT), softsign(FusedManifestoHardLUT), "hard"),
    (softsign(ManifestoSoftLUT), softsign(FusedManifestoSoftLUT), "soft"),
]
PATTERNS = [(4, 4), (1, 4), (4, 1)]


def _normrel(a, b):
    return (a.double() - b.double()).norm().item() / max(b.double().norm().item(), 1e-12)


def _grads(m, x, go):
    xr = x.clone().requires_grad_(True)
    y = m(xr)
    gx, gw, gts, gtl = torch.autograd.grad(
        (y * go).sum(), (xr, m.weights, m.log_soft_score_temp, m.log_select_temp)
    )
    return y.detach(), gx, gw, gts, gtl


def _check(RefCls, FusedCls, h_in, h_out, B, *, anchor_mode):
    spec = LUTSpec(h_in=h_in, h_out=h_out, tph=6, nap=5, d_in=12, d_out=8, anchor_mode=anchor_mode)
    ref = RefCls(spec, seed=0, weight_init_std=1.0).cuda()
    fused = FusedCls(spec, seed=0, weight_init_std=1.0).cuda()
    fused.load_state_dict(ref.state_dict())

    g = torch.Generator(device="cuda").manual_seed(100 + B)
    x = torch.randn(B, spec.h_in, spec.d_in, device="cuda", generator=g)

    # fp32, same cells, sums in a different order: rounding-level differences only.
    ref.eval(); fused.eval()
    with torch.no_grad():
        assert torch.allclose(ref(x), fused(x), atol=1e-5, rtol=1e-4), f"eval {(h_in, h_out, B)}"

    ref.train(); fused.train()
    go = torch.randn(B, h_out, spec.d_out, device="cuda", generator=g)
    yr, gxr, gwr, gtsr, gtlr = _grads(ref, x, go)
    yf, gxf, gwf, gtsf, gtlf = _grads(fused, x, go)
    assert torch.allclose(yr, yf, atol=1e-5, rtol=1e-4), f"train-fwd {(h_in, h_out, B)}"
    assert _normrel(gxf, gxr) < 1e-4, f"grad_x {(h_in, h_out, B)}"
    assert _normrel(gwf, gwr) < 1e-4, f"grad_w {(h_in, h_out, B)}"
    assert abs((gtsf - gtsr).item()) < 1e-4 * (abs(gtsr.item()) + 1e-6), "grad_t_soft"
    assert abs((gtlf - gtlr).item()) < 1e-4 * (abs(gtlr.item()) + 1e-6), "grad_t_sel"


@pytest.mark.parametrize("anchor_mode", ["pairs", "single"])
@pytest.mark.parametrize("RefCls,FusedCls,name", PAIRS)
@pytest.mark.parametrize("h_in,h_out", PATTERNS)
@pytest.mark.parametrize("B", [1, 128, 24576])
def test_fused_equiv_cuda_f32(RefCls, FusedCls, name, h_in, h_out, B, anchor_mode, needs_manifesto_ext):
    _check(RefCls, FusedCls, h_in, h_out, B, anchor_mode=anchor_mode)


@pytest.mark.parametrize("FusedCls", [softsign(FusedManifestoHardLUT), softsign(FusedManifestoSoftLUT)])
@pytest.mark.parametrize("anchor_mode", ["pairs", "single"])
def test_fused_bf16_runs_and_matches_fp32(FusedCls, anchor_mode, needs_manifesto_ext):
    """bf16 is supported and does NOT raise; its grads match an fp32 run fed the SAME bf16-rounded input, to bf16
    tolerance (discrete decisions can't flip)."""
    spec = LUTSpec(h_in=4, h_out=4, tph=6, nap=5, d_in=12, d_out=8, anchor_mode=anchor_mode)
    m16 = FusedCls(spec, seed=0, weight_init_std=1.0).cuda().to(torch.bfloat16).train()
    m32 = FusedCls(spec, seed=0, weight_init_std=1.0).cuda().train()
    m32.load_state_dict({k: v.float() for k, v in m16.state_dict().items()})
    x16 = (torch.randn(128, 4, 12, device="cuda") * 2).to(torch.bfloat16)
    x32 = x16.float()  # same rounded values fed to the fp32 run
    y16, gx16, gw16, gts16, gtl16 = _grads(m16, x16, torch.ones(128, 4, 8, device="cuda", dtype=torch.bfloat16))
    y32, gx32, gw32, gts32, gtl32 = _grads(m32, x32, torch.ones(128, 4, 8, device="cuda"))
    assert torch.isfinite(gx16.float()).all() and torch.isfinite(gw16.float()).all()
    assert _normrel(y16, y32) < 5e-2
    assert _normrel(gx16, gx32) < 1e-1
    assert _normrel(gw16, gw32) < 1e-1


def test_fused_softsign_parameters():
    spec = LUTSpec(h_in=2, h_out=2, tph=4, nap=3, d_in=6, d_out=5)
    for FusedCls in (FusedManifestoHardLUT, FusedManifestoSoftLUT):
        m = FusedCls(spec, seed=0, weight_init_std=1.0, weight="softsign")
        assert {"log_soft_score_temp", "log_select_temp"} <= {n for n, _ in m.named_parameters()}
