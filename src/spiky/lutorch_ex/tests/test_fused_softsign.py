"""Every fused soft-sign backend (auto / tier1 / native) must equal the pure cartridge (oracle).

Covers forward value (eval + train) and gradients wrt input, weights AND both learned
temperatures, for both cartridges, both anchor modes, fp32 + fp64, CPU (tier1 fallback) and
H100 (native). fp64 is bit-exact modulo float accumulation order (native scatter/atomicAdd).
bf16 (kept — a real H100 speedup) is checked against an fp32 reference fed the same
bf16-rounded input.
"""
import pytest
import torch

from spiky.lutorch_ex import (
    FusedSoftSignHardLUT,
    FusedSoftSignSmoothLUT,
    LUTSpec,
    SoftSignHardLUT,
    SoftSignSmoothLUT,
)

PAIRS = [
    (SoftSignHardLUT, FusedSoftSignHardLUT, "hard"),
    (SoftSignSmoothLUT, FusedSoftSignSmoothLUT, "smooth"),
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


def _check(PureCls, FusedCls, h_in, h_out, B, dev, dtype, backend, *, exact, anchor_mode):
    spec = LUTSpec(h_in=h_in, h_out=h_out, tph=6, nap=5, d_in=12, d_out=8, anchor_mode=anchor_mode)
    pure = PureCls(spec, seed=0, weight_init_std=1.0).to(dev).to(dtype)
    fused = FusedCls(spec, seed=0, weight_init_std=1.0, backend=backend).to(dev).to(dtype)
    fused.load_state_dict(pure.state_dict())

    g = torch.Generator(device=dev).manual_seed(100 + B)
    x = torch.randn(B, spec.h_in, spec.d_in, device=dev, dtype=dtype, generator=g)

    pure.eval(); fused.eval()
    fa, fr = (1e-10, 1e-9) if exact else (1e-5, 1e-4)
    with torch.no_grad():
        assert torch.allclose(pure(x), fused(x), atol=fa, rtol=fr), f"eval {backend} {(h_in,h_out,B)}"

    pure.train(); fused.train()
    go = torch.randn(B, h_out, spec.d_out, device=dev, dtype=dtype, generator=g)
    yp, gxp, gwp, gtsp, gtlp = _grads(pure, x, go)
    yf, gxf, gwf, gtsf, gtlf = _grads(fused, x, go)
    assert torch.allclose(yp, yf, atol=fa, rtol=fr), f"train-fwd {backend} {(h_in,h_out,B)}"
    if exact:
        assert torch.allclose(gxp, gxf, atol=1e-8, rtol=1e-6), f"grad_x {backend}"
        assert torch.allclose(gwp, gwf, atol=1e-8, rtol=1e-6), f"grad_w {backend}"
        assert torch.allclose(gtsp, gtsf, atol=1e-8, rtol=1e-6), f"grad_t_soft {backend}"
        assert torch.allclose(gtlp, gtlf, atol=1e-8, rtol=1e-6), f"grad_t_sel {backend}"
    else:
        assert _normrel(gxf, gxp) < 1e-4, f"grad_x {backend} {(h_in,h_out,B)}"
        assert _normrel(gwf, gwp) < 1e-4, f"grad_w {backend} {(h_in,h_out,B)}"
        assert abs((gtsf - gtsp).item()) < 1e-4 * (abs(gtsp.item()) + 1e-6), f"grad_t_soft {backend}"
        assert abs((gtlf - gtlp).item()) < 1e-4 * (abs(gtlp.item()) + 1e-6), f"grad_t_sel {backend}"


@pytest.mark.parametrize("anchor_mode", ["pairs", "single"])
@pytest.mark.parametrize("PureCls,FusedCls,name", PAIRS)
@pytest.mark.parametrize("h_in,h_out", PATTERNS)
@pytest.mark.parametrize("B", [1, 128, 24576])
@pytest.mark.parametrize("backend", ["auto", "tier1"])
def test_fused_equiv_cpu_f32(PureCls, FusedCls, name, h_in, h_out, B, backend, anchor_mode):
    _check(PureCls, FusedCls, h_in, h_out, B, "cpu", torch.float32, backend,
           exact=False, anchor_mode=anchor_mode)


@pytest.mark.parametrize("anchor_mode", ["pairs", "single"])
@pytest.mark.parametrize("PureCls,FusedCls,name", PAIRS)
@pytest.mark.parametrize("h_in,h_out", PATTERNS)
@pytest.mark.parametrize("backend", ["auto", "tier1"])
def test_fused_equiv_cpu_f64(PureCls, FusedCls, name, h_in, h_out, backend, anchor_mode):
    _check(PureCls, FusedCls, h_in, h_out, 128, "cpu", torch.float64, backend,
           exact=True, anchor_mode=anchor_mode)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("anchor_mode", ["pairs", "single"])
@pytest.mark.parametrize("PureCls,FusedCls,name", PAIRS)
@pytest.mark.parametrize("h_in,h_out", PATTERNS)
@pytest.mark.parametrize("B", [1, 128, 24576])
@pytest.mark.parametrize("backend", ["auto", "tier1", "native"])
def test_fused_equiv_cuda_f32(PureCls, FusedCls, name, h_in, h_out, B, backend, anchor_mode):
    _check(PureCls, FusedCls, h_in, h_out, B, "cuda", torch.float32, backend,
           exact=False, anchor_mode=anchor_mode)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("anchor_mode", ["pairs", "single"])
@pytest.mark.parametrize("PureCls,FusedCls,name", PAIRS)
@pytest.mark.parametrize("backend", ["native"])
def test_fused_equiv_cuda_f64_native(PureCls, FusedCls, name, backend, anchor_mode):
    """fp64 native vs pure: forward value byte-identical, grads within accumulation-order tol."""
    _check(PureCls, FusedCls, 4, 4, 128, "cuda", torch.float64, backend,
           exact=True, anchor_mode=anchor_mode)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("anchor_mode", ["pairs", "single"])
def test_fused_f64_forward_value_byte_identical(anchor_mode):
    """The fused forward value is byte-identical to the pure read in fp64 (0.0 difference)."""
    spec = LUTSpec(h_in=4, h_out=4, tph=6, nap=5, d_in=12, d_out=8, anchor_mode=anchor_mode)
    for PureCls, FusedCls, _ in PAIRS:
        p = PureCls(spec, seed=0, weight_init_std=1.0).cuda().double().train()
        f = FusedCls(spec, seed=0, weight_init_std=1.0).cuda().double().train()
        f.load_state_dict(p.state_dict())
        x = torch.randn(256, 4, 12, device="cuda", dtype=torch.float64)
        with torch.no_grad():
            assert (p(x) - f(x)).abs().max().item() == 0.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("FusedCls", [FusedSoftSignHardLUT, FusedSoftSignSmoothLUT])
@pytest.mark.parametrize("anchor_mode", ["pairs", "single"])
def test_fused_bf16_runs_and_matches_fp32(FusedCls, anchor_mode):
    """bf16 is supported (kept for its H100 speedup) and does NOT raise; its grads match an fp32
    reference fed the SAME bf16-rounded input, to bf16 tolerance (discrete decisions can't flip)."""
    spec = LUTSpec(h_in=4, h_out=4, tph=6, nap=5, d_in=12, d_out=8, anchor_mode=anchor_mode)
    m16 = FusedCls(spec, seed=0, weight_init_std=1.0).cuda().to(torch.bfloat16).train()
    m32 = FusedCls(spec, seed=0, weight_init_std=1.0).cuda().train()
    m32.load_state_dict({k: v.float() for k, v in m16.state_dict().items()})
    x16 = (torch.randn(128, 4, 12, device="cuda") * 2).to(torch.bfloat16)
    x32 = x16.float()  # same rounded values fed to the fp32 reference
    y16, gx16, gw16, gts16, gtl16 = _grads(m16, x16, torch.ones(128, 4, 8, device="cuda", dtype=torch.bfloat16))
    y32, gx32, gw32, gts32, gtl32 = _grads(m32, x32, torch.ones(128, 4, 8, device="cuda"))
    assert torch.isfinite(gx16.float()).all() and torch.isfinite(gw16.float()).all()
    assert _normrel(y16, y32) < 5e-2
    assert _normrel(gx16, gx32) < 1e-1
    assert _normrel(gw16, gw32) < 1e-1


def test_fused_defaults_and_backend_kwarg():
    spec = LUTSpec(h_in=2, h_out=2, tph=4, nap=3, d_in=6, d_out=5)
    for FusedCls in (FusedSoftSignHardLUT, FusedSoftSignSmoothLUT):
        m = FusedCls(spec, seed=0, weight_init_std=1.0)
        assert m.backend == "auto"
        assert {"log_soft_score_temp", "log_select_temp"} <= {n for n, _ in m.named_parameters()}


# ---- backend validation: every accepted name runs; a wrong-but-plausible name raises ----------------
from spiky.lutorch_ex.cartridges._native_ops import LPROJ_EXT_NAME, native_available  # noqa: E402
from spiky.lutorch_ex.tests._native_required import require_extension  # noqa: E402

_TRAINABLE = {"auto", "tier1", "native"}               # 'pure_eval' is the eval-only read


@pytest.mark.parametrize("FusedCls,backend",
                         [(C, b) for C in (FusedSoftSignHardLUT, FusedSoftSignSmoothLUT) for b in C._BACKENDS])
def test_every_accepted_backend_runs(FusedCls, backend):
    if backend == "native":       # no CUDA device: skip; extension unavailable: skip, or fail when strict
        require_extension(native_available(torch.device("cuda")), LPROJ_EXT_NAME)
    dev = "cuda" if backend == "native" else "cpu"
    spec = LUTSpec(h_in=2, h_out=2, tph=4, nap=3, d_in=6, d_out=5)
    m = FusedCls(spec, seed=0, weight_init_std=1.0, backend=backend).to(dev)
    x = torch.randn(8, 2, 6, device=dev, requires_grad=True)
    m.eval()
    with torch.no_grad():
        assert m(x).shape == (8, 2, 5)
    if backend in _TRAINABLE:
        m.train()
        m(x).sum().backward()
        assert m.weights.grad is not None and x.grad is not None


@pytest.mark.parametrize("FusedCls", [FusedSoftSignHardLUT, FusedSoftSignSmoothLUT])
@pytest.mark.parametrize("bad,hint", [("pure", "this class uses 'pure_eval'"), ("fastest", None)])
def test_invalid_backend_raises(FusedCls, bad, hint):
    spec = LUTSpec(h_in=2, h_out=2, tph=4, nap=3, d_in=6, d_out=5)
    with pytest.raises(ValueError) as ei:
        FusedCls(spec, seed=0, backend=bad)
    msg = str(ei.value)
    assert FusedCls.__name__ in msg and repr(bad) in msg
    assert all(repr(b) in msg for b in FusedCls._BACKENDS)
    if hint is not None:
        assert hint in msg
