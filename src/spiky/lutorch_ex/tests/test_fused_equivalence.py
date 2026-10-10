"""Every fused backend (auto / tier1 / native) must equal the pure cartridge (the oracle)."""
import pytest
import torch

from spiky.lutorch_ex import (
    FusedManifestoHardLUT,
    FusedManifestoSoftLUT,
    LUTSpec,
    ManifestoHardLUT,
    ManifestoSoftLUT,
)

PAIRS = [
    (ManifestoHardLUT, FusedManifestoHardLUT, "hard"),
    (ManifestoSoftLUT, FusedManifestoSoftLUT, "soft"),
]
PATTERNS = [(4, 4), (1, 4), (4, 1)]


def _normrel(a, b):
    return (a - b).norm().item() / max(a.norm().item(), 1e-12)


def _check(PureCls, FusedCls, h_in, h_out, B, dev, dtype, backend, *, exact, anchor_mode="pairs"):
    spec = LUTSpec(h_in=h_in, h_out=h_out, tph=6, nap=5, d_in=12, d_out=8, anchor_mode=anchor_mode)
    pure = PureCls(spec, seed=0, weight_init_std=1.0).to(dev).to(dtype)
    fused = FusedCls(spec, seed=0, weight_init_std=1.0, backend=backend).to(dev).to(dtype)

    torch.manual_seed(100 + B)
    x = torch.randn(B, spec.h_in, spec.d_in, device=dev, dtype=dtype)

    pure.eval(); fused.eval()
    with torch.no_grad():
        ye_p, ye_f = pure(x), fused(x)
    fa, fr = (1e-10, 1e-9) if exact else (1e-5, 1e-4)
    assert torch.allclose(ye_p, ye_f, atol=fa, rtol=fr), f"eval {backend} {(h_in,h_out,B)}"

    pure.train(); fused.train()
    xp = x.clone().requires_grad_(True); xf = x.clone().requires_grad_(True)
    yp, yf = pure(xp), fused(xf)
    assert torch.allclose(yp.detach(), yf.detach(), atol=fa, rtol=fr), f"train-fwd {backend}"
    yp.float().pow(2).sum().backward()
    yf.float().pow(2).sum().backward()
    if exact:
        assert torch.allclose(xp.grad, xf.grad, atol=1e-8, rtol=1e-6)
        assert torch.allclose(pure.weights.grad, fused.weights.grad, atol=1e-8, rtol=1e-6)
    else:
        assert _normrel(xp.grad, xf.grad) < 1e-4, f"grad_x {backend} {(h_in,h_out,B)}"
        assert _normrel(pure.weights.grad, fused.weights.grad) < 1e-4, f"grad_w {backend} {(h_in,h_out,B)}"


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


# ---- backend validation: every accepted name runs; a wrong-but-plausible name raises ----------------
from spiky.lutorch_ex.cartridges._fused_manifesto_cuda import fused_manifesto_ext  # noqa: E402
from spiky.lutorch_ex.cartridges._native_ops import native_available  # noqa: E402

_TRAINABLE = {"auto", "tier1", "native", "pure"}       # 'pure_eval' is the eval-only read


def _run_backend(FusedCls, backend):
    if backend == "native" and not (torch.cuda.is_available() and native_available(torch.device("cuda"))):
        pytest.skip("native lutorch_cuda ops not available")
    if backend == "cuda" and not (torch.cuda.is_available() and fused_manifesto_ext() is not None):
        pytest.skip("fused_manifesto CUDA extension not available")
    dev = "cuda" if backend in ("native", "cuda") else "cpu"
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


@pytest.mark.parametrize("FusedCls,backend",
                         [(C, b) for C in (FusedManifestoHardLUT, FusedManifestoSoftLUT) for b in C._BACKENDS])
def test_every_accepted_backend_runs(FusedCls, backend):
    _run_backend(FusedCls, backend)


@pytest.mark.parametrize("FusedCls,bad,hint", [
    (FusedManifestoHardLUT, "pure", "this class uses 'pure_eval'"),
    (FusedManifestoSoftLUT, "pure_eval", "this class uses 'pure'"),
    (FusedManifestoHardLUT, "fastest", None),
    (FusedManifestoSoftLUT, "fastest", None),
])
def test_invalid_backend_raises(FusedCls, bad, hint):
    spec = LUTSpec(h_in=2, h_out=2, tph=4, nap=3, d_in=6, d_out=5)
    with pytest.raises(ValueError) as ei:
        FusedCls(spec, seed=0, backend=bad)
    msg = str(ei.value)
    assert FusedCls.__name__ in msg and repr(bad) in msg
    assert all(repr(b) in msg for b in FusedCls._BACKENDS)
    if hint is not None:
        assert hint in msg
