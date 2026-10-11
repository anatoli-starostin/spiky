"""The fused Manifesto classes must equal their reference classes (the oracle), forward and backward, on CUDA.

The fused classes run only on their CUDA kernels (spec rule 3), so these tests need the extension
(tests/_requires_extension.py): a skip without a CUDA device, a failure with one but no extension.
"""
import pytest
import torch

from spiky.lutorch_ex import (
    FusedManifestoHardLUT,
    FusedManifestoSoftLUT,
    LUTSpec,
    ManifestoHardLUT,
    ManifestoSoftLUT,
)
from spiky.lutorch_ex.tests._requires_extension import needs_manifesto_ext  # noqa: F401 (fixture)

PAIRS = [
    (ManifestoHardLUT, FusedManifestoHardLUT, "hard"),
    (ManifestoSoftLUT, FusedManifestoSoftLUT, "soft"),
]
PATTERNS = [(4, 4), (1, 4), (4, 1)]


def _normrel(a, b):
    return (a - b).norm().item() / max(a.norm().item(), 1e-12)


def _check(RefCls, FusedCls, h_in, h_out, B, *, anchor_mode="pairs"):
    spec = LUTSpec(h_in=h_in, h_out=h_out, tph=6, nap=5, d_in=12, d_out=8, anchor_mode=anchor_mode)
    ref = RefCls(spec, seed=0, weight_init_std=1.0).cuda()
    fused = FusedCls(spec, seed=0, weight_init_std=1.0).cuda()

    torch.manual_seed(100 + B)
    x = torch.randn(B, spec.h_in, spec.d_in, device="cuda")

    ref.eval(); fused.eval()
    with torch.no_grad():
        ye_r, ye_f = ref(x), fused(x)
    # fp32, same cells, sums in a different order: rounding-level differences only.
    assert torch.allclose(ye_r, ye_f, atol=1e-5, rtol=1e-4), f"eval {(h_in, h_out, B)}"

    ref.train(); fused.train()
    xr = x.clone().requires_grad_(True); xf = x.clone().requires_grad_(True)
    yr, yf = ref(xr), fused(xf)
    assert torch.allclose(yr.detach(), yf.detach(), atol=1e-5, rtol=1e-4), "train-fwd"
    yr.pow(2).sum().backward()
    yf.pow(2).sum().backward()
    assert _normrel(xr.grad, xf.grad) < 1e-4, f"grad_x {(h_in, h_out, B)}"
    assert _normrel(ref.weights.grad, fused.weights.grad) < 1e-4, f"grad_w {(h_in, h_out, B)}"


@pytest.mark.parametrize("anchor_mode", ["pairs", "single"])
@pytest.mark.parametrize("RefCls,FusedCls,name", PAIRS)
@pytest.mark.parametrize("h_in,h_out", PATTERNS)
@pytest.mark.parametrize("B", [1, 128, 24576])
def test_fused_equiv_cuda_f32(RefCls, FusedCls, name, h_in, h_out, B, anchor_mode, needs_manifesto_ext):
    _check(RefCls, FusedCls, h_in, h_out, B, anchor_mode=anchor_mode)
