"""bfloat16 / float16 dtype coverage.

bfloat16 / float16 is a **fused-cartridge feature** (FusedManifestoHardLUT / FusedManifestoSoftLUT and
ProjectionMHL wrapping them; CUDA only -- the fused classes run only on their kernels). There it is mixed precision: the discrete addressing (margins,
argmin j*, packed cell index) runs in float32 and every reduction accumulates in float32; only the
stored LUT table, the inputs, and the final output carry bfloat16. So against a float32 reference
*fed the same input values* the only differences are bfloat16 storage round-off and the final
output cast — a few tenths of a percent — NOT the discrete LUT-row flips that bfloat16 *input
rounding* can cause near a boundary (an inherent, correct property of a bfloat16 front-end). The
reference therefore uses ``x_ref = x_bf16.float()`` (identical values), isolating what bfloat16
support must get right.

The REFERENCE cartridges (ManifestoHardLUT / ManifestoSoftLUT) accept bfloat16 / float16 too: the
same maths in float32 on float32 views, one rounding per leaf (bit-exact against the float32 run in
test_reference_low_precision.py). ProjectionMHL wrapping either runs in bfloat16.
"""
import pytest
import torch

from spiky.lutorch_ex import (
    FusedManifestoHardLUT,
    FusedManifestoSoftLUT,
    LUTSpec,
    ManifestoHardLUT,
    ManifestoSoftLUT,
    ProjectionMHL,
)
from spiky.lutorch_ex.tests._requires_extension import needs_manifesto_ext  # noqa: F401 (fixture)

FUSED = [FusedManifestoHardLUT, FusedManifestoSoftLUT]
PURE = [ManifestoHardLUT, ManifestoSoftLUT]
PATTERNS = [(4, 4), (1, 4), (4, 1)]
DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
TOL = 5e-2  # loose: observed ~3e-3; headroom for bfloat16 round-off across shapes/devices


def _rel(a, b):
    return (a.float() - b.float()).norm().item() / max(b.float().norm().item(), 1e-12)


@pytest.mark.parametrize("anchor_mode", ["pairs", "single"])
@pytest.mark.parametrize("Cls", FUSED)
@pytest.mark.parametrize("h_in,h_out", PATTERNS)
@pytest.mark.parametrize("B", [1, 128])
def test_fused_bf16_matches_fp32_reference(anchor_mode, Cls, h_in, h_out, B, needs_manifesto_ext):
    device = "cuda"
    spec = LUTSpec(h_in=h_in, h_out=h_out, tph=6, nap=5, d_in=12, d_out=8, anchor_mode=anchor_mode)
    ref = Cls(spec, seed=0, weight_init_std=1.0).to(device)                   # float32
    bf = Cls(spec, seed=0, weight_init_std=1.0).to(device).to(torch.bfloat16)  # bfloat16
    assert bf.weights.dtype == torch.bfloat16 and bf.anchor_a.dtype == torch.long

    torch.manual_seed(100 + B)
    x = torch.randn(B, h_in, 12, device=device).to(torch.bfloat16)  # shared bfloat16-rounded values
    xr = x.float().clone().requires_grad_(True)
    xb = x.clone().requires_grad_(True)

    ref.train(); bf.train()
    yr, yb = ref(xr), bf(xb)
    assert yb.dtype == torch.bfloat16 and yb.shape == yr.shape
    assert _rel(yb, yr) < TOL, f"train fwd rel={_rel(yb, yr):.2e}"
    go = torch.randn_like(yr)
    gxr, gwr = torch.autograd.grad(yr, (xr, ref.weights), go)
    gxb, gwb = torch.autograd.grad(yb, (xb, bf.weights), go.to(torch.bfloat16))
    assert gxb.dtype == torch.bfloat16 and gwb.dtype == torch.bfloat16
    assert _rel(gxb, gxr) < TOL, f"grad_x rel={_rel(gxb, gxr):.2e}"
    assert _rel(gwb, gwr) < TOL, f"grad_w rel={_rel(gwb, gwr):.2e}"

    ref.eval(); bf.eval()
    with torch.no_grad():
        yer, yeb = ref(xr), bf(xb)
    assert yeb.dtype == torch.bfloat16
    assert _rel(yeb, yer) < TOL, f"eval rel={_rel(yeb, yer):.2e}"


@pytest.mark.parametrize("anchor_mode", ["pairs", "single"])
def test_fused_bf16_projection_mhl(anchor_mode, needs_manifesto_ext):
    device = "cuda"
    spec = LUTSpec(h_in=2, h_out=2, tph=3, nap=4, d_in=6, d_out=5, anchor_mode=anchor_mode)
    refc = FusedManifestoHardLUT(spec, seed=0, weight_init_std=1.0)
    bfc = FusedManifestoHardLUT(spec, seed=0, weight_init_std=1.0)
    ref = ProjectionMHL(refc, d_model=10, bias=True).to(device)
    bf = ProjectionMHL(bfc, d_model=10, bias=True).to(device).to(torch.bfloat16)
    bf.load_state_dict({k: v.to(torch.bfloat16) for k, v in ref.state_dict().items()})

    torch.manual_seed(7)
    x = torch.randn(32, 10, device=device).to(torch.bfloat16)
    xr = x.float().clone().requires_grad_(True)
    xb = x.clone().requires_grad_(True)
    ref.train(); bf.train()
    yr, yb = ref(xr), bf(xb)
    assert yb.dtype == torch.bfloat16 and yb.shape == (32, 10)
    assert _rel(yb, yr) < TOL
    yr.float().pow(2).sum().backward()
    yb.float().pow(2).sum().backward()
    assert bf.compress.weight.grad.dtype == torch.bfloat16
    assert _rel(bf.compress.weight.grad, ref.compress.weight.grad) < TOL
    assert _rel(bfc.weights.grad, refc.weights.grad) < TOL


@pytest.mark.parametrize("Cls", FUSED)
def test_fused_bf16_cuda_training_step(Cls, needs_manifesto_ext):
    """A bfloat16 training step runs end-to-end through the fused kernels (bfloat16 table upconverted in registers,
    float32 accumulators) and stays bfloat16 with finite grads."""
    spec = LUTSpec(h_in=4, h_out=4, tph=6, nap=5, d_in=12, d_out=8)
    m = Cls(spec, seed=0, weight_init_std=1.0).cuda().to(torch.bfloat16).train()
    x = torch.randn(256, 4, 12, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    y = m(x)
    assert y.dtype == torch.bfloat16
    y.float().pow(2).sum().backward()
    assert m.weights.grad is not None and torch.isfinite(m.weights.grad.float()).all()
    assert m.weights.grad.dtype == torch.bfloat16 and x.grad.dtype == torch.bfloat16


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("Cls", PURE)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_reference_cartridge_accepts_low_precision(device, Cls, dtype):
    """bfloat16 / float16 params OR inputs run on the reference cartridges and give the input dtype."""
    spec = LUTSpec(h_in=2, h_out=2, tph=4, nap=3, d_in=6, d_out=5)
    m = Cls(spec, seed=0, weight_init_std=1.0).to(device).to(dtype)
    assert m(torch.randn(8, 2, 6, device=device)).dtype == torch.float32      # low table, float32 input
    m32 = Cls(spec, seed=0, weight_init_std=1.0).to(device)
    assert m32(torch.randn(8, 2, 6, device=device, dtype=dtype)).dtype == dtype


@pytest.mark.parametrize("Cls", FUSED)
def test_reference_and_fused_agree_in_bf16(Cls, needs_manifesto_ext):
    """The bfloat16 contract is the same in both: float32 maths, one rounding. So the reference and the fused class on
    the same bfloat16 table and input agree to accumulation order plus that rounding."""
    from spiky.lutorch_ex import ManifestoHardLUT as _H, ManifestoSoftLUT as _S
    Ref = _H if Cls is FusedManifestoHardLUT else _S
    spec = LUTSpec(h_in=4, h_out=4, tph=6, nap=5, d_in=12, d_out=8)
    f = Cls(spec, seed=0, weight_init_std=1.0).cuda().to(torch.bfloat16).train()
    r = Ref(spec, seed=0, weight_init_std=1.0).cuda().to(torch.bfloat16).train()
    r.load_state_dict(f.state_dict())
    x = torch.randn(128, 4, 12, device="cuda").to(torch.bfloat16)
    go = torch.randn(128, 4, 8, device="cuda").to(torch.bfloat16)
    out = []
    for m in (f, r):
        xi = x.clone().requires_grad_(True)
        y = m(xi)
        gx, gw = torch.autograd.grad(y, (xi, m.weights), go)
        out.append((y, gx, gw))
    for a, b, what in zip(out[0], out[1], ("value", "grad x", "grad W")):
        assert a.dtype == b.dtype == torch.bfloat16, what
        assert _rel(a, b) < 2 * 2.0 ** -8, f"{what}: rel {_rel(a, b):.2e}"
