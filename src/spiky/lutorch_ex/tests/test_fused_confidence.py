"""FusedConfidenceLUT (hand-written CUDA) vs ConfidenceLUT: forward value and every gradient, both read_top_n, both
anchor modes, with and without table dropout, across the launch knobs."""
import pytest
import torch

from spiky.lutorch_ex.cartridges.confidence import ConfidenceLUT
from spiky.lutorch_ex.cartridges.fused_confidence import FusedConfidenceLUT, CudaKnobs, fused_confidence_ext
from spiky.lutorch_ex.lut_spec import LUTSpec

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or fused_confidence_ext() is None,
                                reason="needs CUDA and the fused_confidence extension")


def _pair(n, anchor_mode="pairs", rate=0.0, h_in=4, h_out=4, tph=16, nap=6, d=24, knobs=None):
    spec = LUTSpec(h_in=h_in, h_out=h_out, tph=tph, nap=nap, d_in=d, d_out=d, anchor_mode=anchor_mode)
    kw = dict(seed=3, read_top_n=n, table_dropout_rate=rate, weight_init_std=0.5, beta_init=1.7, gamma_init=0.8,
              read_tau_init=0.6)
    ref = ConfidenceLUT(spec, **kw).cuda().train()
    cud = FusedConfidenceLUT(spec, knobs=knobs, **kw).cuda().train()
    cud.load_state_dict(ref.state_dict())
    return ref, cud


def _run(ref, cud, B=96, keep=None):
    torch.manual_seed(0)
    x = torch.randn(B, ref.spec.h_in, ref.spec.d_in, device="cuda")
    go = torch.randn(B, ref.spec.h_out, ref.spec.d_out, device="cuda")
    if keep is not None:
        p = 1.0 - ref.table_dropout_rate
        ref._table_dropout_mask = lambda B_, dev, dt: keep.to(dt) / p
        cud._keep_flags = lambda B_, dev: keep
    out = {}
    for name, mod in (("ref", ref), ("cuda", cud)):
        xi = x.clone().requires_grad_(True)
        y = mod._forward_impl(xi) if name == "ref" else mod(xi)    # ref: eager (same numerics as compiled)
        y.backward(go)
        grads = {k: p.grad.clone() for k, p in mod.named_parameters() if p.grad is not None}
        out[name] = (y.detach(), xi.grad.clone(), grads)
        mod.zero_grad(set_to_none=True)
    return out


def _close(a, b, what, rtol=1e-4):
    err = (a - b).norm() / b.norm().clamp_min(1e-30)
    assert err < rtol, f"{what}: rel-norm error {err:.3e}"


@pytest.mark.parametrize("n", [1, 2])
@pytest.mark.parametrize("anchor_mode", ["pairs", "single"])
@pytest.mark.parametrize("rate", [0.0, 0.3])
def test_matches_confidence_lut(n, anchor_mode, rate):
    ref, cud = _pair(n, anchor_mode, rate)
    keep = None
    if rate > 0:
        g = torch.Generator(device="cuda").manual_seed(1)
        keep = torch.rand(96, ref.spec.n_groups, ref.spec.tph, device="cuda", generator=g) < 1 - rate
    o = _run(ref, cud, keep=keep)
    (yr, gxr, pr), (yc, gxc, pc) = o["ref"], o["cuda"]
    _close(yc, yr, "forward")
    _close(gxc, gxr, "grad x")
    assert pr.keys() == pc.keys()
    for k in pr:
        _close(pc[k], pr[k], f"grad {k}")


@pytest.mark.parametrize("knobs", [
    CudaKnobs(fwd_threads=32, bwd_threads=32, rows_per_cta=1, vec=1, vec_atomics=False),
    CudaKnobs(fwd_threads=64, bwd_threads=96, rows_per_cta=3, vec=2, vec_atomics=True),
    CudaKnobs(fwd_threads=256, bwd_threads=512, rows_per_cta=7, vec=4, vec_atomics=True),
])
@pytest.mark.parametrize("n", [1, 2])
def test_knobs_do_not_change_the_result(knobs, n):
    ref, cud = _pair(n, knobs=knobs)
    o = _run(ref, cud, B=50)                                     # B not a multiple of rows_per_cta
    (yr, gxr, pr), (yc, gxc, pc) = o["ref"], o["cuda"]
    _close(yc, yr, "forward")
    _close(gxc, gxr, "grad x")
    for k in pr:
        _close(pc[k], pr[k], f"grad {k}")


def test_fan_in_and_d24_geometry():
    """Fan-in routing (h_out = 1) and the d24 LUT-FFN geometry (h 16, d 48, tph 64, nap 8)."""
    ref, cud = _pair(1, h_in=4, h_out=1, tph=8, nap=4, d=16)
    o = _run(ref, cud)
    _close(o["cuda"][0], o["ref"][0], "fan-in forward")
    _close(o["cuda"][1], o["ref"][1], "fan-in grad x")
    for n in (1, 2):
        ref, cud = _pair(n, h_in=16, h_out=16, tph=64, nap=8, d=48)
        o = _run(ref, cud, B=64)
        _close(o["cuda"][0], o["ref"][0], f"d24 n={n} forward")
        _close(o["cuda"][1], o["ref"][1], f"d24 n={n} grad x")
        for k in o["ref"][2]:
            _close(o["cuda"][2][k], o["ref"][2][k], f"d24 n={n} grad {k}")


def test_falls_back_off_cuda_fp32():
    spec = LUTSpec(h_in=2, h_out=2, tph=4, nap=3, d_in=8, d_out=8, anchor_mode="pairs")
    cud = FusedConfidenceLUT(spec, seed=0).double()
    ref = ConfidenceLUT(spec, seed=0).double()
    x = torch.randn(5, 2, 8, dtype=torch.float64)
    torch.testing.assert_close(cud.eval()(x), ref.eval()(x))


# -- bf16 tables ------------------------------------------------------------------------------------------------------
# bf16 keeps 8 significant bits, so its unit roundoff is u = 2**-8 (3.9e-3). On the 'cuda' path every intermediate is
# fp32; only the leaves are rounded once to bf16: the output (bf16 input), grad x, grad W, grad beta / gamma / tau.
# One rounding gives a rel-norm error of about u / sqrt(3) (uniform rounding error); BF16_TOL = 2u bounds it with
# margin. FP32_TOL is the existing fp32 tolerance.
BF16_TOL = 2.0 * 2.0 ** -8
FP32_TOL = 1e-4


def _rel(a, b):
    return ((a.double() - b.double()).norm() / b.double().norm().clamp_min(1e-30)).item()


def _model(n, dtype, anchor_mode="pairs", backend="auto", knobs=None, h=4, tph=16, nap=6, d=24, rate=0.0):
    spec = LUTSpec(h_in=h, h_out=h, tph=tph, nap=nap, d_in=d, d_out=d, anchor_mode=anchor_mode)
    m = FusedConfidenceLUT(spec, seed=3, read_top_n=n, table_dropout_rate=rate, weight_init_std=0.5, beta_init=1.7,
                           gamma_init=0.8, read_tau_init=0.6, backend=backend, knobs=knobs)
    return m.to(device="cuda", dtype=dtype).train()


def _fwd_bwd(mod, x, go, eager_ref=False):
    xi = x.clone().requires_grad_(True)
    y = mod._forward_impl(xi) if eager_ref else mod(xi)
    y.backward(go.to(y.dtype))
    grads = {k: p.grad.clone() for k, p in mod.named_parameters() if p.grad is not None}
    mod.zero_grad(set_to_none=True)
    return {"y": y.detach(), "grad x": xi.grad.clone(), **{f"grad {k}": v for k, v in grads.items()}}


def _fp64_reference(mod, x, go):
    """ConfidenceLUT in fp64 on the SAME (already rounded) parameters and input: the exact answer for this storage."""
    ref = ConfidenceLUT(mod.spec, seed=3, read_top_n=mod.read_top_n, table_dropout_rate=0.0).cuda().double().train()
    ref.load_state_dict({k: v.double() if v.is_floating_point() else v for k, v in mod.state_dict().items()})
    return _fwd_bwd(ref, x.double(), go.double(), eager_ref=True)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("n", [1, 2])
@pytest.mark.parametrize("anchor_mode", ["pairs", "single"])
def test_accuracy_vs_fp64_per_dtype(dtype, n, anchor_mode):
    """Cartridge, input and output in `dtype` vs fp64 on identical parameters and input."""
    torch.manual_seed(0)
    mod = _model(n, dtype, anchor_mode)
    x = torch.randn(96, 4, 24, device="cuda").to(dtype)
    go = torch.randn(96, 4, 24, device="cuda").to(dtype)
    got, want = _fwd_bwd(mod, x, go), _fp64_reference(mod, x, go)
    assert got.keys() == want.keys()
    tol = FP32_TOL if dtype == torch.float32 else BF16_TOL
    errs = {k: _rel(got[k], want[k]) for k in want}
    print(f"\n[{dtype}, n={n}, {anchor_mode}] rel-norm vs fp64: " + "  ".join(f"{k}={e:.2e}" for k, e in errs.items()))
    for k, e in errs.items():
        assert got[k].dtype == dtype, f"{k}: dtype {got[k].dtype}, expected {dtype} (output / grads follow the dtype)"
        assert e < tol, f"{k}: rel-norm error {e:.3e} >= {tol:.1e}"


@pytest.mark.parametrize("n", [1, 2])
def test_bf16_cuda_matches_dense_bf16(n):
    """The CUDA bf16 path against the dense ConfidenceLUT path on the same bf16 table ('pure' backend: the
    ConfidenceLUT math in fp32 on the bf16-stored parameters). Same storage, so the gap is accumulation order plus
    the final bf16 rounding of the outputs -- it must not be larger than one bf16 rounding."""
    torch.manual_seed(0)
    cud = _model(n, torch.bfloat16, backend="cuda")
    dense = _model(n, torch.bfloat16, backend="pure")
    dense.load_state_dict(cud.state_dict())
    x = torch.randn(96, 4, 24, device="cuda").to(torch.bfloat16)
    go = torch.randn(96, 4, 24, device="cuda").to(torch.bfloat16)
    a, b = _fwd_bwd(cud, x, go), _fwd_bwd(dense, x, go)
    errs = {k: _rel(a[k], b[k]) for k in b}
    print(f"\n[bf16 cuda vs dense bf16, n={n}] rel-norm: " + "  ".join(f"{k}={e:.2e}" for k, e in errs.items()))
    for k, e in errs.items():
        assert a[k].dtype == b[k].dtype, f"{k}: dtype {a[k].dtype} vs {b[k].dtype}"
        assert e < BF16_TOL, f"{k}: rel-norm gap {e:.3e} >= {BF16_TOL:.1e}"


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_grad_w_accumulates_in_fp32_under_heavy_row_collision(dtype):
    """Every one of 2**17 tokens addresses the SAME cell of every table (identical inputs), so each touched grad-W row
    is a sum of 131,072 positive contributions. An fp32 accumulator gets it to fp32-summation accuracy: the worst case
    for N terms is N * 2**-24 (7.8e-3 here), typically far less, so the bound is 1e-3. Then one bf16 rounding for a
    bf16 table. A bf16 accumulator stalls once the running sum is ~2**8 times a single contribution and ends up wrong
    by most of its value -- far above either bound, so narrowing the accumulator makes this test fail."""
    torch.manual_seed(0)
    spec_kw = dict(h=1, tph=4, nap=3, d=8)
    mod = _model(1, dtype, **spec_kw)
    B = 2 ** 17
    x = torch.randn(1, 1, 8, device="cuda").expand(B, 1, 8).contiguous().to(dtype)
    go = (1.0 + 0.01 * torch.rand(B, 1, 8, device="cuda")).to(dtype)
    got = _fwd_bwd(mod, x, go)["grad weights"]
    want = _fp64_reference(mod, x, go)["grad weights"]
    touched = want.abs().sum(-1) > 0
    assert int(touched.sum()) == mod.spec.tph                                # exactly one cell per table
    e = _rel(got[touched], want[touched])
    tol = 1e-3 if dtype == torch.float32 else BF16_TOL
    print(f"\n[{dtype}] grad W, {B} tokens on one cell per table: rel-norm vs fp64 {e:.2e} (tol {tol:.1e})")
    assert got.dtype == dtype
    assert e < tol, f"grad W rel-norm error {e:.3e} >= {tol:.1e}: the accumulator lost precision"


@pytest.mark.parametrize("vec", [1, 2, 4, 8])
@pytest.mark.parametrize("n", [1, 2])
def test_bf16_vector_widths_agree(vec, n):
    """Every bf16 vector width (8 = one 16-byte load) gives the dense-bf16 result."""
    torch.manual_seed(0)
    cud = _model(n, torch.bfloat16, backend="cuda", knobs=CudaKnobs(vec_bf16=vec, rows_per_cta=3))
    dense = _model(n, torch.bfloat16, backend="pure")
    dense.load_state_dict(cud.state_dict())
    x = torch.randn(50, 4, 24, device="cuda").to(torch.bfloat16)
    go = torch.randn(50, 4, 24, device="cuda").to(torch.bfloat16)
    a, b = _fwd_bwd(cud, x, go), _fwd_bwd(dense, x, go)
    for k in b:
        assert _rel(a[k], b[k]) < BF16_TOL, f"vec {vec}: {k}"


def test_backend_argument():
    spec = LUTSpec(h_in=2, h_out=2, tph=4, nap=3, d_in=8, d_out=8, anchor_mode="pairs")
    with pytest.raises(ValueError, match="backend='native' is not valid here"):
        FusedConfidenceLUT(spec, backend="native")
    with pytest.raises(RuntimeError, match="backend='cuda' needs a CUDA input"):
        FusedConfidenceLUT(spec, backend="cuda")(torch.randn(3, 2, 8))                     # CPU input
    # 'pure' and 'cuda' agree (fp32).
    torch.manual_seed(0)
    x = torch.randn(40, 4, 24, device="cuda")
    go = torch.randn(40, 4, 24, device="cuda")
    pure, cuda_ = _model(2, torch.float32, backend="pure"), _model(2, torch.float32, backend="cuda")
    cuda_.load_state_dict(pure.state_dict())
    a, b = _fwd_bwd(cuda_, x, go), _fwd_bwd(pure, x, go)
    for k in b:
        assert _rel(a[k], b[k]) < FP32_TOL, k


def test_confidence_lut_error_names_fused_confidence():
    spec = LUTSpec(h_in=2, h_out=2, tph=4, nap=3, d_in=8, d_out=8, anchor_mode="pairs")
    with pytest.raises(TypeError, match="FusedConfidenceLUT"):
        ConfidenceLUT(spec).to(torch.bfloat16)(torch.randn(3, 2, 8, dtype=torch.bfloat16))
