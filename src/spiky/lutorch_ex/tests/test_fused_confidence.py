"""FusedConfidenceLUT (hand-written CUDA) vs ConfidenceLUT: forward value and every gradient, both read_top_n, both
anchor modes, with and without table dropout, across the launch knobs."""
import pytest
import torch

from spiky.lutorch_ex.cartridges.confidence import ConfidenceLUT
from spiky.lutorch_ex.cartridges.fused_confidence import FusedConfidenceLUT, CudaKnobs
from spiky.lutorch_ex.lut_spec import LUTSpec
from spiky.lutorch_ex.tests._requires_extension import requires_extension

# No CUDA device: skip. A CUDA device without the extension: fail (tests/_requires_extension.py).
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")


@pytest.fixture(autouse=True)
def _needs_fused_confidence_ext():
    requires_extension("confidence")


def _pair(n, anchor_mode="pairs", rate=0.0, h_in=4, h_out=4, tph=16, nap=6, d=24, knobs=None):
    spec = LUTSpec(h_in=h_in, h_out=h_out, tph=tph, nap=nap, d_in=d, d_out=d, anchor_mode=anchor_mode)
    kw = dict(seed=3, read_top_n=n, table_dropout_rate=rate, weight_init_std=0.5, beta_init=1.7, gamma_init=0.8,
              **({"read_tau_init": 0.6} if n == 2 else {}))
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


def test_no_fallback_off_cuda():
    """No fallback (spec rule 3): a CPU input and an fp64 input raise, naming the case and the reference class."""
    spec = LUTSpec(h_in=2, h_out=2, tph=4, nap=3, d_in=8, d_out=8, anchor_mode="pairs")
    with pytest.raises(RuntimeError, match=r"the input is on cpu.*Construct ConfidenceLUT"):
        FusedConfidenceLUT(spec, seed=0).eval()(torch.randn(5, 2, 8))
    with pytest.raises(RuntimeError, match=r"input dtype torch.float64.*table dtype torch.float64"):
        FusedConfidenceLUT(spec, seed=0).cuda().double().eval()(torch.randn(5, 2, 8, device="cuda",
                                                                            dtype=torch.float64))


# -- bf16 tables ------------------------------------------------------------------------------------------------------
# bf16 keeps 8 significant bits, so its unit roundoff is u = 2**-8 (3.9e-3). In the kernels every intermediate is
# fp32; only the leaves are rounded once to bf16: the output (bf16 input), grad x, grad W, grad beta / gamma / tau.
# One rounding gives a rel-norm error of about u / sqrt(3) (uniform rounding error); BF16_TOL = 2u bounds it with
# margin. FP32_TOL is the existing fp32 tolerance.
BF16_TOL = 2.0 * 2.0 ** -8
FP32_TOL = 1e-4


def _rel(a, b):
    return ((a.double() - b.double()).norm() / b.double().norm().clamp_min(1e-30)).item()


def _model(n, dtype, anchor_mode="pairs", knobs=None, h=4, tph=16, nap=6, d=24, rate=0.0):
    spec = LUTSpec(h_in=h, h_out=h, tph=tph, nap=nap, d_in=d, d_out=d, anchor_mode=anchor_mode)
    m = FusedConfidenceLUT(spec, seed=3, read_top_n=n, table_dropout_rate=rate, weight_init_std=0.5, beta_init=1.7,
                           gamma_init=0.8, knobs=knobs, **({"read_tau_init": 0.6} if n == 2 else {}))
    return m.to(device="cuda", dtype=dtype).train()


def _dense_bf16(cud, x, go):
    """The dense answer for a bf16 cartridge: ConfidenceLUT in fp32 on the same bf16-stored parameters and bf16 input,
    every result then rounded to bf16 once (what the kernels' leaves get)."""
    ref = ConfidenceLUT(cud.spec, seed=3, read_top_n=cud.read_top_n, table_dropout_rate=0.0).cuda().train()
    ref.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in cud.state_dict().items()})
    return {k: v.to(torch.bfloat16) for k, v in _fwd_bwd(ref, x.float(), go.float()).items()}


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
    """The kernels on a bf16 table against the dense ConfidenceLUT math in fp32 on the same bf16-stored parameters
    (:func:`_dense_bf16`). Same storage, so the gap is accumulation order plus the final bf16 rounding of the outputs
    -- it must not be larger than one bf16 rounding."""
    torch.manual_seed(0)
    cud = _model(n, torch.bfloat16)
    x = torch.randn(96, 4, 24, device="cuda").to(torch.bfloat16)
    go = torch.randn(96, 4, 24, device="cuda").to(torch.bfloat16)
    a, b = _fwd_bwd(cud, x, go), _dense_bf16(cud, x, go)
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
    cud = _model(n, torch.bfloat16, knobs=CudaKnobs(vec_bf16=vec, rows_per_cta=3))
    x = torch.randn(50, 4, 24, device="cuda").to(torch.bfloat16)
    go = torch.randn(50, 4, 24, device="cuda").to(torch.bfloat16)
    a, b = _fwd_bwd(cud, x, go), _dense_bf16(cud, x, go)
    for k in b:
        assert _rel(a[k], b[k]) < BF16_TOL, f"vec {vec}: {k}"





# -- bf16 input read in-kernel, bf16 output stored in-kernel, identity in_head gather skipped -------------------------
@pytest.mark.parametrize("table", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("pattern", [(4, 4), (4, 1), (1, 4)])          # per-head, fan-in, fan-out
@pytest.mark.parametrize("n", [1, 2])
def test_bf16_input_is_bitwise_the_fp32_input_rounded_once(table, pattern, n):
    """A bf16 input goes to the kernels as is (upconverted when staged: exact). Per-head / fan-out store the bf16
    output in-kernel; fan-in sums the groups in fp32 and casts. All must be bitwise the fp32 run on the same values
    with its output rounded once, which is what the path did before (cast in, cast out). grad x comes back bf16."""
    h_in, h_out = pattern
    spec = LUTSpec(h_in=h_in, h_out=h_out, tph=16, nap=6, d_in=24, d_out=24, anchor_mode="pairs")
    m = FusedConfidenceLUT(spec, seed=3, read_top_n=n, weight_init_std=0.5, beta_init=1.7, gamma_init=0.8,
                           **({"read_tau_init": 0.6} if n == 2 else {})).to(device="cuda", dtype=table)
    torch.manual_seed(0)
    xb = torch.randn(70, h_in, 24, device="cuda").to(torch.bfloat16)
    go = torch.randn(70, h_out, 24, device="cuda").to(torch.bfloat16).float()   # same go for both runs
    for mode in ("eval", "train"):
        getattr(m, mode)()
        with torch.no_grad():
            yb, yf = m(xb), m(xb.float())
        assert yb.dtype == torch.bfloat16 and yf.dtype == torch.float32
        assert torch.equal(yb, yf.to(torch.bfloat16)), mode
    a, b = _fwd_bwd(m, xb, go), _fwd_bwd(m, xb.float(), go)
    assert a["grad x"].dtype == torch.bfloat16
    # The backward accumulates grad z with shared-memory atomics (not bit-reproducible run to run), so the gradients
    # are compared to one bf16 rounding of the fp32 run.
    for k in a:
        assert _rel(a[k], b[k].to(a[k].dtype)) < BF16_TOL, k


# -- d_out the kernels cannot launch at the default knobs: fitted (vec halved, threads raised) or not a cuda case -----
@pytest.mark.parametrize("mode", ["train", "eval"])
@pytest.mark.parametrize("d_out", [50, 260, 300])
@pytest.mark.parametrize("n", [1, 2])
def test_d_out_not_divisible_by_4_or_wider_than_256_runs_on_cuda(d_out, n, mode):
    """d_out 50 (not divisible by the vector width 4) and 260 / 300 (more than 64 threads x 4 per row) used to crash
    the kernels at launch ("d_out must be divisible by vec" / "threads must be >= d_out / vec"). The launch is now
    fitted to d_out exactly as for the Manifesto kernels, so they run and agree with ConfidenceLUT."""
    spec = LUTSpec(h_in=2, h_out=2, tph=8, nap=4, d_in=12, d_out=d_out, anchor_mode="pairs")
    kw = dict(seed=3, read_top_n=n, weight_init_std=0.5, beta_init=1.7, gamma_init=0.8,
              **({"read_tau_init": 0.6} if n == 2 else {}))
    ref = ConfidenceLUT(spec, **kw).cuda()
    cud = FusedConfidenceLUT(spec, **kw).cuda()
    cud.load_state_dict(ref.state_dict())
    torch.manual_seed(0)
    x = torch.randn(33, 2, 12, device="cuda")
    go = torch.randn(33, 2, d_out, device="cuda")
    getattr(ref, mode)(); getattr(cud, mode)()
    if mode == "eval":
        with torch.no_grad():
            yc, yr = cud(x), ref._forward_impl(x)
        assert _rel(yc, yr) < FP32_TOL
        return
    a, b = _fwd_bwd(cud, x, go), _fwd_bwd(ref, x, go, eager_ref=True)
    for k in b:
        assert _rel(a[k], b[k]) < FP32_TOL, k


def test_d_out_one_cta_cannot_cover_is_not_a_cuda_case():
    """d_out 4100 needs 1056 threads per CTA at vec 4 (> 1024): it raises saying why, instead of crashing at launch."""
    spec = LUTSpec(h_in=1, h_out=1, tph=2, nap=3, d_in=6, d_out=4100, anchor_mode="pairs")
    x = torch.randn(3, 1, 6, device="cuda")
    with torch.no_grad(), pytest.raises(RuntimeError, match=r"d_out=4100 needs 1056 threads.*Construct ConfidenceLUT"):
        FusedConfidenceLUT(spec, seed=0).cuda().eval()(x)
