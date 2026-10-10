"""The 'cuda' backend of FusedManifestoHardLUT / FusedManifestoSoftLUT (csrc/fused_manifesto.cu).

Correctness gate: the new backend must reproduce the existing implementations (value, input gradient, table gradient
and, inside ProjectionMHL, the projection gradients) for both cartridges, both anchor modes, with and without table
dropout, fp32 and bf16, at small geometries and at the canonical h16 d48 tph64 nap8. The weight-gradient scatter uses
fp32 atomics, so run-to-run results differ in summation order; that is measured here, not assumed.
"""
import pytest
import torch

import spiky.lutorch_ex as lx
from spiky.lutorch_ex.cartridges import _fallback
from spiky.lutorch_ex.cartridges import _fused_manifesto_cuda as fmc
from spiky.lutorch_ex.cartridges._fused_manifesto_cuda import ManifestoCudaKnobs, fused_manifesto_ext
from spiky.lutorch_ex.cartridges._native_ops import LPROJ_EXT_NAME, native_available
from spiky.lutorch_ex.lut_spec import LUTSpec
from spiky.lutorch_ex.tests._native_required import optional_extension, require_extension

# No CUDA device: skip. A CUDA device without the extension: skip, or fail under SPIKY_LUTORCH_REQUIRE_NATIVE=1.
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")


@pytest.fixture(autouse=True)
def _needs_fused_manifesto_ext():
    require_extension(fused_manifesto_ext() is not None, fmc.EXT_NAME)

HARD, SOFT = lx.FusedManifestoHardLUT, lx.FusedManifestoSoftLUT
# (h_in, h_out, d, tph, nap): two small geometries, a fan-in one and the canonical one.
GEOMS = {
    "small": (2, 2, 8, 4, 3),
    "mid": (4, 4, 24, 16, 6),
    "fan_in": (4, 1, 16, 8, 5),
    "canonical": (16, 16, 48, 64, 8),
}


def _spec(geom, anchor_mode="pairs"):
    h_in, h_out, d, tph, nap = GEOMS[geom]
    return LUTSpec(h_in=h_in, h_out=h_out, tph=tph, nap=nap, d_in=d, d_out=d, anchor_mode=anchor_mode)


def _make(cls, spec, backend, rate=0.0, dtype=torch.float32, **kw):
    # A larger init std than the default so the tables' differences dominate rounding in the comparisons.
    return cls(spec, seed=7, backend=backend, weight_init_std=0.5, table_dropout_rate=rate, **kw).cuda().train().to(dtype)


def _run(mod, x, go, seed=1234):
    """Forward + backward; the RNG is reseeded so table dropout draws the same keep pattern on every backend."""
    xx = x.detach().clone().requires_grad_(True)
    mod.zero_grad(set_to_none=True)
    torch.manual_seed(seed)
    y = mod(xx)
    y.backward(go.to(y.dtype))
    return y.detach().float(), xx.grad.float(), mod.weights.grad.float()


def _rel(a, b):
    return ((a - b).abs().max() / b.abs().max().clamp_min(1e-30)).item()


def _inputs(spec, B, dtype=torch.float32, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(B, spec.h_in, spec.d_in, device="cuda", generator=g).to(dtype)
    go = torch.randn(B, spec.h_out, spec.d_out, device="cuda", generator=g)
    return x, go


def _refs(cls):
    refs = ["tier1"] + (["native"] if optional_extension(native_available(torch.device("cuda")), LPROJ_EXT_NAME)
                        else [])
    return refs + (["pure"] if cls is SOFT else [])


# --- fp32 equivalence against every existing backend --------------------------------------------------------------

@pytest.mark.parametrize("cls", [HARD, SOFT], ids=["hard", "soft"])
@pytest.mark.parametrize("geom", ["small", "mid", "fan_in", "canonical"])
@pytest.mark.parametrize("anchor_mode", ["pairs", "single"])
@pytest.mark.parametrize("rate", [0.0, 0.3])
def test_cuda_matches_existing_backends_fp32(cls, geom, anchor_mode, rate):
    spec = _spec(geom, anchor_mode)
    x, go = _inputs(spec, 512 if geom == "canonical" else 192)
    got = _run(_make(cls, spec, "cuda", rate), x, go)
    for ref in _refs(cls):
        want = _run(_make(cls, spec, ref, rate), x, go)
        for what, a, b in zip(("value", "grad x", "grad W"), got, want):
            assert _rel(a, b) < 2e-5, f"{cls.__name__} cuda vs {ref} {what}: rel {_rel(a, b):.2e}"


@pytest.mark.parametrize("pure_cls, cls", [(lx.ManifestoHardLUT, HARD), (lx.ManifestoSoftLUT, SOFT)], ids=["hard", "soft"])
@pytest.mark.parametrize("anchor_mode", ["pairs", "single"])
@pytest.mark.parametrize("geom", ["small", "mid"])
def test_cuda_fp32_matches_the_pure_cartridge_in_fp64(pure_cls, cls, anchor_mode, geom):
    """The oracle: the pure ManifestoHardLUT / ManifestoSoftLUT in float64 (same seed -> same anchors and tables)."""
    spec = _spec(geom, anchor_mode)
    x, go = _inputs(spec, 160)
    oracle = pure_cls(spec, seed=7, weight_init_std=0.5).cuda().train().double()
    xo = x.double().requires_grad_(True)
    yo = oracle(xo)
    yo.backward(go.double())
    got = _run(_make(cls, spec, "cuda"), x, go)
    for what, a, b in zip(("value", "grad x", "grad W"), got, (yo.detach(), xo.grad, oracle.weights.grad)):
        assert _rel(a.double(), b) < 1e-5, f"{what}: rel {_rel(a.double(), b):.2e}"


def test_soft_oracle_gradient_is_the_true_derivative_gradcheck():
    """Soft's input / table gradient is the exact derivative of its blend. The kernels are fp32-only, so gradcheck
    (which needs float64) runs on the float64 pure oracle the cuda backend is checked against above; inputs are
    kept away from the sign / argmin decision boundaries, where the function is not differentiable."""
    spec = _spec("small")
    m = lx.ManifestoSoftLUT(spec, seed=7, weight_init_std=0.5).cuda().train().double()
    g = torch.Generator(device="cuda").manual_seed(3)
    x = (torch.randn(6, spec.h_in, spec.d_in, device="cuda", generator=g, dtype=torch.float64)).requires_grad_(True)
    assert torch.autograd.gradcheck(lambda t: m(t), (x,), eps=1e-6, atol=1e-6)


@pytest.mark.parametrize("cls", [HARD, SOFT], ids=["hard", "soft"])
@pytest.mark.parametrize("ref", ["tier1", "native"])
def test_projection_gradients_match(cls, ref):
    """Inside ProjectionMHL: the compress / decompress weight and bias gradients and the input gradient."""
    if ref == "native":
        require_extension(native_available(torch.device("cuda")), LPROJ_EXT_NAME)
    spec = _spec("mid")
    res = {}
    for be in ("cuda", ref):
        torch.manual_seed(5)
        proj = lx.ProjectionMHL(_make(cls, spec, be, rate=0.2), d_model=96).cuda().train()
        torch.manual_seed(6)
        with torch.no_grad():
            proj.decompress.weight.normal_(0, 0.1)        # zero-init by default: give it gradient to pass back
        x = torch.randn(64, 96, device="cuda", generator=torch.Generator(device="cuda").manual_seed(8),
                        requires_grad=True)
        torch.manual_seed(9)
        y = proj(x)
        y.square().sum().backward()
        res[be] = {"y": y.detach(), "x": x.grad} | {n: p.grad for n, p in proj.named_parameters()}
    for k in res["cuda"]:
        assert _rel(res["cuda"][k], res[ref][k]) < 5e-5, f"{k}: rel {_rel(res['cuda'][k], res[ref][k]):.2e}"


# --- bf16 -------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("cls", [HARD, SOFT], ids=["hard", "soft"])
@pytest.mark.parametrize("anchor_mode", ["pairs", "single"])
@pytest.mark.parametrize("geom", ["mid", "fan_in", "canonical"])   # fan_in: the fp32-output path (groups summed)
def test_cuda_bf16_matches_fp32_math_on_the_bf16_values(cls, geom, anchor_mode):
    """A bf16 table + input: identical to the fp32 cuda backend run on the same (bf16-representable) values, up to
    the bf16 rounding of the outputs (the kernels upconvert and accumulate in fp32)."""
    spec = _spec(geom, anchor_mode)
    x, go = _inputs(spec, 256, dtype=torch.bfloat16)
    lo = _make(cls, spec, "cuda", rate=0.2, dtype=torch.bfloat16)
    hi = _make(cls, spec, "cuda", rate=0.2)
    with torch.no_grad():
        hi.weights.copy_(lo.weights.float())
    got, want = _run(lo, x, go), _run(hi, x.float(), go)
    for what, a, b in zip(("value", "grad x", "grad W"), got, want):
        assert _rel(a, b) < 1e-2, f"{what}: rel {_rel(a, b):.2e}"      # bf16 output / grad rounding: 2**-8


@pytest.mark.parametrize("cls", [HARD, SOFT], ids=["hard", "soft"])
def test_cuda_bf16_matches_native_bf16(cls):
    require_extension(native_available(torch.device("cuda")), LPROJ_EXT_NAME)
    spec = _spec("canonical")
    x, go = _inputs(spec, 256, dtype=torch.bfloat16)
    got = _run(_make(cls, spec, "cuda", rate=0.2, dtype=torch.bfloat16), x, go)
    want = _run(_make(cls, spec, "native", rate=0.2, dtype=torch.bfloat16), x, go)
    # native rounds the deciding margin to bf16 before its input-gradient coefficient (_native_ops._flatten); the
    # cuda backend keeps it fp32. Both values / table grads are fp32-accumulated then rounded to bf16 once.
    for what, a, b, tol in zip(("value", "grad x", "grad W"), got, want, (1e-2, 3e-2, 1e-2)):
        assert _rel(a, b) < tol, f"{what}: rel {_rel(a, b):.2e}"


@pytest.mark.parametrize("cls", [HARD, SOFT], ids=["hard", "soft"])
def test_bf16_table_grad_accumulates_in_fp32_under_heavy_row_collision(cls):
    """Every sample addresses the same cells: thousands of tiny contributions into one row. A bf16 accumulator would
    stall; the fp32 buffer (cast once) must match the fp32 run to bf16 resolution."""
    spec = _spec("small")
    B = 1 << 14
    x = torch.ones(B, spec.h_in, spec.d_in, device="cuda") * torch.arange(spec.d_in, device="cuda")
    go = torch.full((B, spec.h_out, spec.d_out), 1e-3, device="cuda")
    lo = _make(cls, spec, "cuda", dtype=torch.bfloat16)
    hi = _make(cls, spec, "cuda")
    with torch.no_grad():
        hi.weights.copy_(lo.weights.float())
    a, b = _run(lo, x.bfloat16(), go)[2], _run(hi, x, go)[2]
    assert _rel(a, b) < 1e-2


# --- determinism, knobs, dispatch ---------------------------------------------------------------------------------

@pytest.mark.parametrize("cls", [HARD, SOFT], ids=["hard", "soft"])
def test_forward_is_deterministic_backward_differs_only_in_summation_order(cls):
    """The forward has no atomics (fixed-order stripe reduction): bit-identical across runs. The backward scatters
    with fp32 atomics (table grad, global) and shared-memory atomics (grad z): NOT bit-deterministic; the run-to-run
    difference must stay at fp32 summation-order level."""
    spec = _spec("canonical")
    x, go = _inputs(spec, 2048)
    m = _make(cls, spec, "cuda", rate=0.2)
    runs = [_run(m, x, go) for _ in range(4)]
    for r in runs[1:]:
        assert torch.equal(r[0], runs[0][0]), "forward must be bit-identical"
        assert _rel(r[1], runs[0][1]) < 1e-5
        assert _rel(r[2], runs[0][2]) < 1e-5


@pytest.mark.parametrize("knobs", [
    dict(fwd_threads=32, bwd_threads=32, rows_per_cta=1, vec=1),
    dict(fwd_threads=128, bwd_threads=256, rows_per_cta=3, vec=2),
    dict(fwd_threads=64, bwd_threads=64, rows_per_cta=4, vec=4, vec_atomics=False),
])
@pytest.mark.parametrize("cls", [HARD, SOFT], ids=["hard", "soft"])
def test_knobs_do_not_change_the_result(knobs, cls):
    spec = _spec("mid")
    x, go = _inputs(spec, 200)
    base = _run(_make(cls, spec, "cuda", rate=0.3), x, go)
    got = _run(_make(cls, spec, "cuda", rate=0.3, knobs=ManifestoCudaKnobs(**knobs)), x, go)
    for a, b in zip(got, base):
        assert _rel(a, b) < 1e-5


@pytest.mark.parametrize("vec", [1, 2, 4, 8])
def test_bf16_vector_widths_agree(vec):
    spec = _spec("mid")
    x, go = _inputs(spec, 128, dtype=torch.bfloat16)
    base = _run(_make(SOFT, spec, "cuda", dtype=torch.bfloat16), x, go)
    got = _run(_make(SOFT, spec, "cuda", dtype=torch.bfloat16, knobs=ManifestoCudaKnobs(vec_bf16=vec)), x, go)
    for a, b in zip(got, base):
        assert _rel(a, b) < 1e-2


def test_auto_picks_cuda_in_training_and_in_eval():
    spec = _spec("mid")
    x = torch.randn(64, spec.h_in, spec.d_in, device="cuda")
    big = torch.randn(SOFT._LARGE_BATCH, spec.h_in, spec.d_in, device="cuda")
    h, s = HARD(spec).cuda().train(), SOFT(spec).cuda().train()
    h(x), s(x)
    assert (h.last_backend, s.last_backend) == ("cuda", "cuda")
    s(big)
    assert s.last_backend == "cuda"                    # training: cuda at large batch too (replaces tier1 there)
    h.eval(), s.eval()
    with torch.no_grad():
        for m in (h, s):
            for inp in (x, big):                       # eval: cuda at every batch size (replaces pure / tier1)
                m(inp)
                assert m.last_backend == "cuda"


def test_eval_non_cuda_inputs_keep_the_old_eval_paths_quietly(recwarn):
    """A legitimate non-cuda case in eval (an fp16 table has no kernel; a CPU input) falls through to the previous
    eval choice without any fallback warning."""
    spec = _spec("mid")
    h = HARD(spec).cuda().eval().half()
    s = SOFT(spec).cuda().eval().half()
    with torch.no_grad():
        h(torch.randn(64, spec.h_in, spec.d_in, device="cuda").half())
        s(torch.randn(64, spec.h_in, spec.d_in, device="cuda").half())
        hc, sc = HARD(spec).eval(), SOFT(spec).eval()
        hc(torch.randn(8, spec.h_in, spec.d_in))
        sc(torch.randn(8, spec.h_in, spec.d_in))
    assert (h.last_backend, s.last_backend, hc.last_backend, sc.last_backend) == ("pure_eval", "pure", "pure_eval", "pure")
    assert not [w for w in recwarn if "FAST CUDA PATH" in str(w.message)]


@pytest.mark.parametrize("d_out,vec", [(4100, 4), (2050, 2), (1025, 1)])
@pytest.mark.parametrize("cls", [HARD, SOFT], ids=["hard", "soft"])
def test_wide_d_out_is_not_a_cuda_case(cls, d_out, vec):
    """A d_out one CTA cannot cover (> 1024 threads at the widest vector dividing it: d_out % 4 == 0 -> vec 4, cap
    4096; d_out % 4 == 2 -> vec 2, cap 2048; odd -> vec 1, cap 1024) is not a cuda case: 'auto' falls back in training
    and in eval instead of failing at launch, and an explicit 'cuda' says why."""
    spec = LUTSpec(h_in=1, h_out=1, tph=2, nap=2, d_in=4, d_out=d_out, anchor_mode="pairs")
    x = torch.randn(4, 1, 4, device="cuda")
    m = cls(spec).cuda().train()
    m(x).sum().backward()
    assert m.last_backend != "cuda"
    m.eval()
    with torch.no_grad():
        m(x)
    assert m.last_backend != "cuda"
    with pytest.raises(RuntimeError, match=rf"d_out={d_out} is too wide for the fused kernel \(max {vec * 1024} "):
        cls(spec, backend="cuda").cuda().train()(x)


@pytest.mark.parametrize("pure_cls, cls", [(lx.ManifestoHardLUT, HARD), (lx.ManifestoSoftLUT, SOFT)], ids=["hard", "soft"])
@pytest.mark.parametrize("d_out", [4096, 1026])
def test_cuda_matches_the_fp64_oracle_at_the_d_out_boundary(pure_cls, cls, d_out):
    """Eligible widths at the edge of a launch class: d_out 4096 (vec 4, exactly 1024 threads per CTA, the largest
    eligible width at the default knobs) and d_out 1026 (vec 2, the 2-wide load path). Value, grad x and grad W of the
    cuda backend in fp32 against the pure cartridge in float64."""
    spec = LUTSpec(h_in=2, h_out=2, tph=4, nap=3, d_in=8, d_out=d_out, anchor_mode="pairs")
    x, go = _inputs(spec, 96)
    m = _make(cls, spec, "cuda")
    got = _run(m, x, go)
    assert m.last_backend == "cuda"
    oracle = pure_cls(spec, seed=7, weight_init_std=0.5).cuda().train().double()
    xo = x.double().requires_grad_(True)
    yo = oracle(xo)
    yo.backward(go.double())
    for what, a, b in zip(("value", "grad x", "grad W"), got, (yo.detach(), xo.grad, oracle.weights.grad)):
        assert _rel(a.double(), b) < 1e-5, f"d_out {d_out} {what}: rel {_rel(a.double(), b):.2e}"


def test_explicit_cuda_backend_errors_off_cuda():
    spec = _spec("small")
    m = HARD(spec, backend="cuda")
    with pytest.raises(RuntimeError, match="The input is on the CPU, but the fused kernel runs on CUDA only"):
        m(torch.randn(4, spec.h_in, spec.d_in))


def test_fp16_table_has_no_kernel_and_auto_falls_back_quietly(recwarn):
    spec = _spec("mid")
    m = HARD(spec).cuda().train().half()
    m(torch.randn(8, spec.h_in, spec.d_in, device="cuda").half()).float().sum().backward()
    assert m.last_backend != "cuda"
    assert not [w for w in recwarn if "FAST CUDA PATH" in str(w.message)]


def test_missing_extension_is_reported_then_auto_moves_on(monkeypatch):
    """The extension 'fails to build': auto reports the classified cause once and runs the next backend; strict mode
    raises instead; an explicit backend='cuda' raises with the cause."""
    _fallback._reset_for_tests()
    monkeypatch.setattr(fmc, "_TRIED", True)
    monkeypatch.setattr(fmc, "_EXT", None)
    _fallback.record_build_failure(fmc.EXT_NAME, ModuleNotFoundError("No module named 'setuptools'", name="setuptools"))
    spec = _spec("mid")
    x = torch.randn(8, spec.h_in, spec.d_in, device="cuda")
    monkeypatch.delenv(_fallback.STRICT_ENV, raising=False)
    m = HARD(spec).cuda().train()
    with pytest.warns(RuntimeWarning, match="lutorch_ex_fused_manifesto"):
        m(x)
    assert m.last_backend in ("native", "tier1")
    monkeypatch.setenv(_fallback.STRICT_ENV, "1")
    with pytest.raises(_fallback.NativeUnavailableError, match="setuptools_missing"):
        SOFT(spec).cuda().train()(x)
    monkeypatch.delenv(_fallback.STRICT_ENV)
    with pytest.raises(_fallback.NativeUnavailableError, match="lutorch_ex_fused_manifesto"):
        HARD(spec, backend="cuda").cuda().train()(x)
    # eval: the same loud fallback (to the previous eval path), and a hard error under the strict flag
    _fallback._reset_for_tests()
    _fallback.record_build_failure(fmc.EXT_NAME, ModuleNotFoundError("No module named 'setuptools'", name="setuptools"))
    e = SOFT(spec).cuda().eval()
    with torch.no_grad():
        with pytest.warns(RuntimeWarning, match="lutorch_ex_fused_manifesto"):
            e(x)
        assert e.last_backend == "pure"
        monkeypatch.setenv(_fallback.STRICT_ENV, "1")
        with pytest.raises(_fallback.NativeUnavailableError, match="setuptools_missing"):
            HARD(spec).cuda().eval()(x)
    _fallback._reset_for_tests()


def test_state_dict_unchanged_by_the_cuda_buffers():
    """The int16 anchor copies are non-persistent: checkpoints are interchangeable with the other backends."""
    spec = _spec("small")
    assert set(HARD(spec).state_dict()) == set(lx.ManifestoHardLUT(spec).state_dict())
    assert set(SOFT(spec).state_dict()) == set(lx.ManifestoSoftLUT(spec).state_dict())
