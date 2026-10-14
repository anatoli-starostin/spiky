"""FusedManifestoHardLUT / FusedManifestoSoftLUT: their CUDA kernels (csrc/fused_manifesto.cu).

Correctness gate: the kernels must reproduce the reference classes ManifestoHardLUT / ManifestoSoftLUT (value, input gradient, table gradient,
with weight="softsign" the two temperatures' gradients, and, inside ProjectionMHL, the projection gradients) for both
cartridges and both weight functions, both anchor modes, with and without table dropout, float32 and bfloat16, at small
geometries and at the canonical h16 d48 tph64 nap8. The weight-gradient scatter uses
float32 atomics, so run-to-run results differ in summation order; that is measured here, not assumed.
"""
import pytest
import torch

import spiky.lutorch_ex as lx
from spiky.lutorch_ex.cartridges import _fallback
from spiky.lutorch_ex.cartridges import _fused_manifesto_cuda as fmc
from spiky.lutorch_ex.cartridges._fused_manifesto_cuda import ManifestoCudaKnobs
from spiky.lutorch_ex.lut_spec import LUTSpec
from spiky.lutorch_ex.tests._requires_extension import requires_extension

# No CUDA device: skip. A CUDA device without the extension: fail (tests/_requires_extension.py).
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")


@pytest.fixture(autouse=True)
def _needs_fused_manifesto_ext():
    requires_extension("manifesto")

HARD, SOFT = lx.FusedManifestoHardLUT, lx.FusedManifestoSoftLUT
#: The reference class of each fused class (same state_dict, same seed -> same anchors and tables).
REF = {HARD: lx.ManifestoHardLUT, SOFT: lx.ManifestoSoftLUT}
WEIGHTS = ["rational", "softsign"]
#: The temperature gradients are sums over every (sample, group, table): a looser bound than the per-element ones.
TEMP_TOL = 1e-4
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


def _make(cls, spec, rate=0.0, dtype=torch.float32, **kw):
    # A larger init std than the default so the tables' differences dominate rounding in the comparisons.
    return cls(spec, seed=7, weight_init_std=0.5, table_dropout_rate=rate, **kw).cuda().train().to(dtype)


def _run(mod, x, go, seed=1234):
    """Forward + backward -> {"value", "grad x", "grad W"} (+ the two temperature gradients for weight="softsign").
    The RNG is reseeded so table dropout draws the same keep pattern in the fused and the reference class."""
    xx = x.detach().clone().requires_grad_(True)
    mod.zero_grad(set_to_none=True)
    torch.manual_seed(seed)
    y = mod(xx)
    y.backward(go.to(y.dtype))
    return _grads(mod, y, xx)


def _grads(mod, y, xx):
    out = {"value": y.detach().float(), "grad x": xx.grad.float(), "grad W": mod.weights.grad.float()}
    if mod.weight == "softsign":
        out["grad T_soft"] = mod.log_soft_score_temp.grad.float().reshape(1)
        out["grad T_sel"] = mod.log_select_temp.grad.float().reshape(1)
    return out


def _check(got, want, tol, msg=""):
    for k, b in want.items():
        a = got[k]
        t = max(tol, TEMP_TOL) if k.startswith("grad T") else tol
        assert _rel(a.double(), b.double()) < t, f"{msg} {k}: rel {_rel(a.double(), b.double()):.2e}"


def _rel(a, b):
    return ((a - b).abs().max() / b.abs().max().clamp_min(1e-30)).item()


def _inputs(spec, B, dtype=torch.float32, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(B, spec.h_in, spec.d_in, device="cuda", generator=g).to(dtype)
    go = torch.randn(B, spec.h_out, spec.d_out, device="cuda", generator=g)
    return x, go


# --- float32 equivalence against the reference classes ----------------------------------------------------------------

@pytest.mark.parametrize("weight", WEIGHTS)
@pytest.mark.parametrize("cls", [HARD, SOFT], ids=["hard", "soft"])
@pytest.mark.parametrize("geom", ["small", "mid", "fan_in", "canonical"])
@pytest.mark.parametrize("anchor_mode", ["pairs", "single"])
@pytest.mark.parametrize("rate", [0.0, 0.3])
def test_fused_matches_the_reference_class_fp32(cls, geom, anchor_mode, rate, weight):
    spec = _spec(geom, anchor_mode)
    x, go = _inputs(spec, 512 if geom == "canonical" else 192)
    got = _run(_make(cls, spec, rate, weight=weight), x, go)
    want = _run(_make(REF[cls], spec, rate, weight=weight), x, go)
    # float32 both sides, same cells; only the summation order differs.
    _check(got, want, 2e-5, f"{cls.__name__}({weight}) vs {REF[cls].__name__}")


@pytest.mark.parametrize("weight", WEIGHTS)
@pytest.mark.parametrize("pure_cls, cls", [(lx.ManifestoHardLUT, HARD), (lx.ManifestoSoftLUT, SOFT)], ids=["hard", "soft"])
@pytest.mark.parametrize("anchor_mode", ["pairs", "single"])
@pytest.mark.parametrize("geom", ["small", "mid"])
@pytest.mark.parametrize("rate", [0.0, 0.3])
def test_cuda_fp32_matches_the_pure_cartridge_in_fp64(pure_cls, cls, anchor_mode, geom, weight, rate):
    """The oracle: the reference ManifestoHardLUT / ManifestoSoftLUT in float64 (same seed -> same anchors and tables), with
    the same table-dropout draw."""
    spec = _spec(geom, anchor_mode)
    x, go = _inputs(spec, 160)
    oracle = pure_cls(spec, seed=7, weight_init_std=0.5, table_dropout_rate=rate, weight=weight).cuda().train()
    oracle = oracle.double()
    xo = x.double().requires_grad_(True)
    torch.manual_seed(1234)
    keep = oracle._table_dropout_mask(160, "cuda", torch.float32)           # the draw _run makes, reproduced
    if keep is not None:
        oracle._table_dropout_mask = lambda B, dev, dt: keep.to(dt)
    yo = oracle(xo)
    yo.backward(go.double())
    got = _run(_make(cls, spec, rate, weight=weight), x, go)
    _check(got, _grads(oracle, yo, xo), 1e-5, f"{cls.__name__}({weight})")


def test_soft_oracle_gradient_is_the_true_derivative_gradcheck():
    """Soft's input / table gradient is the exact derivative of its blend. The kernels are float32-only, so gradcheck
    (which needs float64) runs on the float64 reference oracle the fused class is checked against above; inputs are
    kept away from the sign / argmin decision boundaries, where the function is not differentiable."""
    spec = _spec("small")
    m = lx.ManifestoSoftLUT(spec, seed=7, weight_init_std=0.5).cuda().train().double()
    g = torch.Generator(device="cuda").manual_seed(3)
    x = (torch.randn(6, spec.h_in, spec.d_in, device="cuda", generator=g, dtype=torch.float64)).requires_grad_(True)
    assert torch.autograd.gradcheck(lambda t: m(t), (x,), eps=1e-6, atol=1e-6)


@pytest.mark.parametrize("weight", WEIGHTS)
@pytest.mark.parametrize("cls", [HARD, SOFT], ids=["hard", "soft"])
def test_projection_gradients_match(cls, weight):
    """Inside ProjectionMHL: the compress / decompress weight and bias gradients and the input gradient (and every
    other parameter's, the temperatures included), fused against the reference class."""
    spec = _spec("mid")
    res = {}
    for name, c in (("fused", cls), ("reference", REF[cls])):
        torch.manual_seed(5)
        proj = lx.ProjectionMHL(_make(c, spec, rate=0.2, weight=weight), d_model=96).cuda().train()
        torch.manual_seed(6)
        with torch.no_grad():
            proj.decompress.weight.normal_(0, 0.1)        # zero-init by default: give it gradient to pass back
        x = torch.randn(64, 96, device="cuda", generator=torch.Generator(device="cuda").manual_seed(8),
                        requires_grad=True)
        torch.manual_seed(9)
        y = proj(x)
        y.square().sum().backward()
        res[name] = {"y": y.detach(), "x": x.grad} | {n: p.grad for n, p in proj.named_parameters()}
    for k in res["fused"]:
        tol = TEMP_TOL if "temp" in k else 5e-5
        assert _rel(res["fused"][k], res["reference"][k]) < tol, \
            f"{k}: rel {_rel(res['fused'][k], res['reference'][k]):.2e}"


# --- bfloat16
#-------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("weight", WEIGHTS)
@pytest.mark.parametrize("cls", [HARD, SOFT], ids=["hard", "soft"])
@pytest.mark.parametrize("anchor_mode", ["pairs", "single"])
@pytest.mark.parametrize("geom", ["mid", "fan_in", "canonical"])   # fan_in: the float32-output path (groups summed)
def test_cuda_bf16_matches_fp32_math_on_the_bf16_values(cls, geom, anchor_mode, weight):
    """A bfloat16 table + input: identical to the float32 fused run on the same (bfloat16-representable) values, up to
    the bfloat16 rounding of the outputs (the kernels upconvert and accumulate in float32)."""
    spec = _spec(geom, anchor_mode)
    x, go = _inputs(spec, 256, dtype=torch.bfloat16)
    lo = _make(cls, spec, rate=0.2, dtype=torch.bfloat16, weight=weight)
    hi = _make(cls, spec, rate=0.2, weight=weight)
    hi.load_state_dict({k: v.float() for k, v in lo.state_dict().items()})   # the same bfloat16-representable values
    _check(_run(lo, x, go), _run(hi, x.float(), go), 1e-2)                   # bfloat16 output / grad rounding: 2**-8


@pytest.mark.parametrize("cls", [HARD, SOFT], ids=["hard", "soft"])
def test_bf16_table_grad_accumulates_in_fp32_under_heavy_row_collision(cls):
    """Every sample addresses the same cells: thousands of tiny contributions into one row. A bfloat16 accumulator would
    stall; the float32 buffer (cast once) must match the float32 run to bfloat16 resolution."""
    spec = _spec("small")
    B = 1 << 14
    x = torch.ones(B, spec.h_in, spec.d_in, device="cuda") * torch.arange(spec.d_in, device="cuda")
    go = torch.full((B, spec.h_out, spec.d_out), 1e-3, device="cuda")
    lo = _make(cls, spec, dtype=torch.bfloat16)
    hi = _make(cls, spec)
    with torch.no_grad():
        hi.weights.copy_(lo.weights.float())
    a, b = _run(lo, x.bfloat16(), go)["grad W"], _run(hi, x, go)["grad W"]
    assert _rel(a, b) < 1e-2


# --- determinism, knobs, dispatch ---------------------------------------------------------------------------------

@pytest.mark.parametrize("weight", WEIGHTS)
@pytest.mark.parametrize("cls", [HARD, SOFT], ids=["hard", "soft"])
def test_forward_is_deterministic_backward_differs_only_in_summation_order(cls, weight):
    """The forward has no atomics (fixed-order stripe reduction): bit-identical across runs. The backward scatters
    with float32 atomics (table grad, global) and shared-memory atomics (grad z): NOT bit-deterministic; the run-to-run
    difference must stay at float32 summation-order level."""
    spec = _spec("canonical")
    x, go = _inputs(spec, 2048)
    m = _make(cls, spec, rate=0.2, weight=weight)
    runs = [_run(m, x, go) for _ in range(4)]
    for r in runs[1:]:
        assert torch.equal(r["value"], runs[0]["value"]), "forward must be bit-identical"
        _check({k: v for k, v in r.items() if k != "value"}, {k: v for k, v in runs[0].items() if k != "value"}, 1e-5)


@pytest.mark.parametrize("knobs", [
    dict(fwd_threads=32, bwd_threads=32, rows_per_cta=1, vec=1),
    dict(fwd_threads=128, bwd_threads=256, rows_per_cta=3, vec=2),
    dict(fwd_threads=64, bwd_threads=64, rows_per_cta=4, vec=4, vec_atomics=False),
])
@pytest.mark.parametrize("weight", WEIGHTS)
@pytest.mark.parametrize("cls", [HARD, SOFT], ids=["hard", "soft"])
def test_knobs_do_not_change_the_result(knobs, cls, weight):
    spec = _spec("mid")
    x, go = _inputs(spec, 200)
    base = _run(_make(cls, spec, rate=0.3, weight=weight), x, go)
    got = _run(_make(cls, spec, rate=0.3, knobs=ManifestoCudaKnobs(**knobs), weight=weight), x, go)
    _check(got, base, 1e-5)


@pytest.mark.parametrize("vec", [1, 2, 4, 8])
def test_bf16_vector_widths_agree(vec):
    spec = _spec("mid")
    x, go = _inputs(spec, 128, dtype=torch.bfloat16)
    base = _run(_make(SOFT, spec, dtype=torch.bfloat16), x, go)
    got = _run(_make(SOFT, spec, dtype=torch.bfloat16, knobs=ManifestoCudaKnobs(vec_bf16=vec)), x, go)
    _check(got, base, 1e-2)


def test_non_cuda_cases_raise_naming_the_reference_class():
    """No fallback: a float16 table has no kernel and a CPU input is not a CUDA case; both raise, in
    training and in eval, naming the reference classes."""
    spec = _spec("mid")
    for cls in (HARD, SOFT):
        for train in (True, False):
            m = cls(spec).cuda().train(train).half()
            with torch.no_grad(), pytest.raises(RuntimeError, match=r"table dtype torch\.float16 .*ManifestoHardLUT / "
                                                                     r"ManifestoSoftLUT"):
                m(torch.randn(64, spec.h_in, spec.d_in, device="cuda").half())
            c = cls(spec).train(train)
            with torch.no_grad(), pytest.raises(RuntimeError, match="The input is on the CPU"):
                c(torch.randn(8, spec.h_in, spec.d_in))


@pytest.mark.parametrize("d_out,vec", [(4100, 4), (2050, 2), (1025, 1)])
@pytest.mark.parametrize("cls", [HARD, SOFT], ids=["hard", "soft"])
def test_wide_d_out_is_not_a_cuda_case(cls, d_out, vec):
    """A d_out one CTA cannot cover (> 1024 threads at the widest vector dividing it: d_out % 4 == 0 -> vec 4, cap
    4096; d_out % 4 == 2 -> vec 2, cap 2048; odd -> vec 1, cap 1024) is not a case the kernels cover: it raises, in
    training and in eval, saying why, instead of failing at launch."""
    spec = LUTSpec(h_in=1, h_out=1, tph=2, nap=2, d_in=4, d_out=d_out, anchor_mode="pairs")
    x = torch.randn(4, 1, 4, device="cuda")
    msg = rf"d_out={d_out} is too wide for the fused kernel \(max {vec * 1024} "
    with pytest.raises(RuntimeError, match=msg):
        cls(spec).cuda().train()(x)
    with torch.no_grad(), pytest.raises(RuntimeError, match=msg):
        cls(spec).cuda().eval()(x)


@pytest.mark.parametrize("weight", WEIGHTS)
@pytest.mark.parametrize("pure_cls, cls", [(lx.ManifestoHardLUT, HARD), (lx.ManifestoSoftLUT, SOFT)], ids=["hard", "soft"])
@pytest.mark.parametrize("d_out", [4096, 1026])
def test_cuda_matches_the_fp64_oracle_at_the_d_out_boundary(pure_cls, cls, d_out, weight):
    """Eligible widths at the edge of a launch class: d_out 4096 (vec 4, exactly 1024 threads per CTA, the largest
    eligible width at the default knobs) and d_out 1026 (vec 2, the 2-wide load path). Value, grad x and grad W of the
    fused class in float32 against the reference class in float64."""
    spec = LUTSpec(h_in=2, h_out=2, tph=4, nap=3, d_in=8, d_out=d_out, anchor_mode="pairs")
    x, go = _inputs(spec, 96)
    got = _run(_make(cls, spec, weight=weight), x, go)
    oracle = pure_cls(spec, seed=7, weight_init_std=0.5, weight=weight).cuda().train().double()
    xo = x.double().requires_grad_(True)
    yo = oracle(xo)
    yo.backward(go.double())
    _check(got, _grads(oracle, yo, xo), 1e-5, f"d_out {d_out}")


def test_missing_extension_raises_in_training_and_in_eval(monkeypatch):
    """The extension 'fails to build': both classes raise with the classified cause, in training and in eval."""
    _fallback._reset_for_tests()
    monkeypatch.setattr(fmc.EXT, "tried", True)
    monkeypatch.setattr(fmc.EXT, "module", None)
    _fallback.record_build_failure(fmc.EXT.name, ModuleNotFoundError("No module named 'setuptools'", name="setuptools"))
    spec = _spec("mid")
    x = torch.randn(8, spec.h_in, spec.d_in, device="cuda")
    for cls in (HARD, SOFT):
        with pytest.raises(_fallback.NativeUnavailableError, match="setuptools_missing"):
            cls(spec).cuda().train()(x)
        with torch.no_grad(), pytest.raises(_fallback.NativeUnavailableError, match="lutorch_ex_fused_manifesto"):
            cls(spec).cuda().eval()(x)
    _fallback._reset_for_tests()


@pytest.mark.parametrize("weight", WEIGHTS)
def test_state_dict_unchanged_by_the_cuda_buffers(weight):
    """The int16 anchor copies are non-persistent: checkpoints are interchangeable with the reference classes."""
    spec = _spec("small")
    assert set(HARD(spec, weight=weight).state_dict()) == set(lx.ManifestoHardLUT(spec, weight=weight).state_dict())
    assert set(SOFT(spec, weight=weight).state_dict()) == set(lx.ManifestoSoftLUT(spec, weight=weight).state_dict())
