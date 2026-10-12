"""Spec amendment D: the fused-vs-reference equivalence harness, the one REQUIRED check binding each fused class to its
reference class.

Per cartridge (FusedManifestoHardLUT / FusedManifestoSoftLUT with both weights, FusedConfidenceLUT with read_top_n 1
and 2, FusedQuantisedConfidenceLUT), per dtype (fp32; bf16 table + input; fp16 input on an fp32 table; bf16 table with
an fp32 input; fp16 input on a bf16 table -- every table / input combination the fused classes declare), per anchor mode
and head pattern, wherever the fused class covers the case (the quantised op: fp32, pairs, read_top_n=2 only): the fused
class and its reference class, loaded with the same state_dict, are fed the same input and output gradient, in
training and in eval, and compared on every leaf: the value, the input gradient, the table gradient and every learned
scalar's gradient (soft-sign temperatures; confidence beta / gamma / tau).

Tolerances (relative norm error ``||a - b|| / ||b||``), each from the one way the two can legitimately differ:
* Both sides take the same discrete decisions: addressing runs in fp32 on the same values in both (a margin is one
  fp32 subtraction), so the cells read are identical and only the arithmetic around them can differ.
* fp32 leaf -- summation order. The kernels reduce in a different order (and the table gradient with atomics), so
  each sum carries a different fp32 round-off, a random walk of O(sqrt(K) * 2**-24) relative for K terms. Element-wise
  leaves sum K <= 10**4 terms (~1e-6): ``FP32_TOL = 1e-5``.
* fp32 learned scalar -- an ACCURACY test against fp64. A learned scalar's gradient is ONE sum over every (sample,
  group, table, output), K = B * n_groups * tph * d_out terms (up to ~2.5e7 here), and it cancels: |sum| is down to
  2.9e-4 of S (below) for a soft-sign temperature at fan-in batch 24576. Round-off scales with the sum of |terms|, not
  with |sum|: for a K-term sum in any order, |error| <= lambda * sqrt(K) * u * sum |x_i| with high probability
  (Higham & Mary, "A new approach to probabilistic rounding error analysis", SISC 2019), lambda = 4, u = 2**-24.
  So each side is compared with t, the gradient of the reference class run in fp64 (eager, same state, input and
  output gradient): ``|side - t| < max(max(FP32_SCALAR_TOL, u_input) * |t|, 4 * sqrt(K) * 2**-24 * S)``, with S the
  sum over SCALE_CHUNKS batch chunks of |chunk gradient| (fp64) -- a lower bound on sum |x_i|, so the bound is never
  looser than the derivation. The two sides are also compared with each other within that bound. Both sides being
  wrong the same way fails here; a difference test would not see it. (Before: ``max(FP32_SCALAR_TOL,
  4 * sqrt(K) * 2**-24)`` RELATIVE to |sum|, i.e. |sum| in place of sum |x_i|. That holds only without cancellation:
  that temperature measured 1.1e-3 against 6.3e-4.) The fp64 reference adds ~13 s over the matrix.
* bf16 / fp16 leaf -- one rounding. Both classes compute in fp32 and round each leaf once (fused: in the kernel's
  epilogue; reference: tests/test_reference_low_precision.py proves it bit-exact). Two roundings of fp32 values that
  differ only by the fp32 tolerance above land on the same low-precision value except where they straddle a
  rounding boundary, and then differ by one ulp. One ulp of v is 2u * 2**floor(log2|v|), between u|v| and 2u|v|:
  u = 2**-8 (bf16) / 2**-11 (fp16) is HALF an ulp at the bottom of a binade. The bound is therefore ``2u``. (Before:
  u. On a tensor leaf a lone straddle dilutes in the norm and u held; a scalar leaf is one value, and one straddle
  there always exceeds u -- measured 4.95e-3 = 1.27u on a bf16 temperature of 0.79.) A learned scalar in a bf16 module
  is a bf16 leaf: fused vs reference within 2u|reference| plus the fp32 bound above (the fp32 values may differ by
  that before rounding), and each side vs fp64 within the fp32 bound plus the one final rounding, u * (|t| + it).
* Leaves that stay fp32 in a low-precision run (every parameter's gradient when only the input is fp16) take the
  fp32 tolerances, widened to the input dtype's u: they are computed from the rounded output gradient and input.

Measured 2026-10-12 (RTX 5090) over the 672-case matrix, worst error as a share of its bound: fp32 value 2.6%, grad x
3.6%, grad W 8.5%; bf16 value 0.5%, grad x 1.2%, grad W 2.2%; fp16 value 2.5%, grad x 2.0%; fp32 learned scalars vs
fp64 1.9% (fused vs reference 1.4%); bf16 learned scalars fused vs reference within one ulp (equal in 10 of 10 repeats
of the worst case), vs fp64 up to 76%. That 76% is by construction: the final rounding alone may use half an ulp,
nearly all of u|t| when |t| sits just above a power of two; the bound is rigorous (the fp32 bound plus that
rounding), so it does not fail by chance. The scalar bound is loose where the sum cancels: up to 1.84|t| at fan-in
batch 24576, where a fused scalar has to be off by ~3x to fail (by 3-30% at the canonical geometry; a tensor leaf by
1e-5). Quantised vs the eager reference: integers and values bit-identical. (The harness found one
real double rounding: a low-precision input gathered for several groups (fan-out) summed their input gradients in the
input dtype, 78% of u; the fused classes now cast before the gather.)

The quantised pair, in two layers. Its value is a staircase: the power-of-two exponents q and k' are floors (round
half up) of fp32 transcendental expressions, so two implementations that differ by an ulp anywhere in the expression
can floor a near-boundary value differently (one table's weight then changes by a factor 2). Two independent sources,
tested separately:
* LAYER 1, the oracle -- fused vs the EAGER (uncompiled) reference: BIT FOR BIT, no tolerance, no gate. The reference's
  _pow2.blend_exponents spells out the kernel's operation order (left-fold sums, true fp32 divisions; see its
  docstring for the trade-off), so the same fp32 operations give the same integers, and the read the same values.
  Measured: 0 integer disagreements and bit-identical output rows in 28 runs (8.4M tables each; nap 3, 5, 6, 7, 8; two
  geometries; init and after training). Gradients keep the fp32 tolerances above (the backward's sums differ in order).
* LAYER 2, compile trust -- the compiled reference vs the eager reference, in fp64 on purpose: in fp32 Inductor's
  divide-by-constant rewrite (a reciprocal multiply that Triton contracts with the next add into one FMA, in k') and
  Triton's approximate fp32 division (div.full.f32, in q) flip ~1e-7 of the exponents (the left folds and the
  transcendentals compile bit-identically); in fp64 none flip (0 in 192 runs), so a tight bound checks that compile
  kept the code logic; layer 1 pins the fp32 numbers.
Constructed exact ties check the rule itself: reference and kernel both round half up. Natural exact ties: zero.

Kernel-specific checks (launch knobs, determinism, the d_out launch boundary, bf16 accumulation under row collision,
table dropout) live in test_fused_manifesto_cuda.py, test_fused_confidence.py and test_table_dropout_and_cell_tv.py.
"""
import pytest
import torch

from spiky.lutorch_ex import (ConfidenceLUT, FusedConfidenceLUT, FusedManifestoHardLUT, FusedManifestoSoftLUT,
                              FusedQuantisedConfidenceLUT, LUTSpec, ManifestoHardLUT, ManifestoSoftLUT,
                              QuantisedConfidenceLUT)
from spiky.lutorch_ex.tests._requires_extension import requires_extension

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")

FP32_TOL = 1e-5
FP32_SCALAR_TOL = 1e-4
FP32_U = 2.0 ** -24
U = {torch.bfloat16: 2.0 ** -8, torch.float16: 2.0 ** -11}
#: Batch chunks of a learned scalar's cancellation scale S (module docstring).
SCALE_CHUNKS = 32
TENSOR_LEAVES = ("value", "grad x", "grad weights")
SEED = 7

#: (id, fused class, reference class, kwargs, extension)
CARTRIDGES = [
    ("ManifestoHard", FusedManifestoHardLUT, ManifestoHardLUT, {}, "manifesto"),
    ("ManifestoSoft", FusedManifestoSoftLUT, ManifestoSoftLUT, {}, "manifesto"),
    ("ManifestoHard[softsign]", FusedManifestoHardLUT, ManifestoHardLUT, dict(weight="softsign"), "manifesto"),
    ("ManifestoSoft[softsign]", FusedManifestoSoftLUT, ManifestoSoftLUT, dict(weight="softsign"), "manifesto"),
    ("Confidence n=1", FusedConfidenceLUT, ConfidenceLUT,
     dict(read_top_n=1, beta_init=1.7, gamma_init=0.8), "confidence"),
    ("Confidence n=2", FusedConfidenceLUT, ConfidenceLUT,
     dict(read_top_n=2, beta_init=1.7, gamma_init=0.8, read_tau_init=0.6), "confidence"),
    ("Quantised n=2", FusedQuantisedConfidenceLUT, QuantisedConfidenceLUT,
     dict(read_top_n=2, beta_init=1.7, gamma_init=0.8, read_tau_init=0.6), "quantised"),
]
#: (id, table dtype, input dtype)
DTYPES = [("fp32", torch.float32, torch.float32), ("bf16", torch.bfloat16, torch.bfloat16),
          ("fp16-input", torch.float32, torch.float16), ("bf16-table", torch.bfloat16, torch.float32),
          ("fp16-input-bf16-table", torch.bfloat16, torch.float16)]
#: What each fused class covers (everything not listed: all dtypes, both anchor modes).
COVERS = {"Quantised n=2": dict(dtypes={"fp32"}, anchor_modes={"pairs"})}
#: (id, h_in, h_out, d, tph, nap, batch); d is d_in = d_out, or a (d_in, d_out) pair. The fan-in / fan-out rows at
#: batch 1 and 24576 and at d_in != d_out are the combinations the superseded test_fused_equivalence.py covered.
GEOMS = [("small-per-head", 4, 4, 12, 6, 5, 128), ("fan-in", 4, 1, 12, 6, 5, 128), ("fan-out", 1, 4, 12, 6, 5, 128),
         ("batch-1", 4, 4, 12, 6, 5, 1), ("canonical", 16, 16, 48, 64, 8, 512), ("large-batch", 4, 4, 12, 6, 5, 24576),
         ("fan-in-batch-1", 4, 1, 12, 6, 5, 1), ("fan-out-batch-1", 1, 4, 12, 6, 5, 1),
         ("fan-in-large-batch", 4, 1, 12, 6, 5, 24576), ("fan-out-large-batch", 1, 4, 12, 6, 5, 24576),
         ("fan-in-d12-to-8", 4, 1, (12, 8), 6, 5, 128), ("fan-out-d12-to-8", 1, 4, (12, 8), 6, 5, 128)]
#: Rows run in some anchor modes only (everything not listed: both). d_in != d_out with single anchors is covered by
#: test_single_anchor.py::test_single_fused_equals_reference_f32.
GEOM_ANCHORS = {"fan-in-d12-to-8": {"pairs"}, "fan-out-d12-to-8": {"pairs"}}


def _rel(a, b):
    return ((a.double() - b.double()).norm() / b.double().norm().clamp_min(1e-30)).item()


def _tol(leaf_dtype: torch.dtype, input_dtype: torch.dtype) -> float:
    """The relative-norm tolerance of one tensor leaf (module docstring): one rounding straddle, else fp32."""
    if leaf_dtype in U:
        return 2.0 * U[leaf_dtype]
    return max(FP32_TOL, U.get(input_dtype, 0.0))


def _scalar_truth(Ref, kw, spec, state, x, go):
    """{leaf: (t, S)} for every learned-scalar leaf: t its gradient from the reference class in fp64 (eager, the same
    state, input and output gradient), S the cancellation scale -- the sum of |gradient| over SCALE_CHUNKS batch
    chunks, a lower bound on the sum of |terms| of that gradient's sum (module docstring)."""
    m = _Eager(Ref(spec, seed=SEED, **kw).cuda().double())
    m.m.load_state_dict({k: v.double() for k, v in state.items()})
    m.m.train()

    def scalars(xc, gc):
        return {k: v for k, v in _leaves(m, xc.double(), gc.double()).items() if k not in TENSOR_LEAVES}
    t = scalars(x, go)
    chunks = [scalars(xc, gc) for xc, gc in zip(x.chunk(min(x.shape[0], SCALE_CHUNKS)),
                                                go.chunk(min(x.shape[0], SCALE_CHUNKS)))]
    return {k: (float(t[k]), float(sum(c[k].abs() for c in chunks))) for k in t}


def _checks(a, b, truth, input_dtype, n_terms):
    """(leaf, check, error, bound) for every leaf: a = fused, b = reference (module docstring)."""
    for leaf in b:
        if leaf in truth:                            # a learned scalar: accuracy of each side, then the two sides
            t, S = truth[leaf]
            u = U.get(b[leaf].dtype, 0.0)
            fp32 = max(max(FP32_SCALAR_TOL, U.get(input_dtype, 0.0)) * abs(t), 4.0 * n_terms ** 0.5 * FP32_U * S)
            for side, v in (("fused vs fp64", a[leaf]), ("reference vs fp64", b[leaf])):
                yield leaf, side, abs(float(v) - t), fp32 + u * (abs(t) + fp32)
            yield leaf, "fused vs reference", abs(float(a[leaf]) - float(b[leaf])), fp32 + 2.0 * u * abs(float(b[leaf]))
        else:
            yield leaf, "fused vs reference", _rel(a[leaf], b[leaf]), _tol(b[leaf].dtype, input_dtype)


def _leaves(m, x, go):
    xi = x.clone().requires_grad_(True)
    y = m(xi)
    y.backward(go)
    out = {"value": y.detach(), "grad x": xi.grad}
    out.update({f"grad {n}": p.grad for n, p in m.named_parameters() if p.grad is not None})
    m.zero_grad(set_to_none=True)
    return out


def _cases():
    for cart in CARTRIDGES:
        cov = COVERS.get(cart[0], {})
        for dt in DTYPES:
            for anchor_mode in ("pairs", "single"):
                if dt[0] in cov.get("dtypes", {dt[0]}) and anchor_mode in cov.get("anchor_modes", {anchor_mode}):
                    for geom in GEOMS:
                        if anchor_mode not in GEOM_ANCHORS.get(geom[0], {anchor_mode}):
                            continue
                        yield pytest.param(cart, dt, anchor_mode, geom,
                                           id=f"{cart[0]}-{dt[0]}-{anchor_mode}-{geom[0]}")


def _pair(cart, dt, anchor_mode, geom, seed=SEED):
    name, Fused, Ref, kw, ext = cart
    _, table_dtype, input_dtype = dt
    _, h_in, h_out, d, tph, nap, B = geom
    d_in, d_out = d if isinstance(d, tuple) else (d, d)
    requires_extension(ext)
    torch._dynamo.reset()          # every case compiles the reference fresh (else Dynamo's recompile limit -> eager)
    spec = LUTSpec(h_in=h_in, h_out=h_out, tph=tph, nap=nap, d_in=d_in, d_out=d_out, anchor_mode=anchor_mode)
    ref = Ref(spec, seed=seed, weight_init_std=0.5, **kw).cuda().to(table_dtype)
    fused = Fused(spec, seed=seed, weight_init_std=0.5, **kw).cuda().to(table_dtype)
    fused.load_state_dict(ref.state_dict())
    g = torch.Generator(device="cuda").manual_seed(B + h_in + seed)
    x = torch.randn(B, h_in, d_in, device="cuda", generator=g).to(input_dtype)
    go = torch.randn(B, h_out, d_out, device="cuda", generator=g).to(input_dtype)
    return ref, fused, x, go


class _Eager(torch.nn.Module):
    """The reference's uncompiled forward (``_forward_impl``) with its parameters: the quantised layer-1 oracle."""

    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        return self.m._forward_impl(x)

    def named_parameters(self, *a, **k):
        return self.m.named_parameters(*a, **k)

    def zero_grad(self, set_to_none=True):
        self.m.zero_grad(set_to_none=set_to_none)


@pytest.mark.parametrize("cart,dt,anchor_mode,geom", list(_cases()))
def test_fused_matches_reference(cart, dt, anchor_mode, geom):
    name, Fused, Ref, kw, ext = cart
    input_dtype = dt[2]
    ref, fused, x, go = _pair(cart, dt, anchor_mode, geom)
    quantised = Fused is FusedQuantisedConfidenceLUT
    oracle = _Eager(ref) if quantised else ref     # quantised: layer 1, the EAGER reference

    ref.train(); fused.train()
    a, b = _leaves(fused, x, go), _leaves(oracle, x, go)
    assert a.keys() == b.keys(), f"{name}: leaves {sorted(a)} vs {sorted(b)}"
    ref.eval(); fused.eval()
    with torch.no_grad():
        a["eval value"], b["eval value"] = fused(x), oracle(x)
        if quantised:
            a["eval value (integer read)"], b["eval value (integer read)"] = fused(x), ref.forward_int(x)
            for leaf in ("value", "eval value"):     # layer 1: bit for bit, no tolerance
                assert torch.equal(a[leaf], b[leaf]), f"{name} {leaf}: differs from the eager reference"
    for leaf in b:
        assert a[leaf].dtype == b[leaf].dtype, f"{name} {leaf}: dtype {a[leaf].dtype} vs {b[leaf].dtype}"
    scalars = [leaf for leaf in b if leaf not in TENSOR_LEAVES and not leaf.startswith("eval ")]
    truth = _scalar_truth(Ref, kw, fused.spec, ref.state_dict(), x, go) if scalars else {}
    n_terms = x.shape[0] * fused.spec.n_groups * fused.spec.tph * fused.spec.d_out
    for leaf, check, e, bound in _checks(a, b, truth, input_dtype, n_terms):
        assert e < bound, f"{name} {leaf} ({check}): error {e:.2e} >= {bound:.1e}"


# -- the quantised pair at the champion scale: two layers -------------------------------------------------------------
#
# Why two layers. The exponents q and k' are floor(x + 1/2) of fp32 transcendental expressions; a value within an ulp
# of a boundary floors differently in two implementations that differ by an ulp anywhere. Two independent sources:
# the compiled graph's arithmetic (compiled vs eager reference: Inductor's divide-by-constant rewrite, a reciprocal
# multiply contracted into an FMA, in k'; Triton's approximate fp32 division in q), and any difference in operation
# order between the kernel and the eager reference. The second is removed by construction -- the reference spells out
# the kernel's order (_pow2.blend_exponents) -- so:
#   layer 1 -- fused vs the EAGER reference: BIT FOR BIT, integers and output rows, no tolerance, no gate. A CUDA or
#              torch upgrade that changes libdevice (which both sides call) would fail here, by design;
#   layer 2 -- compiled vs eager reference, in fp64: a check that torch.compile preserved the code logic (below).
# Library code is not switched by any test; layer 2 builds the cartridge in fp64 through its public dtype support.

CHAMPION = dict(h_in=16, h_out=16, tph=64, nap=8, d_in=48, d_out=48, anchor_mode="pairs")
QUANT_KW = dict(weight_init_std=0.5, read_top_n=2, beta_init=1.7, gamma_init=0.8, read_tau_init=0.6)
#: Layer 2 bounds (fp64; see the comment above the layer-2 test). Measured over 192 runs -- nap 3, 4, 5, 6, 7, 8 on the
#: champion geometry (8,192 vectors) and on h8 tph32 d64 (16,384 vectors), 8 seeds each, at init and after 200 Adam
#: steps: no row moved by more than 2.3e-16 (so no exponent flipped in any run); eval and training value at most
#: 6.1e-17, grad x 2.2e-16, grad weights 2.8e-19, the learned scalars beta / gamma / tau 6.0e-14 (each one sum over
#: every sample, group, table and output, with cancellation). A flipped exponent would move its row by ~1e-2.
FP64_ROW_BOUND = 1e-13          # every output row: no flip anywhere (worst 2.3e-16)
FP64_TENSOR_BOUND = 1e-14       # eval, value, grad x, grad weights (worst 2.2e-16, ~45x headroom)
FP64_SCALAR_BOUND = 1e-12       # beta / gamma / tau gradients (worst 6.0e-14, ~17x headroom)


def _champion_pair(seed):
    requires_extension("quantised")
    torch._dynamo.reset()
    spec = LUTSpec(**CHAMPION)
    ref = QuantisedConfidenceLUT(spec, seed=seed, **QUANT_KW).cuda()
    fused = FusedQuantisedConfidenceLUT(spec, seed=seed, **QUANT_KW).cuda()
    fused.load_state_dict(ref.state_dict())
    x = torch.randn(8192, 16, 48, device="cuda", generator=torch.Generator(device="cuda").manual_seed(1234 + seed))
    return ref, fused, x


@pytest.mark.parametrize("seed", [0, 1])
def test_quantised_layer1_fused_bit_exact_vs_eager_reference(seed):
    """LAYER 1 (oracle): at the champion scale (8.4M tables), the op's integers (q, k', skip, drop) equal the eager
    reference definition's (_pow2.blend_exponents on the same margins) on EVERY table, and the eager reference's output
    equals the fused class's bit for bit, in eval and in the training forward. No tolerance, no gate."""
    from spiky.lutorch_ex.cartridges import _pow2, _pow2_int8
    ref, fused, x = _champion_pair(seed)
    with torch.no_grad():
        z, u, c, j, ustar, calt = ref._addresses(x)
        beta, gamma = ref._betagamma(u.dtype)
        tau = ref.log_read_tau.exp()
        g0 = torch.zeros((), device="cuda")
        cfg = ref._quant
        m = u.abs()
        qr, kr, skr, dr = _pow2.blend_exponents(m, m.min(-1, keepdim=True).values, tau, g0, beta, gamma, cfg)
        _, cells, kq = torch.ops.lutorch_ex.p2_scalars(u, tau, g0, beta, gamma, cfg["lo"], cfg["hi"], cfg["Q"])
        qo, ko, sko, do = _pow2_int8._integers_from(cells, kq, cfg["Q"])
        differ = (qr != qo) | (kr != ko) | (skr != sko) | (dr != do)
        assert int(differ.sum()) == 0, f"{int(differ.sum())} integer disagreements with the kernel"
        for train in (False, True):
            ref.train(train); fused.train(train)
            assert torch.equal(ref._forward_impl(x), fused(x)), f"train={train}: output differs from the eager reference"


@pytest.mark.parametrize("seed", [0, 1])
def test_quantised_layer2_compiled_reference_matches_eager_in_fp64(seed):
    """LAYER 2 (compile trust): the compiled reference vs the eager reference, both in fp64, at the champion geometry.

    We run this comparison in fp64 on purpose. The quantised read rounds its power-of-two exponents to integers, and
    in fp32 torch.compile's slightly different evaluation of the same formula lands a few values per million tables on
    the other side of a rounding boundary, flipping an integer; any bound loose enough to absorb that says little. In
    fp64 the two evaluations differ by ~1e-16, so no value is that close to a boundary and no integer flips (0 flips in
    192 measured runs). What is left to compare is the code logic itself: if the compiled reference matches the eager
    reference in fp64, torch.compile has preserved the logic, which is what this layer asserts. We believe that logic
    then works the same in fp32 -- the fp32 numbers will of course differ slightly, and layer 1 is what pins the fp32
    path, bit for bit against the fused kernel. The cartridge is built in fp64 through its normal dtype support
    (``.double()``); no library code is switched. Bounds and the measurements behind them: the FP64_* constants."""
    ref, _, x = _champion_pair(seed)
    ref = ref.double()
    x = x.double()
    with torch.no_grad():
        ref.eval()
        ye = ref._forward_impl(x)
        torch._dynamo.reset()
        yc = ref(x)                                        # compiled
    rows = (yc - ye).norm(dim=-1) / ye.norm(dim=-1).clamp_min(1e-300)
    assert float(rows.max()) <= FP64_ROW_BOUND, f"a row moved by {float(rows.max()):.2e} (an exponent flipped?)"
    e = _rel(yc, ye)
    assert e <= FP64_TENSOR_BOUND, f"eval relative norm {e:.2e} > {FP64_TENSOR_BOUND:.0e}"
    ref.train()
    go = torch.randn(x.shape[0], 16, 48, device="cuda", dtype=torch.float64,
                     generator=torch.Generator(device="cuda").manual_seed(99))
    res = []
    for f in (ref._forward_impl, ref):                     # eager, then compiled
        torch._dynamo.reset()
        xi = x.clone().requires_grad_(True)
        y = f(xi)
        y.backward(go)
        res.append({"value": y.detach(), "grad x": xi.grad} | {n: p.grad.clone() for n, p in ref.named_parameters()})
        ref.zero_grad(set_to_none=True)
    for leaf in res[0]:
        bound = FP64_SCALAR_BOUND if res[0][leaf].numel() == 1 else FP64_TENSOR_BOUND
        e = _rel(res[1][leaf], res[0][leaf])
        assert e <= bound, f"train {leaf}: relative norm {e:.2e} > {bound:.0e}"


#: A row whose tables were dropped identically on both sides differs only by fp32 rounding (measured <= 1e-6 with
#: room to spare); one table dropped on a single side moves its row by ~1/64.
ROW_SAME = 1e-6


def _injected_mask(m, mask):
    """Make ``m`` draw ``mask`` (``[B, G, tph]``, inverted scale) as its table-dropout keep-mask, with the library's own
    gate (TRAIN + grad only). Needed for an apples-to-apples comparison: the fused forward is compiled, and Inductor
    generates ``torch.rand`` with its own Philox RNG, so a shared seed does NOT give the eager reference and the
    compiled fused class the same mask; an identical explicit mask does. Test-side, on the instance only."""
    def draw(B, device, dtype):
        if not (m.training and torch.is_grad_enabled()):
            return None
        return mask.to(device=device, dtype=dtype)
    m._table_dropout_mask = draw


@pytest.mark.parametrize("seed", [0, 1])
def test_quantised_layer1_with_table_dropout(seed):
    """LAYER 1 with table dropout ACTIVE (rate 0.3), at the champion scale: driven by one identical explicit mask (see
    _injected_mask), the fused class and the eager reference agree on WHICH tables are dropped (no output row moves by
    more than ROW_SAME: a table dropped on one side only would move its row by ~1/64) and on the values, every leaf
    within the harness's fp32 tolerances. Not bit for bit, unlike layer 1 at rate 0: the survivors are scaled by
    1/0.7, which fp32 cannot represent exactly, and the compiled fused graph and the eager reference round that
    product differently (measured over 4 seeds: no row above 1e-6; value 2.1e-7 and grad x 1.4e-7 relative norm, 2%
    and 1.4% of FP32_TOL; grad weights identical; the learned scalars below 0.03% of their bound). And eval is a no-op
    for both: with the library's own mask (no injection) the rate-0.3 cartridges evaluate bit for bit like rate-0
    copies."""
    requires_extension("quantised")
    torch._dynamo.reset()
    spec = LUTSpec(**CHAMPION)
    rate, B = 0.3, 8192
    ref = QuantisedConfidenceLUT(spec, seed=seed, table_dropout_rate=rate, **QUANT_KW).cuda()
    fused = FusedQuantisedConfidenceLUT(spec, seed=seed, table_dropout_rate=rate, **QUANT_KW).cuda()
    fused.load_state_dict(ref.state_dict())
    g = torch.Generator(device="cuda").manual_seed(1234 + seed)
    x = torch.randn(B, 16, 48, device="cuda", generator=g)
    go = torch.randn(B, 16, 48, device="cuda", generator=g)
    keep = (torch.rand(B, spec.n_groups, spec.tph, device="cuda", generator=g) < 1.0 - rate).float() / (1.0 - rate)
    assert 0.6 < float((keep > 0).float().mean()) < 0.8               # the mask really drops ~30% of tables
    # eval first, with the library's own mask: a no-op in both classes
    with torch.no_grad():
        for cls, m in ((QuantisedConfidenceLUT, ref), (FusedQuantisedConfidenceLUT, fused)):
            m0 = cls(spec, seed=seed, table_dropout_rate=0.0, **QUANT_KW).cuda()
            m0.load_state_dict(m.state_dict())
            m.eval(); m0.eval()
            assert torch.equal(m(x), m0(x)), f"{cls.__name__}: dropout changed the eval output"
    _injected_mask(ref, keep)
    _injected_mask(fused, keep)
    ref.train(); fused.train()
    oracle = _Eager(ref)
    a, b = _leaves(fused, x, go), _leaves(oracle, x, go)
    rows = (a["value"] - b["value"]).norm(dim=-1) / b["value"].norm(dim=-1).clamp_min(1e-30)
    assert float(rows.max()) <= ROW_SAME, f"a row moved by {float(rows.max()):.2e}: a table dropped on one side only?"
    n_terms = B * spec.n_groups * spec.tph * spec.d_out
    for leaf in b:                 # fp32 throughout; the scalars keep the no-cancellation bound, stricter (measured above)
        tol = max(FP32_SCALAR_TOL, 4.0 * n_terms ** 0.5 * FP32_U) if leaf not in TENSOR_LEAVES else FP32_TOL
        e = _rel(a[leaf], b[leaf])
        assert e < tol, f"{leaf}: rel-norm error {e:.2e} >= {tol:.1e}"
    # the dropped tables contribute nothing: the masked value differs from the unmasked one
    with torch.no_grad():
        _injected_mask(fused, torch.ones_like(keep))
        assert not torch.equal(fused(x), a["value"]), "the mask had no effect"


def _tie_margins(target_q=None, target_k=None, nap=4):
    """Margins d [1, 1, 1, nap] (and tau) whose fp32 pre-floor value is EXACTLY an integer (an exact tie):
    * q: mv * (2/(tau ln2)) + 1/2 == target_q exactly, for the fp32 c = 2/(tau ln2) both sides compute;
    * k': gamma = 0 and q >= 8 (so c_q = 0): log2(S) + 1/2 == target_k exactly, S an fp32 value with log2(S) exactly
      target_k - 1/2 in fp32 (searched around 2^(target_k - 1/2))."""
    from spiky.lutorch_ex.cartridges import _pow2
    tau = torch.tensor(0.6, device="cuda")
    c = 2.0 / (tau * _pow2.LN2)                                     # the fp32 constant both sides compute
    if target_q is not None:                                        # mv so that fp32(mv * c + 0.5) == target_q
        mv = ((target_q - 0.5) / c).reshape(1)
        for _ in range(64):
            v = mv * c + 0.5
            if float(v) == target_q:
                break
            mv = torch.nextafter(mv, mv + (1 if float(v) < target_q else -1))
        assert float(mv * c + 0.5) == target_q
        d = torch.cat([mv, mv * 4, mv * 4, mv * 4])[:nap]           # mv is the smallest margin
        return d.reshape(1, 1, 1, nap), tau
    # k': S with fp32 log2(S) == target_k - 1/2 exactly; margins 2, 2, 2, S - 6 sum to S exactly (S in [8, 16))
    base = torch.tensor(2.0 ** (target_k - 0.5), device="cuda").view(torch.int32)
    cand = (base + torch.arange(-4096, 4097, device="cuda", dtype=torch.int32)).view(torch.float32)  # +-4096 ulps
    hit = cand[torch.log2(cand) == target_k - 0.5]
    assert hit.numel(), "no fp32 S with an exact log2 found"
    s = hit[0]
    d = torch.stack([torch.tensor(2.0, device="cuda")] * 3 + [s - 6.0])
    assert float(d.sum()) == float(s)
    return d.reshape(1, 1, 1, 4), tau


def test_quantised_exact_ties_round_half_up():
    """The tie rule: an fp32 pre-floor value exactly on a boundary rounds UP (floor(x + 1/2)), in the reference
    definition and in the kernel alike. Natural exact ties were measured at zero (module docstring); these are
    constructed. q: x == 3 exactly -> q = 3. k': gamma = 0, q >= 8, log2 S + 1/2 == 4 exactly -> k' = 4 (inside the
    window, so neither clamped nor skipped)."""
    requires_extension("quantised")
    from spiky.lutorch_ex.cartridges import _pow2, _pow2_int8
    cfg = _pow2.resolve_quant_config("p2_int8")
    g0 = torch.zeros((), device="cuda")
    beta, gamma1 = torch.tensor(1.7, device="cuda"), torch.tensor(0.8, device="cuda")
    d, tau = _tie_margins(target_q=3.0)
    m = d.abs()
    qr = _pow2.blend_exponents(m, m.min(-1, keepdim=True).values, tau, g0, beta, gamma1, cfg)[0]
    _, cells, kq = torch.ops.lutorch_ex.p2_scalars(d, tau, g0, beta, gamma1, cfg["lo"], cfg["hi"], cfg["Q"])
    qo = _pow2_int8._integers_from(cells, kq, cfg["Q"])[0]
    assert float(qr) == 3.0 and float(qo) == 3.0, (float(qr), float(qo))
    d, tau = _tie_margins(target_k=4)
    gamma0 = torch.zeros((), device="cuda")
    m = d.abs()
    q, k = _pow2.blend_exponents(m, m.min(-1, keepdim=True).values, tau, g0, beta, gamma0, cfg)[:2]
    assert float(q) >= 8, "the construction needs q >= 8 (c_q = 0)"
    _, cells, kq = torch.ops.lutorch_ex.p2_scalars(d, tau, g0, beta, gamma0, cfg["lo"], cfg["hi"], cfg["Q"])
    qo, ko = _pow2_int8._integers_from(cells, kq, cfg["Q"])[:2]
    assert float(k) == 4.0 and float(ko) == 4.0 and float(qo) == float(q), (float(k), float(ko))
