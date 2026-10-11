"""Spec amendment D: the fused-vs-reference equivalence harness, the one REQUIRED check binding each fused class to its
reference class.

Per cartridge (FusedManifestoHardLUT / FusedManifestoSoftLUT with both weights, FusedConfidenceLUT with read_top_n 1
and 2, FusedQuantisedConfidenceLUT), per dtype (fp32; bf16 table + input; fp16 input on an fp32 table), per anchor mode
and head pattern, wherever the fused class covers the case (the quantised op: fp32, pairs, read_top_n=2 only): the fused
class and its reference class, loaded with the same state_dict, are fed the same input and output gradient, in
training and in eval, and compared on every leaf: the value, the input gradient, the table gradient and every learned
scalar's gradient (soft-sign temperatures; confidence beta / gamma / tau).

Tolerances (relative norm error ``||a - b|| / ||b||``), each from the one way the two can legitimately differ:
* Both sides take the same discrete decisions: addressing runs in fp32 on the same values in both (a margin is one
  fp32 subtraction), so the cells read are identical and only the arithmetic around them can differ.
* fp32 leaf -- summation order. The kernels reduce in a different order (and the table gradient with atomics), so
  each sum carries a different fp32 round-off, a random walk of O(sqrt(K) * 2**-24) relative for K terms. Element-wise
  leaves sum K <= 10**4 terms (~1e-6): ``FP32_TOL = 1e-5``. A learned scalar's gradient is ONE sum over every (sample,
  group, table, output), K = B * n_groups * tph * d_out (up to ~2.5e7 here), the worst-conditioned reduction in the
  cartridge: ``max(FP32_SCALAR_TOL = 1e-4, 4 * sqrt(K) * 2**-24)``, a 4-sigma bound on that walk.
* bf16 / fp16 leaf -- one rounding. Both classes compute in fp32 and round each leaf once (fused: in the kernel's
  epilogue; reference: tests/test_reference_low_precision.py proves it bit-exact). Two roundings of fp32 values that
  differ only by the fp32 tolerance above land on the same low-precision value except where they straddle a
  rounding boundary, and then differ by one ulp. One ulp everywhere would be a relative norm error of at most
  u = 2**-8 (bf16) / 2**-11 (fp16), so ``u`` bounds it with a wide margin. A learned scalar in a bf16 module is a
  bf16 leaf as well, so the same bound applies.
* Leaves that stay fp32 in a low-precision run (every parameter's gradient when only the input is fp16) take the
  fp32 tolerances, widened to the input dtype's u: they are computed from the rounded output gradient and input.

Measured 2026-10-11 (RTX 5090), worst error as a share of its tolerance over the whole matrix: fp32 value 3%, grad x
2%, grad W 9%, learned scalars at most ~30% (one soft-sign temperature at batch 24576 reached 1.02e-4, which is why
the scalar bound scales with K); bf16 value 5%, grad x 5%, grad W 3%; fp16-input grad x 7%; quantised vs the eager
EAGER reference (gated): integers exact, values bit-identical. (The harness found one
real double rounding: a low-precision input gathered for several groups (fan-out) summed their input gradients in the
input dtype, 78% of u; the fused classes now cast before the gather.)

The quantised pair, in two layers. Its value is a staircase: the power-of-two exponents
q = floor(mv*2/(tau ln2) + 1/2) and k' = floor(log2 S + gamma*sum logsigmoid(beta m)/ln2 - c_q + 1/2) are rounded half
up from fp32 transcendental expressions, and no two fp32 implementations agree on them to the last ulp, so a value
within a few ulps of a rounding boundary can floor differently (one table's weight changes by a factor 2). Two
independent sources do that, so they are tested separately (the comment above the champion-scale tests explains why):
* LAYER 1, the oracle -- fused vs the EAGER (uncompiled) reference. The only difference is the kernel's own
  log1p / exp / log2 (nvcc, --fmad=false) vs torch's CUDA kernels: measured at the champion scale (8.4M tables per
  run, 4 seeds, init and after 200 steps) 0-3 integer disagreements per run, all within 1e-7 (fp64) of a boundary.
  Gated at ``QUANT_MARGIN = 1e-5`` (fp64 distance of both pre-floor values to a boundary; 1e-6 already gave zero in all
  8 runs, 1e-7 did not; margins relative to ulps fail even at 64 ulps, the pre-floor value being a sum of large terms
  whose error is absolute): on retained tables the integers are EXACT, and retained output rows are BIT-IDENTICAL
  (99.73% of rows at the champion scale). The rows below gate the same way (gated rows get a zero output gradient, so
  they contribute to no gradient; gradients keep the fp32 tolerances above, as the backward's sums differ in order).
* LAYER 2, compile trust -- the compiled reference vs the eager reference. A TOLERANCE test: Inductor reassociates the
  transcendentals, so the shipped compiled reference flips ~1e-7 of tables; the test asserts compile did not break the
  maths, not that it is bit-reproducible.
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
U = {torch.bfloat16: 2.0 ** -8, torch.float16: 2.0 ** -11}
QUANT_MARGIN = 1e-5             # the quantised rows' gate: fp64 distance of each pre-floor value to a boundary

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
          ("fp16-input", torch.float32, torch.float16)]
#: What each fused class covers (everything not listed: all dtypes, both anchor modes).
COVERS = {"Quantised n=2": dict(dtypes={"fp32"}, anchor_modes={"pairs"})}
#: (id, h_in, h_out, d, tph, nap, batch)
GEOMS = [("small-per-head", 4, 4, 12, 6, 5, 128), ("fan-in", 4, 1, 12, 6, 5, 128), ("fan-out", 1, 4, 12, 6, 5, 128),
         ("batch-1", 4, 4, 12, 6, 5, 1), ("canonical", 16, 16, 48, 64, 8, 512), ("large-batch", 4, 4, 12, 6, 5, 24576)]


def _rel(a, b):
    return ((a.double() - b.double()).norm() / b.double().norm().clamp_min(1e-30)).item()


def _tol(leaf: str, leaf_dtype: torch.dtype, input_dtype: torch.dtype, n_terms: int = 0) -> float:
    """The tolerance of one leaf (module docstring); ``n_terms`` = K of a learned scalar's gradient sum."""
    scalar = leaf not in ("value", "grad x", "grad weights")
    if leaf_dtype in U:
        return U[leaf_dtype]
    base = max(FP32_SCALAR_TOL, 4.0 * n_terms ** 0.5 * 2.0 ** -24) if scalar else FP32_TOL
    return max(base, U.get(input_dtype, 0.0))


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
                        yield pytest.param(cart, dt, anchor_mode, geom,
                                           id=f"{cart[0]}-{dt[0]}-{anchor_mode}-{geom[0]}")


def _pair(cart, dt, anchor_mode, geom, seed=7):
    name, Fused, Ref, kw, ext = cart
    _, table_dtype, input_dtype = dt
    _, h_in, h_out, d, tph, nap, B = geom
    requires_extension(ext)
    torch._dynamo.reset()          # every case compiles the reference fresh (else Dynamo's recompile limit -> eager)
    spec = LUTSpec(h_in=h_in, h_out=h_out, tph=tph, nap=nap, d_in=d, d_out=d, anchor_mode=anchor_mode)
    ref = Ref(spec, seed=seed, weight_init_std=0.5, **kw).cuda().to(table_dtype)
    fused = Fused(spec, seed=seed, weight_init_std=0.5, **kw).cuda().to(table_dtype)
    fused.load_state_dict(ref.state_dict())
    g = torch.Generator(device="cuda").manual_seed(B + h_in + seed)
    x = torch.randn(B, h_in, d, device="cuda", generator=g).to(input_dtype)
    go = torch.randn(B, h_out, d, device="cuda", generator=g).to(input_dtype)
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


def _quantised_row_gate(mod, x, margin=QUANT_MARGIN):
    """[B, h_out] bool: True where every table feeding the output row is more than ``margin`` (fp64) from a rounding
    boundary of q and k' (see the module docstring). Test-side only: recomputes the pre-floor values from the module's
    margins in fp64; the library is not involved beyond _addresses."""
    import torch.nn.functional as F
    from spiky.lutorch_ex.cartridges import _pow2
    with torch.no_grad():
        z, u, c, j, ustar, calt = mod._addresses(x)
        m = u.abs().double()
        tau = mod.log_read_tau.double().exp()
        beta, gamma = mod.confidence_log_beta.double().exp(), mod.confidence_log_gamma.double().exp()
        xq = m.min(-1).values * (2.0 / (tau * _pow2.LN2)) + 0.5
        q = torch.clamp(torch.floor(xq), 0, _pow2.J)
        cq = torch.where(q < _pow2.C, torch.log2(1.0 + torch.pow(2.0, -q)), torch.zeros_like(q))
        xk = torch.log2(m.sum(-1)) + gamma * F.logsigmoid(beta * m).sum(-1) / _pow2.LN2 - cq + 0.5
        clear = ((xq - torch.round(xq)).abs() > margin) & ((xk - torch.round(xk)).abs() > margin)   # [B, G, tph]
        group_ok = clear.all(-1)                                                                     # [B, G]
        rows = torch.ones(x.shape[0], mod.spec.h_out, dtype=torch.bool, device=x.device)
        for g in range(mod.spec.n_groups):                       # a fan-in row sums several groups
            rows[:, mod.out_head[g]] &= group_ok[:, g]
    return rows


@pytest.mark.parametrize("cart,dt,anchor_mode,geom", list(_cases()))
def test_fused_matches_reference(cart, dt, anchor_mode, geom):
    name, Fused, Ref, kw, ext = cart
    input_dtype = dt[2]
    ref, fused, x, go = _pair(cart, dt, anchor_mode, geom)
    oracle, keep = ref, None
    if Fused is FusedQuantisedConfidenceLUT:        # layer 1: the EAGER reference, away from rounding boundaries
        oracle, keep = _Eager(ref), _quantised_row_gate(ref, x)
        go = go * keep.unsqueeze(-1).to(go.dtype)  # a gated row contributes to no gradient

    ref.train(); fused.train()
    a, b = _leaves(fused, x, go), _leaves(oracle, x, go)
    assert a.keys() == b.keys(), f"{name}: leaves {sorted(a)} vs {sorted(b)}"
    ref.eval(); fused.eval()
    with torch.no_grad():
        a["eval value"], b["eval value"] = fused(x), oracle(x)
        if keep is not None:
            a["eval value (integer read)"], b["eval value (integer read)"] = fused(x), ref.forward_int(x)
            for leaf in ("value", "eval value", "eval value (integer read)"):
                a[leaf], b[leaf] = a[leaf][keep], b[leaf][keep]
            for leaf in ("value", "eval value"):     # layer 1: retained values are bit-identical, not just close
                assert torch.equal(a[leaf], b[leaf]), f"{name} {leaf}: retained rows differ from the eager reference"
    for leaf in b:
        assert a[leaf].dtype == b[leaf].dtype, f"{name} {leaf}: dtype {a[leaf].dtype} vs {b[leaf].dtype}"
        tol = _tol(leaf.replace("eval ", "").replace(" (integer read)", ""), b[leaf].dtype, input_dtype,
                   n_terms=x.shape[0] * fused.spec.n_groups * fused.spec.tph * fused.spec.d_out)
        e = _rel(a[leaf], b[leaf])
        assert e < tol, f"{name} {leaf}: rel-norm error {e:.2e} >= {tol:.1e}"


# -- the quantised pair at the champion scale: two layers -------------------------------------------------------------
#
# Why two layers. The exponents q and k' are floor(x + 1/2) of fp32 transcendental expressions, and two independent
# sources move such a value by a few ulps: Inductor reassociating log2 / logsigmoid / exp (compiled vs eager
# reference), and the kernel's nvcc log1p / exp / log2 (--fmad=false) vs torch's CUDA kernels (fused vs eager
# reference). Near a boundary either one flips an integer: ~1e-7 of tables each, measured at 8.4M tables per run,
# zero overlap across seeds, zero natural exact ties. The old single comparison (fused vs compiled reference, with an
# eager-reference variant) confounded the two, and its "fused == eager reference bit for bit" held only at
# harness-size inputs (<= 0.6M tables, where a ~1e-7 event rarely lands); it is false at scale. So:
#   layer 1 -- fused vs the EAGER reference: exact (integers, and output rows bit for bit) on every table more than
#              QUANT_MARGIN from a boundary in fp64; the near-boundary remainder is counted, not asserted;
#   layer 2 -- compiled vs eager reference: a tolerance test ("compile did not break the maths"), bounds below.
# All fp32; library code untouched (the gate is recomputed here in fp64; the eager reference is its _forward_impl).

CHAMPION = dict(h_in=16, h_out=16, tph=64, nap=8, d_in=48, d_out=48, anchor_mode="pairs")
QUANT_KW = dict(weight_init_std=0.5, read_top_n=2, beta_init=1.7, gamma_init=0.8, read_tau_init=0.6)
#: A flipped exponent moves its row by >= ~6e-5 relative (one of 64 tables, weights within the 2^8 window, x2);
#: fp32 summation order moves a row by <= 8.8e-8 (measured). Rows above this are flips.
ROW_FLIP = 1e-6
#: Layer 2 bounds, compiled vs eager reference. Worst measured over 8 runs at the champion scale: flipped rows 2.4e-7
#: per table, eval relative norm 3.1e-4, training (value, input and parameter gradients) 2.9e-3; the bounds leave
#: ~4x, ~3x and ~3x headroom. A broken compile (a wrong op, a dropped term) moves every row, far beyond them.
FLIP_RATE_BOUND = 1e-6
EVAL_REL_BOUND = 1e-3
TRAIN_REL_BOUND = 1e-2


def _champion_pair(seed):
    requires_extension("quantised")
    torch._dynamo.reset()
    spec = LUTSpec(**CHAMPION)
    ref = QuantisedConfidenceLUT(spec, seed=seed, **QUANT_KW).cuda()
    fused = FusedQuantisedConfidenceLUT(spec, seed=seed, **QUANT_KW).cuda()
    fused.load_state_dict(ref.state_dict())
    x = torch.randn(8192, 16, 48, device="cuda", generator=torch.Generator(device="cuda").manual_seed(1234 + seed))
    return ref, fused, x


def _clear_tables(mod, x, margin=QUANT_MARGIN):
    """[B, G, tph] bool: both pre-floor values more than ``margin`` (fp64) from a boundary; and the margins u."""
    import torch.nn.functional as F
    from spiky.lutorch_ex.cartridges import _pow2
    z, u, c, j, ustar, calt = mod._addresses(x)
    m = u.abs().double()
    tau = mod.log_read_tau.double().exp()
    beta, gamma = mod.confidence_log_beta.double().exp(), mod.confidence_log_gamma.double().exp()
    xq = m.min(-1).values * (2.0 / (tau * _pow2.LN2)) + 0.5
    q = torch.clamp(torch.floor(xq), 0, _pow2.J)
    cq = torch.where(q < _pow2.C, torch.log2(1.0 + torch.pow(2.0, -q)), torch.zeros_like(q))
    xk = torch.log2(m.sum(-1)) + gamma * F.logsigmoid(beta * m).sum(-1) / _pow2.LN2 - cq + 0.5
    return ((xq - torch.round(xq)).abs() > margin) & ((xk - torch.round(xk)).abs() > margin), u


@pytest.mark.parametrize("seed", [0, 1])
def test_quantised_layer1_fused_exact_vs_eager_reference_away_from_boundaries(seed):
    """LAYER 1 (oracle): on every table clear of QUANT_MARGIN the op's integers (q, k', skip, drop) equal the eager
    reference definition's (_pow2.blend_exponents on the same margins) exactly, and the eager reference's output rows
    built only from such tables equal the fused class's bit for bit. The near-boundary remainder is counted."""
    from spiky.lutorch_ex.cartridges import _pow2, _pow2_int8
    ref, fused, x = _champion_pair(seed)
    with torch.no_grad():
        clear, u = _clear_tables(ref, x)
        beta, gamma = ref._betagamma(u.dtype)
        tau = ref.log_read_tau.exp()
        g0 = torch.zeros((), device="cuda")
        cfg = ref._quant
        m = u.abs()
        qr, kr, skr, dr = _pow2.blend_exponents(m, m.min(-1, keepdim=True).values, tau, g0, beta, gamma, cfg)
        _, cells, kq = torch.ops.lutorch_ex.p2_scalars(u, tau, g0, beta, gamma, cfg["lo"], cfg["hi"], cfg["Q"])
        qo, ko, sko, do = _pow2_int8._integers_from(cells, kq, cfg["Q"])
        differ = (qr != qo) | (kr != ko) | (skr != sko) | (dr != do)
        assert clear.float().mean() > 0.9999, f"the gate keeps only {clear.float().mean():.6f} of tables"
        assert int((differ & clear).sum()) == 0, f"{int((differ & clear).sum())} integer disagreements off-boundary"
        ref.eval(); fused.eval()
        ye, yf = ref._forward_impl(x), fused(x)            # the EAGER reference
        rows = clear.all(-1)                               # h_out == n_groups at the champion geometry
        assert torch.equal(ye[rows], yf[rows]), "retained rows are not bit-identical to the eager reference"
        print(f"\n[layer 1, seed {seed}] tables {clear.numel()}, clear {clear.float().mean():.6f}, rows kept "
              f"{rows.float().mean():.5f}, near-boundary integer disagreements (counted) {int((differ & ~clear).sum())}")


@pytest.mark.parametrize("seed", [0, 1])
def test_quantised_layer2_compiled_reference_within_bounds_of_eager(seed):
    """LAYER 2 (compile trust): the shipped compiled reference vs the eager reference, ungated. A tolerance test --
    Inductor reassociates the transcendentals, so near-boundary exponents flip (~1e-7 of tables); asserted is that
    compile did not break the maths: flipped rows per table <= FLIP_RATE_BOUND, eval relative norm <= EVAL_REL_BOUND,
    training (value and every gradient) <= TRAIN_REL_BOUND (measured worst and headroom at the constants)."""
    ref, _, x = _champion_pair(seed)
    tables = x.shape[0] * ref.spec.n_groups * ref.spec.tph
    with torch.no_grad():
        ref.eval()
        ye = ref._forward_impl(x)
        torch._dynamo.reset()
        yc = ref(x)                                        # compiled
    rows = (yc - ye).norm(dim=-1) / ye.norm(dim=-1).clamp_min(1e-30)
    rate = int((rows > ROW_FLIP).sum()) / tables
    assert rate <= FLIP_RATE_BOUND, f"flipped-row rate {rate:.2e} > {FLIP_RATE_BOUND:.0e}"
    e = float((yc - ye).norm() / ye.norm())
    assert e <= EVAL_REL_BOUND, f"eval relative norm {e:.2e} > {EVAL_REL_BOUND:.0e}"
    ref.train()
    go = torch.randn(x.shape[0], 16, 48, device="cuda", generator=torch.Generator(device="cuda").manual_seed(99))
    res = []
    for f in (ref._forward_impl, ref):                     # eager, then compiled
        torch._dynamo.reset()
        xi = x.clone().requires_grad_(True)
        y = f(xi)
        y.backward(go)
        res.append({"value": y.detach(), "grad x": xi.grad} | {n: p.grad.clone() for n, p in ref.named_parameters()})
        ref.zero_grad(set_to_none=True)
    for leaf in res[0]:
        e = _rel(res[1][leaf], res[0][leaf])
        assert e <= TRAIN_REL_BOUND, f"train {leaf}: relative norm {e:.2e} > {TRAIN_REL_BOUND:.0e}"


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
