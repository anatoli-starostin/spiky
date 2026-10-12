# lutorch_ex: target architecture for the cartridges

**Status:** agreed goal state, implemented on #160 (rules 1-5, amendments A-E). The code on `main` (36aa7a13, the
#159 merge) does not follow it yet; what #160 lands, and what is deferred, is listed at the end. Each item below is
marked **DECIDED** or **OPEN**.

## Anatoly's five rules (the contract) - DECIDED

1. **Manifesto and SoftSign are merged** (the Gen-2 fold): `rational | softsign` becomes a weight-function parameter,
   not separate classes. The fold is a net deletion of three files, landing in two places (reference and fused).
   - The parameter selects the neighbour-weight function: `weight="rational"` is the Gen-1 `U = 0.5/(1+|u*|)`;
     `weight="softsign"` is the Gen-2 learned-temperature `w`.
   - Landed on #160: `softsign_base.py`, `softsign_hard.py`, `softsign_smooth.py` (reference) and `fused_softsign.py`
     (fused) are gone (four files out, one test helper in).
2. **Reference and fused implementations of each cartridge live in different files.** This is a recommendation, not
   an absolute: some reuse is expected and fine, and parts of the base class are already reused.
3. **A fused implementation works only with the CUDA extension compiled and installed; otherwise it raises.** It may
   be as complicated as needed: efficiency is king.
4. **A reference implementation stays reasonably efficient, but readability is what matters.** It tries
   `torch.compile` for efficiency and falls back to pure PyTorch where compile is not possible.
5. **There is no "backend".** The concept is too confusing and is dropped entirely: no `backend=` argument anywhere.

## Agreed amendments and consequences - DECIDED

- **A. Share the scaffolding, duplicate only the maths.** The config dataclass, geometry validation, anchor-pair
  construction, address packing and parameter registration live in ONE common home and are reused. The reference and
  fused files differ only in how forward and backward compute.
- **B. The shared base is scaffolding only:** state and validation, no executable forward (abstract).
  - *Why:* if a fused class could inherit a runnable pure-PyTorch forward, a missing kernel would resolve silently up
    the MRO. That is a quiet fallback again, breaking rule 3 by inheritance instead of by an `if`.
  - Anatoly's wording: agreed, "we don't force it". This is the structural guard, not a mandate to contort the class
    hierarchy.
- **C. The reference is a load-bearing path, not a test fixture.** Under rule 3, a machine without the compiled
  extension has no fast path at all, so the reference is the only thing that runs there. It must be correct at the
  real dtypes and geometries.
  - **bf16 reference support: DECIDED, landed on #160.** The reference classes accept bf16 / fp16 with the fused
    contract: fp32 maths (on fp32 views of the parameters and input), exactly one rounding per leaf; bit-identical
    to the fp32 run rounded once -- measured at test sizes (tests/test_reference_low_precision.py), and on CUDA under torch.use_deterministic_algorithms (otherwise the fp32 run's own atomic scatters are not reproducible).
- **D. The fused-vs-reference equivalence test is the only thing binding the two files** once they are separate by
  design. It is per cartridge, with explicit per-dtype tolerances (value, input gradient, table gradient, learned
  scalars). It is a REQUIRED check, not optional hygiene.
- **E. The API is explicit class choice at construction.** You import and instantiate the fused cartridge, and it
  raises if the extension is not there. So the class names ARE the public API and need a convention we will not
  regret.
  - **Naming: DECIDED, scheme 1 (the `Fused` prefix).** See below.

## Naming - DECIDED: the `Fused` prefix

`<Family><Variant>LUT` for the reference, the same name with a `Fused` prefix for the CUDA implementation. After the
fold, SoftSign's "Smooth" (the same variant as Manifesto's "Soft") becomes `Soft`.

| family | reference | fused (CUDA only) |
|---|---|---|
| Manifesto, hard read + surrogate gradient | `ManifestoHardLUT(weight="rational" \| "softsign")` | `FusedManifestoHardLUT(weight=...)` |
| Manifesto, blended read | `ManifestoSoftLUT(weight="rational" \| "softsign")` | `FusedManifestoSoftLUT(weight=...)` |
| Confidence | `ConfidenceLUT` | `FusedConfidenceLUT` |
| Confidence, quantised | `QuantisedConfidenceLUT` | `FusedQuantisedConfidenceLUT` (`DeployedQuantisedConfidenceLUT` stays the inference-only deploy form, torch only) |

- **Rejected:** an implementation suffix (`ManifestoHardRefLUT` / `ManifestoHardCudaLUT`), which renames every class,
  and namespaces (`lutorch_ex.reference.X` / `lutorch_ex.cuda.X`), which change every import and make the two forms
  identical in a `repr`.
- **The `SoftSign*` names: DECIDED, no aliases, no shims, no deprecation wrappers.** The fold renames them cleanly
  (one name per thing). No checkpoint and no model code depends on them: state_dicts hold no class names, and the
  learned temperatures stay `log_soft_score_temp` / `log_select_temp`. References in the report `.tex` and in
  experiment / profiling scripts are left dangling on purpose until the final cleanup; they are listed in #160.

## `LUTORCH_EX_NO_CUDA_EXT` under this spec - gone

Before this spec it was read in three places: `_cuda_ext.load_extension` (skip building every JIT extension),
`_fallback.disabled_on_purpose` / `is_deliberate` (make the resulting fallback quiet), and `_pow2_int8.load` (skip the
p2_scalars extension, with the deprecated alias `SPIKY_P2_CUDA_DISABLE`; plus the module switch
`_pow2_int8.set_enabled`). Under rules 3 and 5 all of them **are gone**, with no exception:
- A fused class raises when its kernel cannot load, so there is no fallback left to make quiet or to force.
- No reference class uses an extension (`ManifestoHard/Soft`, `ConfidenceLUT`, `QuantisedConfidenceLUT` and
  `DeployedQuantisedConfidenceLUT` call none).
- **The replacement lever** is explicit class choice (E): to run without the extension (CI, a CPU-only box),
  construct the reference class. Tests of a fused class skip without a CUDA device and fail on one without the
  extension (`tests/_requires_extension.py`).

## Gaps between `main` and this spec (for the PRs that land it)

**Landed on #160:**
- **Rule 1** (the fold): `ManifestoHardLUT` / `ManifestoSoftLUT` and their `Fused` twins take
  `weight="rational" | "softsign"`; the four SoftSign files are gone, with no aliases; the `fused_manifesto.cu`
  kernels take the soft-sign temperatures (their gradients included).
- **Rules 3 + 5:** no `backend=` anywhere. The fused classes run only their CUDA kernels and raise otherwise: a
  case the kernels do not cover names the case and the reference class, a missing extension names the classified
  cause. The `pure` / `pure_eval` / `tier1` / `native` paths are deleted, with `_native_ops.py`,
  `_native_softsign.py` and their CUDA sources (`lprojection`, single-anchor input grad, `softsign_surrogate_grad`).
  `LUTORCH_EX_NO_CUDA_EXT` is gone (above). `LUTORCH_EX_NO_COMPILE=1` stays, deliberately: it runs the reference
  classes uncompiled -- the same class and the same maths, no implementation choice -- for when torch.compile is in
  the way (debugging, an unsupported toolchain, a slow cold compile). It does not select between implementations, so
  the class still is the implementation (rule 5). The other environment reads are build or launch settings of the
  same code: `TORCH_EXTENSIONS_DIR`, `SPIKY_CXX_STD`, the `LUTORCH_EX_MANI_CUDA_*` / `LUTORCH_EX_CONF_CUDA_*` kernel
  launch knobs, and the build-failure log directory.
- **Rules 3 + 5, the quantised path:** `QuantisedConfidenceLUT` is the reference (the torch definition, `_pow2`, every
  case; no extension, no op, no monolith). `FusedQuantisedConfidenceLUT` (new) carries the `lutorch_ex::p2_scalars`
  op: the n=2 training monolith and the n=2 eval read. It raises without the extension (classified cause, including
  an architecture outside its validated list) and on a case the op does not cover (read_top_n=1, single anchors, a
  CPU input, a non-fp32 input or table). `_pow2_int8.set_enabled` / `_enabled` and the env-var switch are gone. Both
  share `QuantisedScaffold` (config, fake-quant tables, the integer read, deployment export) and the state_dict;
  `from_deployment` rebuilds the (torch) `DeployedQuantisedConfidenceLUT`. The extension (`csrc/pow2_int8_read.cu`)
  is not an int8 read despite its name: it computes only the per-table p2 integers; int8 tables are read by the torch
  definition (`_pow2.int_blend_read`). The file and op names are kept (renaming them would churn the build cache
  and the op registry for no behaviour change).
- **Decided consequences of rules 3 + 5 (accepted by Anatoly):**
  - The `native` lprojection kernels are gone. They were a second, slower CUDA implementation and the only CUDA path
    with fp64; fp64 now runs on the reference classes only (CPU or CUDA).
  - A fused class no longer runs on the CPU. CPU coverage lives with the reference classes; fused tests skip without
    a CUDA device.
  - The reference quantised path is slower than the old default, which took the op for n=2 eval and the monolith for
    n=2 training on CUDA; that speed now lives in `FusedQuantisedConfidenceLUT`; the bit-exact exponent order
    (Consequence D below) adds to it. #159's snapshot of record vs #160's (`benchmarks/snapshot_160.md`, RTX 5090):
    8,192 vectors eval 2.52 -> 2.97 ms, train total 12.43 -> 12.56 ms; 32,768 eval 9.66 -> 11.76 ms, train total
    48.56 -> 50.74 ms, eval peak 1.66 -> 2.03 GiB, train peak 3.64 -> 4.17 GiB. Not to be fixed in #160.
- **Consequence B:** `ManifestoLUT` is scaffolding only; its `forward()` raises `NotImplementedError`. The
  reference classes take their (compiled) forward from `ReferenceForward` (`cartridges/reference_base.py`).
  `ConfidenceLUT` and `FusedConfidenceLUT` both build on `ConfidenceBase`; the fused class no longer inherits
  `ConfidenceLUT`'s forward.
- **Consequence C:** the reference classes accept bf16 / fp16: fp32 maths, one rounding per leaf, bit-identical to
  the fp32 run rounded once -- measured at test sizes (tests/test_reference_low_precision.py), and on CUDA under torch.use_deterministic_algorithms (otherwise the fp32 run's own atomic scatters are not reproducible). `QuantisedConfidenceLUT` keeps rejecting low precision (outside this
  spec's families).
- **Consequence D:** `tests/test_fused_vs_reference.py` is the one fused-vs-reference harness: per cartridge (both
  Manifesto twins with both weights, FusedConfidenceLUT n=1 and 2, FusedQuantisedConfidenceLUT), per dtype (fp32,
  bf16, fp16 input) and anchor mode wherever the fused class covers them, per head pattern, on every leaf, with
  explicit tolerances justified in its docstring. Its dtypes are every table / input combination the fused classes
  declare: fp32; bf16 table + bf16 input; fp16 input on an fp32 table; a bf16 table with an fp32 input; an fp16 input
  on a bf16 table. It replaced
  `test_fused_equivalence.py` and `test_fused_manifesto_softsign.py`, and found (and the fused classes now fix) a
  double rounding of a low-precision input's gradient in the fan-out pattern.
- **Consequence D, the quantised pair: bit-exact, in two layers.** The power-of-two exponents are round-half-up
  floors of fp32 transcendental expressions, so two implementations that differ by an ulp anywhere in the expression
  can floor a near-boundary value differently (one table's weight changes 2x).
  - *Layer 1, the oracle -- fused vs the eager (uncompiled) reference: bit for bit, no tolerance, no gate.* The
    reference's `_pow2.blend_exponents` spells out the kernel's operation order (`csrc/pow2_scalars.cuh`): left-fold
    sums of the nap terms, true fp32 tensor divisions (a Python-scalar divisor makes torch multiply by the reciprocal;
    `2.0 / t` is `reciprocal(t) * 2`). Measured: 0 integer disagreements and bit-identical output rows (eval and the
    training forward) in 28 runs of 8.4M tables -- nap 3, 5, 6, 7 and 8 on the champion geometry and h8 tph32 nap4 d64,
    4 seeds, at init and after 200 training steps. Before, with torch's own reduction order, they disagreed on 0-3
    tables per run (the summation order and the reciprocal-multiply division; the transcendentals were identical).
  - *Layer 2, compile trust -- the compiled reference vs the eager reference, in fp64 on purpose.* In fp32 Inductor's
    reassociation of the transcendentals lands a few values per million tables on the other side of a rounding
    boundary (measured: 52 of 192 runs had a flipped row; flip-free rows <= 1.1e-7, flipped rows >= 5.8e-3), so any
    fp32 bound only absorbs flips. In fp64 the two evaluations differ by ~1e-16 and nothing flips (0 in 192 runs: nap 3
    to 8 on two geometries, 8 seeds, at init and after training). The test therefore builds the cartridge in fp64
    (through its normal dtype support; no library switch) and asserts tight bounds -- every row <= 1e-13, eval / value
    / grad x / grad weights <= 1e-14, learned scalars <= 1e-12 (worst 2.3e-16, 2.2e-16 and 6.0e-14): compile
    preserved the code logic. The fp32 numerics differ slightly, as expected; layer 1 pins them bit for bit.
  - Constructed exact ties: the reference and the kernel both round half up. Natural exact ties: zero.
  - With table dropout active (rate 0.3), driven by one identical explicit mask on both sides (the compiled fused
    graph draws `torch.rand` from Inductor's own RNG, so a shared seed would not give the same mask): the fused class
    drops exactly the tables the reference drops (no output row moves by more than 1e-6) and agrees on every leaf
    within the harness's fp32 tolerances -- not bit for bit, since the survivors' 1/0.7 scale is inexact in fp32 and
    the two graphs round that product differently (measured: value 2.1e-7, grad x 1.4e-7, grad weights identical).
    Eval is a no-op in both. The fused class also passes the dropout effect / unbiasedness / dropped-table-zero-grad
    tests and the int32-vs-int64 index test.
  - **The trade-off, accepted on purpose:** (1) the reference partly encodes the kernel's operation order, so their
    agreement is a weaker form of independence than with torch's own reduction; (2) a left fold is less accurate than
    torch's pairwise reduction: relative error of S, mean 3.40e-8 vs 2.97e-8, max 2.7e-7 vs 1.6e-7; (3) the
    bit-agreement relies on the same libdevice log2 / log1p / exp in torch's build and in the extension's nvcc build,
    measured identical on this toolkit but not guaranteed across upgrades -- which is why the comparison is bit-exact:
    such an upgrade fails the test instead of being absorbed by a tolerance. Cost: part of the reference quantised
    path's slowdown in the decided consequences above (unoptimised, by decision).
  - **History:** an earlier version of this section (and the harness, and #160's text) claimed bit-identity with the
    eager reference when it held only at test sizes; that was corrected, and the claim now holds and is tested at
    scale. The non-quantised pairs have no rounded transcendental in their path and are unaffected.
- **Consequence E:** every fused class (`FusedManifestoHardLUT`, `FusedManifestoSoftLUT`, `FusedConfidenceLUT`) is
  exported from `lutorch_ex/__init__.py`.

**Deferred, outside #160 (on purpose):**
- The quantised-optimisation follow-up PR: `DeployedQuantisedConfidenceLUT`'s eval at 32,768 vectors (17.18 ms in
  the #160 snapshot vs 11.76 ms for the reference `QuantisedConfidenceLUT` and 9.67 ms for the fused class; it was
  2.6x its float twin in #159), int8 deploy at n=2, the n=1 variant, and the reference quantised path's speed,
  including its explicit left fold (eval peak 2.03 GiB at 32,768 vs 1.66 GiB in #159). The rounding-boundary item is RESOLVED on #160 (the
  bit-exact layer 1 above); the fp64-exponent exploration (`wip/quant-fp64-exponents`, unpushed) is superseded.
- Re-measure the snapshot of record on the H100 (it is RTX 5090 only).
- References to the removed `SoftSign*` names in the report `.tex`, experiment scripts, JSON and README outside
  lutorch_ex, and the nanochat profiling scripts (which also pass the removed `backend=`).
