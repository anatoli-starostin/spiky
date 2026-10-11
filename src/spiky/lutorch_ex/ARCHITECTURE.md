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
    to the fp32 run rounded once.
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
  `LUTORCH_EX_NO_CUDA_EXT` is gone (above).
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
    n=2 training on CUDA; that speed now lives in `FusedQuantisedConfidenceLUT`. Measured in the #160 snapshot of
    record (`benchmarks/snapshot_160.md`): at 32,768 vectors eval 9.66 -> 10.09 ms (+4.5%), train total 48.56 ->
    49.22 ms (+1.4%), train peak 3.64 -> 4.38 GiB; at 8,192 within 1%. Not to be fixed in #160.
- **Consequence B:** `ManifestoLUT` is scaffolding only; its `forward()` raises `NotImplementedError`. The
  reference classes take their (compiled) forward from `ReferenceForward` (`cartridges/reference_base.py`).
  `ConfidenceLUT` and `FusedConfidenceLUT` both build on `ConfidenceBase`; the fused class no longer inherits
  `ConfidenceLUT`'s forward.
- **Consequence C:** the reference classes accept bf16 / fp16: fp32 maths, one rounding per leaf, bit-identical to
  the fp32 run rounded once (`tests/test_reference_low_precision.py`). `QuantisedConfidenceLUT` keeps rejecting low
  precision (outside this spec's families).
- **Consequence D:** `tests/test_fused_vs_reference.py` is the one fused-vs-reference harness: per cartridge (both
  Manifesto twins with both weights, FusedConfidenceLUT n=1 and 2, FusedQuantisedConfidenceLUT), per dtype (fp32,
  bf16, fp16 input) and anchor mode wherever the fused class covers them, per head pattern, on every leaf, with
  explicit tolerances justified in its docstring. It replaced
  `test_fused_equivalence.py` and `test_fused_manifesto_softsign.py`, and found (and the fused classes now fix) a
  double rounding of a low-precision input's gradient in the fan-out pattern.
- **Consequence D, the quantised pair: two layers.** The power-of-two exponents are round-half-up floors of fp32
  transcendental expressions, and no two implementations agree on them to the last ulp, so a value within a few ulps
  of a rounding boundary can floor differently (one table's weight changes 2x). Two independent sources, tested
  separately:
  - *Layer 1, the oracle -- fused vs the eager (uncompiled) reference.* Only the kernel's own log1p / exp / log2
    (nvcc, `--fmad=false`) vs torch's CUDA kernels differ: 0-3 integer disagreements per 8.4M tables at the champion
    scale (4 seeds, at init and after training), all within 1e-7 (fp64) of a boundary. On every table more than
    1e-5 (fp64) from a boundary the integers are exact and the output rows bit-identical (99.996% of tables, 99.73%
    of rows); the remainder is counted, not asserted.
  - *Layer 2, compile trust -- the compiled reference vs the eager reference.* Inductor reassociates the
    transcendentals and flips ~1e-7 of tables; a tolerance test ("compile did not break the maths"): flipped rows
    <= 1e-6 per table, eval relative norm <= 1e-3, training <= 1e-2 (worst measured 2.4e-7, 3.1e-4, 2.9e-3).
  - Constructed exact ties: the reference and the kernel both round half up. Natural exact ties: zero.
  - **Correction.** An earlier version of this section (and the harness, and #160's text) said the fused class is
    bit-identical to the eager reference. That held only at test sizes (<= 0.6M tables, where a 1e-7 event rarely
    lands); at the champion scale it is false. The non-quantised pairs have no rounded transcendental in their path
    and are unaffected. No library code reads or exposes any test switch: the gate is recomputed in the tests.
- **Consequence E:** every fused class (`FusedManifestoHardLUT`, `FusedManifestoSoftLUT`, `FusedConfidenceLUT`) is
  exported from `lutorch_ex/__init__.py`.

**Deferred, outside #160 (on purpose):**
- The quantised-optimisation follow-up PR: `DeployedQuantisedConfidenceLUT` 2.6x slower than its float twin at 32k
  tokens, int8 deploy at n=2, the n=1 variant, the reference quantised path's speed, and the rounding-boundary
  item: true bit-exactness of the quantised pair needs the kernel and the reference to share one transcendental
  implementation (fp64 on both sides, or torch's CUDA math ported into the kernel) vs the documented status quo
  (exact away from boundaries, bounded at them). Measured for fp64 exponents in the reference alone (branch
  `wip/quant-fp64-exponents`, unpushed, the starting point): 0 compiled-vs-eager flips, but 0-6 disagreements per
  8.4M tables with the fp32 kernel, and on the RTX 5090 (sm_120, fp64 at 1/64 of fp32) quantised eval 4.5x and
  train +80% slower; cheap on H100 (sm_90, fp64 at 1/2).
- References to the removed `SoftSign*` names in the report `.tex`, experiment scripts, JSON and README outside
  lutorch_ex, and the nanochat profiling scripts (which also pass the removed `backend=`).
