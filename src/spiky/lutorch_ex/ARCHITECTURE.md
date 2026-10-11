# lutorch_ex: target architecture for the cartridges

**Status:** agreed goal state, not yet implemented. Future PRs land against this spec. The code on `main` (36aa7a13, the
#159 merge) does not follow it yet; the gaps are listed at the end. Each item below is marked **DECIDED** or **OPEN**.

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
| Confidence, quantised | `QuantisedConfidenceLUT` | none yet (`DeployedQuantisedConfidenceLUT` stays the inference-only deploy form) |

- **Rejected:** an implementation suffix (`ManifestoHardRefLUT` / `ManifestoHardCudaLUT`), which renames every class,
  and namespaces (`lutorch_ex.reference.X` / `lutorch_ex.cuda.X`), which change every import and make the two forms
  identical in a `repr`.
- **The `SoftSign*` names: DECIDED, no aliases, no shims, no deprecation wrappers.** The fold renames them cleanly
  (one name per thing). No checkpoint and no model code depends on them: state_dicts hold no class names, and the
  learned temperatures stay `log_soft_score_temp` / `log_select_temp`. References in the report `.tex` and in
  experiment / profiling scripts are left dangling on purpose until the final cleanup; they are listed in #160.

## `LUTORCH_EX_NO_CUDA_EXT` under this spec

Before this spec it was read in three places: `_cuda_ext.load_extension` (skip building every JIT extension),
`_fallback.disabled_on_purpose` / `is_deliberate` (make the resulting fallback quiet), and `_pow2_int8` (skip the
p2 / int8 extension, with the deprecated alias `SPIKY_P2_CUDA_DISABLE`).

Under rules 3 and 5 the first two **are gone**:
- A fused class raises when its kernel cannot load, so there is no fallback left to make quiet.
- No reference class loads an extension (`ManifestoHard/Soft` and `ConfidenceLUT` import none).
- **The replacement lever** is explicit class choice (E): to run without the extension (CI, a CPU-only box),
  construct the reference class. Tests of a fused class skip without a CUDA device and fail on one without the
  extension (`tests/_requires_extension.py`).
- **One residual, outside this spec's families:** `QuantisedConfidenceLUT` registers the p2 op with a built-in pure
  fallback, and `DeployedQuantisedConfidenceLUT` reads through the int8 extension. Until the quantised-cartridges
  work splits these into a reference and a fused form, the variable (or `SPIKY_P2_CUDA_DISABLE`) still gates that one
  path.

## Gaps between `main` and this spec (for the PRs that land it)

**Landed on #160:**
- **Rule 1** (the fold): `ManifestoHardLUT` / `ManifestoSoftLUT` and their `Fused` twins take
  `weight="rational" | "softsign"`; the four SoftSign files are gone, with no aliases; the `fused_manifesto.cu`
  kernels take the soft-sign temperatures (their gradients included).
- **Rules 3 + 5:** no `backend=` anywhere. The fused classes run only their CUDA kernels and raise otherwise: a
  case the kernels do not cover names the case and the reference class, a missing extension names the classified
  cause. The `pure` / `pure_eval` / `tier1` / `native` paths are deleted, with `_native_ops.py`,
  `_native_softsign.py` and their CUDA sources (`lprojection`, single-anchor input grad, `softsign_surrogate_grad`).
  `LUTORCH_EX_NO_CUDA_EXT` survives only on the p2 / int8 path (above).
- **Consequence B:** `ManifestoLUT` is scaffolding only; its `forward()` raises `NotImplementedError`. The
  reference classes take their (compiled) forward from `ReferenceForward` (`cartridges/reference_base.py`).
  `ConfidenceLUT` and `FusedConfidenceLUT` both build on `ConfidenceBase`; the fused class no longer inherits
  `ConfidenceLUT`'s forward.
- **Consequence C:** the reference classes accept bf16 / fp16: fp32 maths, one rounding per leaf, bit-identical to
  the fp32 run rounded once (`tests/test_reference_low_precision.py`). `QuantisedConfidenceLUT` keeps rejecting low
  precision (outside this spec's families).

- **Consequence D:** `tests/test_fused_vs_reference.py` is the one fused-vs-reference harness: per cartridge (both
  Manifesto twins with both weights, FusedConfidenceLUT n=1 and 2), per dtype (fp32, bf16, fp16 input), per anchor
  mode and head pattern, on every leaf, with explicit tolerances justified in its docstring. It replaced
  `test_fused_equivalence.py` and `test_fused_manifesto_softsign.py`, and found (and the fused classes now fix) a
  double rounding of a low-precision input's gradient in the fan-out pattern.

**Still open:**
- **Consequence E:** `FusedConfidenceLUT` is not exported from `lutorch_ex/__init__.py`.
