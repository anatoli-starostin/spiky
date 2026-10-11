# lutorch_ex: target architecture for the cartridges

**Status:** agreed goal state, not yet implemented. Future PRs land against this spec. The code on `main` (36aa7a13, the
#159 merge) does not follow it yet; the gaps are listed at the end. Each item below is marked **DECIDED** or **OPEN**.

## Anatoly's five rules (the contract) - DECIDED

1. **Manifesto and SoftSign are merged** (the Gen-2 fold): `rational | softsign` becomes a weight-function parameter,
   not separate classes. The fold is a net deletion of three files, landing in two places (reference and fused).
   - The parameter selects the neighbour-weight function: `weight="rational"` is the Gen-1 `U = 0.5/(1+|u*|)`;
     `weight="softsign"` is the Gen-2 learned-temperature `w`.
   - On `main` the Gen-2 files are `softsign_base.py`, `softsign_hard.py`, `softsign_smooth.py` (reference) and
     `fused_softsign.py` (fused).
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
  - **bf16 reference support: DECIDED, NOT YET IMPLEMENTED.** Today the plain cartridges raise on bf16 and only the
    fused ones accept it, the exact reverse of rule 4. This gap is to be closed.
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
- **The `SoftSign*` names: no deprecated aliases** (Anatoly's preference) unless something outside the package
  depends on them. No checkpoint and no model code does (state_dicts hold no class names; the learned temperatures
  stay `log_soft_score_temp` / `log_select_temp`). The only references are documentation and experiment / profiling
  scripts, listed in #160; final call pending.

## `LUTORCH_EX_NO_CUDA_EXT` under this spec

Today it is read in three places:
- `_cuda_ext.load_extension` skips building the five JIT extensions (`lprojection`, single-anchor input grad,
  soft-sign surrogate, `fused_manifesto`, `fused_confidence`);
- `_fallback.disabled_on_purpose` / `is_deliberate` turn the resulting fallback from a `NativeUnavailableError` into a
  quiet, deliberate one;
- `_pow2_int8` skips the p2 / int8 extension (with the deprecated alias `SPIKY_P2_CUDA_DISABLE`).

Under rules 3 and 5 it **disappears**:
- A fused class raises when its kernel cannot load, so there is no fallback left to make quiet.
- No reference class loads an extension (`ManifestoHard/Soft`, `SoftSignHard/Smooth` and `ConfidenceLUT` import
  none today).
- **The replacement lever** is explicit class choice (E): to run without the extension (CI, a CPU-only box),
  construct the reference class. Tests of a fused class skip when its extension cannot load.
- **One residual, outside this spec's families:** `QuantisedConfidenceLUT` registers the p2 op with a built-in pure
  fallback, and `DeployedQuantisedConfidenceLUT` reads through the int8 extension. Until the quantised-cartridges
  work splits these into a reference and a fused form, the variable (or `SPIKY_P2_CUDA_DISABLE`) still gates that one
  path.

## Gaps between `main` and this spec (for the PRs that land it)

- **Rule 5:** the `Fused*` classes have a `backend=` argument and fall back internally:
  - FusedManifesto*: `pure` / `pure_eval` / `tier1` / `native` / `cuda`;
  - FusedSoftSign*: `pure_eval` / `tier1` / `native`;
  - FusedConfidenceLUT: `pure` / `cuda`.

  That also breaks rule 3. One consequence measured in #159: an explicit `backend="pure_eval"` trains silently with
  no input gradient on FusedManifestoHard / FusedSoftSignHard.
- **Consequence B:** `ManifestoLUT` (the shared base) has a runnable forward.
- **Rules 1 and 3:** the SoftSign family has no CUDA kernel of its own. Its fast path is the generic `lprojection`
  extension plus a surrogate-tail kernel, so a `FusedManifesto*(weight="softsign")` needs the `fused_manifesto.cu`
  kernels to take the two soft-sign temperatures.
- **Consequence C:** the reference classes raise on bf16.
- **Consequence D:** fused-vs-reference equivalence tests exist (`test_fused_equivalence`, `test_fused_softsign`,
  `test_fused_confidence`, `test_fused_manifesto_cuda`), but not uniformly per cartridge and per dtype with explicit
  tolerances.
- **Consequence E:** `FusedConfidenceLUT` is not exported from `lutorch_ex/__init__.py`.
