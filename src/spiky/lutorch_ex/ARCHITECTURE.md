# lutorch_ex: target architecture for the cartridges

**Status:** agreed target, not yet implemented. Future PRs land against this spec. The current code (`main` at
54de5377) does not follow it yet; the gaps are listed at the end.

## Rules

1. **Manifesto and SoftSign are one cartridge family**, with the neighbour-weight function as a parameter:
   `weight="rational"` (the Gen-1 `U = 0.5/(1+|u*|)`) or `weight="softsign"` (the Gen-2 learned-temperature `w`).
   The fold happens in both the reference and the fused implementation.
   *Why:* the two families differ only in that one function. Today the addressing, routing, `_star` and the two-cell
   read are duplicated line for line between them.
2. **Reference and fused implementations live in separate files.** This is a recommendation, not an absolute.
   Shared, non-math scaffolding SHOULD be reused through a common base class: the config dataclass, geometry
   validation, anchor construction, address packing and parameter registration. Only the forward / backward maths is
   deliberately duplicated.
   *Why:* each file can then be read on its own, and the shared state cannot drift between the two.
3. **A fused implementation runs only with its CUDA extension built and loaded.** If the kernel is unavailable (not
   built, an unsupported dtype, a non-CUDA tensor), it RAISES. It never falls back to an unfused path. It may be as
   complicated as needed; efficiency is king.
   *Why:* a silent slow path produced invalid timings once already (the setuptools incident behind PR #158).
4. **A reference implementation stays reasonably efficient, but readability is its primary goal.** It attempts
   `torch.compile` for speed and runs plain PyTorch when compile is not possible.
   *Why:* it is the specification of the maths, and the path on any machine without the extension.
5. **There is no `backend` concept and no `backend=` argument.** The model chooses explicitly, by constructing the
   class it wants.
   *Why:* a per-call dispatcher across 3–5 backends hid which code ran and made "what is the fast path?" a
   runtime question.

## Consequences (agreed, part of the spec)

- **The shared base class has no runnable forward.** It is abstract and holds state and validation only, so a
  missing kernel has nowhere in the MRO to resolve to silently. Rule 3 is enforced by structure, not by convention.
- **The reference implementation is load-bearing, not a test fixture.** On a machine without the extension it is
  the only path, so it must support the real dtypes and geometries, including bf16. Today it is the reverse: the
  plain cartridges raise on bf16 and only the `Fused*` classes accept it. Closing that bf16 gap is part of the work.
- **Separate files mean nothing structural keeps them in step.** A per-cartridge fused-vs-reference numerical
  equivalence test, with explicit per-dtype tolerances (value, input gradient, table gradient, learned scalars), is
  the binding mechanism and a REQUIRED check for every cartridge.
- **Selection is by explicit class construction, so class NAMES are the public API.**

## Naming (PROPOSAL, for Anatoli to overrule)

Keep the scheme the package already uses: `<Family><Variant>LUT` for the reference implementation, and the same name
with a `Fused` prefix for the CUDA implementation. Five classes already follow it (`FusedManifestoHardLUT`,
`FusedManifestoSoftLUT`, `FusedSoftSignHardLUT`, `FusedSoftSignSmoothLUT`, `FusedConfidenceLUT`).

| family | reference | fused (CUDA only) |
|---|---|---|
| Manifesto, hard read + surrogate gradient | `ManifestoHardLUT(weight="rational" \| "softsign")` | `FusedManifestoHardLUT(weight=...)` |
| Manifesto, blended read | `ManifestoSoftLUT(weight="rational" \| "softsign")` | `FusedManifestoSoftLUT(weight=...)` |
| Confidence | `ConfidenceLUT` | `FusedConfidenceLUT` |
| Confidence, quantised | `QuantisedConfidenceLUT` | none yet (`DeployedQuantisedConfidenceLUT` stays the inference-only deploy form) |

- **Soft vs smooth.** SoftSign's "Smooth" is the same variant as Manifesto's "Soft" (the blend is the value), so the
  merged family keeps one word, `Soft`.
- **Migration.** `SoftSignHardLUT` / `SoftSignSmoothLUT` and their `Fused` twins become thin deprecated aliases
  (`weight="softsign"`) for one release, then go. State_dict keys do not change: the learned temperatures stay
  `log_soft_score_temp` / `log_select_temp`, present only with `weight="softsign"`.
- **Rejected alternative:** a suffix or namespace scheme (`ManifestoHardRef` / `ManifestoHardCuda`, or
  `lutorch_ex.reference.ManifestoHard` / `lutorch_ex.fused.ManifestoHard`). It would rename every existing public class
  and break every import and doc for no gain over the `Fused` prefix already in use. The namespace version also makes
  the two classes indistinguishable at a glance in a model's `repr`.

## Gaps between `main` and this spec (for the PRs that land it)

- The `Fused*` classes have a `backend=` argument and fall back internally:
  - FusedManifesto*: `pure` / `tier1` / `native` / `cuda`;
  - FusedSoftSign*: `pure_eval` / `tier1` / `native`;
  - FusedConfidenceLUT: `pure` / `cuda`.

  This breaks rules 3 and 5.
- `ManifestoLUT` (the shared base) has a runnable forward. That breaks the first consequence.
- The SoftSign family has no CUDA kernel of its own: its fast path is the generic `lprojection` extension plus a
  surrogate-tail kernel. Under rule 3 a `FusedManifesto*(weight="softsign")` needs the `fused_manifesto.cu` kernels to
  take the two soft-sign temperatures.
- The reference classes raise on bf16. That breaks the second consequence.
- Fused-vs-reference equivalence tests exist (`test_fused_equivalence`, `test_fused_softsign`,
  `test_fused_confidence`, `test_fused_manifesto_cuda`), but not uniformly per cartridge and per dtype with explicit
  tolerances.
