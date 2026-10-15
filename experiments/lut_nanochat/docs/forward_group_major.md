# Forward group-major CTA order ("option (b)", forward half): implementation guide

Sources: branch `exp/group_major_cta` = **f1d5d03a**. Its merge-base with the landed f88dd711 is **a07e9578** (not
f88dd711): option (b) forked from a07e9578 and contains BOTH halves. The diff below is `git diff a07e9578 f1d5d03a`.
The backward half is already on `cleanup/lutorch_ex_cosmetics` as f88dd711 (identical index math, unconditional); patch
only the forward half below. Records: `fq_cuda/stagec/p5/` (option (b), int64 index math), `fq_cuda/stagec/p4/`
("lever 0": group-major forward AND backward on every arch, 32-bit index math), `fq_cuda/staged/` (the landed
backward-only state).

## 1. The forward change (f1d5d03a), separated from the backward half

f1d5d03a has two parts. (A) The forward kernels take the CTA order as an argument chosen by a per-arch table. (B) The
backward kernels are always group-major. (B) is f88dd711. Patch only (A).

### 1a. The index math: 4 forward sites (line numbers at the current tip, a872a68f)

| kernel | file:lines (current tip) | row-block size R |
|---|---|---|
| `manifesto_fwd_body` | `csrc/fused_manifesto.cu:222-223` | `p.rows_per_cta` |
| `confidence_fwd_kernel` | `csrc/fused_confidence.cu:190-191` | `p.rows_per_cta` |
| `quant_fwd_kernel` | `csrc/pow2_int8_read.cu:248-249` | `QF_ROWS` |
| `deploy_fwd_kernel` | `csrc/pow2_int8_read.cu:668-669` | `qf::QF_ROWS` |

Each site, before (current) → after (f1d5d03a):

```diff
-    const int g = blockIdx.x % p.G;
-    const int b0 = static_cast<int>((static_cast<int64_t>(blockIdx.x) / p.G) * R);
+    // CTA order, Params::group_major (the per-arch default, _cuda_ext._ARCH_DEFAULTS): group-major keeps
+    // one group's CTAs consecutive, so the CTAs in flight read ~1-2 groups' table slices (L2 locality);
+    // interleaved walks the groups. One uniform choice per CTA, outside every loop.
+    const int64_t cta = static_cast<int64_t>(blockIdx.x);
+    const int64_t nb = (p.B + R - 1) / R;          // CTAs per group
+    const int g = static_cast<int>(p.group_major ? cta / nb : cta % p.G);
+    const int b0 = static_cast<int>((p.group_major ? cta % nb : cta / p.G) * R);
```

### 1b. The flag plumbing (f1d5d03a)

- **`Params` gains `int group_major;`** in 4 structs: `fused_manifesto.cu:46-55`, `fused_confidence.cu:37-49`,
  `pow2_int8_read.cu:109-121` (qf::Params, quant) and `:615-623` (deploy Params). In f1d5d03a it is added after the
  `int B, G, …` line. In qf::Params it is added before `float qlo, qhi;`.
- **The host entry points gain a trailing `bool group_major` and set `p.group_major = group_major;`** after the params
  are built:
  - `manifesto_fwd(..., bool out_bf16, bool group_major)` (`fused_manifesto.cu:478-481`);
  - `confidence_fwd(..., bool out_bf16, bool group_major)` (`fused_confidence.cu:462-466`);
  - `quant_fwd(..., bool save_tables, bool group_major)` (`pow2_int8_read.cu:822-827`);
  - `deploy_fwd(..., int64_t Q, bool group_major)` (`pow2_int8_read.cu:908-911`).
  - The `m.def` lines are unchanged: pybind takes the new argument positionally.
- **The Python call sites pass the flag** (current tip):
  - `_fused_manifesto_cuda.py:68` `ext.manifesto_fwd(..., vec, out_bf16, forward_group_major("MANI", z.device))`
  - `_fused_confidence_cuda.py:49` `ext.confidence_fwd(..., out_bf16, forward_group_major("CONF", z.device))`
  - `_fused_quantised_cuda.py:191` `ext.quant_fwd(z, anc_a16, anc_b16, *args, save, forward_group_major("QUANT", z.device))`
  - `deployed_quantised_confidence.py:104` `ext.deploy_fwd(..., cfg["Q"], forward_group_major("QUANT", z.device))`
  - `tests/test_fused_vs_reference.py:504` calls `ext.quant_fwd` directly and must pass the new argument too.
- **The per-arch table** (f1d5d03a, `cartridges/_cuda_ext.py`, after `class CudaKnobs`), verbatim:

```python
_ARCH_DEFAULTS = {(9, 0): {"cta_order": "group"}, (12, 0): {"cta_order": "interleaved"}}
_ARCH_FALLBACK = {"cta_order": "interleaved"}
_CTA_ORDERS = ("interleaved", "group")

def resolve_cta_order(capability, env_value: Optional[str] = None) -> str:
    if env_value is not None:
        if env_value not in _CTA_ORDERS:
            raise ValueError(f"CTA order {env_value!r}: expected one of {', '.join(_CTA_ORDERS)}")
        return env_value
    return _ARCH_DEFAULTS.get(tuple(capability), _ARCH_FALLBACK)["cta_order"]

@functools.lru_cache(maxsize=None)
def _forward_group_major(family: str, device_index: int) -> bool:
    env = f"LUTORCH_EX_{family}_CUDA_CTA_ORDER"
    try:
        order = resolve_cta_order(torch.cuda.get_device_capability(device_index), os.environ.get(env))
    except ValueError as e:
        raise ValueError(f"{env}: {e}") from None
    return order == "group"

def forward_group_major(family: str, device: torch.device) -> bool:
    return _forward_group_major(family, device.index if device.index is not None else torch.cuda.current_device())
```
  (plus `import functools`). Families: `MANI`, `CONF`, `QUANT` (QUANT covers both quant_fwd and deploy_fwd). The
  override is `LUTORCH_EX_{MANI,CONF,QUANT}_CUDA_CTA_ORDER=group|interleaved`. An unlisted arch gets interleaved, the
  current behaviour.
- **Design choice in f1d5d03a:** a runtime `Params` field, not a template parameter. A template flag would double
  every forward instantiation and the JIT build time. The select is one uniform branch per CTA, at kernel entry.

## 2. Mechanism

- **Grid:** `G × nb` CTAs, `nb = ceil(B / R)` row blocks per group. Each CTA reads its group's table slice
  `W[g]` (tph · 2^nap · d_out floats; 3 MiB at h16 tph64 nap8 d48) for R rows.
- **Interleaved (current forward):** `g = cta % G`, `b0 = (cta / G)·R`. Consecutive CTAs walk the groups, so the CTAs
  resident at any moment touch every group's slice, i.e. the whole table (48 MiB at that geometry).
- **Group-major:** `g = cta / nb`, `b0 = (cta % nb)·R`. One group's CTAs are consecutive, so the resident CTAs touch
  ~1–2 groups' 3 MiB slices.
- **Effect** (f1d5d03a commit message; `stagec/p5/h100_ncu.md`): the RTX 5090's 96 MB L2 holds the whole 48 MiB
  table either way. The H100's 50 MB L2 does not. Interleaved, the H100 re-streams the table from DRAM; group-major,
  the working set fits in L2.

## 3. Kernels, launch config, call sites

- **Affected forward kernels (4):** `manifesto_fwd` (FusedManifestoHard/Soft, both uncertainty_func values),
  `confidence_fwd` (FusedConfidenceLUT n=1/2), `quant_fwd` (FusedQuantisedConfidenceLUT n=1/2), `deploy_fwd`
  (DeployedQuantisedConfidenceLUT n=1/2).
- **Grid dimensions: unchanged.** The host computes `grid = G · ceil(B / R)` (`fused_manifesto.cu:493`,
  `fused_confidence.cu:477`, `pow2_int8_read.cu:839`, `:949`). The kernel recomputes `nb = ceil(B / R)` with the same
  formula, so `cta / nb ∈ [0, G)` and `cta % nb ∈ [0, nb)` cover the grid exactly once. No divisibility assumption:
  G need not divide anything, and the last row block of each group is partial exactly as before (the existing
  in-kernel row bound handles it).
- **h_out / n_groups:** the kernels index groups of `z [B, G, d_in]` (G = n_groups); routing to output heads (fan-in
  sum) happens after, in `_route`, unchanged. Group-major is equally valid for fan-in and fan-out.
- **Host glue:** no launch-config change is needed in `_fused_ops.py` or `_launch` / `_row_launch`. Only the four
  call sites above gain the flag argument.

## 4. Correctness: forward is bit-identical in either order

- Each output element is still written by one CTA with the same arithmetic; only which CTA index owns it changes.
- **Records, option (b) vs a07e9578, exact `torch.equal`, every fused row, champion + large, eval and train-mode
  forward:**
  - RTX 5090 (`stagec/p5/bitid.md`): 16/16 cells IDENTICAL in both columns, "FORWARD MISMATCHES: 0". Note: on the
    RTX 5090 the option-(b) table chooses interleaved, so this run checks the backward half and the plumbing, not
    the group-major forward (`stagec/p6/bitid.md`, the int64 variant re-checked, is the same case). For the
    group-major forward ON the RTX 5090 the evidence is f1d5d03a's `test_cta_order.py` CUDA cases, which run both
    orders and compare them bit-for-bit (GPU suite 1715 passed, per its commit message). The lever-0 dumps
    `stagec/p4/dump_gm.pt` vs `dump_base*.pt` exist, but their comparison output was not saved to a file.
  - H100 (`stagec/p5/h100_hop.txt`): 16/16 IDENTICAL (group-major forward active).
  - f1d5d03a's message: 32/32 identical on both cards.
- **No forward row was ever non-identical.**
- **Backward-only differences** (reassociated per-CTA scalar partials), not part of a forward patch: on the H100,
  `log_soft_score_temp` gradients of the softsign Manifesto rows differ by up to 9.7e-3 / 1.4e-2 relative
  (`p5/h100_hop.txt`); β/γ/τ ~1e-7–1.5e-6.
- **Tests:**
  - f1d5d03a's `tests/test_cta_order.py` (43 tests): table resolution, the unlisted-arch fallback, the env
    override per family and precedence, a bad value raising, a source check, and on CUDA *both forward orders
    bit-identical* for 7 classes (FusedManifestoHard/Soft, FusedConfidence n=1/2, FusedQuantised n=1/2, Deployed)
    × 2 batch sizes.
  - The CURRENT `tests/test_cta_order.py` (from f88dd711) pins the 4 forward sites to the *interleaved* regex
    (`_FWD`). It will fail after the patch until its source check is updated; take f1d5d03a's version as the
    template.
  - Coverage of the forward against the references: `tests/test_fused_vs_reference.py`,
    `tests/test_fused_manifesto_cuda.py`, `tests/test_fused_confidence.py`, `tests/test_deploy.py`.

## 5. Measured effect

All GPU kernel ms per step, bench_snapshot methodology. *combined.md is a single older session (2 interleaved
passes per card, 2026-10-14) at base a07e9578*; the published records (results_2026-10-15, f88dd711) are a
different session.

**H100 forward kernels, ncu** (`stagec/p5/h100_ncu.md`; manifesto_fwd and confidence_fwd only; quant_fwd and
deploy_fwd were not profiled there):

| geometry | kernel (row) | µs before → after | L2 hit % | DRAM read ÷ compulsory |
|---|---|---|---|---|
| champion | manifesto_fwd (Hard) | 412 → 307 (−25%) | 50 → 92 | 6.2× → 0.7× |
| champion | manifesto_fwd (Soft) | 615 → 426 (−31%) | 45 → 96 | 13.4× → 0.8× |
| champion | confidence_fwd (n=1) | 359 → 333 (−7%) | 50 → 92 | 6.1× → 0.7× |
| champion | confidence_fwd (n=2) | 670 → 446 (−33%) | 47 → 96 | 12.8× → 0.8× |
| large | manifesto_fwd (Hard) | 1566 → 1135 (−28%) | 50 → 96 | 9.7× → 0.6× |
| large | manifesto_fwd (Soft) | 2431 → 1627 (−33%) | 44 → 98 | 21.8× → 0.6× |
| large | confidence_fwd (n=1) | 1404 → 1254 (−11%) | 50 → 96 | 9.8× → 0.6× |
| large | confidence_fwd (n=2) | 2687 → 1686 (−37%) | 47 → 98 | 20.6× → 0.6× |

**H100/RTX 5090, eval and train, large** (`stagec/p5/combined.md`; option (b) = backward + H100 forward group-major,
RTX 5090 forward interleaved):

| row | eval H100/5090 before → after | train H100/5090 before → after |
|---|---|---|
| FusedManifestoHard | 1.52 → 1.11 | 1.39 → 1.26 |
| FusedManifestoSoft | 1.30 → 0.86 | 1.42 → 1.02 |
| FusedManifestoHard·softsign | 2.07 → 1.31 | 1.71 → 1.47 |
| FusedManifestoSoft·softsign | 1.35 → 0.85 | 1.84 → 1.23 |
| FusedConfidence n=1 | 1.28 → 1.14 | 1.36 → 1.27 |
| FusedConfidence n=2 | 1.39 → 0.86 | 1.63 → 1.18 |
| FusedQuantised n=1 | 1.90 → 1.44 | 1.52 → 1.18 |
| FusedQuantised n=2 | 1.69 → 1.27 | 1.63 → 1.01 |
| Deployed (n=2) | 1.33 → 1.35 | – |
| Deployed n=1 | 1.38 → 1.41 | – |

The landed backward-only state, for comparison (published records): large eval 1.27–2.03, train 1.12–1.57.
The forward part is what moves eval.

**H100 eval, forward group-major vs baseline** (`stagec/p5/h100_table.md`, large): FusedManifestoHard −27.4%,
FusedManifestoSoft −33.9%, ·softsign −36.8% / −37.3%, FusedConfidence n=1 −11.0%, n=2 −37.8%, FusedQuantised n=1
−23.2%, n=2 −25.7%; **Deployed +1.9% (n=2), +3.0% (n=1): slower.**

**RTX 5090 with the forward group-major** (`stagec/p4/rtx5090_table.md`, lever 0, 32-bit index math, eval GPU ms):
- large fused eval **+0.7% to +8.4% slower**: FusedQuantised n=2 +8.4%, FusedConfidence n=2 +4.8%,
  FusedManifestoSoft +3.3%.
- champion fused eval −4.7% to +6.0%: FusedQuantised n=2 +6.0%, n=1 +4.2%; Manifesto/Confidence n=1 −4.0% to −4.7%.
- Deployed −0.8% to +0.4%.
- This is why f1d5d03a keeps sm_120 interleaved. RTX 5090 noise floor in that session: median 0.26%, p90 1.12%.

## 6. Risks and unresolved items

- **Occupancy / register allocation:** the per-row gain is sensitive to nvcc register allocation.
  `stagec/p5/v32_h100_threeway.md`: the 32-bit form of the same index math vs the int64 form on the H100, e.g.
  FusedQuantised n=1 large eval −14.7% (32-bit) vs −23.4% (int64); FusedManifestoHard·softsign large train −21.3% vs
  −17.3%; Deployed +6.4% vs +1.9%. f1d5d03a's message: "a follow-up to pin occupancy per kernel is planned". That
  follow-up was never done. Use the int64 form (above) to reproduce the p5 numbers.
- **sm_90 vs sm_120:** group-major forward helps the H100 (−7% to −38% per fused row) and costs the RTX 5090 up to
  +8.4% at large batch, hence the per-arch table. Other architectures were never measured. An unlisted arch falls back
  to interleaved.
- **Deployed int8 read (`deploy_fwd`):** under group-major it is slower on the H100 (+1.8% to +3.2% int64, up to +6.4%
  32-bit; `p5/v32_h100_threeway.md`) and neutral on the RTX 5090. f1d5d03a still routes it through the QUANT table
  entry, which on the H100 means group-major. Consider giving deploy its own family, or leaving it interleaved.
- **read_top_n=2 vs 1:** the n=2 forward rows gain most on the H100 (FusedConfidence n=2 −37.8% vs n=1 −11.0%) and
  cost most on the RTX 5090 (FusedQuantised n=2 +8.4%, FusedConfidence n=2 +4.8%). Same kernels, templated on N; no
  different correctness behaviour.
- **Not profiled:** quant_fwd and deploy_fwd have no ncu L2 / DRAM table (p5's ncu covers manifesto / confidence only).
- **Measurement scope:** one H100 VM session, which no longer exists. combined.md is a single older session.
  Re-measure on any target card before relying on the ratios.
- **Quick-experiment parts:** the per-arch table holds only two entries, set from one session per card. The
  occupancy follow-up was never done. The README lines f1d5d03a added describe the table; check them against the
  rewritten README.

## 7. Gate after patching (green suite BEFORE any timed benchmark)

From the repository root, with a fresh extension cache so the edited `.cu` files are rebuilt:

```bash
export TORCH_EXTENSIONS_DIR=$HOME/.cache/torch_ext_fwd_gm   # fresh build
export TRITON_CACHE_DIR=$HOME/.cache/triton
python -m pytest src/spiky/lutorch_ex/tests -q                                   # full GPU suite
CUDA_VISIBLE_DEVICES= python -m pytest src/spiky/lutorch_ex/tests -q             # CPU suite
python -m pytest src/spiky/lutorch_ex/tests/test_cta_order.py \
                 src/spiky/lutorch_ex/tests/test_fused_vs_reference.py \
                 src/spiky/lutorch_ex/tests/test_fused_manifesto_cuda.py \
                 src/spiky/lutorch_ex/tests/test_fused_confidence.py \
                 src/spiky/lutorch_ex/tests/test_deploy.py -q                    # the bit-exactness / parity subset
```

- On the H100 run it twice, once with `LUTORCH_EX_{MANI,CONF,QUANT}_CUDA_CTA_ORDER=group` and once with
  `=interleaved`, so both orders are exercised on that card. The CUDA cases in f1d5d03a's `test_cta_order.py`
  compare the two orders bit-for-bit.
- The forward bit-identity check against the pre-patch tree, as recorded: the scratch scripts
  `fq_cuda/stagec/p4_dump.py` (dump every fused row's eval and train-mode forward) and `fq_cuda/stagec/p4_compare.py`
  (exact torch.equal of two dumps).
- Only after all of that is green, time with
  `python -m spiky.lutorch_ex.benchmarks.bench_snapshot --geometries champion,large --dtypes float32 --rounds 3`,
  interleaved with the unpatched tree in one session.
