# Optimising FusedManifestoHardLUT / FusedManifestoSoftLUT the way FusedConfidenceLUT was optimised

Analysis and plan only: **no kernel was changed**. Written 2026-10-10 on `research/lut_nanochat_v2`, which carries
`main` at a041d9d2 (after PR #157). Claims are tagged *(code)* (read from the source, with file and line),
*(measured)* (run here), *(arithmetic)* or *(estimate)*.

Hardware: RTX 5090 (sm_120, 32 GB GDDR7, 96 MB L2), torch 2.9.1+cu130, CUDA 13.0. DRAM peak: 1,792 GB/s *(spec:
512-bit × 28 Gbps)*. The H100 was **not** run for the Manifesto cartridges.

Geometry: the current reference is **h16 / tph64 / nap8, batch 32,768 vectors** (the nanochat micro-batch). At
**r = 64 this is not constructible** (d = 4, C(4,2) = 6 anchor pairs < nap 8; `canonical_full_coverage_pairs`
refuses), so the primary point here is **r = 128 (h16 d8)**, the smallest standard r that h16 / nap8 allows. The d24
LUT-FFN width h16 d48 (r = 768) is the second point, and the superseded h8 d8 (r = 64) is kept as a secondary
reference.

## 1. What the measurements say (findings first)

Cartridge-only training step, 32,768 vectors, table dropout 0.2, CUDA-event median of 3 interleaved rounds × 3 × 20
iterations *(measured; script and raw output in `../profiling/results/gpustar-rtx5090-2026-10-10-manifesto/`: `bench_manifesto.py`, `bench_5090_h16r128.txt` (h16 r = 128), `bench_5090.txt` (h8 r = 64 and d48); the compile and geometry probes are there too)*. Times in ms. "Gather" is
forward-table-row bytes / forward time, i.e. tokens · G · tph · cells · d · element size over the forward time
*(arithmetic)*.

**h16 tph64 nap8 d8 (r = 128), fp32 table:**

| cartridge (backend) | forward | fwd+bwd | backward | peak GiB | gather GB/s |
|---|---|---|---|---|---|
| FusedManifestoHard (auto = native) | 6.540 | 19.151 | 12.611 | 3.93 | 164 |
| FusedManifestoHard (tier1) | 50.658 | 57.926 | 7.269 | 6.16 | 21 |
| FusedManifestoSoft (auto = tier1) | 7.721 | 40.991 | 33.271 | 3.65 | 278 |
| FusedManifestoSoft (native) | 9.642 | 22.740 | 13.098 | 5.28 | 223 |
| ManifestoHard (pure) | 54.164 | 77.360 | 23.196 | 8.02 | 20 |
| ManifestoSoft (pure) | 52.662 | 88.491 | 35.829 | 7.27 | 41 |
| ConfidenceLUT n=1 (compiled) | 3.038 | 20.665 | 17.627 | 1.46 | 353 |
| **FusedConfidenceLUT n=1 (CUDA, #157)** | **0.775** | **4.051** | 3.276 | **0.17** | **1,385** |
| **FusedConfidenceLUT n=2 (CUDA, #157)** | **0.970** | **4.565** | 3.595 | 0.17 | 2,215 |

bf16 table at the same point: FusedManifestoHard native 6.702 / 18.714; FusedManifestoSoft tier1 7.747 / 40.856,
native 8.185 / 20.677; FusedConfidenceLUT n=1 0.785 / 4.101, n=2 0.939 / 4.601 (forward / fwd+bwd) *(measured)*.

**h16 tph64 nap8 d48 (r = 768, the d24 LUT-FFN width), fp32:** FusedManifestoHard native 8.697 / 53.846 (peak 9.20
GiB); FusedManifestoSoft tier1 11.596 / 50.068 (4.54 GiB), native 30.415 / 78.304 (15.44 GiB); ConfidenceLUT n=1
compiled 4.895 / 24.709; FusedConfidenceLUT n=1 1.339 / 6.746 (0.36 GiB), n=2 2.190 / 10.401. The pure Manifesto
cartridges run out of memory on the 32 GB card here *(measured)*.

**Superseded h8 tph64 nap8 d8 (r = 64), fp32:** FusedManifestoHard native 3.197 / 9.516; FusedManifestoSoft tier1
3.653 / 20.055, native 4.739 / 11.287; FusedConfidenceLUT n=1 0.361 / 1.971 *(measured)*.

Findings:
- **The Manifesto twins are 4.7–8× slower than FusedConfidenceLUT per training step, although their math is
  simpler** *(measured)*.
  - At h16 r = 128: Hard native 19.2 ms vs 4.1 ms, Soft (best backend) 22.7 ms vs 4.6 ms.
  - At d48: 53.8 ms vs 6.7 ms (Hard), 50.1 ms vs 10.4 ms (Soft, against n=2).
  - They use 20–40× more memory: 3.9 GiB vs 0.17 GiB at r = 128 *(measured)*.
- **None of them is at a DRAM roofline. The tables are L2-resident at batch 32,768.**
  - Footprints: the table is 2,097,152 entries at h16 r = 128, i.e. 8.4 MB fp32 / 4.2 MB bf16, and 12,582,912
    entries (50.3 MB fp32 / 25.2 MB bf16) at d48. Both fit the 5090's 96 MB L2 *(arithmetic)*.
  - Each table row is read ~128 times per forward at n=1, since 33.5M reads land on 262,144 rows *(measured
    2026-10-10, occupancy histogram)*.
  - The twin's forward gather reaches 4.8–5.9 TB/s at d48, which is 2.7–3.3× the 1,792 GB/s DRAM peak. That is only
    possible from L2 *(measured + arithmetic)*.
  - Arithmetic intensity of the read is ≈ 0.5 FLOP/byte in fp32 (one FMA per 4-byte element) and ≈ 1 in bf16
    *(arithmetic)*.
  - So the relevant ceiling is L2 bandwidth and latency, not HBM. The Manifesto twins' forward reaches only 164–278
    GB/s at r = 128 (9–16% of even the DRAM peak) and 741–1,111 GB/s at d48.
  - **Their cost is structure (launches, materialised intermediates, scalar per-element atomics), not bytes.**
    Kernel restructuring is the lever, not data layout or reuse *(estimate, from the measurements above)*.
- **At small d (r = 128) even the twin is not bandwidth-bound.** It reaches 1.4–2.2 TB/s, against 4.8–5.9 TB/s at
  d48. With d = 8 a table row is 32 bytes, so the per-table addressing work (8 margins, the score) costs as much as
  the gather. This regime is addressing- and latency-bound, so the h16 r = 128 priorities differ from d48's
  *(estimate)*.
- **`FusedManifestoSoft`'s `auto` picks the wrong backend at h16 r = 128.** At batch ≥ 4096 it picks tier1 *(code:
  `fused_manifesto_soft.py` L63–68)*, which costs 41.0 ms here, while native costs 22.7 ms. At d48 tier1 is the right
  choice (50.1 vs 78.3). The heuristic was tuned on an older H100 geometry *(code comment)*.

## 2. Why FusedConfidenceLUT is fast *(code: `src/spiky/lutorch_ex/cartridges/csrc/fused_confidence.cu`, `fused_confidence.py`; PR #157)*

- **Fusion boundaries.** There is one forward kernel (`confidence_fwd_kernel`, L163) and one backward kernel
  (`confidence_bwd_kernel`, L239) per step. The Inductor-compiled ConfidenceLUT runs 3 forward + 9 backward
  kernels at d24 (PR #157).
  - Each kernel does, in one launch: addressing (margins, MSB-first address, least-confident bit), score, blend
    weight, the gather-sum over tph, and in the backward the grad-W scatter, the score / blend / margin gradient
    into z, and the β / γ / τ gradients.
  - **Nothing per-table goes to global memory.** The backward recomputes the address and score from z
    (`address_table`, L129–161) instead of saving them, so peak memory is 0.17–0.36 GiB.
- **Thread / block decomposition.**
  - One CTA owns one group g and `rows_per_cta` consecutive samples (L180, L190).
  - Phase A runs one thread per table (addressing).
  - Phase B is "stripe × chunk": `nchunk = d_out / VEC` threads cover one table row with VEC-wide loads, and
    `nstripe = blockDim / nchunk` stripes take tables in turn (L166–167, L188–215).
  - Phase C reduces the stripes through shared memory (L218–221).
  - Launch knobs (`CudaKnobs`): 64 / 64 threads, rows_per_cta 4, vec 4 (fp32) / vec_bf16 4. They are explicit,
    with env overrides; the sweep winner on both the 5090 and the H100.
- **Memory strategy.**
  - The group's anchors are staged in shared memory as **int16** once per CTA (the reference reads int64 per
    element), and the z row is staged in shared memory.
  - Table rows are read with `__ldg` vector loads: float4 for fp32 (L76–80); 2/4/8/16-byte loads for bf16,
    upconverted in registers (L85–100).
  - Shared-memory offsets are 16-byte aligned (`Smem`, L50–68).
  - The table stays L2-resident through reuse across the batch (section 1); there are no explicit L2 controls (the
    persisting-L2 prototype was tried and removed in f92901af).
- **Warp primitives.** `__shfl_xor_sync` is used only in `block_sum`, for the β / γ / τ partials (L227–237). The dot
  products r = W[c] · go are reduced with shared-memory atomics (L299, L307). Address packing uses plain shifts and
  ORs per thread, not ballot / popcount.
- **dtype.**
  - Table T ∈ {fp32, bf16}, dispatched on the table dtype (`dispatch_table`).
  - All arithmetic and accumulation is fp32.
  - grad W is **always an fp32 buffer** (`zeros(W.sizes(), fp32)`), cast once to the table dtype after the full
    accumulation. A bf16 accumulator would stall: in a simulation of 2¹⁷ collisions it reached 128 against an exact
    48,739.
- **Backward strategy.**
  - The grad-W scatter uses **fp32 atomics**, vectorised (float4 / float2 `atomicAdd` on sm_90+, L110–127). That
    costs 58–109% if switched off (PR #157).
  - The z-row gradient accumulates in shared memory with shared atomics and is written once (L352–358). There are
    no global atomics for grad z and no zero-filled `[B, G, d_in]` buffers.
  - β / γ / τ are per-CTA partials, summed on the host.
  - It is not deterministic: atomic order varies, agreeing to ~1e-7 rel-norm.
  - There is no straight-through estimator: ConfidenceLUT's gradient is the exact derivative of its score-weighted
    value.
- **What it does NOT do (headroom).**
  - No `torch.library` registration: 6 graph breaks under whole-model `torch.compile` (measured 2026-10-10).
  - No warp-shuffle reduction of the dot products.
  - Not tuned for small d. At d = 8 a 64-thread CTA has `nchunk = 2` (vec 4), so only 2 lanes cover a row.
  - No fused table-dropout RNG (`torch.rand` outside the kernel).
  - No H100-specific tuning beyond the shared knob defaults.

## 3. The Manifesto twins today *(code: `cartridges/fused_manifesto_hard.py`, `fused_manifesto_soft.py`, `_native_ops.py`, `_fused_ops.py`, `csrc/lprojection.cu`)*

- **FusedManifestoHardLUT, `auto`, training = `native`** (`fused_manifesto_hard.py` L61–66, L77). Per step:
  - **Addressing:** a separately compiled Inductor graph (`_addr`). It returns z, u `[B, G, tph, nap]`, c, j\*, |u\*|
    and c′ as materialised tensors, then `_star` gathers the deciding anchors (more `[B, G, tph]` tensors).
  - **Forward value:** `fused_hard_read`, i.e. `F.embedding_bag` (an ATen kernel) (`_native_ops.py` L290).
  - **Extra setup tensors:** `_table_indices` (an arange expanded to `[B·nt]`, L163–165) and `batch_off` (a
    `repeat_interleave`, L291).
  - **Backward, first step:** `grad_grp` is **expanded and materialised to `[B, G·tph, d_out]`** (L305–308). That is
    6.4 GB at d48 and 1.07 GB at r = 128, fp32 *(arithmetic)*.
  - **Backward kernels:** `lprojection_backward_na1_nonsmooth` (one thread per (b, t, o), one **scalar** fp32
    `atomicAdd` per element, `lprojection.cu` L688–715), then a carriers kernel (one thread per (b, t), looping
    over d_out, so each thread reads its own row and the loads are not coalesced across the warp, L835–874), then
    `anchor_pairs_lookup_backward_all` (one thread per (b, t), two **global** atomics into grad z, L447–491).
  - **Kernel style:** int64 offsets throughout; no vector loads, no shared memory, no warp primitives in
    `lprojection.cu`.
- **FusedManifestoSoftLUT, `auto` at batch ≥ 4096 = `tier1`** (`fused_manifesto_soft.py` L63–73).
  - Forward: one `embedding_bag` with `per_sample_weights = [1−U, U]` over concatenated `[c, c′]` indices.
  - Backward: plain autograd, i.e. ATen's sort-based `_embedding_bag_dense_backward` for grad W plus the autograd
    tail through U and the compiled addressing into z.
  - The `native` backend (lprojection smooth forward + na1-smooth backward) has the same structure as Hard native,
    plus a separate weights kernel and the same `[B, nt, d_out]` expansion (L253–268).
- **Where they lose against the twin, item by item:**
  1. 6–10+ launches per step instead of 2.
  2. Materialised addressing intermediates (u is `[B, G, tph, nap]`).
  3. The `[B, nt, d_out]` gradient expansion.
  4. Scalar atomics: no vectorisation, and fp32 even when the table is bf16 (correct, but unvectorised).
  5. Uncoalesced per-thread row loops in the carriers kernel.
  6. Global atomics for grad z.
  7. int64 index math.
  8. For Soft tier1: a sort-based dense backward.

  bf16 is handled correctly: fp32 addressing, fp32-accumulated reads, and the kernels dispatch on the weight dtype
  *(code)*.

## 4. What transfers

Hard and Soft Manifesto share the twin's addressing exactly. The **least-confident-bit machinery (j\*, c′) is NOT
Confidence-specific**: it lives in the shared base `ManifestoLUT._addresses`, and Gen-1 Manifesto introduced it
*(code: `manifesto_base.py`, `_addresses`)*. What Manifesto lacks is the confidence **score**, and its neighbour
weight is the fixed U = 0.5/(1+|u\*|) instead of the learned σ(−2|u\*|/τ) *(code: `uncertainty.py`)*.

| twin optimisation | Hard Manifesto | Soft Manifesto | why |
|---|---|---|---|
| one fwd + one bwd kernel, recompute from z | transfers | transfers | family-agnostic |
| CTA = group × rows_per_cta, stripe × chunk read, knobs | as-is | as-is | family-agnostic |
| int16 anchors + z row in shared memory | as-is | as-is | same anchors / addressing |
| addressing in-kernel (c, j\*, c′) | as-is | as-is | shared base math |
| score s, β / γ (logσ, exp per margin) | **not applicable** | **not applicable** | no score in Gen-1 |
| neighbour weight | U = 0.5/(1+\|u\*\|), forward weight 1 (value) | U for both cells (forward and backward) | Gen-1 rational uncertainty, no learned τ |
| vectorised gather-sum forward (fp32 / bf16) | as-is (n=1-like, weight 1) | as-is (n=2-like, weights 1−U, U) | family-agnostic |
| fp32 grad-W accumulator + vector atomics | adapt: scatter only c (weight 1) | as-is (both cells, 1−U and U) | Hard's table gradient is hard (c only) |
| dot products ⟨go, W[c]⟩, ⟨go, W[c′]⟩ | **needed** (both, for the surrogate) | needed (both) | the input gradient uses W[c] − W[c′] |
| grad-z in shared memory, one store per row | adapt: **only the j\* pair**, coefficient −U′(\|u\*\|)·⟨go, W[c] − W[c′]⟩ (straight-through) | adapt: only the j\* pair, through U (exact derivative) | Gen-1 input gradient touches 2 anchors per table, not 2·nap |
| per-CTA scalar partials | **not applicable** | **not applicable** | no learned scalars |
| table dropout (keep flags, 1/keep scale) | as-is (scales the table's contribution) | as-is | same mechanism |

The Manifesto backward kernel would therefore be **simpler** than the twin's: no score chain, about 2 shared atomics
per table instead of 2·nap, no scalar partials. Its phase B is the same *(estimate)*.

## 5. Ordered optimisation plan (nothing implemented; needs Anatoly's go-ahead)

1. **Geometry-aware `auto` for FusedManifestoSoft** (cheap win).
   - Pick native at small d (it is 1.8× faster at h16 r = 128: 22.7 vs 41.0 ms) and keep tier1 at d48.
   - Expected −45% step at r = 128 *(measured gap)*. Effort: a one-line threshold plus a bench check.
   - Numerics: native vs tier1 differ only in fp32 summation order; the equivalence tests cover both.
   - Not a bf16 prerequisite.
2. **Drop the `[B, G·tph, d_out]` gradient expansion** in NativeHard / NativeSoft (cheap kernel touch).
   - Index g = t / tph inside the kernels instead of materialising.
   - Saves 1.07 GB (r = 128) to 6.4 GB (d48) of peak memory and a full-size copy per step *(arithmetic)*. Expected
     speedup is modest, a few ms at d48 *(estimate)*.
   - Effort: small. Numerics: none (same values). Not a bf16 prerequisite.
3. **Manifesto modes in the twin's kernels** (kernel rewrite, the main item).
   - Add MANIFESTO_HARD / MANIFESTO_SOFT as a mode template on `fused_confidence.cu`: a new `address_table` branch
     with no score, plus a lighter phase C.
   - Expected training step:
     - h16 r = 128: ≈ 3–4 ms vs 19.2 (Hard) / 22.7 (Soft), **≈ 5–6×**;
     - d48: ≈ 6–10 ms vs ≈ 50–54, **≈ 5–8×**;
     - peak memory ≈ 0.2–0.4 GiB instead of 4–9 GiB.

     This is an estimate from the twin's measured times, since the Manifesto math is lighter.
   - Effort: ~80–150 kernel lines, a Python twin class, oracle tests against ManifestoHard / ManifestoSoft, and
     bench rows.
   - Numerics: fp32 re-association of the atomics, about 1e-7 rel-norm against the oracle, like the twin.
   - **The straight-through Hard gradient must reproduce `ManifestoHardLUT._combine` exactly**: value = hard read;
     table gradient on c only; input gradient through U on the j\* pair. Any deviation would change trained models,
     so it gets an explicit oracle test.
   - It brings a bf16 table path with an fp32 accumulator, but the Manifesto twins already accept bf16, so it is
     **not a prerequisite** for Stage 2a bf16, only a speedup of it.
4. **Register the CUDA kernels as `torch.library` custom ops** (twin and Manifesto modes).
   - Removes the 6–7 graph breaks under nanochat's whole-model compile (measured 2026-10-10). Needed for
     `fullgraph=True`.
   - Worth it only if a profile of the real d24 step shows the breaks cost time; not measured yet.
   - Effort: medium (fake / meta functions, autograd registration). Numerics: none.
   - It IS the prerequisite for the "survives whole-model compile without breaks" part of Stage 2a; the bf16 part
     does not need it.
5. **Small-d tuning** (r = 128, d = 8).
   - The twin covers a 32-byte row with only 2 lanes, so rows per CTA, the stripe mapping, warp-shuffle dot products
     and possibly two rows per thread are worth sweeping.
   - Expected 10–30% on the twin and the Manifesto modes at d = 8 *(estimate)*. Effort: knob sweep plus a small
     kernel change. Numerics: summation order only.

Not measured here: the H100, the real d24 step with these cartridges inside nanochat, the cost of the graph breaks,
and real (trained) activations; all timings use Gaussian inputs.
