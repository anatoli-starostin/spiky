# FusedQuantisedConfidenceLUT as pure CUDA: Stage 0, measurement and inventory

Branch `perf/lutorch_ex_fused_quantised_cuda`. Measurement only: no kernel or cartridge is changed by this note. RTX 5090 (sm_120), torch 2.9.1+cu130, code = `main` at 31c4ef97.

**Sources:**
- Totals are `benchmarks/snapshot_162.json` (the snapshot of record, same protocol).
- Per-kernel breakdowns: torch.profiler, one fresh process per cell, the same construction as bench_snapshot (seed=1, `torch.manual_seed(0)` inputs, Dynamo reset). After the first call and 10 warm-up iterations, 5 eval forwards and 5 train steps were recorded.
- "Non-kernel" is the share of CUDA-event wall time not covered by kernel time (launch and host gaps).

**Geometry:** champion h16 tph64 nap8 d_in = d_out 48, K = 2^8 cells per table, at 8,192 and 32,768 vectors.

**bf16:**
- `FusedQuantisedConfidenceLUT` raises ("not a case the fused op covers": fp32 input and table only).
- `DeployedQuantisedConfidenceLUT` raises (fp32/fp64 inference only).
- Their bf16 cells are not measurable today; FusedConfidenceLUT's bf16 cells are given.

## Summary

- **The gather is the problem, not the quantisation.** At 32,768 vectors, fp32 eval, FusedQuantised spends 7.15 ms of 9.59 ms of kernel time in PyTorch's generic `embedding_bag` forward. FusedConfidenceLUT does the same two-cell, 64-table gather-and-blend, plus all its addressing and scoring, in **one 1.85 ms kernel**.
- **The exponent derivation is ~2.3 ms**, and almost all of it is implementation debt, not arithmetic. It writes the `[B, 16, 64, 8]` fp32 margin tensor (1 GiB) to global memory because it is the input of a custom op, re-reads it in a one-thread-per-table kernel, and builds index and weight tensors for `embedding_bag`.
- **The genuinely quantisation-specific work** is a handful of extra scalar operations per table (log2 S, two divisions, floor, c_q, clamps, ldexp) plus the per-call fake-quant of the table: ~0.1–0.2 ms at 32,768.
- **Realistic target at 32,768:**
  - eval ≈ **2.0 ms**, from 9.68 (≈ 4.8×);
  - train total ≈ **10 ms**, from 48.5 (≈ 4.8×), the backward done the way FusedConfidenceLUT does it.
- **DeployedQuantisedConfidenceLUT is slower** (16.8 ms) because its int8 shift-add read, compiled by Inductor, takes 13.9 ms. Its derivation costs the same as FusedQuantised's.

## The goal state (Anatoly), and whether it is reachable

1. **FusedQuantisedConfidenceLUT only "a hair" slower than FusedConfidenceLUT.** Reachable to within the quantisation arithmetic, which is bounded by measurement: **≤ +0.45 ms at 32k (≤ +24%) and ≤ +0.11 ms at 8k (≤ +23%)**, expected far less once it overlaps the gather (§G1). "A hair" (a few %) is plausible but not guaranteed; a Stage 1 prototype decides it.
2. **DeployedQuantisedConfidenceLUT eval faster than its parent, and ideally than FusedConfidenceLUT.** Faster than the parent: yes, by a wide margin. Faster than FusedConfidenceLUT fp32 (1.88 ms at 32k): **likely**, ≈ 0.9–1.3 ms extrapolated. Faster than FusedConfidenceLUT bf16 (1.21 ms): borderline for n=2, likely for n=1. See §G2; the numbers are an extrapolation, not a measurement.
3. **An n=1 deployed variant.** A design is in §G3. I argue against a separate class, with reference to Rule 5.

### G0. Input independence, settled: the margins, q and k' are PER SAMPLE

- `u = z_a − z_b` with `z = x[:, self.in_head, :]`: `ManifestoLUT._addresses`, quoted in §D1. The training monolith's `d = (z.gather(2, idx_a) - z.gather(2, idx_b))` is the same.
- The `[B, G, tph, nap]` shape carries the batch dimension because each margin is a difference of two input coordinates.
- `p2::table_scalars` derives q from min |d| and k' from Σ|d| and Σ logsigmoid(β|d|), per (token, group, table).
- **Exponent precomputation at deploy time is off the table.** Deployment can precompute only the table side (`packed` int8 and the per-(head, channel) exponent e), which it already does. **Goal 2 has to come from less table traffic and less bookkeeping, not less arithmetic.**

### G1. Goal 1: the residual cost of quantisation inside a fused kernel

- **Measured:** `p2_scalar_kernel` on L2-resident margins (0.5M / 1M / 2M tables: 0.0435 / 0.0492 / 0.0636 ms per call). The marginal cost is **0.0134 ms per million tables** for the *whole* `p2::table_scalars`, its stores included.
  - At 8k (8.4M tables): ≤ 0.11 ms. At 32k (33.6M tables): ≤ 0.45 ms.
- **That is an upper bound on the quantisation-specific part,** because it includes work FusedConfidence's phase A already does: the margins fold, Σ|u|, Σ logsigmoid and argmin.
  - The genuinely extra operations are log2 S, two divisions, two floors, c_q = log2(1+2^-q), the clamps and two ldexp. That is ~4–5 SFU-class operations against phase A's ~18 (8 × exp + log1p, the score exp, the blend sigmoid).
  - So the true increment is roughly a quarter of phase A's arithmetic. Inside a gather-bound kernel it partly overlaps the other CTAs' loads.
- **No extra table traffic:** the fake-quant table has the same shape and bytes as the fp32 table FusedConfidence reads. Its per-call rebuild is 0.04 ms (or once per step).
- **Verdict:** a fused quantised kernel lands at **FusedConfidence + (0 … 0.45) ms at 32k**, i.e. +0% to +24%, expected toward the low end. "A hair" is physically plausible, not physically guaranteed: the gap is not irreducible, but its size depends on how well the extra SFU work hides under the gather. Stage 1 should build the fused forward first and measure exactly this before the backward.

### G2. Goal 2: the levers for deployed eval (32k, fp32 input)

Measured today: Deployed 16.80 ms (snapshot_162; 17.18 in snapshot_160); FusedQuantised 9.68; post-#162 reference 9.98; FusedConfidence fp32 1.88 / bf16 1.21.

| lever | mechanism | expected effect at 32k |
|---|---|---|
| fused kernel (§E item 1) instead of Inductor's int8 read | one launch; phase A in registers; no 1 GiB margins; no 128-chunk split | 16.8 → ~FusedConfidence-class time; removes 13.9 ms of read + 2.5 ms of derivation kernels |
| **int8 rows** (48 B vs 192 B fp32 / 96 B bf16) | the gather is the bound (§C); fewer bytes per row | FusedConfidence fp32 → bf16 halves the row bytes and drops 1.877 → 1.214 ms (linear fit: ~0.55 ms fixed + 0.0069 ms per byte per row). int8 extrapolates to **~0.88 ms (n=2)** |
| **n=1** (one row per table instead of two) | halves the gather | extrapolates to **~0.72 ms** |
| integer accumulation (int32 shift-add) | exact and order-independent; no FMA | ~0 (not the bound), but it frees the arithmetic order entirely |
| no training bookkeeping | no straight-through weights, no fake-quant rebuild, no saved tensors | −0.04 ms (fake-quant) and the psw stores |
| fewer launches | 134 → 1 | ~0.2 ms (the 128 `cat`s) |
| quantisation scalar maths | as §G1 | +0 … +0.45 ms |

**Sum:**
- Deployed n=2 ≈ **0.9–1.3 ms**, n=1 ≈ **0.75–1.2 ms**.
- Against the parent's target of ~2.0 (FusedQuantised after §E item 1) and 9.68 today: **goal 2 vs the parent is reachable.**
- Against FusedConfidence fp32 (1.88): **reachable** on these numbers.
- Against FusedConfidence bf16 (1.21): borderline for n=2, likely for n=1.

**Assumptions, stated:** the extrapolation is linear in row bytes from two measured points (fp32, bf16). L2 sector granularity (32 B) means a 48 B row costs 2 sectors, not 1.5, which is already pessimistic in the fit's favour. The quantisation maths is assumed to be at the upper end of §G1.

### G3. The n=1 deployed variant: design, and the case against a separate class

**Measured accuracy** (champion geometry, 8,192 vectors, relative L2 output error; every model loaded with the same weights; "trained" = 200 Adam steps on a fixed synthetic regression target with the float n=2 model; at init the numbers are within ±0.07 of these):

| comparison (trained weights) | whole tensor | row median | row p99 |
|---|---|---|---|
| quant n=2 vs float n=2 (the cost of quantisation) | 0.223 | 0.223 | 0.349 |
| quant n=1 vs float n=1 | 0.207 | 0.204 | 0.317 |
| **quant n=1 vs quant n=2** | **0.642** | 0.679 | 1.094 |
| float n=1 vs float n=2 | 0.553 | 0.586 | 0.923 |

**Reading:**
- Swapping n=2 → n=1 on the same weights moves the output by ~55–64%. That is **far more than quantisation does (~21–22%)**.
- **So n=1 is a different model, not an approximation of a trained n=2 model:** a deployed n=1 cartridge is only meaningful for a model *trained* as n=1.
- The library already supports that: `QuantisedConfidenceLUT(read_top_n=1)` "stays constructible and trainable, just not exportable".
- The accuracy that matters is therefore model-level (bpb of an n=1-trained vs an n=2-trained model). That is a training experiment, not run in Stage 0; output-level numbers cannot answer it.

**Design proposal (argued against a separate class):**
- **Rule 5 forbids a user-facing *implementation* choice:** the same function computed different ways (`backend=`, a fallback, runtime negotiation).
- **n is not an implementation choice; it is part of the model definition,** like `weight="rational" | "softsign"`, which #160 deliberately made a constructor parameter of one class, and like `read_top_n`, already a parameter of ConfidenceLUT / QuantisedConfidenceLUT / FusedConfidenceLUT.
- **The deployed class does not even take it from the user:** `DeployedQuantisedConfidenceLUT` is built from an export payload whose `meta["read_top_n"]` it already stores (and today rejects unless 2).
- **So the Rule-5-consistent design is:**
  - make `QuantisedConfidenceLUT(read_top_n=1).to_deployment()` exportable;
  - have `DeployedQuantisedConfidenceLUT` run the n=1 read when the payload says so (one kernel template parameter, as FusedConfidence's `<N>`);
  - extend the Amendment D harness and the snapshot with Deployed rows for read_top_n 1 and 2.
- **A separate class would be the inconsistent choice:** it would make n a class axis on the deployed side while it is a parameter on every trainable class.
- **If Anatoly still prefers a separate class:** the name in the convention would be `DeployedQuantisedConfidenceLUT1` / `…Top1`, exported per Amendment E, with its own harness rows and snapshot rows. The kernel is the same template either way.

## A. Inventory

### FusedQuantisedConfidenceLUT (`cartridges/quantised_confidence.py`)

**Custom op: `lutorch_ex::p2_scalars`.** Python glue in `cartridges/_pow2_int8.py`; CUDA in `csrc/pow2_int8_read.cu` and `csrc/pow2_scalars.cuh`; built with `--fmad=false`.
- **Kernel:** `p2_scalar_kernel`, one launch, **one table per thread**, 1024 threads per block. Each thread runs `p2::table_scalars`:
  - over its nap margins: c1, the first argmin mj and mv, the left folds S and ls (sequential, in that thread);
  - then q, c_q, k', skip / drop, c2 and the shift byte.
- **Inputs:** the signed margins `d [..., T, nap]` fp32, which must be materialised; one-element tensors tau, g, beta, gamma; the ints lo, hi, Q.
- **Outputs:** `psw [..., 2]` fp32 (the straight-through cell weights 2^k', 2^(k'−q)), `cells [..., 3]` uint8 (c1, c2, shift codes), `kq [..., 2]` int8 (k', q).
- **Backward (registered autograd):** recomputes `_pow2.ste_cell_weights` in torch with the op's integers held fixed and calls `torch.autograd.grad` on it. This is autograd over torch ops, inside the op's backward.
- This is the only custom op on the quantised path.

**Eval** (`_forward_impl`, compiled by Inductor around the op):
- `_addresses`: the input gather, the margins `u`, address c, j*, |u_j*|, c_alt (torch, compiled);
- `p2_scalars(u, ...)` (CUDA);
- `_fake_quant_tables` → `_pow2.ste_tables`: per (head, channel) amax, quantise, scale (torch, compiled, **every call**; input-independent);
- `_blend_read`: global cell indices and psw concatenation (torch, compiled), then **`F.embedding_bag(per_sample_weights=…)`** (ATen kernel);
- `_route`: identity at the champion.

**Train** (`_quant_read_monolith`, compiled with `dynamic=None`): the same pieces flattened into one graph (margins → `p2_scalars` → `ste_tables` → offsets-based `embedding_bag`), so aot-autograd builds one backward. The backward is ATen's `embedding_bag` backward (per-sample-weight gradient plus the sort-based weight gradient), the op's autograd-recompute backward, and Inductor kernels for the STE / score gradient and `ste_tables`.

### DeployedQuantisedConfidenceLUT

Torch only, no extension, compiled by Inductor:
- `_addresses`;
- `_pow2.blend_exponents`, the torch definition of the integers (left folds), under `no_grad`;
- global indices;
- `_pow2.int_blend_read`: int8 rows gathered, widened to int32, shifted left by k'+6 / k'+6−q, summed over the 2·tph cells of each bag. It is called with the default `chunk_bags=4096` (the eager chunking) even under compile.

### Kernel launches (measured)

| cartridge | per eval forward (8k / 32k) | per train step (8k / 32k) |
|---|---|---|
| FusedQuantised fp32 | 12 / 12 | 49 / 49 |
| FusedConfidence fp32 | **1 / 1** | 9 / 10 |
| FusedConfidence bf16 | 4 / 4 | 17 / 18 |
| Deployed fp32 | 5 / **134** | — |

At 32k, 128 of Deployed's launches are `triton_poi_fused_cat_view_5`: the `chunk_bags=4096` chunking concatenating 128 chunks.

### FusedConfidenceLUT: what makes it fast (`csrc/fused_confidence.cu`, `cartridges/fused_confidence.py`)

**Structure:**
- **One forward kernel and one hand-written backward kernel.** Around them sit only tiny torch ops: zero-fill of the gradient buffers, the sum of the per-CTA scalar partials, gradient accumulation adds.
- **Blocking:** one CTA per (group g, `rows_per_cta` = 4 consecutive samples); 64 threads per CTA by default (`CudaKnobs`, the sweep winners on the RTX 5090 and H100).

**Phase A, thread per table:**
- The z row is staged once into shared memory (fp32; a bf16 input is upconverted while staged), and the group's int16 anchors are staged once per CTA.
- Each thread computes its table's margins from shared z, the MSB-first address, j*, the confidence score (sum |u|, sum logsigmoid) and the blend weight v, all **in registers**.
- The nap reduction is a **sequential, unrolled loop in one thread** (MAX_NAP = 16).
- The per-table results go to shared memory: `cs`, `cas` (flat rows) and `w1`, `w2`. **Nothing per table touches global memory.**

**Phase B, stripe × chunk:**
- `nchunk = d_out / VEC` threads cover one table row, and `nstripe = threads / nchunk` stripes split the tph tables (stripe s takes t = s, s+nstripe, …).
- Each thread accumulates `acc[VEC]` in registers with **vectorised read-only loads**: `__ldg` float4 for fp32 (VEC = 4), and 2/4/8/16-byte loads upconverted in registers for a bf16 table.
- FMAs into fp32 accumulators.

**Phase C:**
- The stripe partials go to shared memory and are summed **in a fixed stripe order**, so the forward is deterministic.
- One coalesced store of the output row, or a bf16 store with one rounding.

**Backward:**
- Recomputes phase A from z (nothing saved but the inputs).
- Phase B: the weight-gradient scatter with **atomics** (float4 vector atomics on sm_90+, a knob) and the per-table dot products W[c]·go via shared-memory atomics.
- Phase C: the score / blend / margin gradients scattered into the z row in shared memory (shared atomics), one store of grad z.
- The β / γ / τ gradients as per-CTA partials summed on the host side, so there are no contended global atomics.
- The table gradient is not bit-reproducible run to run; the forward is.

**Other properties:**
- Shared memory ~4.5 KB per CTA at the champion; small 64-thread CTAs. Occupancy was not measured: no Nsight Compute run in this stage.
- No caching hints beyond `__ldg`.
- The table (50 MB fp32 at the champion) stays resident in the 96 MB L2.

## B. Profile

**Totals** (`snapshot_162.json`, median ms / peak GiB):

| cartridge | dtype | vectors | eval | train fwd | train bwd | train total |
|---|---|---|---|---|---|---|
| FusedQuantised | fp32 | 8,192 | 2.521 / 0.415 | 2.507 / 0.352 | 9.884 / 0.946 | 12.390 / 0.946 |
| FusedQuantised | fp32 | 32,768 | 9.683 / 1.658 | 9.430 / 1.406 | 39.069 / 3.641 | 48.500 / 3.641 |
| FusedQuantised | bf16 | both | not a case the op covers | | | |
| FusedConfidence n=2 | fp32 | 8,192 | 0.473 / 0.023 | 0.538 / 0.023 | 1.923 / 0.094 | 2.459 / 0.094 |
| FusedConfidence n=2 | fp32 | 32,768 | 1.877 / 0.094 | 1.975 / 0.094 | 7.473 / 0.236 | 9.450 / 0.236 |
| FusedConfidence n=2 | bf16 | 8,192 | 0.314 / 0.012 | 0.385 / 0.012 | 1.770 / 0.094 | 2.152 / 0.094 |
| FusedConfidence n=2 | bf16 | 32,768 | 1.214 / 0.047 | 1.430 / 0.047 | 7.022 / 0.236 | 8.452 / 0.236 |
| Deployed | fp32 | 8,192 / 32,768 | 2.909 / 0.414 · 16.802 / 1.656 | — | — | — |
| Deployed | bf16 | both | refused (fp32 / fp64 only) | | | |

**FusedQuantised, 32,768 fp32 eval** (kernel time 9.59 ms, wall 9.63, non-kernel 0.5%, 12 launches):

| phase | kernels | ms |
|---|---|---|
| gather + blend | ATen `EmbeddingBag_updateOutputKernel_sum_mean` | **7.150** |
| margins + address | Inductor `triton_per_fused_…gather…min…sum_5` (margins `u`, c, j*, c_alt; writes the 1 GiB `u`) | 1.028 |
| exponent derivation | `p2_scalar_kernel` (reads `u`, writes psw / cells / kq) | 0.957 |
| read set-up | Inductor index / psw concatenation kernels (×3) | 0.411 |
| table fake-quant (input-independent) | `amax` ×2 + scalar exps | 0.043 |

8,192 fp32 eval (2.46 ms of kernels, 12 launches): `embedding_bag` 1.79, margins 0.25, p2 0.24, set-up 0.14, fake-quant 0.04.

**FusedQuantised, 32,768 fp32 train step** (kernel time 44.29 ms, non-kernel 0.2%, 49 launches):

| phase | ms |
|---|---|
| forward margins (Inductor) + `p2_scalar_kernel` + index set-up | 1.08 + 0.95 + 0.82 + 0.82 = 3.67 |
| forward gather (`embedding_bag` forward) | 7.21 |
| **backward: per-sample-weight gradient** (`_embedding_bag_per_sample_weights_backward_kernel`, dot(go, W[idx]) for all 2·T bag entries) | **20.29** |
| backward: table gradient, sort-based (`compute_grad_weight_bags` 4.72, radix sort 2.85 + 0.17, unique 0.24, `sum_and_scatter` 0.81, arange 0.48, segments 0.07) | 9.34 |
| backward: STE / score / blend / margin gradients (Inductor reductions and pointwise) | 1.42 + 1.38 + 0.26 ×3 ≈ 3.6 |
| rest (adds, fills, memsets, scans) | ≈ 0.4 |

**FusedConfidence:**
- **32,768 fp32:** eval = **1 kernel, 1.852 ms** (non-kernel 0.7%). Train = forward kernel 1.874 + backward kernel 6.947 + ≈ 0.08 of fills / adds / one reduce (10 launches).
- **bf16:** eval 1.197 ms kernel + three tiny copies; train backward kernel 6.56.

**Deployed, 32,768 fp32 eval** (kernel time 16.75 ms, non-kernel 0.4%, 134 launches):

| phase | ms |
|---|---|
| **int8 shift-add read**, two Inductor reduction kernels `…bitwise_left_shift…sum…_3` / `_4` | **7.834 + 6.110 = 13.94** |
| margins + address | 1.088 |
| exponent derivation (`blend_exponents` in Inductor) | 0.948 + 0.435 = 1.38 |
| 128 × `cat` of the `chunk_bags` chunks | 0.219 |
| output cast | 0.114 |

At 8,192: read 2.26 ms of 2.86 (5 launches).

## C. Roofline (RTX 5090: 1,792 GB/s DRAM; ~105 TFLOP/s fp32; 96 MB L2)

32,768 vectors: B·G·tph = T = 33.55M tables; 2·T = 67.1M cell rows read, each 48 fp32 = 192 B.

**FusedQuantised eval, per phase:**

| phase | DRAM bytes (approx.) | FLOPs | bound |
|---|---|---|---|
| margins kernel | x 0.10 GB in; u 1.07 GB + c / c_alt / j* ~0.4 GB out | ~0.5 G | **DRAM, on a materialised intermediate** (1.6 GB / 1.03 ms ≈ 1.5 TB/s, ~85% of peak) |
| p2_scalar_kernel | u 1.07 GB in; psw / cells / kq 0.44 GB out | ~0.7 G transcendental-heavy | DRAM (1.5 GB / 0.96 ms ≈ 1.6 TB/s) |
| read set-up | ~0.8 GB (indices, psw) | ~0 | DRAM |
| embedding_bag | idx + psw 0.54 GB, out 0.10 GB; **12.9 GB of row gathers from the L2-resident 50 MB table** | 6.4 G | **L2 / gather-latency** (12.9 GB / 7.15 ms = 1.8 TB/s effective) |

**FusedConfidence eval as the measured attainable point:**
- DRAM traffic ≈ 0.25 GB (x, out, the table once) → 0.13 TB/s, **~7.5% of DRAM peak**.
- ≈ 11 GFLOP (gather FMAs + score maths) in 1.85 ms → ~6 TFLOP/s, **~6% of fp32 peak**.
- The same 12.9 GB of row gathers at **7.0 TB/s effective from L2**.
- So it is bound by L2 gather throughput / latency, not DRAM and not FLOPs, and it reaches ~3.9× the gather throughput of `embedding_bag`.
- The RTX 5090's L2 bandwidth is not published and was not micro-benchmarked here, so "fraction of achievable L2 peak" is not stated.

**Floor for a perfectly fused FusedQuantised forward at 32,768:**
- **Absolute floor:** DRAM ≈ x in + out = 0.2 GB → **0.11 ms**. SFU for the exponent maths ≈ 0.7 G transcendentals at ~7 T/s (170 SMs × 16 / clk × ~2.6 GHz) → ~0.1 ms, overlappable.
- The gather traffic dominates: at FusedConfidence's demonstrated 7.0 TB/s, the 12.9 GB of L2 gathers cost **1.85 ms**.
- **Practical floor ≈ FusedConfidence's eval + the quantisation-specific scalar work ≈ 1.85 + ~0.15 ≈ 2.0 ms**, from 9.68 today.
- Beyond that, only a smaller gather would help: int8 rows (48 B instead of 192 B) would cut the L2 gather bytes 4×. That is what a deployed int8 kernel can exploit and the fake-quant fp32 table cannot.

## D. Questions

**1. Are the per-table margins (and hence q and k') input-independent? No, they are functions of the input x.**
- `ManifestoLUT._addresses` (`cartridges/manifesto_base.py`), used by FusedQuantised eval and by Deployed:
  ```python
  z = x[:, self.in_head, :]  # route each group to its input head -> [B, G, d_in]
  z_a = z.gather(2, idx_a).reshape(B, G, tph, nap)
  ...
  u = z_a - z_b
  ```
- The training monolith (`FusedQuantisedConfidenceLUT._quant_read_monolith`):
  ```python
  z = x[:, self.in_head, :]
  d = (z.gather(2, idx_a) - z.gather(2, idx_b)).view(B, G, tph, nap)  # signed margins
  ```
- Then `p2::table_scalars` computes q from mv = min |d| and k' from S = Σ|d| and ls = Σ logsigmoid(β|d|).
- So q and k' are per (token, group, table), from the input.
- **Only the table quantisation is input-independent:** the per-(head, channel) exponent e and W_hat (`_pow2.head_chan_exponents`, `quantise_tables`). Deployed already precomputes that (`packed`); FusedQuantised recomputes it every call (0.04 ms).
- **Deployment cannot precompute q / k'.**

**2. Is the `[B, 16, 64, nap]` margin tensor materialised in the fused path? Yes, in eval and in training.**
- It is the input `d` of the custom op `lutorch_ex::p2_scalars`, which reads it from global memory (`p2_scalar_kernel`: `table_scalars(Dm + bt * nap, ...)`). Inductor cannot fuse across a custom-op boundary, so the margins kernel writes it.
- In training it is also **saved for the backward** (`ctx.save_for_backward(d, ...)`).
- At 32,768 that is 1.07 GB. It accounts for most of the eval peak (1.658 GiB, vs 0.094 for FusedConfidence) and for the margins-kernel and p2-kernel DRAM time (~2 ms).

**3. Derivation vs gather / blend.**
- At 32,768 eval: derivation including margins, p2 and read set-up ≈ 2.4 ms (25%); gather / blend 7.15 ms (75%).
- **If the derivation cost nothing, 7.15 ms (+0.04) would remain:** only ~25% faster.
- The gather is the lever; the derivation becomes nearly free once it moves into registers inside the gather kernel, as in FusedConfidence.

**4. Why is Deployed eval (16.8 ms at 32k) slower than FusedQuantised (9.68) and the compiled reference (9.98)?**
- Measured: **13.94 ms of its 16.75 ms of kernels is the int8 shift-add read**, two Inductor reduction kernels (`…bitwise_left_shift…sum…_3` 7.83 ms and `_4` 6.11 ms).
- Its derivation (margins 1.09 + exponents 1.38 = 2.47 ms) costs the same as FusedQuantised's (2.40).
- So the whole difference is the read: Inductor's gather of 2·T int8 rows, widen, shift and reduce is 1.95× slower than `embedding_bag` (7.15) and 7.5× slower than FusedConfidence's whole kernel.
- In addition, Deployed calls `int_blend_read` with the eager default `chunk_bags=4096` under compile. That costs 128 extra `cat` launches (0.22 ms) and splits the read into chunks. The `int8_blend_read` docstring says "compiled callers pass chunk_bags=None"; Deployed does not.

**5. What does the backward do? A mix; no hand-written kernel.**
- **ATen `embedding_bag` backward:**
  - the per-sample-weight gradient, a generic kernel recomputing dot(go, W[idx]): 20.29 ms, 46% of the step;
  - the sort-based table gradient: radix sort, unique, segment sums, ≈ 9.3 ms.
- **The op's registered backward:** `torch.autograd.grad` over `_pow2.ste_cell_weights`, recomputed in torch and compiled by Inductor.
- **Inductor kernels** for the score / blend / margin gradients and `ste_tables`, ≈ 3.6 ms.
- **For comparison:** FusedConfidence's single hand-written backward kernel does the equivalent work, weight scatter + dot products + score / blend / margin / scalar gradients, in **6.95 ms**, vs ≈ 33.3 ms of backward kernels here.

## Key comparison: inherent vs implementation debt (32,768, fp32)

**Eval:** FusedQuantised 9.68 ms vs FusedConfidence 1.88 ms, a 7.8 ms gap.

- **Inherent to quantisation: ≈ 0.15 ms.**
  - The table fake-quant, 0.04 ms measured, input-independent.
  - The extra per-table scalar maths on top of what FusedConfidence already computes (FusedConfidence already does the margins, Σ|u|, Σ logsigmoid, argmin and blend weight): log2 S, two divisions, floor ×2, c_q = log2(1+2^-q), clamps, ldexp. That is ~5 SFU-class ops × 33.5M tables, ≈ 0.03 ms of SFU throughput; allowing for the instruction overhead, ≤ ~0.1 ms.
  - No extra table traffic: the gather reads the same two rows per table. Using int8 rows (Deployed) would *reduce* it.
- **Implementation debt: ≈ 7.6 ms.**
  - The generic `embedding_bag` gather, 7.15 ms vs the ~1.6 ms the FusedConfidence-style stripe gather needs for the same rows: ≈ 5.5 ms.
  - The materialised 1 GiB margins and the separate p2 / set-up kernels, ≈ 2.1 ms (a fused phase A costs ~0.25 ms inside FusedConfidence's kernel).
  - Not launches: non-kernel time is < 1%.
- **Target for the rewrite:** the quantised cartridge with its gather-and-blend done exactly as well as FusedConfidence's, plus the quantisation work, is **≈ 1.88 + 0.15 ≈ 2.0 ms eval**, with an eval peak ≈ FusedConfidence's 0.09 GiB instead of 1.66.

**Training:**
- Train total 48.5 vs 9.45 ms.
- The quantised backward adds only the straight-through ratio factor (b / (s·σ)) and the identity gradient to the fp32 master table on top of FusedConfidence's backward.
- **Target ≈ 9.5 + ~0.5 ≈ 10 ms** (≈ 4.8×), peak ≈ 0.24 GiB instead of 3.64.

## E. Optimisation plan, ranked by expected gain against the 2.0 ms / 10 ms targets

**Determinism rule:** anything feeding q or k' must be deterministic and reproducible in eager torch. The per-table derivation done by one thread, sequentially over nap, is exactly that. The output and gradient accumulations do not feed q or k'.

1. **One fused forward kernel, FusedConfidence-style, with the gather-and-blend in it.**
   - *Mechanism:*
     - one CTA per (group, rows), the z row staged in shared memory;
     - phase A, one thread per table, computes the margins, c / c', and **the p2 integers and the straight-through weights in registers** (the `p2::table_scalars` arithmetic);
     - phase B, the stripe × chunk vectorised `__ldg` float4 gather of the two fake-quant rows weighted 2^k' / 2^(k'−q);
     - phase C, a fixed-order stripe reduction.
   - *Gain:* eval 9.68 → ~2.0 ms at 32k (−7.7 ms); eval peak 1.66 → ~0.1 GiB; the training forward the same.
   - *Risk:* moderate. New kernel code. The fake-quant table must be built first: one tiny input-independent kernel or the existing `ste_tables`, 0.04 ms.
   - *Arithmetic order:*
     - The q / k' derivation can keep today's sequential per-table left fold, which `p2_scalar_kernel` already uses, so `_left_fold` need not change.
     - **The output's accumulation order changes**, from `embedding_bag`'s to stripe partials summed in fixed order. Layer 1 compares output rows bit for bit, so the reference's read must be **re-derived** to that order. A sum of power-of-two-weighted values can still round differently.
   - *Borrowed from FusedConfidence:* the structure, phase A, the stripe gather, vector loads. *New:* the p2 scalar maths in phase A.
2. **A hand-written backward kernel, FusedConfidence-style.**
   - *Mechanism:* recompute phase A from z (integers included). Phase B: the table-gradient scatter (atomics; it does not feed q / k') and the dot products W[c]·go. Phase C: the straight-through score / blend / margin gradients into z in shared memory; per-CTA scalar partials.
   - *Gain:* train backward 39.1 → ~8 ms; train total 48.5 → ~10 ms at 32k; train peak 3.64 → ~0.25 GiB (no saved margins, no sorted index buffers).
   - *Risk:* moderate to high. The straight-through gradient (the ratio b / e with its clamp; the gradient of skipped / dropped cells) must match `_pow2.ste_blend_weights` within the harness's fp32 tolerances. The table gradient becomes atomic, hence non-deterministic, as in FusedConfidence.
   - *Arithmetic order:* gradients only, so no q / k' constraint. The harness compares gradients with tolerances, unchanged.
   - *Borrowed:* the structure, the atomics, the per-CTA scalar partials. *New:* the straight-through ratio terms.
3. **Stop materialising the margins.** This comes for free with item 1; listed separately because it alone removes ≈ 2.1 ms and ~1 GiB. An intermediate step, if wanted: move the margin computation into `p2_scalar_kernel` (read x and the anchors directly), keeping `embedding_bag`.
   - *Arithmetic order:* unchanged (same per-thread fold). *Borrowed:* phase A addressing from z in shared memory.
4. **Deployed: use the item-1 kernel with int8 rows and shifts.**
   - *Mechanism:* gather the two int8 rows, widen in registers, shift-add into int32, store int32.
   - *Gain:* 16.8 → ~1.5–2 ms at 32k. The int8 rows are 48 B instead of 192 B, a 4× smaller gather, and integer accumulation is exactly order-independent.
   - *Risk:* low to moderate. *Arithmetic:* integer, so the order is irrelevant; the integers come from the same phase A.
   - *Borrowed:* the structure. *New:* the int8 / shift path.
   - *Cheap interim fix:* pass `chunk_bags=None` in Deployed's compiled forward. It removes the 128-chunk split and launches (≥ 0.22 ms); the effect on the reduction kernels is unmeasured.
5. **Cache the input-independent table fake-quant per training step** instead of per call (0.04 ms at 32k). Low gain, low priority. *Arithmetic:* unchanged. *New.*
6. **bf16 coverage** (not in the current op): FusedConfidence's bf16 upconvert-in-registers, the input staged as bf16. A capability rather than a speed item. *Borrowed.*

7. **The n=1 deployed read** (§G3): the same kernel with `<N=1>`, plus exporting `QuantisedConfidenceLUT(read_top_n=1)`. *Gain:* ~0.72 ms extrapolated at 32k, for n=1-trained models. *Risk:* low. *Arithmetic:* integer. *Borrowed:* FusedConfidence's `<N>` template.

**Order of work proposed for Stage 1+:**
1. The fused forward kernel (item 1), measured against §G1's bound, before anything else.
2. Deployed on the same kernel (items 4 and 7).
3. The backward (item 2).
4. Re-derive the reference's read order to the kernel's and keep Layer 1 bit-exact.

Not proposed: anything that makes the q / k' inputs depend on reduction order (tree reductions across threads for S or ls). The sequential per-thread fold is already the cheapest deterministic option, because nap = 8 values fit in one thread's registers.
