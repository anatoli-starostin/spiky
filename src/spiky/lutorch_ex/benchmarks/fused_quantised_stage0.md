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

## Stage 1: the fused forward kernel

`csrc/pow2_int8_read.cu` (the same extension, built with `--fmad=false`, gated by the same validated architectures) and `_pow2_int8.FusedQuantRead`. `FusedQuantisedConfidenceLUT`'s public surface is unchanged: no new constructor argument, no switch.

### Kernel structure

**Two launches per forward:**
1. `table_exponent_kernel`: the per-(group, channel) exponent e of the fake-quant table, `ceil(log2(max |W|)) − (bits − 1) + offset` (`_pow2.head_chan_exponents`).
   - It is input-independent and needs a reduction over the whole group's table before any row can be quantised, hence its own launch.
   - It is a max, so the result does not depend on the order. 0.03 ms.
2. `quant_fwd_kernel<VEC>`: everything else, structured as FusedConfidenceLUT's forward.
   - One CTA per (group, 4 consecutive samples), 64 threads (raised to cover one table row for wide d_out).
   - The z row and the group's int16 anchors are staged in shared memory; the 2^e / 2^−e scales go to shared memory per group.
   - **Phase A, one thread per table:** the margins `d_j = z[a_j] − z[b_j]` (one fp32 subtraction, as the reference). Then `p2::table_scalars`, **the same header function `p2_scalar_kernel` runs**: c1, the first argmin, the sequential left folds S and ls, q, c_q, k', skip / drop, c2. Then the cell weights `w1 = skip ? 0 : 2^k'` and `w2 = skip||drop ? 0 : 2^(k'−q)`, times the table-dropout factor when training. **All in registers; the margins never reach global memory.**
   - **Phase B, stripe × chunk:** `nchunk = d_out / VEC` threads per row, `nstripe = threads / nchunk` stripes. `__ldg` float4 loads of the fp32 master rows c1 and c2, fake-quantised in registers: `fq = clamp(floor(v · 2^−e + 0.5), −128, 127) · 2^e`, bit-identical to `ste_tables`' value because power-of-two scaling is exact.
   - **Phase C:** the stripe partials summed in a fixed order, one store.

**Launch count per forward:** 2, from 12.

### The exact reduction order (for Stage 2 to mirror)

The q / k' inputs, per table, in one thread:
- `S = ((m_0 + m_1) + m_2) + …` and `ls = ((l_0 + l_1) + …)` with `m_j = |d_j|` and `l_j = min(0, βm_j) − log1p(exp(−|βm_j|))`;
- q and k' exactly as `p2::table_scalars` (unchanged from `p2_scalar_kernel`, which the reference already mirrors in `_pow2.blend_exponents`, layer 1 bit-exact);
- **no cross-thread reduction feeds q or k'.**

The output, per (sample, group, output channel):
- `p_s = (((0 + w1[s]·fq(c1[s])) + w2[s]·fq(c2[s])) + w1[s+ns]·fq(c1[s+ns])) + …` for t = s, s+ns, … < tph.
- Then `y = ((0 + p_0) + p_1) + … + p_{ns−1}`.
- Here `ns = threads / (d_out / VEC)`, `VEC` = the widest of 4 / 2 / 1 dividing d_out, `threads = max(64, ⌈d_out / VEC⌉ rounded up to 32)`; both are kernel constants, not knobs.
- Champion: VEC 4, 12 chunks, **ns = 5**.
- Every product and add is a separate fp32 operation (no contraction).

### Layer 1 did NOT go red

**Measured: the fused output equals the eager reference bit for bit** (layer 1, seeds 0 and 1, eval and the training forward; all 12 quantised harness cases `torch.equal`).

**Reason: the quantised read is exact in fp32 at these sizes, so the summation order cannot change it.**
- Each term is `2^k' · W_hat · 2^e` or `2^(k'−q) · W_hat · 2^e`, with |W_hat| ≤ 127, k' ∈ [lo, hi] = [−3, 4] and q ≤ Q = 3 (else the cell is dropped).
- So every term is an integer multiple of 2^(e−6) of magnitude at most 127 · 2^(e+4), and a bag of 2·tph = 128 terms sums to at most 128 · 127 · 2^10 = 16,646,144 units of 2^(e−6), **under 2^24**.
- Every partial sum is therefore exactly representable in fp32 and no addition rounds: any order gives the same bits.

**Consequences:**
- **Stage 2's reference re-derivation is not needed for the no-dropout value at tph ≤ 64.**
- Exactness ends when 2·tph · 127 · 2^(hi − lo + Q) ≥ 2^24, i.e. tph > 64 at the p2_int8 preset. For such geometries Stage 2 must mirror the order above.
- With table dropout the factor 1/(1 − rate) is not a power of two, so products round. That case is checked by the layer-1-with-dropout test (fp32 tolerance + ROW_SAME), which passes.
- So Layer 1 is green and keeps zero tolerance. **I recommend Stage 2 be re-scoped** to the cases the exactness argument does not cover (tph > 64, or a preset with wider windows), plus documenting the argument in the reference. It is not needed to make Layer 1 green.

### Measured (RTX 5090, bench_snapshot's own `measure()`, fresh process per cell, two runs each; ms / GiB)

| cell | Stage 0 (snapshot_162) | Stage 1 | FusedConfidence n=2 | vs FusedConfidence |
|---|---|---|---|---|
| 32k eval | 9.683 / 1.658 | **1.940–1.946 / 0.094** | 1.866 / 0.094 | **+4%** (+0.08 ms) |
| 8k eval | 2.521 / 0.415 | **0.526 / 0.023** | 0.473 / 0.023 | **+11%** (+0.05 ms) |
| 32k train total | 48.500 / 3.641 | 50.363–50.407 / 3.734 | 9.318–9.411 / 0.236 | (Stage 4) |
| 8k train total | 12.390 / 0.946 | 12.902 / 0.969 | 2.412–2.473 / 0.094 | (Stage 4) |

- **Goal-1 bound** (≤ +24%, the Stage 0 upper bound on the quantisation arithmetic): **met**. At 32k the residual is +4% by wall time (+8% by kernel time incl. the exponent kernel: 2.005 vs 1.852 ms); at 8k +11%.
- **Effective gather throughput:** 12.9 GB of row reads in 1.977 ms of `quant_fwd_kernel` at 32k = **6.5 TB/s** (FusedConfidence: 7.0 TB/s); 8k likewise 6.5 TB/s.
- **Eval peak:** 1.658 → 0.094 GiB at 32k (FusedConfidence's), because the margins are no longer materialised.
- **Training:** +4% time and +0.09 GiB at 32k against Stage 0. The backward, unchanged until Stage 4, now first recomputes the forward in torch: `FusedQuantRead.backward` runs the torch.compiled `_read_torch` (margins, the p2_scalars op, ste_tables, embedding_bag) and differentiates it. An eager recompute measured 85.4 ms / 7.5 GiB; compiling it brought that back to 50.4 / 3.73.

### How Stage 4's backward gets the margins without materialising them

As FusedConfidenceLUT's backward does:
- The forward saves only z (the routed input, `[B, G, d_in]`), the parameters and the dropout factors.
- The backward kernel, one CTA per (group, rows), re-stages the z row and anchors in shared memory and **recomputes phase A**: the margins in registers / shared memory for the gradient terms (as FusedConfidence's `us` scratch, per CTA, `tph × nap` floats), the integers, the weights.
- Nothing `[B, G, tph, nap]`-sized is stored in global memory at any point.

## Stages 2 + 4: the exactness argument in the reference, and the hand-written backward

### Stage 2 (folded in, documentation + coverage)

- **The exactness argument now lives in the reference**, in `QuantisedConfidenceLUT._blend_read`'s docstring.
  - Every term is W_hat·2^e·2^k' or W_hat·2^e·2^(k'−q), with |W_hat| ≤ 127, k' ∈ [−3, 4] and q ≤ 3. So every term is an integer multiple of 2^(e−6).
  - A bag of 2·tph = 128 terms stays within 128·127·2^10 = 16,646,144 < 2^24, so no partial sum rounds.
  - It names when the argument fails: tph > 64; a wider read window or larger Q; table dropout (1/(1−rate) rounds).
  - `_left_fold` and every reduction order are unchanged.
- **New coverage:** the existing Layer 1 test (`test_quantised_layer1_fused_bit_exact_vs_eager_reference`) is parametrised over `LAYER1_GEOMS = {champion, tph128}`. tph128 is the champion with tph = 128, i.e. 16.8M tables at 8,192 vectors, past the exactness bound.
  - Zero tolerance, unchanged assertions.
  - **Measured: bit-exact** for seeds 0 and 1, eval and training forward, and the integers agree on every table.
  - So no separate tolerance assertion is proposed. The bound is a worst case (every term at |W_hat| = 127 and the top shift, all the same sign), which these weights do not reach. Bit-equality at tph > 64 is measured, not guaranteed. If a future geometry breaks it, Layer 1 will say so.

### Stage 4: `quant_bwd_kernel` (csrc/pow2_int8_read.cu)

**One launch**, structured as FusedConfidenceLUT's backward. One CTA per (group, 4 samples), the same thread count and vector width as the forward.

- **Saved by the forward:** z `[B, G, d_in]`, the fp32 master table, the exponents E `[G, d_out]`, the three log scalars, the dropout factors `[B, G, tph]` (when training with dropout), and the anchors. The anchors are stored as int16, plus a per-group input-dim → anchor-entry CSR map (int32, input-independent, built once in `__init__`).
- **Nothing of size `[B, G, tph, nap]` is ever in global memory.** The margins `us` and their gradients `gd` are per-CTA shared memory, `tph × nap` floats each.
- **Phase A (thread per table):** re-stage z, recompute the margins (kept in shared memory), `p2::table_scalars`, the forward weights w1 / w2 (× dropout), and the straight-through numerators b1 = 2^k', b2 = 2^(k'−q). These use the clamped k', also for skipped and dropped cells, as `_pow2.ste_blend_weights`.
- **Phase B (stripe × chunk, float4 `__ldg`):**
  - `gW[c] += w·go` via fp32 atomics (float4 vector atomics on sm_90+).
  - The per-chunk partial dot products `go · fq(W[c])`, each written to its own shared slot `rp[chunk][t]`.
- **Phase C (thread per table):**
  - `r_c = Σ_chunk rp` in chunk order.
  - The straight-through gradient: `G_c = r_c · mask · b_c / max(e_c, 1e-30)`, with e_1 = s·σ(x), e_2 = s·σ(−x), x = 2mv/τ. Then g_s = G_1σ(x) + G_2σ(−x) and g_x = s(G_1 − G_2)σ(x)σ(−x).
  - The margin gradients `gd[t][j] = sign(u_j)·(g_s·E + g_s·s·γ·β·σ(−βm_j) [+ g_x·2/τ if j is the first argmin])`.
  - This thread's share of the β / γ / τ gradients.
- **Phase D (thread per input dim):** `gz[i]` sums the gd of the anchors that read i, from the CSR map. **No atomics.**

**Exact reduction order of every gradient. Determinism: nothing feeding q or k' uses atomics.**

| gradient | order | deterministic |
|---|---|---|
| grad x (z) | per (sample, group, input dim i): `((0 ± gd[e_0]) ± gd[e_1]) ± …` over the anchor entries reading i in increasing entry index (a-anchors in (t, j) order, then b-anchors; + for a, − for b); then `x[:, in_head]`'s autograd scatter when h_in ≠ n_groups | yes, bit-repeatable |
| per-table dot r_c | `((0 + rp_0) + rp_1) + … + rp_{nchunk−1}`, each `rp = ((0 + fq_0·go_0) + fq_1·go_1) + …` over the VEC lanes | yes |
| log β, log γ, log τ | per thread: tables t = tid, tid + blockDim, … within each of the CTA's 4 rows in order; warp xor-shuffle butterfly (16, 8, 4, 2, 1); warps 0..nw−1 summed in order by thread 0; per-CTA partials `[grid, 3]` summed by `torch.sum(0)` (a fixed-shape reduction) | yes, bit-repeatable |
| grad W | fp32 `atomicAdd` (float4 vector atomics on sm_90+): order **not** deterministic. Allowed (it feeds no q / k' within the step) | no (measured: not bit-repeatable) |

### Gradient correctness

Throwaway script: fused vs the eager reference in fp32, and vs the reference in fp64. Same weights and input, B = 512, seeds 0 / 1. Values are relative norms; the eager32 ~ fp64 column is the reference's own fp32 error, for scale.

| geometry | gradient | fused ~ eager fp32 | fused ~ fp64 | eager fp32 ~ fp64 |
|---|---|---|---|---|
| champion | grad x | 1.9e-7 | 1.5e-7 | 1.5e-7 |
| champion | grad W | 5.3e-8 | 5.0e-8 | 4.9e-8 |
| champion | log β | 2.5e-7 – 4.7e-7 | 5e-9 – 8e-9 | 2.4e-7 – 4.6e-7 |
| champion | log γ | 1.2e-7 – 3.4e-7 | 2.0e-7 – 3.9e-7 | 5e-8 – 3.2e-7 |
| champion | log τ | 9e-8 – 2.3e-6 | 1.4e-7 – 1.8e-6 | 4e-8 – 4.7e-7 |
| tph128 | grad x / W | 2.1e-7 / 4.4e-8 | 1.7e-7 / 5.0e-8 | 1.7e-7 / 4.9e-8 |
| tph128 | β / γ / τ | ≤ 1.4e-6 | ≤ 1.6e-6 | ≤ 2.3e-7 |
| h4 tph20 nap5 d13→6 | grad x / W | 1.8e-7 / 1.4e-7 | 1.3e-7 / 1.3e-7 | 1.5e-7 / 7.5e-8 |
| h4 tph20 nap5 d13→6 | log β (seed 0) | 7.2e-5 | 1.8e-5 | 5.4e-5 |

- The fused gradients are as close to fp64 as the eager fp32 reference is.
- The single 7e-5 outlier (the 6-output geometry's β, seed 0) is the sum's cancellation: the eager fp32 reference is 5.4e-5 from fp64 there too.
- **Bit-identical to the torch path: none**, except log γ at the small geometry for seed 1 (by chance). The kernel sums in a different order from autograd, so bit identity is not expected. The forward value stays bit-identical (Layer 1).
- The suite's own gradient checks pass: the Amendment D harness's fp32 checks against the fp64 truth for the scalars, layer 1 with table dropout (fp32 tolerances, ROW_SAME), test_table_dropout_and_cell_tv (a dropped table gets zero grad). The surrogate / gradcheck tests in test_quantised_confidence.py run on the reference class (fp64), not on the fused kernel, which is fp32 only.

### Measured (RTX 5090, `bench_snapshot.measure`, fresh process per cell, two runs; ms / GiB)

| cell | Stage 1 | **Stage 4** | FusedConfidence n=2 | vs FusedConfidence |
|---|---|---|---|---|
| 32k train total | 50.4 / 3.734 | **9.92–10.02 / 0.236** | 9.35–9.39 / 0.236 | **+6%** (+0.6 ms) |
| 32k train backward | — | 7.84–7.96 | 7.37–7.42 | +7% |
| 8k train total | 12.90 / 0.969 | **2.64–2.71 / 0.094** | 2.43 / 0.094 | **+10%** (+0.24 ms) |
| 32k eval | 1.94 / 0.094 | 1.94–1.99 / 0.094 | 1.87 / 0.094 | +4% (unchanged) |
| 8k eval | 0.526 / 0.023 | 0.526–0.528 / 0.023 | 0.474 / 0.023 | +11% (unchanged) |

- **Against the targets:** 32k **met** (~10 ms target; 10.0 measured, 5.0× faster than Stage 1). 8k **missed**: 2.67 against ~2.4, +10% over FusedConfidence.
- **Peak memory equals FusedConfidence's** at both sizes.
- **Launches per train step: 52 → 10 at 8k, 11 at 32k** (FusedConfidence: 9 / 10). The one extra launch is the table-exponent kernel.
- **Profile at 32k train (kernel time 9.81 ms):** `quant_bwd_kernel` 7.71, `quant_fwd_kernel` 1.94, `table_exponent_kernel` 0.07, then the optimizer-free bench's adds and fills.
- **Where the residual is:** the backward is +0.4–0.6 ms over FusedConfidence's (bench backward medians). Likely causes (not profiled to the instruction level): the fake-quant of every loaded element for the dot products, the extra phase-C transcendentals of the straight-through ratios (σ, the logsigmoid derivative, the divides), and phase D's serial CSR walk.

### Tests

- **Full GPU suite: 1575 passed / 0 failed / 0 skipped** (main: 1573; the 2 new cases are the tph128 Layer 1 parametrisations).
- **CPU suite: 417 passed / 1010 skipped** (main: 417 / 1008; the 2 new GPU-only cases skip on CPU).
- No assertion changed, no tolerance added or widened; Layer 1 stays zero-tolerance.

## Stage 3: DeployedQuantisedConfidenceLUT on the kernel template, and read_top_n = 1

### Part A: `deploy_fwd_kernel<VEC, N>` (csrc/pow2_int8_read.cu)

The Stage 1 template, specialised for the stored int8 rows and for the read width N. **One launch per forward** (from 134 at 32k / 5 at 8k), no backward.
- **Structure:** one CTA per (group, 4 samples), 64 threads. The z row and the group's int16 anchors are staged in shared memory.
- **Phase A (thread per table):** the margins (z[a] − z[b], or z[a] for single anchors), the address c1 (bits d_p > cmp_eps), then `p2::table_scalars` (N = 2) or the new `p2::single_scalars` (N = 1). These give each cell's left shift, k' + 6 and k' + 6 − q, or "not read".
- **Phase B (stripe × chunk):** 8 int8 bytes per load (`int2` `__ldg`: the widest of 8 / 4 / 2 / 1 dividing d_out), widened to int32 in registers and shifted. A cell that is not read (skipped / dropped) is not loaded at all.
- **Phase C:** the stripe partials summed in order. Then ONE int32 → fp32 conversion (round to nearest, as torch's `.to(float32)`): the raw accumulator in units of 2^−6.
- **Exactness and order:** int32 accumulation is exact (|sum| ≤ 2·tph·127·2^10 < 2^31), so the value is order-independent.
- **Bit-identical to the torch definition** (the class's CPU path, `_forward_impl`) at the champion scale for n ∈ {1, 2} × {pairs, single}. This is a new zero-tolerance test, `test_deployed_kernel_bit_exact_vs_torch_definition`. It is also bit-identical to the trained cartridge's own `forward_int` once the folded scale is applied.
- **NOTHING of size `[B, G, tph, nap]`** is in global memory: margins and integers are in registers, shifts and cells in shared memory.
- **N is a template parameter** (two instantiations per vector width). The inner loop has no runtime branch on the read width.
- **Tuning (measured, a scratch ablation outside the repo, all variants bit-identical):**

  | variant (32k eval, ms) | n=2 | n=1 |
  |---|---|---|
  | VEC 16 (first version) | 1.50 | 1.38 |
  | **VEC 8 (kept)** | **1.21** | **0.87** |
  | VEC 4 | 1.46 | 0.91 |
  | VEC 16, 32 threads | 1.53 | 1.10 |
  | VEC 8, 96 threads | 1.27 | 0.98 |
  | VEC 8, 2 / 8 rows per CTA | 1.38 / 1.31 | 0.93 / 0.93 |
  | VEC 16, phase B removed (no row loads) | 1.05 | 1.03 |
  | VEC 16, phase-A maths replaced by constants | 1.32 | 1.22 |

- **Where the time goes:** phase B (the row loads) is the minority. Phase A plus the per-row staging, syncs and stripe sum dominate. The per-table maths alone is ~0.18 ms. Overlapping the nap transcendental chains in `p2::table_scalars` (an unrolled rewrite) measured no gain and was reverted.

**Measured** (RTX 5090, `bench_snapshot.measure`, fresh process per cell, two runs; ms / GiB, eval forward):

| cell | Deployed Stage 0 | **Deployed n=2** | **Deployed n=1** | FusedQuantised n=2 (parent) | FusedConfidence n=2 fp32 | FusedConfidence n=2 bf16 |
|---|---|---|---|---|---|---|
| 32k | 16.802 / 1.656 | **1.252–1.256 / 0.094** | **0.899 / 0.094** | 1.956–2.014 / 0.094 | 1.869–1.873 / 0.094 | 1.206–1.218 / 0.047 |
| 8k | 2.909 / 0.414 | **0.311 / 0.023** | **0.223 / 0.023** | 0.526 / 0.023 | 0.473–0.474 / 0.023 | 0.315 / 0.012 |

- **Deployed n=2 against each reference:**
  - **Stage 0:** 13.4× faster at 32k, 9.4× at 8k.
  - **The fused quantised parent:** beaten, −37% at 32k, −41% at 8k (**goal 2 met**).
  - **FusedConfidence fp32:** beaten, −33% at 32k, −34% at 8k (the "ideally" part of goal 2 met).
  - **FusedConfidence bf16:** **not beaten at 32k**, +3–4% (1.25 vs 1.21; inside the ±6% noise floor, but consistent in both runs). At 8k it is level (0.311 vs 0.315). Its eval peak is twice bf16's (fp32 input and output).
- **The 0.9–1.3 ms target at 32k is met** (1.25).
- **Deployed n=1** beats every cell above, including bf16: 0.899 ms at 32k, 0.223 at 8k.
- **Effective gather bandwidth** (int8 row bytes actually read ÷ kernel time; 32k):
  - n=2: 1.94 cells per table read (the rest are dropped second cells), 3.13 GB in 1.250 ms = **2.50 TB/s**.
  - n=1: 1.61 GB in 0.899 ms = **1.79 TB/s**.
  - The fp32 kernels move 6.5–7.0 TB/s. The int8 table (12.6 MB) is L2-resident, and this kernel is bound by phase A and the per-row structure rather than by the gather. So the row bytes being 4× smaller buys ~1.6× (n=2), not 4×.

### Part B: read_top_n = 1

- **Not a separate class.** `QuantisedConfidenceLUT(read_top_n=1)` is now exportable (`to_deployment` no longer refuses it). The export records `meta["read_top_n"]`, and `DeployedQuantisedConfidenceLUT` takes the read width from there, never from the user. No switch, no negotiation, no fallback.
- **An n = 1 export carries no `log_read_tau`:** there is no blend temperature.
- **The n = 1 integers have one definition in kernel order.** `_pow2.single_exponent` (torch) and `p2::single_scalars` (CUDA) compute k' = floor(log2 S + (g + γ·ls)/ln2 + 1/2) with S and ls left-folded and a true fp32 division by ln2, as `blend_exponents` does for n = 2.
  - **This changes the reference's n=1 rounding expression.** Previously it was `round_half_up(log2(m.sum()) + γ·ls.sum()/LN2)`: torch's reduction order and a reciprocal multiply. A value within an ulp of a rounding boundary can now land on the other side. The n = 2 reference made the same move in #162.
- **`forward_int` is now a real integer read for n = 1** (one row per table shifted by k' + 6, `_pow2.int_single_read`). Before, it returned the fake-quant eval itself.
- **Coverage (test_fakequant_eval_matches_int8_read, extended; n ∈ {1, 2} × {pairs, single} × {cpu, cuda}):**
  - First, the existing check (fake-quant eval ≈ int8 read, unchanged).
  - Then 20 Adam steps, export → `from_deployment` (the registry's rebuilder) → `DeployedQuantisedConfidenceLUT`.
  - Asserted: the deployed raw accumulator × the folded scale 2^(e−6) equals the trained cartridge's `forward_int` **bit for bit** and its eval fake-quant read within the existing 1e-6.
  - test_deploy's file-level export → load round trip gained n=1 cases (pairs and single anchors).
- **No accuracy or bpb claim is made for n = 1.** It is a different model (quant n=1 vs n=2 relative L2 0.64) and is only meaningful trained as n=1; that run is out of scope.

### Deviations to note

- **The deployed class has one device split.** On CUDA it runs the kernel: fp32 input only, nap ≤ 8, d_in ≤ 32767. Anything else raises naming the case, and a missing extension raises the classified NativeUnavailableError (no fallback). On CPU it runs the torch definition (fp32 / fp64), which was its only path before.
  - This is decided by the input's device, not by a user or a runtime negotiation.
  - The two compute the same integers, and the CUDA one is pinned to the torch one bit for bit.
  - **Change in behaviour:** fp64 input on CUDA, accepted before (computed in fp64), now raises.
  - **Change in behaviour:** a CUDA device outside VALIDATED_ARCHES (or without nvcc) now raises instead of compiling the torch read.
- **`cmp_eps`:** the deployed kernel honours it in the address bits. FusedQuantisedConfidenceLUT's forward kernel (Stage 1) does not: it addresses with d > 0 regardless of `cmp_eps`. That gives a different read for a non-zero `cmp_eps` (default 0, which every test and the champion use). Not changed here; flagged.
- **`bench_snapshot`:** the deployed row is now built per read width ("DeployedQuantisedConfidenceLUT" = n=2, the earlier name, plus a new "DeployedQuantisedConfidenceLUT n=1" row). Inference-only cells are recognised by class. bench_compare's row order gained the n=1 row.

### Tests

- **Full GPU suite: 1584 passed / 0 failed / 0 skipped.** Stage 2+4 state: 1575. The +9:
  - 4 `test_deployed_kernel_bit_exact_vs_torch_definition`;
  - 2 deployed without-extension cases and 1 deployed fp64-on-CUDA case in test_extension_unavailable;
  - 2 n=1 cases in test_deploy's round trip.
- **CPU suite: 419 passed / 1017 skipped** (from 417 / 1010). The 2 test_deploy n=1 cases run on CPU; the 7 GPU-only new tests skip there.
- No assertion weakened, no tolerance added or widened. The extended round-trip test adds an exact (`torch.equal`) assertion.

## Investigation: bf16 for the quantised confidence cartridge (no code change)

The question: is bf16 genuinely impossible here, or is the effect small enough to accept? It was asked first for bf16 input activations, then redirected to the design worth having: **bf16 STORED TABLES with fp32 activations and fp32 exponent arithmetic**.

**Methodology (scripts outside the repo):**
- Champion geometry (h16 tph64 nap8 d48), 8,192 vectors, seeds 0 and 1.
- Trained weights, as the G3 n=1 study: 200 Adam steps of the float ConfidenceLUT n=2 on the fixed synthetic regression target. Every model takes those weights.
- Outputs are compared by relative L2 over the whole tensor, the row-wise median / p99, and the single worst row (row = one sample × head).
- Every quantised output below is the read with the cartridge's own integers. The fp32 one equals the cartridge's eval output bit for bit (checked).

### 1. The current scope of the refusal, and where it came from

- **`QuantisedConfidenceLUT` refuses both bf16 / fp16 parameters and bf16 / fp16 input activations.**
  - It sets `_ACCEPTS_LOW_PRECISION = False`, so `ReferenceForward.forward` raises `TypeError("... does not support low precision ...")`.
  - That happens when the input dtype is bf16 / fp16 OR any parameter or buffer is.
- **`FusedQuantisedConfidenceLUT` refuses both, as an implementation limit:** `_require` accepts only an fp32 input and an fp32 table. fp64 is refused too.
- **`DeployedQuantisedConfidenceLUT` refuses bf16 / fp16 input** (`TypeError`, "fp32/fp64 inference only"). On CUDA it takes fp32 input only. It has no table parameter: its rows are int8.
- **`test_rejects_low_precision` asserts, for read_top_n ∈ {1, 2} and dtype ∈ {bf16, fp16}:** a `QuantisedConfidenceLUT(...).to(dtype)` (every parameter, the table included, in low precision) called on an input of that dtype raises `TypeError` matching "does not support low precision".
  - It covers only the reference class, and only the all-low-precision case. A bf16 table with an fp32 input is not tested separately; today it raises through the same check.
  - To allow bf16 tables, the class flag would change, and this test would narrow to the cases still refused (see §4).
- **No technical reason was ever recorded.** The history:
  - The class was added in 5f0e5d8c with "fp32/fp64 only; bf16 raises via the base". At that time every non-fused cartridge raised on low precision. That rule came from 128b1f6c, where Gen-2 bf16 gave no speedup through embedding_bag: a performance reason about a different read, not a numerical one.
  - When #160 Amendment C (016a3a84) gave the reference classes bf16 support, the quantised class was explicitly deferred: "QuantisedConfidenceLUT keeps rejecting low precision (outside the spec's families; the quantised-cartridges work decides)". ARCHITECTURE.md Consequence C and the README say the same.
  - The exponent-flip argument appears nowhere in code, docs or commits. **It is an inherited default plus a deferral, not a reasoned decision.**
- **Amendment C already describes the legal design.** ReferenceForward runs a bf16 table / input as fp32 maths on fp32 views and rounds the output once to the input dtype.
  - For the quantised class that means: the bf16 table upconverted once; the margins, folds, logsigmoid, division by ln2, round_half_up and the exponent e all in fp32; the output in the input's dtype.
  - With an fp32 input, a bf16 table therefore gives exactly the "bf16-rounded table, fp32 arithmetic" read measured below.
  - Enabling it in the reference class is the flag alone. The fused class needs kernel support (§4). Deployed is unaffected.

### 2. What table values feed (confirmed from the code)

- **q, k', skip, drop and the cells c / c_alt depend only on the margins**: z (the activations) and the int16 anchors, plus β, γ, τ (`p2::table_scalars`, `_pow2.blend_exponents`). **The table W never enters them.** Checked: the integers are identical with an fp32 and a bf16-rounded table.
- **The table-derived quantities are:**
  - the exponent `e[g, c] = ceil(log2 max|W[g, ·, ·, c]|) − 7` (`table_exponent_kernel` / `head_chan_exponents`), per (group, channel), not per sample;
  - and the int8 values `W_hat = clamp(round_half_up(W / 2^e), −128, 127)`.
- **So the original objection (round_half_up on q / k' with 8 mantissa bits) does not apply to bf16 tables.** It applies only to bf16 activations or bf16 exponent arithmetic (§5).

### 3. bf16 tables: the numbers (fp32 activations, fp32 exponent arithmetic; both seeds agree to 3 digits)

**A. e flips: none.**
- **0 of 768 (group, channel) exponents change**, in both seeds.
- **The mechanism:** e changes only if bf16 rounding moves max|W| across a power of two. bf16 can round a value at most 2^−9 relative onto 2^n from above. So a flip needs max|W| within 2^−9 (0.2%) above a power of two.
- **Measured distance of the trained max|W| above the power of two below it:** every channel lies between 1.6% and 25% above. Fewer than 4% lie within 6.25%, none within 1.6%.
  - That is a property of these weights (max of 16,384 entries of a std-0.5 table ≈ 2.0–2.5), not a guarantee.
  - Under a uniform log-distribution the expected flip rate would be ~0.3% of channels.
- **A flip is benign in kind:** a static per-channel change of the fake-quant step by ×2 (W_hat of that channel re-quantised on a grid twice as coarse or fine). It is not a per-sample discontinuity, and the export would carry the same e.

**B. But bf16 storage is NOT nearly free here: double rounding into the int8 grid.**
- **7.3% of the int8 W_hat entries change, by exactly ±1** (both seeds), with e unchanged. The cause:
  - W / 2^e lies in [−128, 128) and bf16 keeps 8 significant bits. For |W_hat| ∈ [64, 128) the bf16 grid is 0.5 W_hat units, in [32, 64) it is 0.25.
  - A value in (k + 0.25, k + 0.5) rounds in bf16 to k + 0.5, and round_half_up then takes it to k + 1 instead of k.
  - bf16 and int8 have about the same resolution, so rounding twice moves a fixed fraction of entries. **The hypothesis that the quantised cartridge is where bf16 storage is least harmful is refuted:** it is the place where bf16's grid collides with the int8 grid.
- **Output** (seed 0 / seed 1):

| comparison | rel. L2 | row median | row p99 | worst row |
|---|---|---|---|---|
| **FusedQuantised bf16 table vs fp32 table** | **0.0203 / 0.0203** | 0.0208 | 0.0287 / 0.0289 | 0.037 / 0.039 |
| − with e forced to the fp32 value (rounding inside the fake-quant only) | 0.0203 / 0.0203 | 0.0208 | 0.029 | 0.037 / 0.039 |
| − e flips only | 0 | 0 | 0 | 0 |
| yardstick: quantisation itself, FusedQuantised fp32 vs FusedConfidence fp32 | 0.2227 / 0.2228 | 0.223 | 0.350 | 0.53 / 0.56 |
| yardstick: FusedConfidence bf16 table vs fp32 table (fp32 input) | 0.0017 / 0.0017 | 0.0017 | 0.0024 | 0.0031 |

- **Reading:**
  - The whole bf16-table divergence is the in-grid rounding, 0.020 relative L2. That is **12× what bf16 table storage costs FusedConfidenceLUT (0.0017)**, but **1/11 of what quantisation itself costs (0.223)**: in quadrature it moves the total distance from the float model by ~0.001.
  - It is smooth: no row moves by more than 0.039, and no row by more than 0.1. A bf16 table is simply a different, equally valid int8 table: 7% of entries one step away.
  - The quantised reproduction of the G3 figure is confirmed: 0.2227 / 0.2228, 0.223 previously.

**C. Speed and memory.**
- **Measured:** an ablation of the real kernels with the phase-B table gathers reading half the bytes (bf16 rows of the same shape, upconverted in registers); scratch builds, outside the repo.

| kernel (ms) | fp32 table (now) | bf16-width reads | all table loads removed (floor) |
|---|---|---|---|
| quant_fwd 32k | 1.956 | 1.651 (**−16%**) | 0.921 |
| quant_bwd 32k | 7.875 | 7.154 (**−9%**) | 6.505 |
| quant_fwd 8k | 0.494 | 0.467 (−5%) | 0.254 |
| quant_bwd 8k | 1.829 | 1.942 (+6%, noise or the narrower loads) | 1.627 |

- **Implied trainable class gain:**
  - 32k: eval ~1.94 → ~1.65 ms (still above FusedConfidence bf16's 1.21) and train total ~10.0 → ~9.0 ms (−10%).
  - 8k: about nothing.
- **Memory:** the champion table is 50 MB per layer in fp32, 25 MB in bf16, gradient likewise.
  - That saving exists only if the table PARAMETER itself is bf16, as FusedConfidenceLUT's bf16 table is: no fp32 master.
  - If an fp32 master is kept for the optimiser, a bf16 copy ADDS 25 MB, and the design is only a bandwidth win inside the kernels.
  - A pure-bf16 parameter has its own unmeasured risk. Adam updates smaller than half a bf16 ulp vanish, and for |W_hat| ≥ 64 that is 0.25 int8 steps. So small updates can no longer accumulate across a W_hat boundary.
  - The fp32 master is what makes slow accumulation work today.
- **DeployedQuantisedConfidenceLUT: no upside.** Its rows are already int8; there is no float table to halve. bf16 tables matter only at training time (the export quantises whatever table the model holds, so it stays consistent). Activations are fp32 by the premise.
- **Is it worth the work?**
  - **Trainable class: marginal.** ~16% eval / ~10% train at 32k, nothing at 8k. It does not reach FusedConfidence bf16. The work is a bf16 path in three kernels plus the exponents, plus the training-dynamics question above.
  - **Deployed: no.**

### 4. What a bf16-table design would need

- **Reference:** `_ACCEPTS_LOW_PRECISION` → accept a bf16 table, keep refusing low-precision input. This needs a narrower check than the shared flag (a class-level rule "table may be bf16, input must not"). Or accept both under Amendment C and keep input refusal as a documented choice.
- **Fused:** `quant_fwd` / `quant_bwd` / `table_exponents` read bf16 rows (as FusedConfidence's bf16 kernels do); gW in the table's dtype; `_require` admits a bf16 table.
- **Export:** unchanged. It quantises the model's table, bf16 or not.
- **Tests:**
  - a bf16-table harness row (fused vs eager reference bit for bit, as Layer 1);
  - an Amendment-C exactness test (bf16 table == fp32 run on the bf16-rounded table, output rounded once);
  - and a narrowed `test_rejects_low_precision`.
- **Proposed narrowing, NOT applied:** keep the existing test for bf16 / fp16 INPUT (input of that dtype, table of that dtype or fp32: still `TypeError`). Add a fp16-table case (still refused). Replace the "bf16 table refused" expectation by a positive test: `QuantisedConfidenceLUT(...)` with `weights` in bf16 and an fp32 input runs, and equals the fp32 run on `weights.bfloat16().float()` bit for bit. This narrows an existing assertion, so it is Anatoly's decision.

### 5. bf16 input activations (the first question, kept short): not unsafe, but measurably more fragile than elsewhere, and pointless

The fp32 maths on a bf16-rounded input (Amendment C for the input), against the fp32 input:

| | flips per table | rel. L2 | row p99 | worst row | rows moved > 0.1 |
|---|---|---|---|---|---|
| quant: bf16 input, fp32 exponent maths | q 0.36%, k' 0.36% (all ±1), cells c 0.36%, c_alt 0.76% | 0.048 | 0.19–0.20 | 0.68–0.90 | 6% |
| − of which address changes only | − | 0.022 | 0.11 | 0.34–0.45 | 1.2% |
| − of which exponent changes only | − | 0.043 | 0.19 | 0.68–0.89 | 4.5% |
| quant, Amendment C exactly (bf16 input + bf16 params, fp32 maths, bf16 out) | as above | 0.052–0.053 | 0.20 | 0.69–0.92 | 6.4–6.7% |
| quant, exponent arithmetic itself in bf16 (variant c) | q 0.60%, k' 1.5% | 0.089–0.090 | 0.29 | 0.77–0.92 | 23% |
| yardstick: FusedConfidence fp32 on bf16-rounded input | − | 0.022 | 0.10 | 0.33 | 1.1% |
| yardstick: FusedConfidence bf16 (all) vs fp32 | − | 0.022 | 0.10 | 0.33 | 1.1% |

- **The prediction ("flips routine") is half right.**
  - Per table, flips are rare: 0.36% of exponents. That is the same rate as the address-bit flips every bf16 cartridge already has. The pre-rounding values sit uniformly with respect to the boundaries (2% within 0.01); bf16 input moves them by 0.0025 (median) to 0.012–0.021 (p99).
  - But a row reads 64 tables, so ~59% of rows hold at least one changed table.
  - An exponent flip doubles or halves a table's weight. So the exponent part costs **2× FusedConfidence's whole bf16 cost** (0.043 vs 0.022), with a heavier tail (worst row 0.9 vs 0.33; 4.5% vs 1.1% of rows moved > 0.1).
  - The address part alone reproduces FusedConfidence's bf16 cost exactly (0.022).
- **Doing the exponent arithmetic in bf16 (variant c) is clearly worse:**
  - k' flips rise 4× (1.5%) and the output divergence to 0.09.
  - 0.6% / 1.3% of the bf16 pre-rounding values land exactly on a rounding boundary (bf16's coarse grid).
  - So any bf16 support must keep the exponent maths in fp32. That is Amendment C's rule anyway.
- **Verdict on bf16 input: not unsafe, pointless.** Its cost (0.05) is a quarter of quantisation's own and smooth on average, though heavier-tailed than elsewhere. The residual stream is fp32 (Anatoly's premise), so there is nothing to gain from accepting it. Note that a bf16 activation would not even be protected by the refusal: the same flips happen if a caller upcasts a bf16 tensor to fp32 before the call.

### 6. Verdict

- **bf16 TABLE storage (fp32 activations, fp32 exponent arithmetic) is numerically acceptable.**
  - The deciding number: **0.020 relative L2 (worst row 0.039)** against **0.223 for quantisation itself**, with **no e flips** in the trained weights and no per-sample discontinuity.
  - It is **not** "nearly free": it is 12× FusedConfidence's bf16-table cost (0.0017), from double rounding into the int8 grid. A training run would still have to confirm it at model level (bpb), since 7% of int8 entries move.
- **The upside is small:**
  - ~16% eval / ~10% train at 32k, ~0 at 8k, for the trainable class;
  - 25 MB per layer only with a pure-bf16 table parameter (which carries the small-update risk);
  - nothing for Deployed.
- **Recommendation: follow-up, not this PR.** #164's goal state is met in fp32. bf16 tables would buy at most ~1.7 ms per 32k step, at the cost of three kernel variants, a narrowed test and a training-dynamics question. If pursued, do it as its own PR with a bpb run, in this order:
  1. the reference flag plus the narrowed test;
  2. the fused bf16 table reads.
- **bf16 input activations:** not unsafe, and not worth having. The exponent arithmetic must stay fp32 in any case (bf16 arithmetic: 0.09).
