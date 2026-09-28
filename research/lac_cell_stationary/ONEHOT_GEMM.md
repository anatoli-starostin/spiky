# One-hot GEMM at the study shape — does the old H100 "14–22× slower" survive D = 1024?

**Verdict: it was not a D=48 artifact. Widening the output makes the ratio worse, not
better, and the strongest possible formulation — a single matmul running at 106% of this
GPU's measured dense peak — is still 9.2× slower than the AS gather. The question is
closed on GPU.**

Shape throughout: `T=256` tables, `K=256` rows/table, `D` swept, `N=24,576` tokens, int8
tables, **real cells** from the `exp_n_0196` forward (1.3272 cells actually fetched per
table). Reference is AS `read_cells`, `load16=False`, `block_n=16`. RTX 5090.
ncu is unavailable here (`RmProfilingAdminOnly=1`); traffic is from launch geometry.

## Step 0 — the roofline, computed before any kernel

| quantity | value |
|---|---|
| useful work `N*T*1.3272*D` | 8.5504e9 MAC-equivalent lane updates |
| full one-hot MACs `N*T*K*D` | 1.6493e12 |
| full one-hot FLOPs `2*N*T*K*D` | 3.2985e12 |
| **FLOP-waste multiplier `K/1.3272`** | **192.9×** |
| compacted GEMM waste (`gemm_scoping.py`) | 26.3× (Bn=32) … 103.8× (Bn=256) |

Measured dense throughput on this GPU (`torch.matmul`, square, not a spec sheet):
**bf16 238.9 TFLOP/s, fp16 233.7, int8 250.7 TOP/s**. Note int8 is only 1.05× bf16 here,
not the usual 2× — the int8 tensor-core path is not a lever on this part.

**Break-even, from measured numbers on both sides.** AS sustains
`8.5504e9 / 1.4149 ms = 6.043e12` useful MAC/s. The tensor cores do `238.9/2 = 1.1945e14`
MAC/s. So a GEMM formulation wins only if its waste multiplier is **below 19.8×**.

- full one-hot at 192.9× → **9.8× over break-even**
- best compacted at 26.3× → **1.33× over break-even**

This tightens the earlier `gemm_scoping.py` estimate, which put break-even at ~25–30 from
a rougher scalar-rate figure and therefore called the compacted case "at best break-even".
With both sides measured, **even perfect compaction loses**, though only narrowly.

## Step 1/3 — measured, D sweep

Per-table chunked `bmm` (16 tables per chunk), build and GEMM timed separately:

| D | AS ms | build ms | GEMM ms | total ms | **GEMM/AS** | max abs | max rel |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 48 | 0.2704 | 4.186 | 2.601 | 6.787 | **25.1×** | 693 | 1.8e-3 |
| 128 | 0.2778 | 4.198 | 5.379 | 9.576 | **34.5×** | 856 | 2.2e-3 |
| 256 | 0.4432 | 4.201 | 11.989 | 16.190 | **36.5×** | 856 | 2.2e-3 |
| 512 | 0.8265 | 4.241 | 24.447 | 28.689 | **34.7×** | 856 | 2.1e-3 |
| 1024 | 1.4242 | 4.242 | 49.998 | 54.240 | **38.0×** | 926 | 2.2e-3 |

**The ratio is flat to slightly rising in D.** At D=48 we measure 25.1× on the 5090, in
line with the H100's 14–22× at the same width; at D=1024 it is *worse*.

The build cost is **flat at ~4.2 ms** (it does not depend on D), so the two objections in
the old H100 note apply in different regimes: at D=48 the **build dominates** (4.19 vs 2.60
ms), at D=1024 the **FLOP waste dominates** (4.24 vs 50.0 ms). There is no D at which both
are small.

### The strongest form: one matmul, `[N, T*K] @ [T*K, D]`

Flattening the table axis into the reduction dimension gives `K_eff = 65,536`, the shape
cuBLAS wants:

| | ms | note |
|---|---:|---|
| AS `read_cells` | 1.4149 | |
| selection build | 5.176 | 24,576 × 65,536 bf16 = **3.22 GB** |
| **GEMM** | **12.997** | **253.8 TFLOP/s = 106% of measured square peak** |
| total | 18.172 | **12.8× slower than AS** (GEMM alone **9.2×**) |
| peak memory | 3.95 GB | |
| numerics | max abs 2648, max rel 6.3e-3 | |

The GEMM is at the machine's limit — there is **no implementation slack left to recover**.
It matches the roofline prediction of 9.7× almost exactly, which is the cleanest possible
confirmation that the loss is arithmetic, not engineering.

## Step 2 — correctness, and a prediction of mine that was wrong

I expected bf16 might be **bit-exact** here: both operands are exactly representable (int8
values ≤127 fit bf16's 8-bit mantissa, and 2^sh is a power of two), and cuBLAS is documented
to accumulate bf16 in fp32, which is exact for integers below 2^24 — our largest accumulator
is 1,054,180. **That was wrong.** Measured max abs error is 693–2648 (max rel 1.8e-3 to
6.3e-3), consistent with bf16-precision accumulation somewhere in the chain, not fp32. So
the usual "a matmul reassociates" objection stands, and by measurement rather than by
assumption.

**fp16 produces NaN at every D.** Products reach `1024 × 127 = 130,048`, past fp16's 65,504
maximum, so the intermediate overflows to inf. fp16 cannot express this computation at all.

**The int8 tensor-core path cannot carry the coefficient.** The weight is `2^sh` with sh up
to 10, i.e. up to 1024, which does not fit int8; and the shift is per (token, table, cell),
so it cannot be factored out of the GEMM. A correct int8 form needs one 0/1 GEMM per shift
plane — 8× the FLOPs. Measured as a *speed-only* upper bound with a plain 0/1 selection
(computing the wrong value): **70.16 ms, i.e. 49× slower than AS** — slower than bf16, so
int8 is not a rescue even before correctness.

## The N × D grid — and one-hot DOES win, in a corner, for a reason that disqualifies it

At N = 1 and 16 the selection matrix is 256 or 4,096 values per table rather than 3.22 GB,
so the build objection evaporates — and AS is badly under-occupied (one block of 1024
threads on 170 SMs). Measured (`flat` = the single flattened matmul; ratio > 1 means the
GEMM is **faster**):

| N | D | AS ms | flat build | flat GEMM | flat total | **flat/AS** | TFLOP/s |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 48 | 0.0888 | 0.0393 | 0.0162 | 0.0556 | **1.60× faster** | 0.4 |
| 1 | 128 | 0.0864 | 0.0425 | 0.0165 | 0.0590 | **1.46×** | 1.0 |
| 1 | 256 | 0.0863 | 0.0413 | 0.0215 | 0.0628 | **1.37×** | 1.6 |
| 1 | 512 | 0.0886 | 0.0410 | 0.0251 | 0.0661 | **1.34×** | 2.7 |
| 1 | 1024 | 0.0838 | 0.0406 | 0.0763 | 0.1169 | 0.72× | 1.8 |
| 16 | 48 | 0.1227 | 0.0376 | 0.0163 | 0.0539 | **2.28× faster** | 6.2 |
| 16 | 128 | 0.1132 | 0.0400 | 0.0190 | 0.0590 | **1.92×** | 14.1 |
| 16 | 256 | 0.1089 | 0.0393 | 0.0254 | 0.0647 | **1.68×** | 21.1 |
| 16 | 512 | 0.1050 | 0.0382 | 0.0598 | 0.0980 | **1.07×** | 17.9 |
| 16 | 1024 | 0.1331 | 0.0398 | 0.0940 | 0.1337 | 1.00× | 22.9 |
| 24576 | 48 | 0.2683 | 5.1702 | 1.9540 | 7.1242 | 0.038× | 79.1 |
| 24576 | 1024 | 1.4218 | 5.2068 | 13.1469 | 18.3537 | 0.078× | 250.9 |

**Eight cells where one-hot GEMM is faster, best 2.28× at N=16, D=48.** But look at the
TFLOP/s column: **0.4 to 22.9, i.e. 0.2–10% of the 238.9 peak.** This is not a tensor-core
win. At small N both paths are launch- and latency-bound, and the GEMM wins because it is
*one cuBLAS launch* against a custom kernel running at 1/170th occupancy. AS at N=1 is
**flat in D** (0.0888 / 0.0864 / 0.0863 / 0.0886 / 0.0838 ms) — it is doing no measurable
work at all, just paying launch and occupancy.

The win dies at D=1024 because **the GEMM must read the entire table whatever N is**:
`T*K*D` bf16 is 6.29 MB at D=48 but 134 MB at D=1024, against AS's ~340 KB per token. That
cost is amortised over 24,576 tokens at the top of the ladder and over *one* token at the
bottom, which is exactly why the ratio inverts along the D axis at small N.

**And the corner is already taken.** Split-K at N=1, D=1024 runs in **0.0103 ms**
(`artifacts/splitk_sweep.json`) — **11.4× faster than the best one-hot cell at that width**,
and 5.4× faster than the fastest GEMM cell anywhere in the grid. So the small-N advantage
is real, explicable, and irrelevant: it is an argument against AS's small-batch occupancy,
not for one-hot GEMM, and we already have a better answer to that.

## Answers

1. **Was the 14–22× a D=48 artifact? No.** Reproduced at 25.1× at D=48 on the 5090, and
   the ratio *worsens* to 38× at D=1024 for the same implementation.
2. **At D=1024 the actual ratio is 12.8× slower** for the best formulation (9.2× counting
   only the GEMM), and 38× for the per-table chunked form.
3. **No shape wins.** Break-even needs waste < 19.8×; one-hot is 192.9× and the best
   compacted variant is 26.3×. Since waste is `K/1.3272` and K is fixed by `nap`, the only
   way under break-even would be `K ≲ 26` — i.e. `nap ≲ 4.7`, a 16-row table, far below the
   256-row tables we train.
4. **For TPU.** The two GPU-specific objections do move in the GEMM's favour there: the
   gather is expensive on TPU and the MXU is cheap, and SparseCore's 32-bit-only DMA makes
   an int8 gather awkward. But the 192.9× waste is hardware-independent arithmetic, and the
   3.22 GB selection matrix is worse on TPU, where our 64 MiB table set will **not** be
   VMEM-resident the way it is L2-resident here. For one-hot GEMM to win on TPU, the MXU
   would have to beat TPU's gather path by more than 192.9× — against a machine whose
   SparseCore exists precisely to make gathers fast. **It is unlikely, and this measurement
   does not support trying it.** The compacted variant at 26.3× is the only version with a
   plausible TPU story, and it needs the gather:MXU ratio there to be ~30× more favourable
   than here.
