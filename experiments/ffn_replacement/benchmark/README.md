# FFN-slot inference benchmark harness

Portable rig for measuring whether an FFN-LUT experiment beats the vanilla dense
baseline on **inference wall-clock**. Built on gpustar's RTX 5090; nothing here is
5090-specific — paths come from arguments and the device is whatever CUDA reports.

## READ THIS FIRST: the warm-up is not optional

A GPU idles at a low SM clock and boosts under load. Timing that starts before the
clock has ramped measures the ramp, not the kernel.

On the 5090 (idle 1627 MHz) this artefact **flipped the sign of a headline result**:
`exp_n_0126` was reported as 1.06× vanilla (i.e. losing) when a warmed measurement
puts it at **0.92× (winning)**. Same code, same process, same interleaving — only the
clock state differed.

`bench.burn_in()` runs ≥60 iterations before any timing and is **on by default**.
`--no-warmup` exists only to demonstrate the artefact. Never quote a number produced
with it.

Two other measurement rules baked in:

- **Interleaved A/B in one process.** Every model is timed in alternating rounds
  inside a single process. Cross-process comparisons drift; a 7% baseline shift
  between two processes was observed here.
- **Correctness before timing.** The driver refuses to print timings if the optimized
  model is not bit-exact against the unpatched fp32 model. An optimization that
  changes the output is not a speedup.

  The one *deliberately* approximate path, `--gather-impl cuda-bf16`, does not get a
  weakened assertion — it gets **two** checks. First the same CUDA kernel with an
  **fp32** table must be bit-exact (proving the addressing and pipelining are right),
  then the bf16 rounding must be inside a relative tolerance **on the gather output**.
  Both must pass or nothing is timed.

## Files

| file | what it is |
|---|---|
| `model.py` | Rebuilds an experiment's model from its `config.json` (+ `checkpoint.pt` if asked). Handles both FFN families used by the paper: dense vanilla and CompressionMHL/anchor-pair. |
| `gather.py` | The fused Triton gather+sum kernel replacing `embedding_bag`, plus `patch()` for the FastMultiHeadLut family and `tune()` to re-sweep its config on new hardware. **The default and the portable fallback — needs no compiler.** |
| `gather_cuda.cu` | Optional hand-written CUDA gather+sum: software-pipelined, fp32 (bit-exact) or bf16 (approximate) table. Faster than Triton, but needs `nvcc`. |
| `gather_cuda.py` | Builds `gather_cuda.cu` on first use and wires it in, with `patch()`, `tune()` and `check_table_precision()`. **Never raises on a machine without a toolchain** — `available()` reports why and the driver falls back to Triton. |
| `gather_fused.cu` | **The fastest path on the 5090.** Routing and gather fused into one kernel with the index kept in shared, so the 48–96 MB HBM index write disappears. bf16 table, both routing regimes, row-1-ahead prefetch. |
| `gather_fused.py` | Wiring for the above: regime dispatch from the model, `patch()`, `tune()`, `check_table_precision()`, `supported()`, same never-raises fallback contract. |
| `gather_fused_v2_h100.cu`, `route_v2_h100.cu`, `route_shared_h100.cu`, `H100_OPT_NOTES.md` | nebius's H100 prototypes and sweep notes, which `gather_fused.cu` is adapted from. Kept as the H100 reference; not wired into `run_bench.py`. |
| `hybrid.py` | hybrid-v2 storage: dense weights **stored** bf16, LUT tables and (for FastMHL) LUT input kept fp32 so the native bit-pack kernel stays alive. |
| `bench.py` | `burn_in`, `timeit`, `check_bit_exact`, `slot_breakdown`, `interleaved_ab`, `report`. |
| `run_bench.py` | CLI driver tying it together. |

## Running it

```bash
cd experiments/ffn_replacement/benchmark

# the two CompressionMHL grid cells, against the untied vanilla baseline
python run_bench.py --exp exp_n_0126_grid_H4d48_nap7_tph64
python run_bench.py --exp exp_n_0127_grid_H4d48_nap7_tph128

# load the trained weights (only needed when the measurement depends on values)
python run_bench.py --exp exp_n_0128_grid_H4d48_nap8_tph64 --load-checkpoint

# explicit baseline / batches / rounds; re-tune the gather on new hardware
python run_bench.py --exp exp_n_0126_grid_H4d48_nap7_tph64 \
    --baseline exp_n_0135_untied_vanilla_baseline_16k \
    --batches 12,48,96 --rounds 11 --tune-gather

# the CUDA fast path (needs nvcc; falls back to Triton with a log line if absent)
python run_bench.py --exp exp_n_0126_grid_H4d48_nap7_tph64 --load-checkpoint \
    --gather-impl cuda-bf16 --tune-gather
```

`--gather-impl` picks the gather kernel and **defaults to `triton`**, the existing
behaviour, so nothing changes unless you ask:

| value | bit-exact? | needs a compiler? | notes |
|---|---|---|---|
| `triton` *(default)* | yes | no | the portable path |
| `cuda-fp32` | yes | yes | software pipelining alone, ~1.2–1.3× the Triton gather |
| `cuda-bf16` | **no** | yes | + bf16 table, costs **+0.0001 bpb** — see below |
| `cuda-fused` | **no** | yes | **fastest.** + routing fused in, index kept in shared. Same **+0.0001 bpb** (output is bit-identical to `cuda-bf16`) |

Both `cuda-*` values fall back to `triton` (with a `gather:` line saying so) when the
extension cannot build, or when the model is not a FastMultiHeadLut. Pass
`--require-gather-impl` to make that a hard failure instead, and `--gather-tol` to
change the bf16 tolerance (default `1e-2`; measured 9.4e-4 – 1.8e-3).

`--baseline` defaults to `exp_n_0135_untied_vanilla_baseline_16k` (the untied vanilla
dense-FFN reference, 1.20144 bpb). `--load-checkpoint` is only needed when the
*values* matter; pure timing does not need it, since a gather reads a row whatever is
in it.

Requires `lutorch_cuda` built (the native bit-pack kernel) and Triton. `nvcc` is
needed **only** for `--gather-impl cuda-*`; without it the harness still runs
everything on the Triton path.

## What to report back

1. **Correctness** — the `max|logit diff|` line per batch. Expect `0.000e+00`.
2. **FFN-slot breakdown** at batch 48 — shipped fp32, optimized, baseline slot, and
   the optimized-vs-baseline ratio.
3. **End-to-end ladder** at batches 12/48/96 — median and [min–max] per model, the
   vs-baseline ratio, and whether the intervals are disjoint or overlap.
4. Confirmation the warm-up ran (i.e. that `--no-warmup` was *not* used).

The `disjoint`/`overlap` flag matters: with 11 rounds the intervals often still
overlap, so treat a ratio as a good central estimate rather than a tight bound unless
it says `disjoint`.

## 5090 reference numbers, for comparison against the H100

All bit-exact (`max|logit diff| = 0.000e+00`), warmed, interleaved, seq 512, on the
default Triton path. For the faster approximate path see
[the CUDA gather fast path](#the-cuda-gather-fast-path---gather-impl-cuda-bf16):

| experiment | params | val bpb | slot @48 (optimized) | end-to-end vs vanilla @48 |
|---|---|---|---|---|
| `exp_n_0126` H4/d48 nap7 **tph64** | 39.04 M | 1.20694 | 0.278 ms | **0.92×** (faster) |
| `exp_n_0127` H4/d48 nap7 **tph128** | 48.48 M | 1.19471 | 0.477 ms | **1.04×** (slower) |
| `exp_n_0135` untied vanilla | 35.79 M | 1.20144 | 0.345 ms | 1.00× (the baseline) |

0126 also runs 0.90× at batch 96. The tph 64 → 128 step costs ≈0.12–0.14 on the
ratio, and the slot breakdown attributes essentially all of it to the gather: the
optimized slot grows 0.278 → 0.477 ms, and 6 slots × 0.199 ms ≈ the measured 1.0 ms
end-to-end gap. Dense parts are identical between the two.

What the optimization stack is worth on 0126: the shipped fp32 slot is 0.836 ms; the
fused Triton gather alone takes it to ~0.28 ms (5.2× on the gather stage, 0.703 →
0.135 ms), and hybrid-v2 storage supplies most of the end-to-end gain by keeping
autocast from re-casting dense weights every forward.

## The CUDA gather fast path (`--gather-impl cuda-bf16`)

Gather stage only, batch 48 × seq 512, trained checkpoints, warmed, 5090:

| model | Triton | cuda-fp32 | cuda-bf16 | bf16 vs cuda-fp32 |
|---|---|---|---|---|
| `exp_n_0126` tph64 | 0.1247 ms | 0.1064 ms | **0.0781 ms** | 1.36× |
| `exp_n_0127` tph128 | 0.3024 ms | 0.2621 ms | **0.2538 ms** | 1.03× |
| `exp_n_0128` tph64, 256 rows | 0.1777 ms | 0.1300 ms | **0.0826 ms** | 1.57× |

FFN slot at batch 48, and what it does to the vs-vanilla ratio:

| model | slot, Triton | slot, cuda-bf16 | vs vanilla slot |
|---|---|---|---|
| `exp_n_0126` | 0.2705 ms | **0.2454 ms** | 0.79× → **0.72×** |
| `exp_n_0127` | 0.4664 ms | **0.4137 ms** | 1.36× → **1.21×** |
| `exp_n_0128` | 0.3900 ms | **0.3170 ms** | 1.14× → **0.93×** |

`exp_n_0128` crosses under the vanilla slot for the first time on this path.

**Numerics cost, measured rather than argued: +0.00014 / +0.00011 / +0.00007 bpb**
(0126/0127/0128) — real `val_bpb` on the nanochat val set, fp32 tables reproducing the
recorded value exactly. Relative error on the gather output is 9.4e-4 – 1.8e-3.

> Do **not** gate this on a logit-level tolerance. Rounding the table moves full-model
> logits by ~1.5e-1 *relative* on random tokens — which looks disqualifying and is not:
> the same change costs +0.0001 bpb on real data. A logit check on random tokens
> nearly produced exactly the wrong verdict here. The tolerance belongs on the gather
> output, where the approximation is actually introduced, and the bpb number is what
> decides adoption.

### Why it works — 32-byte **sectors**, not 128-byte lines

Easy to state wrongly, so state it precisely. Every row pitch here is 32 B aligned, so
an fp32 48-wide row is exactly **6 sectors** and a bf16 row exactly **3** — packed or
padded. The bf16 win is halving sectors per gathered row, bounded by 2×.

The 128 B cache-line story is a different (and wrong) model, and it was tested rather
than assumed: padding the row pitch changes how many 128 B lines a row straddles while
holding useful bytes and load-instruction count fixed.

| variant | pitch | lines/row | 0126 | 0127 | 0128 |
|---|---|---|---|---|---|
| fp32 packed | 192 B | 2.0 | 0.1064 | 0.2621 | 0.1300 |
| fp32 padded | 256 B | 2.0 | 0.1064 | 0.2596 | 0.1359 |
| bf16 packed | 96 B | 1.5 | 0.0802 | 0.2615 | 0.0872 |
| bf16 padded | 128 B | 1.0 | **0.0781** | **0.2538** | **0.0826** |

A line-count model predicts the last row is 1.5× the one above it. Measured 1.03–1.06×
— **falsified**. The sector model predicts ≈1.0× for both padding steps and gets it
right. (The padding is kept anyway: it is free and worth 3–6%.)

The second ingredient is **software pipelining**, worth 1.19–1.30× over Triton on its
own (that is the `cuda-fp32` column). The row load's address comes from an in-loop
indirect load of the index, so index and row need *different* prefetch depths — index
2 ahead, row 1 ahead. Triton's `num_stages` drives one depth for the whole loop body
and cannot express that; sweeping it 1..5 moved these shapes by 0.0–0.3%.

### Why `exp_n_0127` gains almost nothing (1.03×)

Its bottleneck is not the table. At tph=128 its int64 index is
24576 × 512 × 8 B = **96 MB — exactly this GPU's L2**, while the bf16 table is 6 MB and
fully resident. It is index-bound, and halving table bytes cannot help an index-bound
kernel. This is a property of tph=128, not of the kernel.

## The fused routing+gather path (`--gather-impl cuda-fused`) — the best on this box

**All three models beat vanilla end-to-end on this path, and `exp_n_0127` crosses over
for the first time.** Adapted from nebius's H100 prototype (`gather_fused_v2_h100.cu`,
`H100_OPT_NOTES.md`) — see the porting notes below, because the 5090 needed three
changes.

The routing (anchor-compare → index) and the gather+sum become **one kernel**, with the
index held in **shared memory**. The 48–96 MB int64 index is therefore never written to
HBM and never read back.

Routing+gather stage, batch 48 × seq 512, trained checkpoints:

| model | native + Triton | native + `cuda-bf16` | **`cuda-fused`** | vs `cuda-bf16` |
|---|---|---|---|---|
| `exp_n_0126` | 0.1697 ms | 0.1213 ms | **0.0894 ms** | **1.357×** |
| `exp_n_0127` | 0.3686 ms | 0.3239 ms | **0.1718 ms** | **1.885×** |
| `exp_n_0128` | 0.2878 ms | 0.1958 ms | **0.0959 ms** | **2.042×** |

FFN slot and end-to-end vs vanilla (vanilla slot 0.338 ms, e2e 7.84 ms):

| model | slot `cuda-bf16` | **slot `cuda-fused`** | e2e `cuda-bf16` | **e2e `cuda-fused`** |
|---|---|---|---|---|
| `exp_n_0126` | 0.2452 (0.73×) | **0.1793 (0.53×)** | 0.93× | **0.88×** |
| `exp_n_0127` | 0.4156 (1.23×) | **0.2608 (0.77×)** | 1.05× | **0.93×** |
| `exp_n_0128` | 0.3171 (0.94×) | **0.1868 (0.55×)** | 0.98× | **0.88×** |

### The mechanism — the HBM index write, measured by control

`ncu` is unavailable here (`ERR_NVGPUCTRPERM`), so the claim is tested rather than
asserted: the kernel has a `spill` flag that runs the identical code and *additionally*
writes the index to HBM, isolating exactly the write the fusion removes.

| model | index size | fused | fused + spill | the write costs | effective |
|---|---|---|---|---|---|
| `exp_n_0126` | 48 MB | 0.0833 | 0.1247 | **+0.0414 ms** | 1.16 TB/s |
| `exp_n_0127` | **96 MB** | 0.1612 | 0.2482 | **+0.0870 ms** | 1.10 TB/s |
| `exp_n_0128` | 48 MB | 0.0918 | 0.1305 | **+0.0387 ms** | 1.24 TB/s |

0127's penalty is 2.10× and 2.25× the other two against an index exactly 2× the size, at
a consistent 1.1–1.2 TB/s — HBM write bandwidth. That is why 0127 gains most: its index
is 24576 × 512 × 8 B = **96 MB, exactly this GPU's L2**, and it was the one model bound
on it.

### Numerics: free relative to `cuda-bf16`

The fused **fp32** path is bit-exact vs native routing + the Triton gather (`0.000e+00`
at batches 1/12/48, all three models) — which also proves the *routing* it replaces is
bit-exact. The fused **bf16** output is **bit-identical (`0.000e+00`) to the non-fused
`cuda-bf16` gather** across every routing-regime × BLOCK_N config, so its accuracy cost
is exactly the already-measured bf16 cost: **+0.00014 / +0.00011 / +0.00007 bpb**. No
separate val run is needed, and none was done — the identity is the evidence.

### Regime dispatch: the routing winner flips, so it is chosen per model

Standalone routing vs the native kernel, all bit-exact:

| model | native | route_shared (v1) | route_v2 |
|---|---|---|---|
| `exp_n_0126` nap7 | 0.0434 | **0.0333 (1.30×)** | 0.0620 (0.70×) |
| `exp_n_0127` nap7 | 0.0888 | **0.0821 (1.08×)** | 0.1200 (0.74×) |
| `exp_n_0128` nap8 | 0.1130 | 0.0748 (1.51×) | **0.0635 (1.78×)** |

v2 (column-major z, token-inner, bank-conflict-free) wins the L1-bound `nap≥8` case and
**regresses 30%** at nap7, where v1 (row-major z, table-inner) wins. `gather_fused.py`
picks v1 below nap 8 and v2 at/above it, automatically from the model. Inside the fused
kernel the gap is far smaller (0126: 0.0894 v1 vs 0.0903 v2) because once the index write
is gone, routing is no longer what the stage is bound on. `--tune-gather` sweeps BLOCK_N
**and both regimes**, and says so if it disagrees with the dispatch rule.

**This also explains the stage-2 anomaly** recorded against the native routing kernel:
0128 (nap8) routed in 0.113 ms vs 0126 (nap7) 0.043 ms — 2.60× for +14% work. nebius's
ncu root-caused it on the H100: the native kernel runs one thread per (token, table) and
re-reads each token's z-columns through L1 *once per table*, and nap8 tips it over the L1
throughput cliff. Staging z in shared fixes it here too: best-kernel ratio is **1.91×**,
down from 2.60×. Reduced, **not eliminated** — nap8 still costs more than the +14% of
work it does, so something residual remains unexplained.

### Porting notes: three things the 5090 needed that the H100 version does not

1. **bf16 table** (fp32 accumulation). On the H100 bf16 *lost*, so nebius fused against
   the fp32/Triton gather; here bf16 wins, and it is the deciding change — fused **fp32**
   is only 0.81× / 1.05× / 1.16× vs `cuda-bf16`, i.e. on 0126 the fusion alone does not
   pay. This is why the H100 notes and this file recommend different combinations, and
   neither is wrong for its own box.
2. **Both routing regimes**, selectable, rather than v2 hardcoded (see above).
3. **uint8 index in shared** instead of int32 — `nap ≤ 8` packs into a byte exactly, which
   cuts the shared index footprint 4× and is what lets tph=128 (0127) run at a useful
   BLOCK_N. Guarded by `TORCH_CHECK` on `nap ≤ 8` and `table_dim ≤ 256`; a model outside
   those falls back to Triton with a message rather than crashing.

Also carried over the row-1-ahead prefetch from `gather_cuda.cu`: the index latency is
gone once it lives in shared, but the *row* latency is not.

For reference, the H100 numbers are in `H100_OPT_NOTES.md`. There, fused_v2 reaches
1.09× / 1.82× / 1.14× the vanilla slot (still losing to dense); here fused+bf16 reaches
0.53× / 0.77× / 0.55× (winning).

## Tune the gather before anything else

**This is the highest-value remaining lever, and it costs nothing.** Config tuning
moves this kernel *more than any layout change tried*: within a single layout, the
gather stage ranged **1.65×** across six `BLOCK_N × num_warps` settings on
`exp_n_0126` (0.1291 → 0.2134 ms) and 1.49× on `exp_n_0127`. No structural change
measured here came close to that.

The optimum will **not** transfer across architectures. Run with `--tune-gather`
rather than inheriting the 5090's `BLOCK_N=128 / num_warps=8`:

```bash
python run_bench.py --exp exp_n_0126_grid_H4d48_nap7_tph64 --tune-gather
```

`gather.tune()` sweeps BLOCK_N × warps × stages and reports the best config; the
driver then uses it for the rest of the run.

`--tune-gather` tunes whichever implementation `--gather-impl` selected. For the CUDA
path it sweeps `(BLOCK_N, threads)` via `gather_cuda.tune()`; the optimum moved when
pipelining was added, and it is **not** the same across models — on the 5090,
`exp_n_0126`/`exp_n_0128` want 256/512 while `exp_n_0127` wants 64/256. Both tuners
sweep against the **real** index: an earlier version tuned on a zeros index, so every
gather hit row 0, cache behaviour was unrepresentative, and it picked a config 23%
slower than the true best.

## Optimization levers tried and rejected (5090)

Three rounds of optimization on the CompressionMHL gather, all measured with the
harness above, all null or worse. Recorded so nobody re-treads them.

| lever | result | why |
|---|---|---|
| ~~**Lower-precision LUT tables** (bf16 / int8)~~ **— OVERTURNED, see below** | ~~null~~ **1.36–1.57× in a hand-written CUDA kernel** | In *Triton* bf16 was ~16% slower despite half the bytes, and that was recorded here four times as "the gather is not bound by table bytes". That conclusion was wrong: it was a property of **Triton's load path**, not of the problem. Reading the row as 16 B vector loads and converting on arrival makes bf16 the single largest win found — see [the CUDA gather fast path](#the-cuda-gather-fast-path---gather-impl-cuda-bf16). int8 was not retried. |
| **Vector-width / lane packing** — remove the 25% masked lanes from D=48 running at BD=64 | **null to −18%** | Neutral at tph64, 18% worse at tph128. The 25% is lane *occupancy*, not wasted work: a masked 64-wide load still issues **one** instruction and coalesces the same 192 contiguous bytes, while splitting into three 16-wide loads triples the instruction count. Wrong axis. |
| **Fusing the decompress Linear** into the gather epilogue | **−70%** (3.4× slower), and not bit-exact | The decompress is only 13–27% of the slot and already runs as a tuned cuBLAS GEMM. Fusing forces one program to own all heads (the decompress mixes them), collapsing the grid by H=4 and blowing up register pressure, and replaces the library GEMM with a hand-rolled `tl.dot`. Also inherently ~2.7× *less* accurate, because a matmul reassociates — bit-exactness and this fusion are in tension by construction. |

### Levers rejected in the CUDA kernel (a later round, same rigor)

| lever | result | why |
|---|---|---|
| **Deeper software pipeline** — index 3 ahead / row 2 ahead instead of 2/1 | **0.72–0.90×** (slower on every model, every config) | Bit-exact, just slower. 128 registers at 512 threads is the entire 64K register file, i.e. exactly one block per SM. The extra buffers buy latency hiding and pay for it in occupancy — a bad trade in a kernel with plenty of parallelism. Worst at the largest tile every time. |
| **Narrow (int32/uint8) gather indices** in the CUDA kernel | **1.02–1.12× at best, i.e. dead** | See the artefact note below — the isolated number is not real. Registers barely move (int64 128 → uint8 122), so there is no occupancy win either. Not worth a native uint8-emitting anchor kernel. |
| **Vectorized index prefetch** — 16 uint8 indices in one `uint4` | **0.79–0.85×** | Breaks the thing that made the kernel fast: the row-1-ahead prefetch has to cross a chunk boundary every 16 tables, where the next index comes from a *different* vector load, so the pipeline stalls once per chunk. Trading 16 cheap index loads for one vector load is a bad deal when index loads were never the bottleneck. |

> **A benchmarking artefact worth internalizing: never time the gather in isolation to
> judge an index-side change.** A loop over a fixed index tensor re-reads the whole
> thing every iteration. For `exp_n_0127` that is 96 MB of int64 against a 96 MB L2 —
> it thrashes exactly at the boundary, and uint8 indices then look **2.13×** faster.
> In a real forward pass the anchor kernel has just *written* that index, so it is
> still resident and the width stops mattering: measured **0.99×** in situ. Time
> `index kernel + gather` together for anything touching the index.

**The common thread, restated after the bf16 result overturned the old one:** the
gather is bound by **32 B sector traffic and instruction issue**. Halving *sectors*
per row (bf16 table) is the biggest win found. Reducing the *number of independent
accesses* still pays. What does not pay: shaving bytes that do not change the sector
count (see the padding table), narrowing an index that is already L2-resident, and
anything that buys latency hiding with occupancy.

## A recurring artefact that will fool a correctness check

**Without a checkpoint, `train.py` zero-initializes `decompress.weight`**, so the FFN
slot emits only the decompress bias and its output is completely independent of the
LUT table. Any logit-diff check then reads a false `0.000e+00` — patched and unpatched
agree because *neither* depends on what you changed.

This has produced a misleading pass three separate times here (a bf16-table accuracy
check, a table-precision sweep, and the fused-epilogue sweep). Precision and
correctness checks must use either a real checkpoint (`--load-checkpoint`) or a
non-trivial decompress weight. Timing is unaffected — a gather reads its row whatever
is in it — so only the *numeric* claims are at risk.

## Things already measured and ruled out

So the H100 side does not re-tread them. All on the 5090, against the dense bf16
addressing GEMM:

- Narrowing the gather index (int32/int16/uint8): no gain **in situ**. The reason
  originally recorded here — "the strided index read costs the same number of 32-byte
  sectors at any width" — is not right; a narrower index genuinely does move fewer
  sectors. It gains nothing because in a real forward pass the index has just been
  written by the anchor kernel and is still L2-resident, so its width is not on the
  critical path. Re-tested in the CUDA kernel where both sides are under our control:
  1.02–1.12× upper bound even assuming a free uint8-emitting producer.
- int8 tensor-core addressing: the GEMM is 2× faster but the stage is 24% slower (the
  int32 accumulator can't fuse) and 4.3× more sign flips.
- 2:4 structured sparsity: sparse tensor cores work on this GPU (1.98× on an 8192³
  GEMM) but lose 6.5× at K=384 — too small to be compute-bound. They start paying
  around K≈4096.
- Sparse/gather addressing in every form tried (CSR, embedding_bag, feature-major
  tiled SpMM, scatter+atomic, a hand-written CUDA masked-sum): 10× to 700× slower
  than the dense GEMM. Top-k truncation down to 2% density never crosses it.
