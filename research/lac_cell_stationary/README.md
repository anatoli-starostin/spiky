# LUT gather dataflows on an RTX 5090 — five arms against the shipped accumulator-stationary read

This directory is one study with one question: **is there a better dataflow than the
output-stationary gather the repo already ships, for reading `n_t` lookup tables per token
and summing the selected rows?** Five arms were written, gated bit-exact against the
shipped kernel, and timed. The shipped one wins at batch, and one new arm wins decisively
at small batch.

Everything here is **inference only**, int8 or fp32 tables, integer or fp32 accumulation.
Nothing was retrained. Nothing left the machine.

## The framing: AS and TS

The two dataflow families, in the vocabulary of the LAC design note:

- **AS — accumulator-stationary.** A thread (or a small lane group) owns an output
  accumulator and walks the tables, fetching one or two rows per table. Reads are
  scattered; writes are private. This is a *prefill*/sequential-sweep shape.
- **TS — table-stationary.** A thread owns table **cells**, holds their values in
  registers for the life of the kernel, and lets the whole batch stream past; each thread
  compares every token's index against the cells it owns. Reads are register-local; the
  cost moves into a compare per (thread, token) and a cross-thread reduction. This is a
  *decode*/random-read shape.

**Today's shipped LightMHL read is AS with a token tile of 1.** Each token's output is
built by one lane group that sweeps all 256 tables; nothing is stationary across tokens.
That is the baseline every arm here is measured against, and it is worth saying plainly
because the interesting design space is precisely the shapes that make something *else*
stationary.

## The headline table

Shape throughout: `T = 256` tables, `K = 256` rows/table, `D = 1024` lanes, `N = 24,576`
tokens, int8 tables, **real indices** from an `exp_n_0196` forward (1.3272 cells actually
fetched per table). RTX 5090, 170 SMs, 96 MB L2.

| arm | best configuration | ms at N = 24,576 | vs AS |
|---|---|---:|---:|
| **AS `read_cells`** (shipped kernel) | `block_n=16`, `UPR=64`, `load16=False` | **1.4187 – 1.4445** | **1.0×** |
| **split-K over tables** | `S = 32–64` | — | **1.6–8.1× faster, but only for B ≤ 1024** |
| scatter, shared accumulate (arm B) | `Bt=4`, `L=4` | 5.116 | 3.6× slower |
| one-hot GEMM | single flattened matmul | 18.17 | 12.8× slower |
| scatter, global atomics (arm A) | `L=1` | 43.23 | 30× slower |
| **TS (cell-stationary)** | `G=2`, `L=64` | 85.6 | **60× slower** |

Same table for AS with `load16=True` (two 8-byte loads per row instead of one 16-byte
load): 1.9985 ms — i.e. the vectorised load is worth 1.4×, and is on by default for the
wide rows added here.

## Arm by arm

### AS — the baseline, and why it is hard to beat

AS sustains 8.55e9 useful lane updates in 1.4187 ms. Counting **request volume** (bytes
the kernel asks for, not bytes DRAM delivers) that is **5,919–6,027 GB/s against an HBM
peak of 1,792 GB/s — 3.3× peak.** The only way to exceed HBM peak is to be served from
cache, so this is the measurement that says the **64 MiB int8 table set is L2-resident**
in the 96 MB L2. The table read is nearly free; the premise that motivates making tables
stationary does not hold on this part.

Table-set sizes matter here and are worth recording: at `T=256, K=256, D=1024` the tables
are **64 MiB in int8** and **256 MiB in fp32**. Only the int8 set fits L2. The fp32
LightMHL path is therefore not merely 4× more traffic, it is on the wrong side of the
residency cliff — measured at 18.13 ms of gather at B=24,576 (`artifacts/shipped_light.json`),
12.8× the int8 kernel.

### TS — scan-bound, and the reason is occupancy

Nine (G, L, Bt) configurations, all gated. The fit is

```
t ≈ 4359 / L  ms
```

**independent of G and of the token tile Bt.** Every configuration gets **1 block/SM = 8
warps**, because holding `G·L/4` table words in registers forces high registers/thread.
The kernel is bound by the scan over tokens, not by the table read — which is the opposite
of what the design predicts. Extrapolating the law to an unreachable `L = 1024` still
leaves TS 3.7× slower than AS.

**Falsification test.** If the loss were layout, blocking the tables and transposing the
cells should move it. Measured: **±3%** (`artifacts/ts_layout.json`). Layout is irrelevant
here; occupancy is the whole story.

**The index distribution closes the G > 1 door independently.** Per-table participation
ratio is 155–172 of 256 rows and entropy 5.28–5.32 bits of the 8 available (a
uniform-index control at the same sample size gives 5.52), so the indices are spread, not
degenerate; and the empirical intra-thread dedup sits **on the birthday bound to within
2.2% at every G from 1 to 256** (`artifacts/index_stats.json`). Owning more cells per thread therefore
buys essentially nothing in coalescing terms — the collisions a larger G could exploit are
no more frequent than chance.

**`__launch_bounds__`'s second argument is load-bearing.** With only the block size given,
ptxas targets maximum occupancy, caps itself at 80–128 registers, and spills the register
table to local memory: every TS configuration spilled 1.7–6.4 kB before this was fixed.
See the comment at the top of `ts_kernel.cu`.

### Scatter — the transpose of AS

Instead of each token pulling its rows, each *cell occurrence* pushes into the output.
Arm A uses global atomics (`REDG`, 2·L per thread, 0 `ATOMG` on sm_120); arm B accumulates
in shared memory and flushes with plain stores. Arm B is 8.4× faster than arm A, and still
3.6× slower than AS. Atomic-free is necessary but not sufficient: the shared tile caps the
token tile, which caps occupancy.

### Split-K — the one real win

AS at small B is **launch- and occupancy-bound, not work-bound**: at B = 1 the kernel is
one block of 1024 threads on a 170-SM GPU (occupancy 0.6%) and its time is *flat in D*.
Splitting the table axis S ways and combining with int32 `atomicAdd` fills the machine:

| B | AS bare ms | best S | split-K ms | speed-up |
|---:|---:|---:|---:|---:|
| 1 | 0.0838 | 64 | **0.0103** | **8.14×** |
| 8 | 0.0982 | 32 | 0.0143 | 6.87× |
| 64 | 0.1343 | 16 | 0.0247 | 5.44× |
| 256 | 0.1398 | 8 | 0.0409 | 3.42× |
| 1024 | 0.1391 | 2 | 0.0880 | 1.58× |
| 2048 | 0.1450 | 2 | 0.1659 | 0.87× (AS wins) |
| 24576 | 1.4659 | 1 | 1.5473 | 0.95× (AS wins) |

The crossover is between B = 1024 and B = 2048, which is where AS's own occupancy
saturates (measured saturation at B = 4096, `artifacts/underfill.json`). **This is the
decode case**, and 8× at B = 1 is the most useful number in the study.

Two corrections worth keeping, because both were mine:

1. An early claim of "~100× headroom at B=1" was **wrong** — it compared against work
   alone and ignored the 5.25 µs empty-launch floor (`artifacts/splitk_sweep.json`,
   `launch_floor_ms`). With the floor in, the ceiling at B = 1 is **16.1×** and split-K
   achieves 8.1× of it.
2. The saturation table **mispredicted split-K's optimal S by 2–8× at small B**
   (`pred_sat_B` vs `best_S` in the same file). The model is useful for the crossover, not
   for picking S.

### One-hot GEMM — closed

Full write-up in [`ONEHOT_GEMM.md`](ONEHOT_GEMM.md). Summary: the FLOP-waste multiplier is
`K / 1.3272 = 192.9×`; break-even against AS, with both sides measured, is **19.8×**. The
strongest formulation — one `[N, T·K] @ [T·K, D]` matmul — runs at **253.8 TFLOP/s, 106%
of this GPU's measured square-matmul peak**, and is still **9.2× slower than AS** (12.8×
with the 3.22 GB selection build). There is no implementation slack left to recover, and
the best *compacted* variant at 26.3× waste still loses. fp16 overflows to NaN; int8
cannot carry the power-of-two coefficient. The question is closed on GPU.

One-hot does win in eight cells of the N × D grid — best 2.28× at N = 16, D = 48 — at
0.2–10% of peak, i.e. by being one cuBLAS launch against an under-occupied kernel. **That
corner is already taken by split-K**, which is 11.4× faster there.

## A discovery about the shipped path

`LightMultiHeadLUT`'s eval path does its gather with **plain `F.embedding_bag(...,
mode="sum", per_sample_weights=score)`**. The three hand-written gathers in the tree
(`gather.py`, `gather_cuda.py`, `gather_fused.py`) all gate on
`isinstance(mod, FastMultiHeadLut)` and **never fire** for `lut_impl=light`. Whatever is
decided about dataflow, that is the code currently running.

The fused int8 kernel could not run the paper's reference geometry at all before this
study: `UPR_MAX = 8` capped the cell width at `D ≤ 128`. `pow2_int8_read.cu` is widened
here to `UPR_MAX = 64` (`D ≤ 1024` at `block_n = 16`), with a deliberately sparse
instantiation matrix, and — the safety half of the change — **`PICK`'s `default:` now
refuses instead of falling through to `PICKU(BN, 8)`**, which for `upr` in 9…63 would have
silently written only the first 128 lanes of every row and left the rest of the output
untouched.

## Methodology, including what it cannot show

Stated because several of these limit what the numbers mean:

- **`ncu` is unavailable on this box** (`RmProfilingAdminOnly=1`). **Every GB/s figure here
  is request volume computed from launch geometry divided by measured time — never a DRAM
  counter.** That is why the 5,919 GB/s figure is evidence of cache residency rather than
  a bandwidth claim.
- **Hot/cold L2 flushing cannot demonstrate residency above B ≈ 64.** A 512 MiB scratch
  write before `start.record()` does evict the 96 MB L2, but at B = 24,576 the kernel
  rebuilds its own residency within 1/201 of the work, so hot and cold converge
  (18.13 vs 18.79 ms on the fp32 path) and the flush proves nothing at batch. The cold
  column is meaningful only in the small-B rows.
- **Timing discipline is per-arm and stated per-arm.** Median of 30–50 iterations, CUDA
  events with the synchronise *outside* the window, 3–5 warmup iterations, every buffer
  allocated and filled outside the measured region. Arms differ in **whether output
  zeroing is inside the timed window**: AS writes its output unconditionally and needs no
  zeroing (`as_bare`), while scatter and split-K accumulate and therefore must zero first.
  Both forms are reported — `as_zero` / `memset` columns exist precisely so the comparison
  can be made either way, and the split-K speed-ups above use `as_bare`, the stricter
  choice.
- **The `torch::empty` arms are gated with a poisoned allocator.** For arms that allocate
  with `torch::empty` and claim to write every element, a sentinel is written into the
  block, the allocator is shown to hand the same block back (`data_ptr` equality), the
  sentinel is shown to be present immediately before launch, and a deliberately cropped
  flush is shown to be *caught*. See `verify_poison.py`. This is how the "every element is
  written" claim rests on evidence rather than on a lucky zero page.
- **All gates are int32 bit-exact, zero tolerance**, against AS `read_cells`, and every
  comparison is run against *both* AS settings (`load16` true and false). Gate counts, all
  with **0 mismatches**: table-stationary **108** (18 fitting configs x 2 index
  distributions x 3 batch sizes, plus 5 over-budget configs checked to *refuse* rather than
  substitute), layout **168** (7 x 4 layouts x 2 x 3), scatter **198** (arm A 30 + arm B
  168), split-K **90** (9 values of S x 2 x 5). One of
  those gates earned its keep: a flush guard `if (tid < FLUSH)` silently dropped the tail
  whenever `M·K/4 > blockDim`, wrong output in 12 configurations, caught before any timing
  was believed.
- **The cells are produced offline and bit-exactly.** `cells_producer.py` is a PyTorch
  transcription of `p2::table_scalars`, matched detail for detail: left folds rather than
  `.sum()`, the *first strict* argmin, `where(x < 0, x, 0)` to reproduce `fmin`'s NaN rule,
  and `~(kr >= lo)` rather than `kr < lo`. `gate_cells.py` checks it against the kernel.
- **The vehicle is `read_cells`, i.e. Act 3 only.** Given a cells tensor, it does the
  shift-add read and nothing else — no indexing, no scoring. That isolates the dataflow
  question, and it means **none of these numbers are end-to-end layer times.**

## Open items

- **A dispatch-on-B path was not written.** The split-K result says the right shipped
  behaviour is "split the table axis when B is small, don't when it is large," with the
  crossover between 1024 and 2048 at this geometry. Choosing S is *not* predictable from
  the occupancy model (see correction 2 above), so it wants a small measured table, not a
  formula.
- **No end-to-end decode-step measurement.** Everything here is Act 3 in isolation.
- **The one remaining experiment worth money is a Pallas SparseCore gather on a v6e-1.**
  The two GPU-specific objections to a gather-heavy design — cheap L2-resident reads and an
  expensive MXU-bypassing scatter — both invert on TPU, and SparseCore exists precisely to
  make gathers fast. Nothing else in this space needs new hardware to answer.

## Files

| file | what it is |
|---|---|
| `lac_kernels.cu`, `lac.py` | the first-pass gather baseline and `cs_v0`…`cs_v3` cell-stationary kernels, plus the two ablation kernels used to attribute runtime |
| `ts_kernel.cu`, `ts.py`, `bench_ts.py`, `gate_ts.py` | the table-stationary arm |
| `bench_layout.py`, `gate_layout.py` | the layout falsification test (blocked tables, transposed cells) |
| `ts_law.py`, `ts_law_check.py`, `ts_budgets.py`, `ts_occupancy.py`, `ts_report.py` | the `t ≈ 4359/L` fit, the smem/register budgets, occupancy, and the written report |
| `scatter_kernel.cu`, `scatter.py`, `bench_scatter.py`, `gate_scatter.py`, `scatter_report.py` | the two scatter arms (global atomics, shared accumulate) |
| `splitk_kernel.cu`, `splitk.py`, `bench_splitk.py`, `gate_splitk.py` | split-K over the table axis, S as a runtime argument |
| `bench_underfill.py` | AS occupancy and the saturation point, plus the empty-launch floor |
| `gemm_scoping.py`, `gemm_roofline.py`, `bench_onehot*.py`, `ONEHOT_GEMM.md` | the one-hot GEMM study |
| `cells_producer.py`, `gate_cells.py`, `extract_margins.py`, `bench_act3.py` | the offline cells producer, its gate, and the Act-3 harness every arm shares |
| `bench_shipped_light.py`, `bench_shipped_int8.py` | the shipped `LightMultiHeadLUT` path as-is, fp32 and p2_int8, hot and cold |
| `verify_poison.py` | proof that the poisoned-allocator gate is real and sensitive |
| `index_stats.py`, `as_opcount.py`, `ptxas_table.py`, `probe_model.py`, `extract_real.py` | index statistics, op counts, registers/spills per kernel, and the real-layer extraction |
| `artifacts/` | every JSON result quoted above |

`make_report.py`, `plot_bench.py`, `plot_dedup.py`, `summarise_log.py`, `diag_attrib.py`,
`probe_time.py`, `crossover_act3.py`, `ts_compile_info.py` and `scatter_compile_info.py` are
first-pass helpers for the `lac_kernels.cu` arm. Note that `make_report.py` splices generated
tables under a `<!-- RESULTS -->` marker, which this hand-written README deliberately does
**not** carry — run it only if you want the generated tables back.

Two artifacts are **not** versioned (see `.gitignore`), both regenerable from the
checkpoint: `artifacts/real_layer.pt` (151 MB, `extract_real.py`) and
`artifacts/real_margins.pt` (4.8 MB, `extract_margins.py`).

## Reproducing

```sh
cd research/lac_cell_stationary
python probe_model.py            # what shape is the real layer
python extract_real.py           # -> artifacts/real_layer.pt   (gitignored)
python extract_margins.py        # -> artifacts/real_margins.pt (gitignored)
python index_stats.py            # -> artifacts/index_stats.json
python gate_cells.py             # offline cells == kernel cells, bit exact
python bench_act3.py             # -> artifacts/act3.json       (the AS baseline)
python gate_ts.py      && python bench_ts.py        # -> artifacts/ts_*.json
python gate_layout.py  && python bench_layout.py    # -> artifacts/ts_layout.json
python gate_scatter.py && python bench_scatter.py   # -> artifacts/scatter_bench.json
python gate_splitk.py  && python bench_splitk.py    # -> artifacts/splitk_sweep.json
python bench_underfill.py                           # -> artifacts/underfill.json
python gemm_roofline.py && python bench_onehot.py \
  && python bench_onehot_big.py && python bench_onehot_grid.py
python bench_shipped_light.py && python bench_shipped_int8.py
```
