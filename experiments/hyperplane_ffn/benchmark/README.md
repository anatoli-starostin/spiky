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

## Files

| file | what it is |
|---|---|
| `model.py` | Rebuilds an experiment's model from its `config.json` (+ `checkpoint.pt` if asked). Handles all three FFN families: dense vanilla, CompressionMHL/anchor-pair, and the pure-ternary hyperplane family. |
| `gather.py` | The fused Triton gather+sum kernel replacing `embedding_bag`, plus `patch()` for the FastMultiHeadLut family and `tune()` to re-sweep its config on new hardware. |
| `gather_ternary.py` | The same gather for the ternary hyperplane family, which is *not* a FastMultiHeadLut subclass and needs its own patch. |
| `hybrid.py` | hybrid-v2 storage: dense weights **stored** bf16, LUT tables and (for FastMHL) LUT input kept fp32 so the native bit-pack kernel stays alive. |
| `bench.py` | `burn_in`, `timeit`, `check_bit_exact`, `slot_breakdown`, `interleaved_ab`, `report`. |
| `run_bench.py` | CLI driver tying it together. |

## Running it

```bash
cd experiments/hyperplane_ffn/benchmark

# the two CompressionMHL grid cells, against the untied vanilla baseline
python run_bench.py --exp exp_n_0126_grid_H4d48_nap7_tph64
python run_bench.py --exp exp_n_0127_grid_H4d48_nap7_tph128

# the ternary family needs its trained weights (realized sparsity is data-dependent)
python run_bench.py --exp exp_g_0053_ternary_t192_inputonly_headdecomp_nap7_tph64 \
                    --load-checkpoint

# explicit baseline / batches / rounds; re-tune the gather on new hardware
python run_bench.py --exp exp_n_0126_grid_H4d48_nap7_tph64 \
    --baseline exp_n_0135_untied_vanilla_baseline_16k \
    --batches 12,48,96 --rounds 11 --tune-gather
```

`--baseline` defaults to `exp_n_0135_untied_vanilla_baseline_16k` (the untied vanilla
dense-FFN reference, 1.20144 bpb). `--load-checkpoint` is only needed when the
*values* matter; pure timing does not need it, since a gather reads a row whatever is
in it.

Requires `lutorch_cuda` built (the native bit-pack kernel) and Triton.

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

All bit-exact (`max|logit diff| = 0.000e+00`), warmed, interleaved, seq 512:

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

## Known limitation: the ternary path is correct but not yet tuned here

`gather_ternary.patch` is bit-exact, and the family dispatch is explicit so it can
never silently no-op (an earlier version matched only `FastMultiHeadLut` and reported
an *unpatched* ternary model as optimized). But the ternary slot measures ~3.5 ms in
this harness against ~0.62 ms from the ad-hoc rig the same optimization achieved
during the original sprint, and that gap has not been chased down. **Do not quote the
harness's ternary numbers as the family's best** until it is. The FastMultiHeadLut
path (0126/0127) reproduces its known numbers and is the one to trust today.

Separately, for the ternary family `--addr-dtype bf16` is faster but **not** bit-exact
and cannot be: near a score of 0 the rounding flips the sign bit and selects a
different table row — a discrete change in output, not a tolerance. `fp32` is the
default for that reason.

## Things already measured and ruled out

So the H100 side does not re-tread them. All on the 5090, against the dense bf16
addressing GEMM:

- Narrowing the gather index (int32/int16): no gain; the strided index read costs the
  same number of 32-byte sectors at any width.
- int8 tensor-core addressing: the GEMM is 2× faster but the stage is 24% slower (the
  int32 accumulator can't fuse) and 4.3× more sign flips.
- 2:4 structured sparsity: sparse tensor cores work on this GPU (1.98× on an 8192³
  GEMM) but lose 6.5× at K=384 — too small to be compute-bound. They start paying
  around K≈4096.
- Sparse/gather addressing in every form tried (CSR, embedding_bag, feature-major
  tiled SpMM, scatter+atomic, a hand-written CUDA masked-sum): 10× to 700× slower
  than the dense GEMM. Top-k truncation down to 2% density never crosses it.
