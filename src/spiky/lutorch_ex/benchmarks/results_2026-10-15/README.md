# Benchmarks, 2026-10-15: RTX 5090 and H100

Every cartridge row of `bench_snapshot` (every public class, with the alternatives the harness lists), float32,
defaults, at the `champion` geometry (h16 tph64 nap8 d48) with 8,192 vectors and the `large` geometry (the same) with
32,768 vectors, at commit `f88dd711` on both cards. Both cards ran the same harness and configuration:

    python -m spiky.lutorch_ex.benchmarks.bench_snapshot --geometries champion,large --dtypes float32 --rounds 3

i.e. one fresh process per cell, 3 rounds with the cell order rotated, 20 warm-up + 100 timed iterations, the L2
cache flushed before every timed iteration; wall time (median of the round medians), GPU kernel time per step, peak
memory. Each card ran this several times, interleaved in one session; a published value is the mean of those passes,
and the passes' round rows are all in the JSON (`publication` says how the record was made, with the noise floor
between passes).

| file | what |
|---|---|
| `rtx5090.json` / `.md` | NVIDIA GeForce RTX 5090 (sm_120, 170 SMs, 96 MiB L2), torch 2.9.1+cu130, driver 595.84, VBIOS 98.02.2E.00.C9, power limit 600 W, max SM clock 3210 MHz; 2026-10-15T06:59:51Z; 3 passes; the full GPU suite passing first on that build (1680 passed) |
| `h100.json` / `.md` | NVIDIA H100 80GB HBM3 (sm_90, 132 SMs, 50 MiB L2), torch 2.9.1+cu130, driver 580.173.02, VBIOS 96.00.A5.00.01, power limit 700 W, max SM clock 1980 MHz; 2026-10-14T21:37:43Z; 2 passes; the full GPU suite passing first on that build (1680 passed) |
| `figures_rtx5090_vs_h100/` | `bench_devices`: wall and GPU time per geometry, training peak memory, `comparison.md` |

Noise floor between passes (|a / b - 1| of the same cell; fused and deployed rows):

| card | wall time median / p90 / max | GPU time median / p90 / max |
|---|---|---|
| RTX 5090 | 0.19% / 0.73% / 1.46% | 0.47% / 2.88% / 6.77% |
| H100 | 0.25% / 0.50% / 0.70% | 0.09% / 0.53% / 0.64% |

On the RTX 5090 the profiler's GPU time scattered more between passes than the wall time; there the wall time is the
steadier figure.

Reading the numbers: wall time includes the CPU's dispatch; where the kernels are short it is dispatch-bound and
GPU time is the steadier figure. A GPU time above 1.05 x the wall median (profiler-inflated, typically a compiled,
kernel-dense train step) is marked `*` in `comparison.md` and drawn hollow; there the wall time is the better number.
No cell of either record is so marked.

Cells with no number, and why:

- `DeployedQuantisedConfidenceLUT` (both rows): no train step and no train peak memory (`-` in the tables). It is an
  inference-only class.
- The `ManifestoHardLUT` / `ManifestoSoftLUT` reference rows (both weightings): the train step is the reference's
  deliberately uncompiled eager path (their eval forward is compiled), not a kernel, so it is not comparable with the
  fused rows' train step.
