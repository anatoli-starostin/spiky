# Benchmarks, 2026-10-14: RTX 5090 and H100

Every cartridge row of `bench_snapshot` (every public class, with the alternatives the harness lists), float32,
defaults, at the `champion` geometry (h16 tph64 nap8 d48) with 8,192 vectors and the `large` geometry (the same) with
32,768 vectors. Both cards ran the same harness and configuration:

    python -m spiky.lutorch_ex.benchmarks.bench_snapshot --geometries champion,large --dtypes float32 --rounds 3

i.e. one fresh process per cell, 3 rounds with the cell order rotated, 20 warm-up + 100 timed iterations, the L2
cache flushed before every timed iteration; wall time (median of the round medians, round range in the JSON), GPU
kernel time per step, peak memory.

| file | what |
|---|---|
| `rtx5090.json` / `.md` | NVIDIA GeForce RTX 5090 (sm_120), torch 2.9.1+cu130, driver 595.84 |
| `h100.json` / `.md` | NVIDIA H100 80GB HBM3 (sm_90), torch 2.9.1+cu130, driver 580.173.02; lutorch_ex installed from its pyproject, the full GPU suite passing first (1668 passed) |
| `figures_rtx5090_vs_h100/` | `bench_devices`: wall and GPU time per geometry, training peak memory, `comparison.md` |

Provenance: the RTX 5090 record names commit `d0136e20`, the harness commit as it was before being rebased onto the
branch as `c4547ce6`; the code of both is identical (only docstrings differ). The H100 record names `b02cf440`
(`c4547ce6` plus a comments-only fix-up). The library code measured is the same on both cards.

Reading the numbers: wall time includes the CPU's dispatch; where the kernels are short it is dispatch-bound and
GPU time is the steadier figure. A GPU time above 1.05 x the wall median (profiler-inflated, typically a compiled,
kernel-dense train step) is marked `*` in `comparison.md` and drawn hollow; there the wall time is the better number.
