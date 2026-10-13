# lutorch_ex performance snapshot of record (#162)

**RTX 5090 only** (sm_120), torch 2.9.1+cu130, CUDA 13.0, driver 595.84; the H100 was not re-measured. Code
`c8e48c36` (#162's library code; no uncommitted changes in lutorch_ex); measured 2026-10-12, the whole matrix in one run.
Champion geometry h16 tph64 nap8 d48 at 8,192 vectors (`champion`) and 32,768 (`large`). Compiled, one fresh process
per cell, a Dynamo reset per cell, median of 30 timed iterations after 10 warm-up; peak GiB above the pre-step
allocation; clocks not locked (observed SM clock 2647 MHz after the champion cells,
2677 MHz after the large ones, 42 C at the end, no throttling; other
processes on the card: desktop only, ~180 MiB). Every class runs its one implementation with its defaults; `status`
says why a cell has no numbers. The IQRs and the environment observations are in `snapshot_162.json`.

Reproduce: `python -m spiky.lutorch_ex.bench_snapshot --geometries champion,large --out snapshot_162.json --md snapshot_162.md`

What #162 changed against `snapshot_160` (the rest of the library is unchanged since):
- `QuantisedConfidenceLUT` compiles with Inductor's realize thresholds raised (a scheduling hint for its left-folded
  exponents, not maths). 32,768 vectors: eval 11.76 -> 9.98 ms, eval peak 2.03 -> 0.90 GiB, train-forward peak 2.33 ->
  1.92 GiB, train total within noise. 8,192: eval 2.97 -> 2.58 ms, peak 0.51 -> 0.26 GiB.
- `ManifestoHardLUT` (both weights) builds its training surrogate with one embedding_bag. 32,768 vectors: training no
  longer runs out of memory (fp32 60.4 ms / 5.09 GiB). 8,192 fp32: train total 26.07 -> 14.91 ms, peak 6.71 -> 1.27
  GiB; train backward is slower, 7.48 -> 8.97 ms (embedding_bag's per-sample-weight backward), the forward much faster.

Run-to-run variation, as in snapshot_160: sub-millisecond fused-class cells move by up to ~15% between sessions with
no code change in their path (the largest here: FusedManifestoSoftLUT weight=softsign bf16 train forward 0.339 -> 0.390
ms, a cell that measured 0.339-0.389 in the repeats behind snapshot_160); treat such single-cell differences as noise.

| geometry | cartridge | dtype | status | eval forward ms | GiB | train forward ms | GiB | train backward ms | GiB | train total ms | GiB |
|---|---|---|---|---|---|---|---|---|---|---|---|
| champion | ManifestoHardLUT | float32 | ok | 0.647 | 0.055 | 5.937 | 1.273 | 8.971 | 1.047 | 14.907 | 1.273 |
| champion | ManifestoHardLUT | bfloat16 | ok | 0.684 | 0.125 | 5.993 | 1.344 | 8.947 | 1.082 | 14.941 | 1.344 |
| champion | ManifestoSoftLUT | float32 | ok | 1.225 | 0.148 | 5.041 | 1.273 | 10.213 | 1.265 | 15.254 | 1.273 |
| champion | ManifestoSoftLUT | bfloat16 | ok | 1.250 | 0.219 | 5.116 | 1.344 | 10.116 | 1.323 | 15.233 | 1.344 |
| champion | FusedManifestoHardLUT | float32 | ok | 0.252 | 0.023 | 0.317 | 0.023 | 1.285 | 0.094 | 1.601 | 0.094 |
| champion | FusedManifestoHardLUT | bfloat16 | ok | 0.192 | 0.012 | 0.251 | 0.012 | 1.149 | 0.094 | 1.400 | 0.094 |
| champion | FusedManifestoSoftLUT | float32 | ok | 0.461 | 0.023 | 0.545 | 0.023 | 1.711 | 0.094 | 2.257 | 0.094 |
| champion | FusedManifestoSoftLUT | bfloat16 | ok | 0.282 | 0.012 | 0.329 | 0.012 | 1.604 | 0.094 | 1.933 | 0.094 |
| champion | ManifestoHardLUT weight=softsign | float32 | ok | 0.668 | 0.055 | 5.973 | 1.273 | 9.276 | 1.047 | 15.249 | 1.273 |
| champion | ManifestoHardLUT weight=softsign | bfloat16 | ok | 0.708 | 0.125 | 6.030 | 1.344 | 9.225 | 1.082 | 15.253 | 1.344 |
| champion | ManifestoSoftLUT weight=softsign | float32 | ok | 1.289 | 0.148 | 5.071 | 1.273 | 10.557 | 1.327 | 15.627 | 1.327 |
| champion | ManifestoSoftLUT weight=softsign | bfloat16 | ok | 1.317 | 0.219 | 5.160 | 1.344 | 10.449 | 1.386 | 15.611 | 1.386 |
| champion | FusedManifestoHardLUT weight=softsign | float32 | ok | 0.268 | 0.023 | 0.328 | 0.023 | 1.591 | 0.094 | 1.911 | 0.094 |
| champion | FusedManifestoHardLUT weight=softsign | bfloat16 | ok | 0.215 | 0.012 | 0.294 | 0.012 | 1.495 | 0.094 | 1.787 | 0.094 |
| champion | FusedManifestoSoftLUT weight=softsign | float32 | ok | 0.478 | 0.023 | 0.568 | 0.023 | 1.849 | 0.094 | 2.417 | 0.094 |
| champion | FusedManifestoSoftLUT weight=softsign | bfloat16 | ok | 0.313 | 0.012 | 0.390 | 0.012 | 1.716 | 0.094 | 2.101 | 0.094 |
| champion | ConfidenceLUT n=1 | float32 | ok | 0.910 | 0.117 | 1.084 | 0.117 | 2.360 | 0.711 | 3.443 | 0.711 |
| champion | ConfidenceLUT n=1 | bfloat16 | ok | 0.945 | 0.188 | 1.081 | 0.199 | 2.379 | 0.746 | 3.461 | 0.746 |
| champion | ConfidenceLUT n=2 | float32 | ok | 1.277 | 0.219 | 1.319 | 0.211 | 4.067 | 0.820 | 5.385 | 0.820 |
| champion | ConfidenceLUT n=2 | bfloat16 | ok | 1.265 | 0.289 | 1.318 | 0.293 | 4.026 | 0.855 | 5.343 | 0.855 |
| champion | FusedConfidenceLUT n=1 | float32 | ok | 0.267 | 0.023 | 0.323 | 0.023 | 1.241 | 0.094 | 1.563 | 0.094 |
| champion | FusedConfidenceLUT n=1 | bfloat16 | ok | 0.205 | 0.012 | 0.273 | 0.012 | 1.154 | 0.094 | 1.418 | 0.094 |
| champion | FusedConfidenceLUT n=2 | float32 | ok | 0.473 | 0.023 | 0.538 | 0.023 | 1.923 | 0.094 | 2.459 | 0.094 |
| champion | FusedConfidenceLUT n=2 | bfloat16 | ok | 0.314 | 0.012 | 0.385 | 0.012 | 1.770 | 0.094 | 2.152 | 0.094 |
| champion | QuantisedConfidenceLUT n=2 | float32 | ok | 2.579 | 0.259 | 3.043 | 0.516 | 9.511 | 1.116 | 12.554 | 1.116 |
| champion | QuantisedConfidenceLUT n=2 | bfloat16 | refused: QuantisedConfidenceLUT does not support low precision | - | - | - | - | - | - | - | - |
| champion | FusedQuantisedConfidenceLUT n=2 | float32 | ok | 2.521 | 0.415 | 2.507 | 0.352 | 9.884 | 0.946 | 12.390 | 0.946 |
| champion | FusedQuantisedConfidenceLUT n=2 | bfloat16 | error: RuntimeError: FusedQuantisedConfidenceLUT: not a case the fused op covers: input dtype torch.bfloat16 (float32 only); table dtype torc | - | - | - | - | - | - | - | - |
| champion | DeployedQuantisedConfidenceLUT | float32 | ok | 2.909 | 0.414 | - | - | - | - | - | - |
| champion | DeployedQuantisedConfidenceLUT | bfloat16 | refused: DeployedQuantisedConfidenceLUT is fp32/fp64 inference only; got input dtype torch.bfloat16 | - | - | - | - | - | - | - | - |
| large | ManifestoHardLUT | float32 | ok | 2.481 | 0.219 | 24.519 | 5.094 | 35.927 | 4.188 | 60.450 | 5.094 |
| large | ManifestoHardLUT | bfloat16 | ok | 2.611 | 0.359 | 24.725 | 5.234 | 35.920 | 4.188 | 60.646 | 5.234 |
| large | ManifestoSoftLUT | float32 | ok | 5.036 | 0.594 | 20.810 | 5.094 | 40.974 | 4.763 | 61.786 | 5.094 |
| large | ManifestoSoftLUT | bfloat16 | ok | 5.067 | 0.734 | 20.981 | 5.234 | 40.945 | 4.857 | 61.927 | 5.234 |
| large | FusedManifestoHardLUT | float32 | ok | 0.991 | 0.094 | 1.077 | 0.094 | 4.866 | 0.234 | 5.943 | 0.234 |
| large | FusedManifestoHardLUT | bfloat16 | ok | 0.750 | 0.047 | 0.847 | 0.047 | 4.494 | 0.234 | 5.342 | 0.234 |
| large | FusedManifestoSoftLUT | float32 | ok | 1.816 | 0.094 | 2.032 | 0.094 | 6.570 | 0.234 | 8.604 | 0.234 |
| large | FusedManifestoSoftLUT | bfloat16 | ok | 1.118 | 0.047 | 1.208 | 0.047 | 6.356 | 0.234 | 7.565 | 0.234 |
| large | ManifestoHardLUT weight=softsign | float32 | ok | 2.559 | 0.219 | 24.782 | 5.094 | 37.725 | 4.188 | 62.509 | 5.094 |
| large | ManifestoHardLUT weight=softsign | bfloat16 | ok | 2.696 | 0.359 | 24.995 | 5.234 | 37.730 | 4.188 | 62.724 | 5.234 |
| large | ManifestoSoftLUT weight=softsign | float32 | ok | 5.378 | 0.594 | 21.072 | 5.094 | 42.774 | 5.013 | 63.847 | 5.094 |
| large | ManifestoSoftLUT weight=softsign | bfloat16 | ok | 5.419 | 0.734 | 21.259 | 5.234 | 42.835 | 5.107 | 64.092 | 5.234 |
| large | FusedManifestoHardLUT weight=softsign | float32 | ok | 1.010 | 0.094 | 1.117 | 0.094 | 6.331 | 0.234 | 7.447 | 0.234 |
| large | FusedManifestoHardLUT weight=softsign | bfloat16 | ok | 0.769 | 0.047 | 0.887 | 0.047 | 5.947 | 0.234 | 6.833 | 0.234 |
| large | FusedManifestoSoftLUT weight=softsign | float32 | ok | 1.837 | 0.094 | 2.000 | 0.094 | 7.010 | 0.234 | 9.010 | 0.234 |
| large | FusedManifestoSoftLUT weight=softsign | bfloat16 | ok | 1.176 | 0.047 | 1.274 | 0.047 | 6.787 | 0.234 | 8.060 | 0.234 |
| large | ConfidenceLUT n=1 | float32 | ok | 3.653 | 0.469 | 3.801 | 0.469 | 9.633 | 2.703 | 13.438 | 2.703 |
| large | ConfidenceLUT n=1 | bfloat16 | ok | 3.746 | 0.609 | 3.792 | 0.656 | 9.639 | 2.844 | 13.421 | 2.844 |
| large | ConfidenceLUT n=2 | float32 | ok | 4.946 | 0.875 | 4.816 | 0.844 | 16.139 | 3.141 | 20.956 | 3.141 |
| large | ConfidenceLUT n=2 | bfloat16 | ok | 5.083 | 1.016 | 4.811 | 1.031 | 15.867 | 3.281 | 20.680 | 3.281 |
| large | FusedConfidenceLUT n=1 | float32 | ok | 1.053 | 0.094 | 1.133 | 0.094 | 4.753 | 0.236 | 5.885 | 0.236 |
| large | FusedConfidenceLUT n=1 | bfloat16 | ok | 0.798 | 0.047 | 0.939 | 0.047 | 4.483 | 0.236 | 5.422 | 0.236 |
| large | FusedConfidenceLUT n=2 | float32 | ok | 1.877 | 0.094 | 1.975 | 0.094 | 7.473 | 0.236 | 9.450 | 0.236 |
| large | FusedConfidenceLUT n=2 | bfloat16 | ok | 1.214 | 0.047 | 1.430 | 0.047 | 7.022 | 0.236 | 8.452 | 0.236 |
| large | QuantisedConfidenceLUT n=2 | float32 | ok | 9.976 | 0.895 | 11.566 | 1.922 | 38.103 | 4.169 | 49.671 | 4.169 |
| large | QuantisedConfidenceLUT n=2 | bfloat16 | refused: QuantisedConfidenceLUT does not support low precision | - | - | - | - | - | - | - | - |
| large | FusedQuantisedConfidenceLUT n=2 | float32 | ok | 9.683 | 1.658 | 9.430 | 1.406 | 39.069 | 3.641 | 48.500 | 3.641 |
| large | FusedQuantisedConfidenceLUT n=2 | bfloat16 | error: RuntimeError: FusedQuantisedConfidenceLUT: not a case the fused op covers: input dtype torch.bfloat16 (float32 only); table dtype torc | - | - | - | - | - | - | - | - |
| large | DeployedQuantisedConfidenceLUT | float32 | ok | 16.802 | 1.656 | - | - | - | - | - | - |
| large | DeployedQuantisedConfidenceLUT | bfloat16 | refused: DeployedQuantisedConfidenceLUT is fp32/fp64 inference only; got input dtype torch.bfloat16 | - | - | - | - | - | - | - | - |

> **Note (2026-10-13):** the benchmark modules have since moved: `spiky.lutorch_ex.bench_snapshot` is now `spiky.lutorch_ex.benchmarks.bench_snapshot` (likewise `bench`, `bench_compare`, `bench_fused_confidence`). The command above is recorded as it was run.
