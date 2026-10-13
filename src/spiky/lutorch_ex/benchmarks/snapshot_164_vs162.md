# lutorch_ex performance snapshot: #164 HEAD, paired with #162 re-measured

**Code.** `e68fcc32`, #164 HEAD (library code identical to a251a82f); no uncommitted changes in lutorch_ex. A second same-day measurement of #164; `snapshot_164.json` is the one paired with #160.

**Hardware and software.** RTX 5090 (sm_120, 170 SMs), torch 2.9.1+cu130, Triton 3.5.1, CUDA 13.0, driver 595.84.

**Session.** Measured 2026-10-13 in ONE session together with snapshot_162_remeasured (#162's code), interleaved cell by cell.
- Each cell ran under each tree's own `python -m spiky.lutorch_ex.bench_snapshot` for that one cell: one fresh process per cell, a Dynamo reset, 10 warm-up + 30 timed iterations, median; peak GiB above the pre-step allocation.
- The two trees alternated which went first.
- Champion geometry h16 tph64 nap8 d48 at 8,192 vectors (`champion`) and 32,768 (`large`).

**Conditions** (logged between every cell):
- SM clock ≥ 2.65 GHz on busy cells (unlocked), 34–43 C.
- **No throttle or power-cap flag at any point.**
- Other processes on the card: desktop only, ~180 MiB.

**No transient.** Nothing moved beyond its noise floor except the cells this PR targets, so no row was re-measured.

Comparison: `figures_162_remeasured_vs_164/` (bench_compare.py --same-session). This is #164's primary comparison, #162 being the parent.

| geometry | cartridge | dtype | status | eval forward ms | GiB | train forward ms | GiB | train backward ms | GiB | train total ms | GiB |
|---|---|---|---|---|---|---|---|---|---|---|---|
| champion | ManifestoHardLUT | float32 | ok | 0.658 | 0.055 | 5.937 | 1.273 | 8.973 | 1.047 | 14.911 | 1.273 |
| champion | ManifestoHardLUT | bfloat16 | ok | 0.699 | 0.125 | 5.990 | 1.344 | 8.945 | 1.082 | 14.936 | 1.344 |
| champion | ManifestoSoftLUT | float32 | ok | 1.221 | 0.148 | 5.041 | 1.273 | 10.206 | 1.265 | 15.245 | 1.273 |
| champion | ManifestoSoftLUT | bfloat16 | ok | 1.249 | 0.219 | 5.117 | 1.344 | 10.105 | 1.323 | 15.222 | 1.344 |
| champion | FusedManifestoHardLUT | float32 | ok | 0.251 | 0.023 | 0.308 | 0.023 | 1.296 | 0.094 | 1.596 | 0.094 |
| champion | FusedManifestoHardLUT | bfloat16 | ok | 0.191 | 0.012 | 0.250 | 0.012 | 1.189 | 0.094 | 1.437 | 0.094 |
| champion | FusedManifestoSoftLUT | float32 | ok | 0.462 | 0.023 | 0.536 | 0.023 | 1.727 | 0.094 | 2.268 | 0.094 |
| champion | FusedManifestoSoftLUT | bfloat16 | ok | 0.287 | 0.012 | 0.342 | 0.012 | 1.603 | 0.094 | 1.954 | 0.094 |
| champion | ManifestoHardLUT weight=softsign | float32 | ok | 0.644 | 0.055 | 5.968 | 1.273 | 9.292 | 1.047 | 15.261 | 1.273 |
| champion | ManifestoHardLUT weight=softsign | bfloat16 | ok | 0.687 | 0.125 | 6.018 | 1.344 | 9.248 | 1.082 | 15.268 | 1.344 |
| champion | ManifestoSoftLUT weight=softsign | float32 | ok | 1.288 | 0.148 | 5.071 | 1.273 | 10.526 | 1.327 | 15.598 | 1.327 |
| champion | ManifestoSoftLUT weight=softsign | bfloat16 | ok | 1.318 | 0.219 | 5.157 | 1.344 | 10.437 | 1.386 | 15.595 | 1.386 |
| champion | FusedManifestoHardLUT weight=softsign | float32 | ok | 0.266 | 0.023 | 0.341 | 0.023 | 1.589 | 0.094 | 1.925 | 0.094 |
| champion | FusedManifestoHardLUT weight=softsign | bfloat16 | ok | 0.215 | 0.012 | 0.294 | 0.012 | 1.495 | 0.094 | 1.789 | 0.094 |
| champion | FusedManifestoSoftLUT weight=softsign | float32 | ok | 0.477 | 0.023 | 0.579 | 0.023 | 1.869 | 0.094 | 2.438 | 0.094 |
| champion | FusedManifestoSoftLUT weight=softsign | bfloat16 | ok | 0.305 | 0.012 | 0.380 | 0.012 | 1.757 | 0.094 | 2.105 | 0.094 |
| champion | ConfidenceLUT n=1 | float32 | ok | 0.900 | 0.117 | 1.064 | 0.117 | 2.371 | 0.711 | 3.434 | 0.711 |
| champion | ConfidenceLUT n=1 | bfloat16 | ok | 0.942 | 0.188 | 1.066 | 0.199 | 2.357 | 0.746 | 3.421 | 0.746 |
| champion | ConfidenceLUT n=2 | float32 | ok | 1.274 | 0.219 | 1.358 | 0.211 | 4.065 | 0.820 | 5.424 | 0.820 |
| champion | ConfidenceLUT n=2 | bfloat16 | ok | 1.280 | 0.289 | 1.321 | 0.293 | 4.017 | 0.855 | 5.336 | 0.855 |
| champion | FusedConfidenceLUT n=1 | float32 | ok | 0.266 | 0.023 | 0.305 | 0.023 | 1.218 | 0.094 | 1.540 | 0.094 |
| champion | FusedConfidenceLUT n=1 | bfloat16 | ok | 0.213 | 0.012 | 0.282 | 0.012 | 1.145 | 0.094 | 1.428 | 0.094 |
| champion | FusedConfidenceLUT n=2 | float32 | ok | 0.474 | 0.023 | 0.541 | 0.023 | 1.929 | 0.094 | 2.464 | 0.094 |
| champion | FusedConfidenceLUT n=2 | bfloat16 | ok | 0.320 | 0.012 | 0.407 | 0.012 | 1.820 | 0.094 | 2.223 | 0.094 |
| champion | QuantisedConfidenceLUT n=2 | float32 | ok | 2.575 | 0.259 | 3.032 | 0.516 | 9.421 | 1.116 | 12.454 | 1.116 |
| champion | QuantisedConfidenceLUT n=2 | bfloat16 | refused: QuantisedConfidenceLUT does not support low precision | - | - | - | - | - | - | - | - |
| champion | FusedQuantisedConfidenceLUT n=2 | float32 | ok | 0.515 | 0.023 | 0.591 | 0.023 | 2.098 | 0.094 | 2.685 | 0.094 |
| champion | FusedQuantisedConfidenceLUT n=2 | bfloat16 | error: RuntimeError: FusedQuantisedConfidenceLUT: not a case the fused op covers: input dtype torch.bfloat16 (float32 only); table dtype torc | - | - | - | - | - | - | - | - |
| champion | DeployedQuantisedConfidenceLUT | float32 | ok | 0.310 | 0.023 | - | - | - | - | - | - |
| champion | DeployedQuantisedConfidenceLUT | bfloat16 | refused: DeployedQuantisedConfidenceLUT is fp32/fp64 inference only; got input dtype torch.bfloat16 | - | - | - | - | - | - | - | - |
| champion | DeployedQuantisedConfidenceLUT n=1 | float32 | ok | 0.220 | 0.023 | - | - | - | - | - | - |
| champion | DeployedQuantisedConfidenceLUT n=1 | bfloat16 | refused: DeployedQuantisedConfidenceLUT is fp32/fp64 inference only; got input dtype torch.bfloat16 | - | - | - | - | - | - | - | - |
| large | ManifestoHardLUT | float32 | ok | 2.538 | 0.219 | 24.521 | 5.094 | 35.947 | 4.188 | 60.467 | 5.094 |
| large | ManifestoHardLUT | bfloat16 | ok | 2.680 | 0.359 | 24.731 | 5.234 | 35.973 | 4.188 | 60.706 | 5.234 |
| large | ManifestoSoftLUT | float32 | ok | 5.018 | 0.594 | 20.801 | 5.094 | 40.974 | 4.763 | 61.778 | 5.094 |
| large | ManifestoSoftLUT | bfloat16 | ok | 5.075 | 0.734 | 20.984 | 5.234 | 40.937 | 4.857 | 61.915 | 5.234 |
| large | FusedManifestoHardLUT | float32 | ok | 0.987 | 0.094 | 1.070 | 0.094 | 4.924 | 0.234 | 5.997 | 0.234 |
| large | FusedManifestoHardLUT | bfloat16 | ok | 0.745 | 0.047 | 0.849 | 0.047 | 4.599 | 0.234 | 5.448 | 0.234 |
| large | FusedManifestoSoftLUT | float32 | ok | 1.812 | 0.094 | 2.088 | 0.094 | 6.745 | 0.234 | 8.833 | 0.234 |
| large | FusedManifestoSoftLUT | bfloat16 | ok | 1.119 | 0.047 | 1.216 | 0.047 | 6.276 | 0.234 | 7.494 | 0.234 |
| large | ManifestoHardLUT weight=softsign | float32 | ok | 2.483 | 0.219 | 24.773 | 5.094 | 37.736 | 4.188 | 62.512 | 5.094 |
| large | ManifestoHardLUT weight=softsign | bfloat16 | ok | 2.619 | 0.359 | 24.994 | 5.234 | 37.761 | 4.188 | 62.754 | 5.234 |
| large | ManifestoSoftLUT weight=softsign | float32 | ok | 5.375 | 0.594 | 21.066 | 5.094 | 42.796 | 5.013 | 63.870 | 5.094 |
| large | ManifestoSoftLUT weight=softsign | bfloat16 | ok | 5.407 | 0.734 | 21.243 | 5.234 | 42.793 | 5.107 | 64.049 | 5.234 |
| large | FusedManifestoHardLUT weight=softsign | float32 | ok | 1.003 | 0.094 | 1.111 | 0.094 | 6.344 | 0.234 | 7.454 | 0.234 |
| large | FusedManifestoHardLUT weight=softsign | bfloat16 | ok | 0.767 | 0.047 | 0.884 | 0.047 | 5.951 | 0.234 | 6.834 | 0.234 |
| large | FusedManifestoSoftLUT weight=softsign | float32 | ok | 1.831 | 0.094 | 2.051 | 0.094 | 7.113 | 0.234 | 9.150 | 0.234 |
| large | FusedManifestoSoftLUT weight=softsign | bfloat16 | ok | 1.137 | 0.047 | 1.218 | 0.047 | 6.812 | 0.234 | 8.029 | 0.234 |
| large | ConfidenceLUT n=1 | float32 | ok | 3.585 | 0.469 | 3.701 | 0.469 | 9.531 | 2.703 | 13.233 | 2.703 |
| large | ConfidenceLUT n=1 | bfloat16 | ok | 3.705 | 0.609 | 3.755 | 0.656 | 9.602 | 2.844 | 13.355 | 2.844 |
| large | ConfidenceLUT n=2 | float32 | ok | 5.138 | 0.875 | 5.000 | 0.844 | 16.219 | 3.141 | 21.210 | 3.141 |
| large | ConfidenceLUT n=2 | bfloat16 | ok | 5.308 | 1.016 | 4.617 | 1.031 | 15.860 | 3.281 | 20.479 | 3.281 |
| large | FusedConfidenceLUT n=1 | float32 | ok | 1.035 | 0.094 | 1.143 | 0.094 | 4.749 | 0.236 | 5.892 | 0.236 |
| large | FusedConfidenceLUT n=1 | bfloat16 | ok | 0.825 | 0.047 | 0.987 | 0.047 | 4.478 | 0.236 | 5.467 | 0.236 |
| large | FusedConfidenceLUT n=2 | float32 | ok | 1.870 | 0.094 | 1.982 | 0.094 | 7.416 | 0.236 | 9.398 | 0.236 |
| large | FusedConfidenceLUT n=2 | bfloat16 | ok | 1.254 | 0.047 | 1.430 | 0.047 | 6.963 | 0.236 | 8.395 | 0.236 |
| large | QuantisedConfidenceLUT n=2 | float32 | ok | 9.987 | 0.895 | 11.548 | 1.922 | 37.756 | 4.169 | 49.311 | 4.169 |
| large | QuantisedConfidenceLUT n=2 | bfloat16 | refused: QuantisedConfidenceLUT does not support low precision | - | - | - | - | - | - | - | - |
| large | FusedQuantisedConfidenceLUT n=2 | float32 | ok | 1.949 | 0.094 | 2.047 | 0.094 | 8.071 | 0.236 | 10.118 | 0.236 |
| large | FusedQuantisedConfidenceLUT n=2 | bfloat16 | error: RuntimeError: FusedQuantisedConfidenceLUT: not a case the fused op covers: input dtype torch.bfloat16 (float32 only); table dtype torc | - | - | - | - | - | - | - | - |
| large | DeployedQuantisedConfidenceLUT | float32 | ok | 1.254 | 0.094 | - | - | - | - | - | - |
| large | DeployedQuantisedConfidenceLUT | bfloat16 | refused: DeployedQuantisedConfidenceLUT is fp32/fp64 inference only; got input dtype torch.bfloat16 | - | - | - | - | - | - | - | - |
| large | DeployedQuantisedConfidenceLUT n=1 | float32 | ok | 0.892 | 0.094 | - | - | - | - | - | - |
| large | DeployedQuantisedConfidenceLUT n=1 | bfloat16 | refused: DeployedQuantisedConfidenceLUT is fp32/fp64 inference only; got input dtype torch.bfloat16 | - | - | - | - | - | - | - | - |

> **Note (2026-10-13):** the benchmark modules have since moved: `spiky.lutorch_ex.bench_snapshot` is now `spiky.lutorch_ex.benchmarks.bench_snapshot` (likewise `bench`, `bench_compare`, `bench_fused_confidence`). The command above is recorded as it was run.
