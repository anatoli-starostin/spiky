# lutorch_ex performance snapshot: #160 re-measured (the BEFORE of #164)

**Code.** `a811390e`, #160's library code: the commit snapshot_160.json was measured at, materialised from git, nothing changed. This is the authoritative before-side for #164's deltas; snapshot_160.json itself is from an older session.

**Hardware and software.** RTX 5090 (sm_120, 170 SMs), torch 2.9.1+cu130, Triton 3.5.1, CUDA 13.0, driver 595.84.

**Session.** Measured 2026-10-13 in ONE session together with snapshot_164, interleaved cell by cell.
- Each cell ran under each tree's own `python -m spiky.lutorch_ex.bench_snapshot` for that one cell, the established procedure: one fresh process per cell, a Dynamo reset, 10 warm-up + 30 timed iterations, median; peak GiB above the pre-step allocation.
- The two trees alternated which went first.
- Champion geometry h16 tph64 nap8 d48 at 8,192 vectors (`champion`) and 32,768 (`large`).

**Conditions** (logged between every cell):
- SM clock 2.65–3.07 GHz (unlocked), 33–42 C.
- **No throttle or power-cap flag at any point.**
- Other processes on the card: desktop only, ~180 MiB.

**One re-measured row.** The first pass recorded ConfidenceLUT n=2 champion eval at 1.743 ms (fp32) and 1.914 ms (bf16) on the #164 side.
- Three further interleaved fresh-process rounds gave 1.28–1.31 ms for #160, for 41df6f25 and for #164 alike, so that first pass was a transient.
- That whole row (both dtypes, both trees) was re-measured once more, interleaved, and replaced in both files: #164 eval 1.300 / 1.279, #160 1.281 / 1.280.

Comparison: `figures_160_vs_164/` (bench_compare.py --same-session).

| geometry | cartridge | dtype | status | eval forward ms | GiB | train forward ms | GiB | train backward ms | GiB | train total ms | GiB |
|---|---|---|---|---|---|---|---|---|---|---|---|
| champion | ManifestoHardLUT | float32 | ok | 0.666 | 0.055 | 17.923 | 6.712 | 7.331 | 3.517 | 25.253 | 6.712 |
| champion | ManifestoHardLUT | bfloat16 | ok | 0.703 | 0.125 | 17.979 | 6.782 | 7.331 | 3.528 | 25.311 | 6.782 |
| champion | ManifestoSoftLUT | float32 | ok | 1.228 | 0.148 | 5.044 | 1.273 | 10.210 | 1.265 | 15.256 | 1.273 |
| champion | ManifestoSoftLUT | bfloat16 | ok | 1.251 | 0.219 | 5.129 | 1.344 | 10.124 | 1.323 | 15.249 | 1.344 |
| champion | FusedManifestoHardLUT | float32 | ok | 0.254 | 0.023 | 0.305 | 0.023 | 1.266 | 0.094 | 1.565 | 0.094 |
| champion | FusedManifestoHardLUT | bfloat16 | ok | 0.193 | 0.012 | 0.254 | 0.012 | 1.165 | 0.094 | 1.419 | 0.094 |
| champion | FusedManifestoSoftLUT | float32 | ok | 0.462 | 0.023 | 0.551 | 0.023 | 1.735 | 0.094 | 2.290 | 0.094 |
| champion | FusedManifestoSoftLUT | bfloat16 | ok | 0.281 | 0.012 | 0.355 | 0.012 | 1.617 | 0.094 | 1.971 | 0.094 |
| champion | ManifestoHardLUT weight=softsign | float32 | ok | 0.651 | 0.055 | 17.960 | 6.743 | 7.657 | 3.579 | 25.619 | 6.743 |
| champion | ManifestoHardLUT weight=softsign | bfloat16 | ok | 0.689 | 0.125 | 17.994 | 6.813 | 7.655 | 3.591 | 25.651 | 6.813 |
| champion | ManifestoSoftLUT weight=softsign | float32 | ok | 1.293 | 0.148 | 5.076 | 1.273 | 10.528 | 1.327 | 15.606 | 1.327 |
| champion | ManifestoSoftLUT weight=softsign | bfloat16 | ok | 1.317 | 0.219 | 5.159 | 1.344 | 10.445 | 1.386 | 15.607 | 1.386 |
| champion | FusedManifestoHardLUT weight=softsign | float32 | ok | 0.267 | 0.023 | 0.332 | 0.023 | 1.595 | 0.094 | 1.914 | 0.094 |
| champion | FusedManifestoHardLUT weight=softsign | bfloat16 | ok | 0.214 | 0.012 | 0.295 | 0.012 | 1.491 | 0.094 | 1.785 | 0.094 |
| champion | FusedManifestoSoftLUT weight=softsign | float32 | ok | 0.478 | 0.023 | 0.582 | 0.023 | 1.889 | 0.094 | 2.456 | 0.094 |
| champion | FusedManifestoSoftLUT weight=softsign | bfloat16 | ok | 0.310 | 0.012 | 0.391 | 0.012 | 1.756 | 0.094 | 2.121 | 0.094 |
| champion | ConfidenceLUT n=1 | float32 | ok | 0.907 | 0.117 | 1.074 | 0.117 | 2.386 | 0.711 | 3.460 | 0.711 |
| champion | ConfidenceLUT n=1 | bfloat16 | ok | 0.942 | 0.188 | 1.069 | 0.199 | 2.392 | 0.746 | 3.462 | 0.746 |
| champion | ConfidenceLUT n=2 | float32 | ok | 1.281 | 0.219 | 1.357 | 0.211 | 4.106 | 0.820 | 5.462 | 0.820 |
| champion | ConfidenceLUT n=2 | bfloat16 | ok | 1.280 | 0.289 | 1.319 | 0.293 | 4.055 | 0.855 | 5.373 | 0.855 |
| champion | FusedConfidenceLUT n=1 | float32 | ok | 0.267 | 0.023 | 0.305 | 0.023 | 1.231 | 0.094 | 1.545 | 0.094 |
| champion | FusedConfidenceLUT n=1 | bfloat16 | ok | 0.204 | 0.012 | 0.272 | 0.012 | 1.143 | 0.094 | 1.414 | 0.094 |
| champion | FusedConfidenceLUT n=2 | float32 | ok | 0.472 | 0.023 | 0.536 | 0.023 | 1.917 | 0.094 | 2.453 | 0.094 |
| champion | FusedConfidenceLUT n=2 | bfloat16 | ok | 0.313 | 0.012 | 0.379 | 0.012 | 1.814 | 0.094 | 2.189 | 0.094 |
| champion | QuantisedConfidenceLUT n=2 | float32 | ok | 2.964 | 0.508 | 2.983 | 0.617 | 9.598 | 1.116 | 12.581 | 1.116 |
| champion | QuantisedConfidenceLUT n=2 | bfloat16 | refused: QuantisedConfidenceLUT does not support low precision | - | - | - | - | - | - | - | - |
| champion | FusedQuantisedConfidenceLUT n=2 | float32 | ok | 2.519 | 0.415 | 2.501 | 0.352 | 9.892 | 0.946 | 12.393 | 0.946 |
| champion | FusedQuantisedConfidenceLUT n=2 | bfloat16 | error: RuntimeError: FusedQuantisedConfidenceLUT: not a case the fused op covers: input dtype torch.bfloat16 (float32 only); table dtype torc | - | - | - | - | - | - | - | - |
| champion | DeployedQuantisedConfidenceLUT | float32 | ok | 2.883 | 0.414 | - | - | - | - | - | - |
| champion | DeployedQuantisedConfidenceLUT | bfloat16 | refused: DeployedQuantisedConfidenceLUT is fp32/fp64 inference only; got input dtype torch.bfloat16 | - | - | - | - | - | - | - | - |
| large | ManifestoHardLUT | float32 | out of memory | 2.544 | 0.219 | - | - | - | - | - | - |
| large | ManifestoHardLUT | bfloat16 | out of memory | 2.681 | 0.359 | - | - | - | - | - | - |
| large | ManifestoSoftLUT | float32 | ok | 5.019 | 0.594 | 20.801 | 5.094 | 40.969 | 4.763 | 61.772 | 5.094 |
| large | ManifestoSoftLUT | bfloat16 | ok | 5.061 | 0.734 | 20.983 | 5.234 | 40.953 | 4.857 | 61.938 | 5.234 |
| large | FusedManifestoHardLUT | float32 | ok | 0.997 | 0.094 | 1.083 | 0.094 | 4.885 | 0.234 | 5.968 | 0.234 |
| large | FusedManifestoHardLUT | bfloat16 | ok | 0.749 | 0.047 | 0.858 | 0.047 | 4.481 | 0.234 | 5.338 | 0.234 |
| large | FusedManifestoSoftLUT | float32 | ok | 1.812 | 0.094 | 2.102 | 0.094 | 6.783 | 0.234 | 8.895 | 0.234 |
| large | FusedManifestoSoftLUT | bfloat16 | ok | 1.119 | 0.047 | 1.219 | 0.047 | 6.417 | 0.234 | 7.637 | 0.234 |
| large | ManifestoHardLUT weight=softsign | float32 | out of memory | 2.492 | 0.219 | - | - | - | - | - | - |
| large | ManifestoHardLUT weight=softsign | bfloat16 | out of memory | 2.622 | 0.359 | - | - | - | - | - | - |
| large | ManifestoSoftLUT weight=softsign | float32 | ok | 5.349 | 0.594 | 21.067 | 5.094 | 42.782 | 5.013 | 63.851 | 5.094 |
| large | ManifestoSoftLUT weight=softsign | bfloat16 | ok | 5.394 | 0.734 | 21.239 | 5.234 | 42.784 | 5.107 | 64.022 | 5.234 |
| large | FusedManifestoHardLUT weight=softsign | float32 | ok | 1.005 | 0.094 | 1.114 | 0.094 | 6.348 | 0.234 | 7.457 | 0.234 |
| large | FusedManifestoHardLUT weight=softsign | bfloat16 | ok | 0.769 | 0.047 | 0.888 | 0.047 | 5.947 | 0.234 | 6.838 | 0.234 |
| large | FusedManifestoSoftLUT weight=softsign | float32 | ok | 1.831 | 0.094 | 2.040 | 0.094 | 7.076 | 0.234 | 9.117 | 0.234 |
| large | FusedManifestoSoftLUT weight=softsign | bfloat16 | ok | 1.146 | 0.047 | 1.220 | 0.047 | 6.794 | 0.234 | 8.016 | 0.234 |
| large | ConfidenceLUT n=1 | float32 | ok | 3.625 | 0.469 | 3.775 | 0.469 | 9.658 | 2.703 | 13.440 | 2.703 |
| large | ConfidenceLUT n=1 | bfloat16 | ok | 3.701 | 0.609 | 3.780 | 0.656 | 9.713 | 2.844 | 13.496 | 2.844 |
| large | ConfidenceLUT n=2 | float32 | ok | 5.140 | 0.875 | 5.050 | 0.844 | 16.161 | 3.141 | 21.216 | 3.141 |
| large | ConfidenceLUT n=2 | bfloat16 | ok | 5.293 | 1.016 | 4.631 | 1.031 | 15.861 | 3.281 | 20.491 | 3.281 |
| large | FusedConfidenceLUT n=1 | float32 | ok | 1.041 | 0.094 | 1.139 | 0.094 | 4.761 | 0.236 | 5.904 | 0.236 |
| large | FusedConfidenceLUT n=1 | bfloat16 | ok | 0.798 | 0.047 | 0.967 | 0.047 | 4.489 | 0.236 | 5.458 | 0.236 |
| large | FusedConfidenceLUT n=2 | float32 | ok | 1.866 | 0.094 | 1.977 | 0.094 | 7.421 | 0.236 | 9.398 | 0.236 |
| large | FusedConfidenceLUT n=2 | bfloat16 | ok | 1.208 | 0.047 | 1.380 | 0.047 | 6.963 | 0.236 | 8.345 | 0.236 |
| large | QuantisedConfidenceLUT n=2 | float32 | ok | 11.391 | 2.031 | 11.223 | 2.328 | 38.377 | 4.169 | 49.603 | 4.169 |
| large | QuantisedConfidenceLUT n=2 | bfloat16 | refused: QuantisedConfidenceLUT does not support low precision | - | - | - | - | - | - | - | - |
| large | FusedQuantisedConfidenceLUT n=2 | float32 | ok | 9.668 | 1.658 | 9.408 | 1.406 | 39.060 | 3.641 | 48.472 | 3.641 |
| large | FusedQuantisedConfidenceLUT n=2 | bfloat16 | error: RuntimeError: FusedQuantisedConfidenceLUT: not a case the fused op covers: input dtype torch.bfloat16 (float32 only); table dtype torc | - | - | - | - | - | - | - | - |
| large | DeployedQuantisedConfidenceLUT | float32 | ok | 16.804 | 1.656 | - | - | - | - | - | - |
| large | DeployedQuantisedConfidenceLUT | bfloat16 | refused: DeployedQuantisedConfidenceLUT is fp32/fp64 inference only; got input dtype torch.bfloat16 | - | - | - | - | - | - | - | - |
