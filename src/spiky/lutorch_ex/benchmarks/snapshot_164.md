# lutorch_ex performance snapshot of record (#164)

**Code.** `a251a82f`, #164 HEAD; no uncommitted changes in lutorch_ex.

**Hardware and software.** RTX 5090 (sm_120, 170 SMs), torch 2.9.1+cu130, Triton 3.5.1, CUDA 13.0, driver 595.84.

**Session.** Measured 2026-10-13 in ONE session together with snapshot_160_remeasured (#160's code), interleaved cell by cell.
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
| champion | ManifestoHardLUT | float32 | ok | 0.660 | 0.055 | 5.944 | 1.273 | 8.982 | 1.047 | 14.927 | 1.273 |
| champion | ManifestoHardLUT | bfloat16 | ok | 0.702 | 0.125 | 5.996 | 1.344 | 8.947 | 1.082 | 14.946 | 1.344 |
| champion | ManifestoSoftLUT | float32 | ok | 1.218 | 0.148 | 5.045 | 1.273 | 10.204 | 1.265 | 15.250 | 1.273 |
| champion | ManifestoSoftLUT | bfloat16 | ok | 1.252 | 0.219 | 5.120 | 1.344 | 10.120 | 1.323 | 15.239 | 1.344 |
| champion | FusedManifestoHardLUT | float32 | ok | 0.254 | 0.023 | 0.296 | 0.023 | 1.317 | 0.094 | 1.608 | 0.094 |
| champion | FusedManifestoHardLUT | bfloat16 | ok | 0.192 | 0.012 | 0.230 | 0.012 | 1.186 | 0.094 | 1.421 | 0.094 |
| champion | FusedManifestoSoftLUT | float32 | ok | 0.463 | 0.023 | 0.544 | 0.023 | 1.730 | 0.094 | 2.279 | 0.094 |
| champion | FusedManifestoSoftLUT | bfloat16 | ok | 0.286 | 0.012 | 0.345 | 0.012 | 1.580 | 0.094 | 1.933 | 0.094 |
| champion | ManifestoHardLUT weight=softsign | float32 | ok | 0.644 | 0.055 | 5.968 | 1.273 | 9.299 | 1.047 | 15.266 | 1.273 |
| champion | ManifestoHardLUT weight=softsign | bfloat16 | ok | 0.690 | 0.125 | 6.021 | 1.344 | 9.235 | 1.082 | 15.258 | 1.344 |
| champion | ManifestoSoftLUT weight=softsign | float32 | ok | 1.286 | 0.148 | 5.074 | 1.273 | 10.526 | 1.327 | 15.602 | 1.327 |
| champion | ManifestoSoftLUT weight=softsign | bfloat16 | ok | 1.315 | 0.219 | 5.157 | 1.344 | 10.441 | 1.386 | 15.601 | 1.386 |
| champion | FusedManifestoHardLUT weight=softsign | float32 | ok | 0.269 | 0.023 | 0.343 | 0.023 | 1.595 | 0.094 | 1.938 | 0.094 |
| champion | FusedManifestoHardLUT weight=softsign | bfloat16 | ok | 0.213 | 0.012 | 0.292 | 0.012 | 1.494 | 0.094 | 1.786 | 0.094 |
| champion | FusedManifestoSoftLUT weight=softsign | float32 | ok | 0.476 | 0.023 | 0.568 | 0.023 | 1.855 | 0.094 | 2.416 | 0.094 |
| champion | FusedManifestoSoftLUT weight=softsign | bfloat16 | ok | 0.311 | 0.012 | 0.390 | 0.012 | 1.780 | 0.094 | 2.132 | 0.094 |
| champion | ConfidenceLUT n=1 | float32 | ok | 0.904 | 0.117 | 1.075 | 0.117 | 2.369 | 0.711 | 3.444 | 0.711 |
| champion | ConfidenceLUT n=1 | bfloat16 | ok | 0.942 | 0.188 | 1.075 | 0.199 | 2.368 | 0.746 | 3.442 | 0.746 |
| champion | ConfidenceLUT n=2 | float32 | ok | 1.300 | 0.219 | 1.356 | 0.211 | 4.069 | 0.820 | 5.425 | 0.820 |
| champion | ConfidenceLUT n=2 | bfloat16 | ok | 1.279 | 0.289 | 1.313 | 0.293 | 4.013 | 0.855 | 5.320 | 0.855 |
| champion | FusedConfidenceLUT n=1 | float32 | ok | 0.266 | 0.023 | 0.328 | 0.023 | 1.223 | 0.094 | 1.549 | 0.094 |
| champion | FusedConfidenceLUT n=1 | bfloat16 | ok | 0.213 | 0.012 | 0.289 | 0.012 | 1.150 | 0.094 | 1.439 | 0.094 |
| champion | FusedConfidenceLUT n=2 | float32 | ok | 0.473 | 0.023 | 0.524 | 0.023 | 1.945 | 0.094 | 2.477 | 0.094 |
| champion | FusedConfidenceLUT n=2 | bfloat16 | ok | 0.317 | 0.012 | 0.407 | 0.012 | 1.833 | 0.094 | 2.226 | 0.094 |
| champion | QuantisedConfidenceLUT n=2 | float32 | ok | 2.573 | 0.259 | 3.032 | 0.516 | 9.433 | 1.116 | 12.464 | 1.116 |
| champion | QuantisedConfidenceLUT n=2 | bfloat16 | refused: QuantisedConfidenceLUT does not support low precision | - | - | - | - | - | - | - | - |
| champion | FusedQuantisedConfidenceLUT n=2 | float32 | ok | 0.516 | 0.023 | 0.591 | 0.023 | 2.100 | 0.094 | 2.691 | 0.094 |
| champion | FusedQuantisedConfidenceLUT n=2 | bfloat16 | error: RuntimeError: FusedQuantisedConfidenceLUT: not a case the fused op covers: input dtype torch.bfloat16 (float32 only); table dtype torc | - | - | - | - | - | - | - | - |
| champion | DeployedQuantisedConfidenceLUT | float32 | ok | 0.310 | 0.023 | - | - | - | - | - | - |
| champion | DeployedQuantisedConfidenceLUT | bfloat16 | refused: DeployedQuantisedConfidenceLUT is fp32/fp64 inference only; got input dtype torch.bfloat16 | - | - | - | - | - | - | - | - |
| champion | DeployedQuantisedConfidenceLUT n=1 | float32 | ok | 0.221 | 0.023 | - | - | - | - | - | - |
| champion | DeployedQuantisedConfidenceLUT n=1 | bfloat16 | refused: DeployedQuantisedConfidenceLUT is fp32/fp64 inference only; got input dtype torch.bfloat16 | - | - | - | - | - | - | - | - |
| large | ManifestoHardLUT | float32 | ok | 2.536 | 0.219 | 24.528 | 5.094 | 35.943 | 4.188 | 60.472 | 5.094 |
| large | ManifestoHardLUT | bfloat16 | ok | 2.678 | 0.359 | 24.740 | 5.234 | 35.968 | 4.188 | 60.704 | 5.234 |
| large | ManifestoSoftLUT | float32 | ok | 5.021 | 0.594 | 20.807 | 5.094 | 40.984 | 4.763 | 61.791 | 5.094 |
| large | ManifestoSoftLUT | bfloat16 | ok | 5.066 | 0.734 | 20.982 | 5.234 | 40.929 | 4.857 | 61.913 | 5.234 |
| large | FusedManifestoHardLUT | float32 | ok | 0.997 | 0.094 | 1.078 | 0.094 | 4.941 | 0.234 | 6.020 | 0.234 |
| large | FusedManifestoHardLUT | bfloat16 | ok | 0.744 | 0.047 | 0.849 | 0.047 | 4.595 | 0.234 | 5.446 | 0.234 |
| large | FusedManifestoSoftLUT | float32 | ok | 1.811 | 0.094 | 2.108 | 0.094 | 6.802 | 0.234 | 8.912 | 0.234 |
| large | FusedManifestoSoftLUT | bfloat16 | ok | 1.118 | 0.047 | 1.227 | 0.047 | 6.415 | 0.234 | 7.635 | 0.234 |
| large | ManifestoHardLUT weight=softsign | float32 | ok | 2.480 | 0.219 | 24.781 | 5.094 | 37.730 | 4.188 | 62.516 | 5.094 |
| large | ManifestoHardLUT weight=softsign | bfloat16 | ok | 2.619 | 0.359 | 24.989 | 5.234 | 37.765 | 4.188 | 62.757 | 5.234 |
| large | ManifestoSoftLUT weight=softsign | float32 | ok | 5.371 | 0.594 | 21.070 | 5.094 | 42.781 | 5.013 | 63.851 | 5.094 |
| large | ManifestoSoftLUT weight=softsign | bfloat16 | ok | 5.401 | 0.734 | 21.239 | 5.234 | 42.781 | 5.107 | 64.026 | 5.234 |
| large | FusedManifestoHardLUT weight=softsign | float32 | ok | 1.002 | 0.094 | 1.110 | 0.094 | 6.333 | 0.234 | 7.443 | 0.234 |
| large | FusedManifestoHardLUT weight=softsign | bfloat16 | ok | 0.765 | 0.047 | 0.883 | 0.047 | 5.948 | 0.234 | 6.832 | 0.234 |
| large | FusedManifestoSoftLUT weight=softsign | float32 | ok | 1.830 | 0.094 | 2.046 | 0.094 | 7.102 | 0.234 | 9.137 | 0.234 |
| large | FusedManifestoSoftLUT weight=softsign | bfloat16 | ok | 1.145 | 0.047 | 1.219 | 0.047 | 6.803 | 0.234 | 8.023 | 0.234 |
| large | ConfidenceLUT n=1 | float32 | ok | 3.618 | 0.469 | 3.740 | 0.469 | 9.539 | 2.703 | 13.280 | 2.703 |
| large | ConfidenceLUT n=1 | bfloat16 | ok | 3.690 | 0.609 | 3.749 | 0.656 | 9.602 | 2.844 | 13.356 | 2.844 |
| large | ConfidenceLUT n=2 | float32 | ok | 5.316 | 0.875 | 4.993 | 0.844 | 16.147 | 3.141 | 21.139 | 3.141 |
| large | ConfidenceLUT n=2 | bfloat16 | ok | 5.296 | 1.016 | 4.618 | 1.031 | 15.863 | 3.281 | 20.478 | 3.281 |
| large | FusedConfidenceLUT n=1 | float32 | ok | 1.042 | 0.094 | 1.146 | 0.094 | 4.752 | 0.236 | 5.900 | 0.236 |
| large | FusedConfidenceLUT n=1 | bfloat16 | ok | 0.828 | 0.047 | 0.988 | 0.047 | 4.486 | 0.236 | 5.473 | 0.236 |
| large | FusedConfidenceLUT n=2 | float32 | ok | 1.870 | 0.094 | 1.975 | 0.094 | 7.394 | 0.236 | 9.370 | 0.236 |
| large | FusedConfidenceLUT n=2 | bfloat16 | ok | 1.230 | 0.047 | 1.426 | 0.047 | 6.969 | 0.236 | 8.399 | 0.236 |
| large | QuantisedConfidenceLUT n=2 | float32 | ok | 10.008 | 0.895 | 11.547 | 1.922 | 37.756 | 4.169 | 49.300 | 4.169 |
| large | QuantisedConfidenceLUT n=2 | bfloat16 | refused: QuantisedConfidenceLUT does not support low precision | - | - | - | - | - | - | - | - |
| large | FusedQuantisedConfidenceLUT n=2 | float32 | ok | 1.921 | 0.094 | 2.049 | 0.094 | 8.095 | 0.236 | 10.142 | 0.236 |
| large | FusedQuantisedConfidenceLUT n=2 | bfloat16 | error: RuntimeError: FusedQuantisedConfidenceLUT: not a case the fused op covers: input dtype torch.bfloat16 (float32 only); table dtype torc | - | - | - | - | - | - | - | - |
| large | DeployedQuantisedConfidenceLUT | float32 | ok | 1.252 | 0.094 | - | - | - | - | - | - |
| large | DeployedQuantisedConfidenceLUT | bfloat16 | refused: DeployedQuantisedConfidenceLUT is fp32/fp64 inference only; got input dtype torch.bfloat16 | - | - | - | - | - | - | - | - |
| large | DeployedQuantisedConfidenceLUT n=1 | float32 | ok | 0.892 | 0.094 | - | - | - | - | - | - |
| large | DeployedQuantisedConfidenceLUT n=1 | bfloat16 | refused: DeployedQuantisedConfidenceLUT is fp32/fp64 inference only; got input dtype torch.bfloat16 | - | - | - | - | - | - | - | - |
