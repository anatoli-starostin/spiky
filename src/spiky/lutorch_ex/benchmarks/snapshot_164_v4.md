# lutorch_ex performance snapshot: #164 HEAD, v4 (the within-branch pair with snapshot_164_prefusedn1_v3)

**Code.** `949d4921`, #164 HEAD (library code 0b3de742); no uncommitted changes in lutorch_ex. A fresh measurement of HEAD in this session: the v3 HEAD snapshot came from a different session and is not differenced against 3e8eec29. `snapshot_164_v3` stays the snapshot of record against #162 / #160.

**Hardware and software.** RTX 5090 (sm_120, 170 SMs), torch 2.9.1+cu130, Triton 3.5.1, CUDA 13.0, driver 595.84.

**Session.** Measured 2026-10-13 in ONE session with the other of the pair, interleaved cell by cell over HEAD's full cell matrix (18 cartridges x {fp32, bf16} x {8k, 32k}), alternating which tree went first. Champion geometry h16 tph64 nap8 d48 at 8,192 (`champion`) and 32,768 (`large`) vectors.

**Commands, as run** (one fresh process per cell; cwd the tree's `src/spiky/lutorch_ex`, `PYTHONPATH` its `src`): `python -m spiky.lutorch_ex.benchmarks.bench_snapshot --geometries G --cartridges NAME --dtypes D --out c.json --md c.md` (its default isolated mode: Dynamo reset, 10 warm-up + 30 timed iterations, median; peak GiB above the pre-step allocation). Both trees' `measure()` are the same code (the two bench_snapshot.py differ only by the FusedQuantised n=1 row and a comment).

**Conditions.** Logged before and after every cell (288 samples over 144 cells): SM up to 3.07 GHz (unlocked), 31-43 C. **No throttle or power-cap flag at any point** (the software power cap 0x4 did not trip on any sample).

Comparison: `figures_164_prefusedn1_vs_164_v3/` (bench_compare.py --same-session; rows only the after side has are drawn as NEW after-only bars).

| geometry | cartridge | dtype | status | eval forward ms | GiB | train forward ms | GiB | train backward ms | GiB | train total ms | GiB |
|---|---|---|---|---|---|---|---|---|---|---|---|
| champion | ManifestoHardLUT | float32 | ok | 0.659 | 0.055 | 5.936 | 1.273 | 8.956 | 1.047 | 14.893 | 1.273 |
| champion | ManifestoHardLUT | bfloat16 | ok | 0.701 | 0.125 | 6.000 | 1.344 | 8.947 | 1.082 | 14.946 | 1.344 |
| champion | ManifestoSoftLUT | float32 | ok | 1.218 | 0.148 | 5.047 | 1.273 | 10.208 | 1.265 | 15.256 | 1.273 |
| champion | ManifestoSoftLUT | bfloat16 | ok | 1.254 | 0.219 | 5.121 | 1.344 | 10.091 | 1.323 | 15.215 | 1.344 |
| champion | FusedManifestoHardLUT | float32 | ok | 0.252 | 0.023 | 0.311 | 0.023 | 1.318 | 0.094 | 1.619 | 0.094 |
| champion | FusedManifestoHardLUT | bfloat16 | ok | 0.192 | 0.012 | 0.253 | 0.012 | 1.192 | 0.094 | 1.436 | 0.094 |
| champion | FusedManifestoSoftLUT | float32 | ok | 0.463 | 0.023 | 0.529 | 0.023 | 1.711 | 0.094 | 2.244 | 0.094 |
| champion | FusedManifestoSoftLUT | bfloat16 | ok | 0.287 | 0.012 | 0.370 | 0.012 | 1.617 | 0.094 | 1.986 | 0.094 |
| champion | ManifestoHardLUT weight=softsign | float32 | ok | 0.643 | 0.055 | 5.970 | 1.273 | 9.295 | 1.047 | 15.264 | 1.273 |
| champion | ManifestoHardLUT weight=softsign | bfloat16 | ok | 0.688 | 0.125 | 6.025 | 1.344 | 9.241 | 1.082 | 15.267 | 1.344 |
| champion | ManifestoSoftLUT weight=softsign | float32 | ok | 1.282 | 0.148 | 5.074 | 1.273 | 10.526 | 1.327 | 15.598 | 1.327 |
| champion | ManifestoSoftLUT weight=softsign | bfloat16 | ok | 1.317 | 0.219 | 5.160 | 1.344 | 10.446 | 1.386 | 15.607 | 1.386 |
| champion | FusedManifestoHardLUT weight=softsign | float32 | ok | 0.265 | 0.023 | 0.341 | 0.023 | 1.589 | 0.094 | 1.919 | 0.094 |
| champion | FusedManifestoHardLUT weight=softsign | bfloat16 | ok | 0.214 | 0.012 | 0.293 | 0.012 | 1.490 | 0.094 | 1.782 | 0.094 |
| champion | FusedManifestoSoftLUT weight=softsign | float32 | ok | 0.476 | 0.023 | 0.570 | 0.023 | 1.861 | 0.094 | 2.424 | 0.094 |
| champion | FusedManifestoSoftLUT weight=softsign | bfloat16 | ok | 0.306 | 0.012 | 0.385 | 0.012 | 1.772 | 0.094 | 2.143 | 0.094 |
| champion | ConfidenceLUT n=1 | float32 | ok | 0.905 | 0.117 | 1.080 | 0.117 | 2.365 | 0.711 | 3.448 | 0.711 |
| champion | ConfidenceLUT n=1 | bfloat16 | ok | 0.941 | 0.188 | 1.073 | 0.199 | 2.374 | 0.746 | 3.446 | 0.746 |
| champion | ConfidenceLUT n=2 | float32 | ok | 1.240 | 0.219 | 1.320 | 0.211 | 4.044 | 0.820 | 5.362 | 0.820 |
| champion | ConfidenceLUT n=2 | bfloat16 | ok | 1.298 | 0.289 | 1.350 | 0.293 | 3.985 | 0.855 | 5.334 | 0.855 |
| champion | FusedConfidenceLUT n=1 | float32 | ok | 0.266 | 0.023 | 0.324 | 0.023 | 1.216 | 0.094 | 1.540 | 0.094 |
| champion | FusedConfidenceLUT n=1 | bfloat16 | ok | 0.213 | 0.012 | 0.282 | 0.012 | 1.152 | 0.094 | 1.427 | 0.094 |
| champion | FusedConfidenceLUT n=2 | float32 | ok | 0.474 | 0.023 | 0.542 | 0.023 | 1.927 | 0.094 | 2.469 | 0.094 |
| champion | FusedConfidenceLUT n=2 | bfloat16 | ok | 0.320 | 0.012 | 0.405 | 0.012 | 1.807 | 0.094 | 2.211 | 0.094 |
| champion | QuantisedConfidenceLUT n=1 | float32 | ok | 1.650 | 0.165 | 1.715 | 0.228 | 5.484 | 0.555 | 7.201 | 0.555 |
| champion | QuantisedConfidenceLUT n=1 | bfloat16 | refused: QuantisedConfidenceLUT does not support low precision | - | - | - | - | - | - | - | - |
| champion | QuantisedConfidenceLUT n=2 | float32 | ok | 2.571 | 0.259 | 3.026 | 0.516 | 9.421 | 1.116 | 12.446 | 1.116 |
| champion | QuantisedConfidenceLUT n=2 | bfloat16 | refused: QuantisedConfidenceLUT does not support low precision | - | - | - | - | - | - | - | - |
| champion | FusedQuantisedConfidenceLUT n=1 | float32 | ok | 0.303 | 0.023 | 0.388 | 0.055 | 1.192 | 0.125 | 1.589 | 0.125 |
| champion | FusedQuantisedConfidenceLUT n=1 | bfloat16 | error: RuntimeError: FusedQuantisedConfidenceLUT: not a case the fused op covers: input dtype torch.bfloat16 (float32 only); table dtype torc | - | - | - | - | - | - | - | - |
| champion | FusedQuantisedConfidenceLUT n=2 | float32 | ok | 0.519 | 0.023 | 0.602 | 0.055 | 1.999 | 0.125 | 2.600 | 0.125 |
| champion | FusedQuantisedConfidenceLUT n=2 | bfloat16 | error: RuntimeError: FusedQuantisedConfidenceLUT: not a case the fused op covers: input dtype torch.bfloat16 (float32 only); table dtype torc | - | - | - | - | - | - | - | - |
| champion | DeployedQuantisedConfidenceLUT | float32 | ok | 0.309 | 0.023 | - | - | - | - | - | - |
| champion | DeployedQuantisedConfidenceLUT | bfloat16 | refused: DeployedQuantisedConfidenceLUT is fp32/fp64 inference only; got input dtype torch.bfloat16 | - | - | - | - | - | - | - | - |
| champion | DeployedQuantisedConfidenceLUT n=1 | float32 | ok | 0.220 | 0.023 | - | - | - | - | - | - |
| champion | DeployedQuantisedConfidenceLUT n=1 | bfloat16 | refused: DeployedQuantisedConfidenceLUT is fp32/fp64 inference only; got input dtype torch.bfloat16 | - | - | - | - | - | - | - | - |
| large | ManifestoHardLUT | float32 | ok | 2.532 | 0.219 | 24.522 | 5.094 | 35.934 | 4.188 | 60.459 | 5.094 |
| large | ManifestoHardLUT | bfloat16 | ok | 2.673 | 0.359 | 24.723 | 5.234 | 35.948 | 4.188 | 60.671 | 5.234 |
| large | ManifestoSoftLUT | float32 | ok | 5.025 | 0.594 | 20.809 | 5.094 | 40.963 | 4.763 | 61.779 | 5.094 |
| large | ManifestoSoftLUT | bfloat16 | ok | 5.059 | 0.734 | 20.982 | 5.234 | 40.916 | 4.857 | 61.897 | 5.234 |
| large | FusedManifestoHardLUT | float32 | ok | 0.988 | 0.094 | 1.074 | 0.094 | 4.955 | 0.234 | 6.028 | 0.234 |
| large | FusedManifestoHardLUT | bfloat16 | ok | 0.744 | 0.047 | 0.843 | 0.047 | 4.593 | 0.234 | 5.425 | 0.234 |
| large | FusedManifestoSoftLUT | float32 | ok | 1.810 | 0.094 | 2.096 | 0.094 | 6.764 | 0.234 | 8.861 | 0.234 |
| large | FusedManifestoSoftLUT | bfloat16 | ok | 1.120 | 0.047 | 1.246 | 0.047 | 6.456 | 0.234 | 7.701 | 0.234 |
| large | ManifestoHardLUT weight=softsign | float32 | ok | 2.481 | 0.219 | 24.788 | 5.094 | 37.742 | 4.188 | 62.529 | 5.094 |
| large | ManifestoHardLUT weight=softsign | bfloat16 | ok | 2.613 | 0.359 | 24.990 | 5.234 | 37.798 | 4.188 | 62.791 | 5.234 |
| large | ManifestoSoftLUT weight=softsign | float32 | ok | 5.378 | 0.594 | 21.073 | 5.094 | 42.798 | 5.013 | 63.874 | 5.094 |
| large | ManifestoSoftLUT weight=softsign | bfloat16 | ok | 5.401 | 0.734 | 21.235 | 5.234 | 42.790 | 5.107 | 64.028 | 5.234 |
| large | FusedManifestoHardLUT weight=softsign | float32 | ok | 1.005 | 0.094 | 1.112 | 0.094 | 6.318 | 0.234 | 7.430 | 0.234 |
| large | FusedManifestoHardLUT weight=softsign | bfloat16 | ok | 0.766 | 0.047 | 0.885 | 0.047 | 5.941 | 0.234 | 6.827 | 0.234 |
| large | FusedManifestoSoftLUT weight=softsign | float32 | ok | 1.835 | 0.094 | 2.024 | 0.094 | 7.051 | 0.234 | 9.073 | 0.234 |
| large | FusedManifestoSoftLUT weight=softsign | bfloat16 | ok | 1.163 | 0.047 | 1.238 | 0.047 | 6.791 | 0.234 | 8.028 | 0.234 |
| large | ConfidenceLUT n=1 | float32 | ok | 3.568 | 0.469 | 3.708 | 0.469 | 9.556 | 2.703 | 13.271 | 2.703 |
| large | ConfidenceLUT n=1 | bfloat16 | ok | 3.724 | 0.609 | 3.825 | 0.656 | 9.551 | 2.844 | 13.384 | 2.844 |
| large | ConfidenceLUT n=2 | float32 | ok | 4.861 | 0.875 | 4.612 | 0.844 | 16.196 | 3.141 | 20.817 | 3.141 |
| large | ConfidenceLUT n=2 | bfloat16 | ok | 4.920 | 1.016 | 4.839 | 1.031 | 15.945 | 3.281 | 20.784 | 3.281 |
| large | FusedConfidenceLUT n=1 | float32 | ok | 1.049 | 0.094 | 1.139 | 0.094 | 4.746 | 0.236 | 5.884 | 0.236 |
| large | FusedConfidenceLUT n=1 | bfloat16 | ok | 0.824 | 0.047 | 0.985 | 0.047 | 4.479 | 0.236 | 5.465 | 0.236 |
| large | FusedConfidenceLUT n=2 | float32 | ok | 1.878 | 0.094 | 1.980 | 0.094 | 7.355 | 0.236 | 9.335 | 0.236 |
| large | FusedConfidenceLUT n=2 | bfloat16 | ok | 1.226 | 0.047 | 1.436 | 0.047 | 6.981 | 0.236 | 8.402 | 0.236 |
| large | QuantisedConfidenceLUT n=1 | float32 | ok | 6.188 | 0.520 | 6.253 | 0.770 | 21.345 | 1.953 | 27.603 | 1.953 |
| large | QuantisedConfidenceLUT n=1 | bfloat16 | refused: QuantisedConfidenceLUT does not support low precision | - | - | - | - | - | - | - | - |
| large | QuantisedConfidenceLUT n=2 | float32 | ok | 9.954 | 0.895 | 11.540 | 1.922 | 37.780 | 4.169 | 49.321 | 4.169 |
| large | QuantisedConfidenceLUT n=2 | bfloat16 | refused: QuantisedConfidenceLUT does not support low precision | - | - | - | - | - | - | - | - |
| large | FusedQuantisedConfidenceLUT n=1 | float32 | ok | 1.205 | 0.094 | 1.367 | 0.219 | 4.418 | 0.361 | 5.794 | 0.361 |
| large | FusedQuantisedConfidenceLUT n=1 | bfloat16 | error: RuntimeError: FusedQuantisedConfidenceLUT: not a case the fused op covers: input dtype torch.bfloat16 (float32 only); table dtype torc | - | - | - | - | - | - | - | - |
| large | FusedQuantisedConfidenceLUT n=2 | float32 | ok | 1.951 | 0.094 | 2.073 | 0.219 | 7.693 | 0.361 | 9.767 | 0.361 |
| large | FusedQuantisedConfidenceLUT n=2 | bfloat16 | error: RuntimeError: FusedQuantisedConfidenceLUT: not a case the fused op covers: input dtype torch.bfloat16 (float32 only); table dtype torc | - | - | - | - | - | - | - | - |
| large | DeployedQuantisedConfidenceLUT | float32 | ok | 1.252 | 0.094 | - | - | - | - | - | - |
| large | DeployedQuantisedConfidenceLUT | bfloat16 | refused: DeployedQuantisedConfidenceLUT is fp32/fp64 inference only; got input dtype torch.bfloat16 | - | - | - | - | - | - | - | - |
| large | DeployedQuantisedConfidenceLUT n=1 | float32 | ok | 0.891 | 0.094 | - | - | - | - | - | - |
| large | DeployedQuantisedConfidenceLUT n=1 | bfloat16 | refused: DeployedQuantisedConfidenceLUT is fp32/fp64 inference only; got input dtype torch.bfloat16 | - | - | - | - | - | - | - | - |
