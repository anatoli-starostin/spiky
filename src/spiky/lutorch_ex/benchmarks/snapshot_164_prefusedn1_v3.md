# lutorch_ex performance snapshot: #164 before the fused n=1 work (3e8eec29), paired with snapshot_164_v4

**Code.** `3e8eec29`, #164 one commit before 0b3de742 (FusedQuantisedConfidenceLUT read_top_n=1). Materialised from git, nothing changed. FusedQuantisedConfidenceLUT n=1 does not exist here (it raised at the forward): no row.

**Hardware and software.** RTX 5090 (sm_120, 170 SMs), torch 2.9.1+cu130, Triton 3.5.1, CUDA 13.0, driver 595.84.

**Session.** Measured 2026-10-13 in ONE session with the other of the pair, interleaved cell by cell over HEAD's full cell matrix (18 cartridges x {fp32, bf16} x {8k, 32k}), alternating which tree went first. Champion geometry h16 tph64 nap8 d48 at 8,192 (`champion`) and 32,768 (`large`) vectors.

**Commands, as run** (one fresh process per cell; cwd the tree's `src/spiky/lutorch_ex`, `PYTHONPATH` its `src`): `python -m spiky.lutorch_ex.benchmarks.bench_snapshot --geometries G --cartridges NAME --dtypes D --out c.json --md c.md` (its default isolated mode: Dynamo reset, 10 warm-up + 30 timed iterations, median; peak GiB above the pre-step allocation). Both trees' `measure()` are the same code (the two bench_snapshot.py differ only by the FusedQuantised n=1 row and a comment).

**Conditions.** Logged before and after every cell (288 samples over 144 cells): SM up to 3.07 GHz (unlocked), 31-43 C. **No throttle or power-cap flag at any point** (the software power cap 0x4 did not trip on any sample).

Comparison: `figures_164_prefusedn1_vs_164_v3/` (bench_compare.py --same-session; rows only the after side has are drawn as NEW after-only bars).

| geometry | cartridge | dtype | status | eval forward ms | GiB | train forward ms | GiB | train backward ms | GiB | train total ms | GiB |
|---|---|---|---|---|---|---|---|---|---|---|---|
| champion | ManifestoHardLUT | float32 | ok | 0.659 | 0.055 | 5.936 | 1.273 | 8.960 | 1.047 | 14.897 | 1.273 |
| champion | ManifestoHardLUT | bfloat16 | ok | 0.702 | 0.125 | 5.999 | 1.344 | 8.947 | 1.082 | 14.949 | 1.344 |
| champion | ManifestoSoftLUT | float32 | ok | 1.220 | 0.148 | 5.054 | 1.273 | 10.212 | 1.265 | 15.265 | 1.273 |
| champion | ManifestoSoftLUT | bfloat16 | ok | 1.253 | 0.219 | 5.123 | 1.344 | 10.093 | 1.323 | 15.220 | 1.344 |
| champion | FusedManifestoHardLUT | float32 | ok | 0.252 | 0.023 | 0.292 | 0.023 | 1.346 | 0.094 | 1.637 | 0.094 |
| champion | FusedManifestoHardLUT | bfloat16 | ok | 0.191 | 0.012 | 0.251 | 0.012 | 1.194 | 0.094 | 1.434 | 0.094 |
| champion | FusedManifestoSoftLUT | float32 | ok | 0.464 | 0.023 | 0.551 | 0.023 | 1.733 | 0.094 | 2.284 | 0.094 |
| champion | FusedManifestoSoftLUT | bfloat16 | ok | 0.288 | 0.012 | 0.350 | 0.012 | 1.617 | 0.094 | 1.987 | 0.094 |
| champion | ManifestoHardLUT weight=softsign | float32 | ok | 0.643 | 0.055 | 5.967 | 1.273 | 9.296 | 1.047 | 15.262 | 1.273 |
| champion | ManifestoHardLUT weight=softsign | bfloat16 | ok | 0.688 | 0.125 | 6.025 | 1.344 | 9.242 | 1.082 | 15.267 | 1.344 |
| champion | ManifestoSoftLUT weight=softsign | float32 | ok | 1.283 | 0.148 | 5.075 | 1.273 | 10.527 | 1.327 | 15.603 | 1.327 |
| champion | ManifestoSoftLUT weight=softsign | bfloat16 | ok | 1.316 | 0.219 | 5.157 | 1.344 | 10.442 | 1.386 | 15.601 | 1.386 |
| champion | FusedManifestoHardLUT weight=softsign | float32 | ok | 0.267 | 0.023 | 0.310 | 0.023 | 1.587 | 0.094 | 1.895 | 0.094 |
| champion | FusedManifestoHardLUT weight=softsign | bfloat16 | ok | 0.214 | 0.012 | 0.292 | 0.012 | 1.491 | 0.094 | 1.783 | 0.094 |
| champion | FusedManifestoSoftLUT weight=softsign | float32 | ok | 0.477 | 0.023 | 0.572 | 0.023 | 1.845 | 0.094 | 2.404 | 0.094 |
| champion | FusedManifestoSoftLUT weight=softsign | bfloat16 | ok | 0.306 | 0.012 | 0.381 | 0.012 | 1.782 | 0.094 | 2.127 | 0.094 |
| champion | ConfidenceLUT n=1 | float32 | ok | 0.903 | 0.117 | 1.064 | 0.117 | 2.365 | 0.711 | 3.428 | 0.711 |
| champion | ConfidenceLUT n=1 | bfloat16 | ok | 0.942 | 0.188 | 1.076 | 0.199 | 2.349 | 0.746 | 3.424 | 0.746 |
| champion | ConfidenceLUT n=2 | float32 | ok | 1.264 | 0.219 | 1.326 | 0.211 | 4.045 | 0.820 | 5.371 | 0.820 |
| champion | ConfidenceLUT n=2 | bfloat16 | ok | 1.297 | 0.289 | 1.357 | 0.293 | 3.986 | 0.855 | 5.342 | 0.855 |
| champion | FusedConfidenceLUT n=1 | float32 | ok | 0.267 | 0.023 | 0.328 | 0.023 | 1.219 | 0.094 | 1.547 | 0.094 |
| champion | FusedConfidenceLUT n=1 | bfloat16 | ok | 0.213 | 0.012 | 0.280 | 0.012 | 1.146 | 0.094 | 1.422 | 0.094 |
| champion | FusedConfidenceLUT n=2 | float32 | ok | 0.475 | 0.023 | 0.542 | 0.023 | 1.938 | 0.094 | 2.471 | 0.094 |
| champion | FusedConfidenceLUT n=2 | bfloat16 | ok | 0.320 | 0.012 | 0.389 | 0.012 | 1.821 | 0.094 | 2.219 | 0.094 |
| champion | QuantisedConfidenceLUT n=1 | float32 | ok | 1.650 | 0.165 | 1.706 | 0.228 | 5.484 | 0.555 | 7.189 | 0.555 |
| champion | QuantisedConfidenceLUT n=1 | bfloat16 | refused: QuantisedConfidenceLUT does not support low precision | - | - | - | - | - | - | - | - |
| champion | QuantisedConfidenceLUT n=2 | float32 | ok | 2.572 | 0.259 | 3.025 | 0.516 | 9.419 | 1.116 | 12.444 | 1.116 |
| champion | QuantisedConfidenceLUT n=2 | bfloat16 | refused: QuantisedConfidenceLUT does not support low precision | - | - | - | - | - | - | - | - |
| champion | FusedQuantisedConfidenceLUT n=2 | float32 | ok | 0.517 | 0.023 | 0.597 | 0.055 | 2.022 | 0.125 | 2.612 | 0.125 |
| champion | FusedQuantisedConfidenceLUT n=2 | bfloat16 | error: RuntimeError: FusedQuantisedConfidenceLUT: not a case the fused op covers: input dtype torch.bfloat16 (float32 only); table dtype torc | - | - | - | - | - | - | - | - |
| champion | DeployedQuantisedConfidenceLUT | float32 | ok | 0.309 | 0.023 | - | - | - | - | - | - |
| champion | DeployedQuantisedConfidenceLUT | bfloat16 | refused: DeployedQuantisedConfidenceLUT is fp32/fp64 inference only; got input dtype torch.bfloat16 | - | - | - | - | - | - | - | - |
| champion | DeployedQuantisedConfidenceLUT n=1 | float32 | ok | 0.221 | 0.023 | - | - | - | - | - | - |
| champion | DeployedQuantisedConfidenceLUT n=1 | bfloat16 | refused: DeployedQuantisedConfidenceLUT is fp32/fp64 inference only; got input dtype torch.bfloat16 | - | - | - | - | - | - | - | - |
| large | ManifestoHardLUT | float32 | ok | 2.540 | 0.219 | 24.527 | 5.094 | 35.947 | 4.188 | 60.470 | 5.094 |
| large | ManifestoHardLUT | bfloat16 | ok | 2.673 | 0.359 | 24.729 | 5.234 | 35.945 | 4.188 | 60.677 | 5.234 |
| large | ManifestoSoftLUT | float32 | ok | 5.039 | 0.594 | 20.809 | 5.094 | 40.972 | 4.763 | 61.780 | 5.094 |
| large | ManifestoSoftLUT | bfloat16 | ok | 5.063 | 0.734 | 20.980 | 5.234 | 40.923 | 4.857 | 61.900 | 5.234 |
| large | FusedManifestoHardLUT | float32 | ok | 0.988 | 0.094 | 1.071 | 0.094 | 4.940 | 0.234 | 6.009 | 0.234 |
| large | FusedManifestoHardLUT | bfloat16 | ok | 0.743 | 0.047 | 0.846 | 0.047 | 4.596 | 0.234 | 5.443 | 0.234 |
| large | FusedManifestoSoftLUT | float32 | ok | 1.809 | 0.094 | 2.063 | 0.094 | 6.685 | 0.234 | 8.743 | 0.234 |
| large | FusedManifestoSoftLUT | bfloat16 | ok | 1.119 | 0.047 | 1.238 | 0.047 | 6.412 | 0.234 | 7.651 | 0.234 |
| large | ManifestoHardLUT weight=softsign | float32 | ok | 2.479 | 0.219 | 24.771 | 5.094 | 37.734 | 4.188 | 62.513 | 5.094 |
| large | ManifestoHardLUT weight=softsign | bfloat16 | ok | 2.612 | 0.359 | 25.004 | 5.234 | 37.801 | 4.188 | 62.808 | 5.234 |
| large | ManifestoSoftLUT weight=softsign | float32 | ok | 5.384 | 0.594 | 21.069 | 5.094 | 42.787 | 5.013 | 63.863 | 5.094 |
| large | ManifestoSoftLUT weight=softsign | bfloat16 | ok | 5.404 | 0.734 | 21.236 | 5.234 | 42.795 | 5.107 | 64.034 | 5.234 |
| large | FusedManifestoHardLUT weight=softsign | float32 | ok | 1.007 | 0.094 | 1.114 | 0.094 | 6.323 | 0.234 | 7.436 | 0.234 |
| large | FusedManifestoHardLUT weight=softsign | bfloat16 | ok | 0.764 | 0.047 | 0.883 | 0.047 | 5.944 | 0.234 | 6.822 | 0.234 |
| large | FusedManifestoSoftLUT weight=softsign | float32 | ok | 1.836 | 0.094 | 2.037 | 0.094 | 7.096 | 0.234 | 9.134 | 0.234 |
| large | FusedManifestoSoftLUT weight=softsign | bfloat16 | ok | 1.166 | 0.047 | 1.242 | 0.047 | 6.783 | 0.234 | 8.029 | 0.234 |
| large | ConfidenceLUT n=1 | float32 | ok | 3.634 | 0.469 | 3.716 | 0.469 | 9.561 | 2.703 | 13.268 | 2.703 |
| large | ConfidenceLUT n=1 | bfloat16 | ok | 3.701 | 0.609 | 3.770 | 0.656 | 9.543 | 2.844 | 13.311 | 2.844 |
| large | ConfidenceLUT n=2 | float32 | ok | 4.811 | 0.875 | 4.584 | 0.844 | 16.152 | 3.141 | 20.739 | 3.141 |
| large | ConfidenceLUT n=2 | bfloat16 | ok | 4.919 | 1.016 | 4.830 | 1.031 | 15.945 | 3.281 | 20.776 | 3.281 |
| large | FusedConfidenceLUT n=1 | float32 | ok | 1.046 | 0.094 | 1.144 | 0.094 | 4.730 | 0.236 | 5.879 | 0.236 |
| large | FusedConfidenceLUT n=1 | bfloat16 | ok | 0.824 | 0.047 | 0.985 | 0.047 | 4.475 | 0.236 | 5.464 | 0.236 |
| large | FusedConfidenceLUT n=2 | float32 | ok | 1.875 | 0.094 | 1.980 | 0.094 | 7.377 | 0.236 | 9.361 | 0.236 |
| large | FusedConfidenceLUT n=2 | bfloat16 | ok | 1.228 | 0.047 | 1.434 | 0.047 | 6.960 | 0.236 | 8.388 | 0.236 |
| large | QuantisedConfidenceLUT n=1 | float32 | ok | 6.185 | 0.520 | 6.254 | 0.770 | 21.339 | 1.953 | 27.594 | 1.953 |
| large | QuantisedConfidenceLUT n=1 | bfloat16 | refused: QuantisedConfidenceLUT does not support low precision | - | - | - | - | - | - | - | - |
| large | QuantisedConfidenceLUT n=2 | float32 | ok | 9.933 | 0.895 | 11.544 | 1.922 | 37.799 | 4.169 | 49.345 | 4.169 |
| large | QuantisedConfidenceLUT n=2 | bfloat16 | refused: QuantisedConfidenceLUT does not support low precision | - | - | - | - | - | - | - | - |
| large | FusedQuantisedConfidenceLUT n=2 | float32 | ok | 1.943 | 0.094 | 2.063 | 0.219 | 7.674 | 0.361 | 9.737 | 0.361 |
| large | FusedQuantisedConfidenceLUT n=2 | bfloat16 | error: RuntimeError: FusedQuantisedConfidenceLUT: not a case the fused op covers: input dtype torch.bfloat16 (float32 only); table dtype torc | - | - | - | - | - | - | - | - |
| large | DeployedQuantisedConfidenceLUT | float32 | ok | 1.252 | 0.094 | - | - | - | - | - | - |
| large | DeployedQuantisedConfidenceLUT | bfloat16 | refused: DeployedQuantisedConfidenceLUT is fp32/fp64 inference only; got input dtype torch.bfloat16 | - | - | - | - | - | - | - | - |
| large | DeployedQuantisedConfidenceLUT n=1 | float32 | ok | 0.891 | 0.094 | - | - | - | - | - | - |
| large | DeployedQuantisedConfidenceLUT n=1 | bfloat16 | refused: DeployedQuantisedConfidenceLUT is fp32/fp64 inference only; got input dtype torch.bfloat16 | - | - | - | - | - | - | - | - |
