# lutorch_ex performance snapshot: #164 HEAD, v3 (THE SNAPSHOT OF RECORD)

**Code.** `41cc6b97`, #164 HEAD (library code 0b3de742: FusedQuantisedConfidenceLUT read_top_n=1); no uncommitted changes in lutorch_ex. The first single-session set covering every cell, the n=1 rows included. Paired with snapshot_162_remeasured_v3, snapshot_160_remeasured_v3 and snapshot_164_presave_v3 (one measurement of HEAD, the same session as all three).

**Hardware and software.** RTX 5090 (sm_120, 170 SMs), torch 2.9.1+cu130, Triton 3.5.1, CUDA 13.0, driver 595.84.

**Session (v3).** Measured 2026-10-13 in ONE session with the other three v3 trees, interleaved cell by cell over HEAD's full cell matrix (18 cartridges x {fp32, bf16} x {8k, 32k}), the tree order rotating (4 cyclic shifts, then reversed). Champion geometry h16 tph64 nap8 d48 at 8,192 (`champion`) and 32,768 (`large`) vectors.

**Commands, as run** (one fresh process per cell; cwd the tree's `src/spiky/lutorch_ex`, `PYTHONPATH` its `src`):
- every row the tree's own CLI knows: `python -m spiky.lutorch_ex.benchmarks.bench_snapshot --geometries G --cartridges NAME --dtypes D --out c.json --md c.md` (its default isolated mode: Dynamo reset, 10 warm-up + 30 timed iterations, median; peak GiB above the pre-step allocation);

**Conditions.** Logged before and after every cell (576 samples over 288 cells): SM up to 3.07 GHz (unlocked), 31-44 C. **No throttle or power-cap flag at any point** (the software power cap 0x4 did not trip on any sample).

**No transient.** No row moved beyond its noise floor in a way that needed re-measuring.

Comparisons: `figures_162_remeasured_vs_164_v3/`, `figures_160_vs_164_v3/`, `figures_164_presave_vs_164_v3/` (bench_compare.py --same-session). The v1 / v2 / presave / addendum files are history and must not be differenced against v3 cells (different sessions).

| geometry | cartridge | dtype | status | eval forward ms | GiB | train forward ms | GiB | train backward ms | GiB | train total ms | GiB |
|---|---|---|---|---|---|---|---|---|---|---|---|
| champion | ManifestoHardLUT | float32 | ok | 0.657 | 0.055 | 5.980 | 1.273 | 8.970 | 1.047 | 14.968 | 1.273 |
| champion | ManifestoHardLUT | bfloat16 | ok | 0.700 | 0.125 | 6.002 | 1.344 | 8.928 | 1.082 | 14.932 | 1.344 |
| champion | ManifestoSoftLUT | float32 | ok | 1.220 | 0.148 | 5.047 | 1.273 | 10.198 | 1.265 | 15.245 | 1.273 |
| champion | ManifestoSoftLUT | bfloat16 | ok | 1.250 | 0.219 | 5.121 | 1.344 | 10.134 | 1.323 | 15.257 | 1.344 |
| champion | FusedManifestoHardLUT | float32 | ok | 0.251 | 0.023 | 0.309 | 0.023 | 1.296 | 0.094 | 1.593 | 0.094 |
| champion | FusedManifestoHardLUT | bfloat16 | ok | 0.191 | 0.012 | 0.249 | 0.012 | 1.183 | 0.094 | 1.431 | 0.094 |
| champion | FusedManifestoSoftLUT | float32 | ok | 0.460 | 0.023 | 0.558 | 0.023 | 1.722 | 0.094 | 2.280 | 0.094 |
| champion | FusedManifestoSoftLUT | bfloat16 | ok | 0.287 | 0.012 | 0.356 | 0.012 | 1.594 | 0.094 | 1.938 | 0.094 |
| champion | ManifestoHardLUT weight=softsign | float32 | ok | 0.662 | 0.055 | 5.974 | 1.273 | 9.305 | 1.047 | 15.281 | 1.273 |
| champion | ManifestoHardLUT weight=softsign | bfloat16 | ok | 0.707 | 0.125 | 6.022 | 1.344 | 9.238 | 1.082 | 15.260 | 1.344 |
| champion | ManifestoSoftLUT weight=softsign | float32 | ok | 1.283 | 0.148 | 5.068 | 1.273 | 10.530 | 1.327 | 15.598 | 1.327 |
| champion | ManifestoSoftLUT weight=softsign | bfloat16 | ok | 1.316 | 0.219 | 5.163 | 1.344 | 10.442 | 1.386 | 15.605 | 1.386 |
| champion | FusedManifestoHardLUT weight=softsign | float32 | ok | 0.268 | 0.023 | 0.342 | 0.023 | 1.589 | 0.094 | 1.918 | 0.094 |
| champion | FusedManifestoHardLUT weight=softsign | bfloat16 | ok | 0.214 | 0.012 | 0.292 | 0.012 | 1.493 | 0.094 | 1.785 | 0.094 |
| champion | FusedManifestoSoftLUT weight=softsign | float32 | ok | 0.477 | 0.023 | 0.585 | 0.023 | 1.883 | 0.094 | 2.454 | 0.094 |
| champion | FusedManifestoSoftLUT weight=softsign | bfloat16 | ok | 0.303 | 0.012 | 0.381 | 0.012 | 1.774 | 0.094 | 2.130 | 0.094 |
| champion | ConfidenceLUT n=1 | float32 | ok | 0.901 | 0.117 | 1.071 | 0.117 | 2.362 | 0.711 | 3.432 | 0.711 |
| champion | ConfidenceLUT n=1 | bfloat16 | ok | 0.944 | 0.188 | 1.079 | 0.199 | 2.359 | 0.746 | 3.438 | 0.746 |
| champion | ConfidenceLUT n=2 | float32 | ok | 1.242 | 0.219 | 1.321 | 0.211 | 4.030 | 0.820 | 5.352 | 0.820 |
| champion | ConfidenceLUT n=2 | bfloat16 | ok | 1.275 | 0.289 | 1.275 | 0.293 | 3.987 | 0.855 | 5.259 | 0.855 |
| champion | FusedConfidenceLUT n=1 | float32 | ok | 0.267 | 0.023 | 0.324 | 0.023 | 1.224 | 0.094 | 1.537 | 0.094 |
| champion | FusedConfidenceLUT n=1 | bfloat16 | ok | 0.213 | 0.012 | 0.285 | 0.012 | 1.153 | 0.094 | 1.428 | 0.094 |
| champion | FusedConfidenceLUT n=2 | float32 | ok | 0.475 | 0.023 | 0.542 | 0.023 | 1.938 | 0.094 | 2.472 | 0.094 |
| champion | FusedConfidenceLUT n=2 | bfloat16 | ok | 0.318 | 0.012 | 0.399 | 0.012 | 1.811 | 0.094 | 2.217 | 0.094 |
| champion | QuantisedConfidenceLUT n=1 | float32 | ok | 1.647 | 0.165 | 1.712 | 0.228 | 5.480 | 0.555 | 7.190 | 0.555 |
| champion | QuantisedConfidenceLUT n=1 | bfloat16 | refused: QuantisedConfidenceLUT does not support low precision | - | - | - | - | - | - | - | - |
| champion | QuantisedConfidenceLUT n=2 | float32 | ok | 2.574 | 0.259 | 3.035 | 0.516 | 9.429 | 1.116 | 12.463 | 1.116 |
| champion | QuantisedConfidenceLUT n=2 | bfloat16 | refused: QuantisedConfidenceLUT does not support low precision | - | - | - | - | - | - | - | - |
| champion | FusedQuantisedConfidenceLUT n=1 | float32 | ok | 0.302 | 0.023 | 0.398 | 0.055 | 1.201 | 0.125 | 1.612 | 0.125 |
| champion | FusedQuantisedConfidenceLUT n=1 | bfloat16 | error: RuntimeError: FusedQuantisedConfidenceLUT: not a case the fused op covers: input dtype torch.bfloat16 (float32 only); table dtype torc | - | - | - | - | - | - | - | - |
| champion | FusedQuantisedConfidenceLUT n=2 | float32 | ok | 0.518 | 0.023 | 0.600 | 0.055 | 2.007 | 0.125 | 2.606 | 0.125 |
| champion | FusedQuantisedConfidenceLUT n=2 | bfloat16 | error: RuntimeError: FusedQuantisedConfidenceLUT: not a case the fused op covers: input dtype torch.bfloat16 (float32 only); table dtype torc | - | - | - | - | - | - | - | - |
| champion | DeployedQuantisedConfidenceLUT | float32 | ok | 0.310 | 0.023 | - | - | - | - | - | - |
| champion | DeployedQuantisedConfidenceLUT | bfloat16 | refused: DeployedQuantisedConfidenceLUT is fp32/fp64 inference only; got input dtype torch.bfloat16 | - | - | - | - | - | - | - | - |
| champion | DeployedQuantisedConfidenceLUT n=1 | float32 | ok | 0.220 | 0.023 | - | - | - | - | - | - |
| champion | DeployedQuantisedConfidenceLUT n=1 | bfloat16 | refused: DeployedQuantisedConfidenceLUT is fp32/fp64 inference only; got input dtype torch.bfloat16 | - | - | - | - | - | - | - | - |
| large | ManifestoHardLUT | float32 | ok | 2.465 | 0.219 | 24.528 | 5.094 | 35.925 | 4.188 | 60.454 | 5.094 |
| large | ManifestoHardLUT | bfloat16 | ok | 2.602 | 0.359 | 24.723 | 5.234 | 35.943 | 4.188 | 60.667 | 5.234 |
| large | ManifestoSoftLUT | float32 | ok | 5.017 | 0.594 | 20.816 | 5.094 | 40.965 | 4.763 | 61.783 | 5.094 |
| large | ManifestoSoftLUT | bfloat16 | ok | 5.089 | 0.734 | 20.981 | 5.234 | 40.914 | 4.857 | 61.893 | 5.234 |
| large | FusedManifestoHardLUT | float32 | ok | 0.992 | 0.094 | 1.074 | 0.094 | 4.910 | 0.234 | 5.982 | 0.234 |
| large | FusedManifestoHardLUT | bfloat16 | ok | 0.745 | 0.047 | 0.848 | 0.047 | 4.590 | 0.234 | 5.437 | 0.234 |
| large | FusedManifestoSoftLUT | float32 | ok | 1.811 | 0.094 | 2.109 | 0.094 | 6.807 | 0.234 | 8.908 | 0.234 |
| large | FusedManifestoSoftLUT | bfloat16 | ok | 1.117 | 0.047 | 1.218 | 0.047 | 6.298 | 0.234 | 7.515 | 0.234 |
| large | ManifestoHardLUT weight=softsign | float32 | ok | 2.560 | 0.219 | 24.790 | 5.094 | 37.720 | 4.188 | 62.509 | 5.094 |
| large | ManifestoHardLUT weight=softsign | bfloat16 | ok | 2.691 | 0.359 | 24.994 | 5.234 | 37.790 | 4.188 | 62.790 | 5.234 |
| large | ManifestoSoftLUT weight=softsign | float32 | ok | 5.358 | 0.594 | 21.085 | 5.094 | 42.792 | 5.013 | 63.875 | 5.094 |
| large | ManifestoSoftLUT weight=softsign | bfloat16 | ok | 5.403 | 0.734 | 21.251 | 5.234 | 42.813 | 5.107 | 64.067 | 5.234 |
| large | FusedManifestoHardLUT weight=softsign | float32 | ok | 1.008 | 0.094 | 1.114 | 0.094 | 6.317 | 0.234 | 7.432 | 0.234 |
| large | FusedManifestoHardLUT weight=softsign | bfloat16 | ok | 0.766 | 0.047 | 0.885 | 0.047 | 5.944 | 0.234 | 6.831 | 0.234 |
| large | FusedManifestoSoftLUT weight=softsign | float32 | ok | 1.837 | 0.094 | 2.032 | 0.094 | 7.077 | 0.234 | 9.105 | 0.234 |
| large | FusedManifestoSoftLUT weight=softsign | bfloat16 | ok | 1.177 | 0.047 | 1.251 | 0.047 | 6.787 | 0.234 | 8.037 | 0.234 |
| large | ConfidenceLUT n=1 | float32 | ok | 3.604 | 0.469 | 3.799 | 0.469 | 9.573 | 2.703 | 13.380 | 2.703 |
| large | ConfidenceLUT n=1 | bfloat16 | ok | 3.701 | 0.609 | 3.742 | 0.656 | 9.602 | 2.844 | 13.341 | 2.844 |
| large | ConfidenceLUT n=2 | float32 | ok | 4.853 | 0.875 | 4.590 | 0.844 | 16.188 | 3.141 | 20.771 | 3.141 |
| large | ConfidenceLUT n=2 | bfloat16 | ok | 5.465 | 1.016 | 4.857 | 1.031 | 15.915 | 3.281 | 20.776 | 3.281 |
| large | FusedConfidenceLUT n=1 | float32 | ok | 1.046 | 0.094 | 1.145 | 0.094 | 4.743 | 0.236 | 5.889 | 0.236 |
| large | FusedConfidenceLUT n=1 | bfloat16 | ok | 0.824 | 0.047 | 0.987 | 0.047 | 4.486 | 0.236 | 5.478 | 0.236 |
| large | FusedConfidenceLUT n=2 | float32 | ok | 1.874 | 0.094 | 1.986 | 0.094 | 7.374 | 0.236 | 9.359 | 0.236 |
| large | FusedConfidenceLUT n=2 | bfloat16 | ok | 1.224 | 0.047 | 1.426 | 0.047 | 6.967 | 0.236 | 8.384 | 0.236 |
| large | QuantisedConfidenceLUT n=1 | float32 | ok | 6.189 | 0.520 | 6.260 | 0.770 | 21.379 | 1.953 | 27.639 | 1.953 |
| large | QuantisedConfidenceLUT n=1 | bfloat16 | refused: QuantisedConfidenceLUT does not support low precision | - | - | - | - | - | - | - | - |
| large | QuantisedConfidenceLUT n=2 | float32 | ok | 9.957 | 0.895 | 11.552 | 1.922 | 37.813 | 4.169 | 49.362 | 4.169 |
| large | QuantisedConfidenceLUT n=2 | bfloat16 | refused: QuantisedConfidenceLUT does not support low precision | - | - | - | - | - | - | - | - |
| large | FusedQuantisedConfidenceLUT n=1 | float32 | ok | 1.179 | 0.094 | 1.355 | 0.219 | 4.412 | 0.361 | 5.772 | 0.361 |
| large | FusedQuantisedConfidenceLUT n=1 | bfloat16 | error: RuntimeError: FusedQuantisedConfidenceLUT: not a case the fused op covers: input dtype torch.bfloat16 (float32 only); table dtype torc | - | - | - | - | - | - | - | - |
| large | FusedQuantisedConfidenceLUT n=2 | float32 | ok | 1.944 | 0.094 | 2.080 | 0.219 | 7.691 | 0.361 | 9.774 | 0.361 |
| large | FusedQuantisedConfidenceLUT n=2 | bfloat16 | error: RuntimeError: FusedQuantisedConfidenceLUT: not a case the fused op covers: input dtype torch.bfloat16 (float32 only); table dtype torc | - | - | - | - | - | - | - | - |
| large | DeployedQuantisedConfidenceLUT | float32 | ok | 1.252 | 0.094 | - | - | - | - | - | - |
| large | DeployedQuantisedConfidenceLUT | bfloat16 | refused: DeployedQuantisedConfidenceLUT is fp32/fp64 inference only; got input dtype torch.bfloat16 | - | - | - | - | - | - | - | - |
| large | DeployedQuantisedConfidenceLUT n=1 | float32 | ok | 0.891 | 0.094 | - | - | - | - | - | - |
| large | DeployedQuantisedConfidenceLUT n=1 | bfloat16 | refused: DeployedQuantisedConfidenceLUT is fp32/fp64 inference only; got input dtype torch.bfloat16 | - | - | - | - | - | - | - | - |
