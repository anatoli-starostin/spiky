# lutorch_ex performance snapshot: #164 before the per-table-integer saving, v3

**Code.** `9a9f0891`. Materialised from git, nothing changed.

**Hardware and software.** RTX 5090 (sm_120, 170 SMs), torch 2.9.1+cu130, Triton 3.5.1, CUDA 13.0, driver 595.84.

**Session (v3).** Measured 2026-10-13 in ONE session with the other three v3 trees, interleaved cell by cell over HEAD's full cell matrix (18 cartridges x {fp32, bf16} x {8k, 32k}), the tree order rotating (4 cyclic shifts, then reversed). Champion geometry h16 tph64 nap8 d48 at 8,192 (`champion`) and 32,768 (`large`) vectors.

**Commands, as run** (one fresh process per cell; cwd the tree's `src/spiky/lutorch_ex`, `PYTHONPATH` its `src`):
- every row the tree's own CLI knows: `python -m spiky.lutorch_ex.bench_snapshot --geometries G --cartridges NAME --dtypes D --out c.json --md c.md` (its default isolated mode: Dynamo reset, 10 warm-up + 30 timed iterations, median; peak GiB above the pre-step allocation);
- `QuantisedConfidenceLUT n=1`, which this tree's CLI predates: `python add_cell.py spiky.lutorch_ex.bench_snapshot G D c.json` -- a driver outside the repo that calls this tree's own `bench_snapshot.measure(name, _conf(QuantisedConfidenceLUT, 1), LUTSpec(**GEOMETRIES[G]), B, D, 10, 30)` and sets geometry / batch, i.e. the per-cell path its CLI runs (`measure()` is the same code in all four trees for a training class: checked before the session).

FusedQuantisedConfidenceLUT n=1 does not exist in this tree (no row).

**Conditions.** Logged before and after every cell (576 samples over 288 cells): SM up to 3.07 GHz (unlocked), 31-44 C. **No throttle or power-cap flag at any point** (the software power cap 0x4 did not trip on any sample).

**No transient.** No row moved beyond its noise floor in a way that needed re-measuring.

Comparisons: `figures_162_remeasured_vs_164_v3/`, `figures_160_vs_164_v3/`, `figures_164_presave_vs_164_v3/` (bench_compare.py --same-session). The v1 / v2 / presave / addendum files are history and must not be differenced against v3 cells (different sessions).

| geometry | cartridge | dtype | status | eval forward ms | GiB | train forward ms | GiB | train backward ms | GiB | train total ms | GiB |
|---|---|---|---|---|---|---|---|---|---|---|---|
| champion | ManifestoHardLUT | float32 | ok | 0.662 | 0.055 | 5.944 | 1.273 | 8.975 | 1.047 | 14.921 | 1.273 |
| champion | ManifestoHardLUT | bfloat16 | ok | 0.699 | 0.125 | 6.001 | 1.344 | 8.922 | 1.082 | 14.926 | 1.344 |
| champion | ManifestoSoftLUT | float32 | ok | 1.220 | 0.148 | 5.046 | 1.273 | 10.206 | 1.265 | 15.253 | 1.273 |
| champion | ManifestoSoftLUT | bfloat16 | ok | 1.250 | 0.219 | 5.122 | 1.344 | 10.132 | 1.323 | 15.253 | 1.344 |
| champion | FusedManifestoHardLUT | float32 | ok | 0.252 | 0.023 | 0.310 | 0.023 | 1.306 | 0.094 | 1.614 | 0.094 |
| champion | FusedManifestoHardLUT | bfloat16 | ok | 0.192 | 0.012 | 0.249 | 0.012 | 1.185 | 0.094 | 1.432 | 0.094 |
| champion | FusedManifestoSoftLUT | float32 | ok | 0.461 | 0.023 | 0.552 | 0.023 | 1.712 | 0.094 | 2.262 | 0.094 |
| champion | FusedManifestoSoftLUT | bfloat16 | ok | 0.286 | 0.012 | 0.341 | 0.012 | 1.620 | 0.094 | 1.962 | 0.094 |
| champion | ManifestoHardLUT weight=softsign | float32 | ok | 0.664 | 0.055 | 5.996 | 1.273 | 9.301 | 1.047 | 15.297 | 1.273 |
| champion | ManifestoHardLUT weight=softsign | bfloat16 | ok | 0.707 | 0.125 | 6.022 | 1.344 | 9.243 | 1.082 | 15.267 | 1.344 |
| champion | ManifestoSoftLUT weight=softsign | float32 | ok | 1.287 | 0.148 | 5.065 | 1.273 | 10.533 | 1.327 | 15.600 | 1.327 |
| champion | ManifestoSoftLUT weight=softsign | bfloat16 | ok | 1.318 | 0.219 | 5.161 | 1.344 | 10.435 | 1.386 | 15.595 | 1.386 |
| champion | FusedManifestoHardLUT weight=softsign | float32 | ok | 0.267 | 0.023 | 0.342 | 0.023 | 1.588 | 0.094 | 1.929 | 0.094 |
| champion | FusedManifestoHardLUT weight=softsign | bfloat16 | ok | 0.213 | 0.012 | 0.257 | 0.012 | 1.489 | 0.094 | 1.742 | 0.094 |
| champion | FusedManifestoSoftLUT weight=softsign | float32 | ok | 0.476 | 0.023 | 0.579 | 0.023 | 1.875 | 0.094 | 2.441 | 0.094 |
| champion | FusedManifestoSoftLUT weight=softsign | bfloat16 | ok | 0.303 | 0.012 | 0.384 | 0.012 | 1.773 | 0.094 | 2.151 | 0.094 |
| champion | ConfidenceLUT n=1 | float32 | ok | 0.900 | 0.117 | 1.086 | 0.117 | 2.361 | 0.711 | 3.452 | 0.711 |
| champion | ConfidenceLUT n=1 | bfloat16 | ok | 0.942 | 0.188 | 1.076 | 0.199 | 2.372 | 0.746 | 3.449 | 0.746 |
| champion | ConfidenceLUT n=2 | float32 | ok | 1.239 | 0.219 | 1.315 | 0.211 | 4.030 | 0.820 | 5.341 | 0.820 |
| champion | ConfidenceLUT n=2 | bfloat16 | ok | 1.276 | 0.289 | 1.276 | 0.293 | 3.989 | 0.855 | 5.264 | 0.855 |
| champion | FusedConfidenceLUT n=1 | float32 | ok | 0.266 | 0.023 | 0.324 | 0.023 | 1.236 | 0.094 | 1.559 | 0.094 |
| champion | FusedConfidenceLUT n=1 | bfloat16 | ok | 0.213 | 0.012 | 0.281 | 0.012 | 1.146 | 0.094 | 1.422 | 0.094 |
| champion | FusedConfidenceLUT n=2 | float32 | ok | 0.474 | 0.023 | 0.536 | 0.023 | 1.925 | 0.094 | 2.444 | 0.094 |
| champion | FusedConfidenceLUT n=2 | bfloat16 | ok | 0.318 | 0.012 | 0.409 | 0.012 | 1.836 | 0.094 | 2.234 | 0.094 |
| champion | QuantisedConfidenceLUT n=1 | float32 | ok | 1.651 | 0.165 | 1.706 | 0.228 | 5.482 | 0.555 | 7.189 | 0.555 |
| champion | QuantisedConfidenceLUT n=1 | bfloat16 | refused: QuantisedConfidenceLUT does not support low precision | - | - | - | - | - | - | - | - |
| champion | QuantisedConfidenceLUT n=2 | float32 | ok | 2.574 | 0.259 | 3.026 | 0.516 | 9.425 | 1.116 | 12.450 | 1.116 |
| champion | QuantisedConfidenceLUT n=2 | bfloat16 | refused: QuantisedConfidenceLUT does not support low precision | - | - | - | - | - | - | - | - |
| champion | FusedQuantisedConfidenceLUT n=2 | float32 | ok | 0.515 | 0.023 | 0.589 | 0.023 | 2.106 | 0.094 | 2.693 | 0.094 |
| champion | FusedQuantisedConfidenceLUT n=2 | bfloat16 | error: RuntimeError: FusedQuantisedConfidenceLUT: not a case the fused op covers: input dtype torch.bfloat16 (float32 only); table dtype torc | - | - | - | - | - | - | - | - |
| champion | DeployedQuantisedConfidenceLUT | float32 | ok | 0.310 | 0.023 | - | - | - | - | - | - |
| champion | DeployedQuantisedConfidenceLUT | bfloat16 | refused: DeployedQuantisedConfidenceLUT is fp32/fp64 inference only; got input dtype torch.bfloat16 | - | - | - | - | - | - | - | - |
| champion | DeployedQuantisedConfidenceLUT n=1 | float32 | ok | 0.220 | 0.023 | - | - | - | - | - | - |
| champion | DeployedQuantisedConfidenceLUT n=1 | bfloat16 | refused: DeployedQuantisedConfidenceLUT is fp32/fp64 inference only; got input dtype torch.bfloat16 | - | - | - | - | - | - | - | - |
| large | ManifestoHardLUT | float32 | ok | 2.535 | 0.219 | 24.518 | 5.094 | 35.953 | 4.188 | 60.470 | 5.094 |
| large | ManifestoHardLUT | bfloat16 | ok | 2.603 | 0.359 | 24.727 | 5.234 | 35.941 | 4.188 | 60.669 | 5.234 |
| large | ManifestoSoftLUT | float32 | ok | 5.012 | 0.594 | 20.814 | 5.094 | 40.976 | 4.763 | 61.791 | 5.094 |
| large | ManifestoSoftLUT | bfloat16 | ok | 5.083 | 0.734 | 20.983 | 5.234 | 40.929 | 4.857 | 61.915 | 5.234 |
| large | FusedManifestoHardLUT | float32 | ok | 0.993 | 0.094 | 1.073 | 0.094 | 4.927 | 0.234 | 6.000 | 0.234 |
| large | FusedManifestoHardLUT | bfloat16 | ok | 0.744 | 0.047 | 0.844 | 0.047 | 4.591 | 0.234 | 5.436 | 0.234 |
| large | FusedManifestoSoftLUT | float32 | ok | 1.810 | 0.094 | 2.138 | 0.094 | 6.874 | 0.234 | 9.009 | 0.234 |
| large | FusedManifestoSoftLUT | bfloat16 | ok | 1.120 | 0.047 | 1.241 | 0.047 | 6.416 | 0.234 | 7.659 | 0.234 |
| large | ManifestoHardLUT weight=softsign | float32 | ok | 2.553 | 0.219 | 24.792 | 5.094 | 37.706 | 4.188 | 62.497 | 5.094 |
| large | ManifestoHardLUT weight=softsign | bfloat16 | ok | 2.688 | 0.359 | 24.994 | 5.234 | 37.784 | 4.188 | 62.781 | 5.234 |
| large | ManifestoSoftLUT weight=softsign | float32 | ok | 5.370 | 0.594 | 21.079 | 5.094 | 42.783 | 5.013 | 63.863 | 5.094 |
| large | ManifestoSoftLUT weight=softsign | bfloat16 | ok | 5.394 | 0.734 | 21.257 | 5.234 | 42.821 | 5.107 | 64.077 | 5.234 |
| large | FusedManifestoHardLUT weight=softsign | float32 | ok | 1.007 | 0.094 | 1.116 | 0.094 | 6.311 | 0.234 | 7.431 | 0.234 |
| large | FusedManifestoHardLUT weight=softsign | bfloat16 | ok | 0.765 | 0.047 | 0.887 | 0.047 | 5.942 | 0.234 | 6.827 | 0.234 |
| large | FusedManifestoSoftLUT weight=softsign | float32 | ok | 1.837 | 0.094 | 2.038 | 0.094 | 7.093 | 0.234 | 9.126 | 0.234 |
| large | FusedManifestoSoftLUT weight=softsign | bfloat16 | ok | 1.172 | 0.047 | 1.253 | 0.047 | 6.796 | 0.234 | 8.045 | 0.234 |
| large | ConfidenceLUT n=1 | float32 | ok | 3.577 | 0.469 | 3.704 | 0.469 | 9.524 | 2.703 | 13.231 | 2.703 |
| large | ConfidenceLUT n=1 | bfloat16 | ok | 3.694 | 0.609 | 3.742 | 0.656 | 9.602 | 2.844 | 13.342 | 2.844 |
| large | ConfidenceLUT n=2 | float32 | ok | 4.828 | 0.875 | 4.587 | 0.844 | 16.170 | 3.141 | 20.740 | 3.141 |
| large | ConfidenceLUT n=2 | bfloat16 | ok | 5.380 | 1.016 | 4.846 | 1.031 | 15.908 | 3.281 | 20.758 | 3.281 |
| large | FusedConfidenceLUT n=1 | float32 | ok | 1.046 | 0.094 | 1.143 | 0.094 | 4.754 | 0.236 | 5.899 | 0.236 |
| large | FusedConfidenceLUT n=1 | bfloat16 | ok | 0.825 | 0.047 | 0.990 | 0.047 | 4.477 | 0.236 | 5.468 | 0.236 |
| large | FusedConfidenceLUT n=2 | float32 | ok | 1.875 | 0.094 | 1.970 | 0.094 | 7.345 | 0.236 | 9.317 | 0.236 |
| large | FusedConfidenceLUT n=2 | bfloat16 | ok | 1.227 | 0.047 | 1.428 | 0.047 | 6.971 | 0.236 | 8.418 | 0.236 |
| large | QuantisedConfidenceLUT n=1 | float32 | ok | 6.199 | 0.520 | 6.255 | 0.770 | 21.382 | 1.953 | 27.637 | 1.953 |
| large | QuantisedConfidenceLUT n=1 | bfloat16 | refused: QuantisedConfidenceLUT does not support low precision | - | - | - | - | - | - | - | - |
| large | QuantisedConfidenceLUT n=2 | float32 | ok | 9.962 | 0.895 | 11.552 | 1.922 | 37.808 | 4.169 | 49.362 | 4.169 |
| large | QuantisedConfidenceLUT n=2 | bfloat16 | refused: QuantisedConfidenceLUT does not support low precision | - | - | - | - | - | - | - | - |
| large | FusedQuantisedConfidenceLUT n=2 | float32 | ok | 1.946 | 0.094 | 2.039 | 0.094 | 7.986 | 0.236 | 10.022 | 0.236 |
| large | FusedQuantisedConfidenceLUT n=2 | bfloat16 | error: RuntimeError: FusedQuantisedConfidenceLUT: not a case the fused op covers: input dtype torch.bfloat16 (float32 only); table dtype torc | - | - | - | - | - | - | - | - |
| large | DeployedQuantisedConfidenceLUT | float32 | ok | 1.252 | 0.094 | - | - | - | - | - | - |
| large | DeployedQuantisedConfidenceLUT | bfloat16 | refused: DeployedQuantisedConfidenceLUT is fp32/fp64 inference only; got input dtype torch.bfloat16 | - | - | - | - | - | - | - | - |
| large | DeployedQuantisedConfidenceLUT n=1 | float32 | ok | 0.891 | 0.094 | - | - | - | - | - | - |
| large | DeployedQuantisedConfidenceLUT n=1 | bfloat16 | refused: DeployedQuantisedConfidenceLUT is fp32/fp64 inference only; got input dtype torch.bfloat16 | - | - | - | - | - | - | - | - |
