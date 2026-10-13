# lutorch_ex performance snapshot: #160 re-measured, v3 (the cumulative BEFORE of #164)

**Code.** `a811390e`, #160's library code. Materialised from git, nothing changed.

**Hardware and software.** RTX 5090 (sm_120, 170 SMs), torch 2.9.1+cu130, Triton 3.5.1, CUDA 13.0, driver 595.84.

**Session (v3).** Measured 2026-10-13 in ONE session with the other three v3 trees, interleaved cell by cell over HEAD's full cell matrix (18 cartridges x {fp32, bf16} x {8k, 32k}), the tree order rotating (4 cyclic shifts, then reversed). Champion geometry h16 tph64 nap8 d48 at 8,192 (`champion`) and 32,768 (`large`) vectors.

**Commands, as run** (one fresh process per cell; cwd the tree's `src/spiky/lutorch_ex`, `PYTHONPATH` its `src`):
- every row the tree's own CLI knows: `python -m spiky.lutorch_ex.bench_snapshot --geometries G --cartridges NAME --dtypes D --out c.json --md c.md` (its default isolated mode: Dynamo reset, 10 warm-up + 30 timed iterations, median; peak GiB above the pre-step allocation);
- `QuantisedConfidenceLUT n=1`, which this tree's CLI predates: `python add_cell.py spiky.lutorch_ex.bench_snapshot G D c.json` -- a driver outside the repo that calls this tree's own `bench_snapshot.measure(name, _conf(QuantisedConfidenceLUT, 1), LUTSpec(**GEOMETRIES[G]), B, D, 10, 30)` and sets geometry / batch, i.e. the per-cell path its CLI runs (`measure()` is the same code in all four trees for a training class: checked before the session).

FusedQuantisedConfidenceLUT n=1 and DeployedQuantisedConfidenceLUT n=1 do not exist in this tree (no row).

**Conditions.** Logged before and after every cell (576 samples over 288 cells): SM up to 3.07 GHz (unlocked), 31-44 C. **No throttle or power-cap flag at any point** (the software power cap 0x4 did not trip on any sample).

**No transient.** No row moved beyond its noise floor in a way that needed re-measuring.

Comparisons: `figures_162_remeasured_vs_164_v3/`, `figures_160_vs_164_v3/`, `figures_164_presave_vs_164_v3/` (bench_compare.py --same-session). The v1 / v2 / presave / addendum files are history and must not be differenced against v3 cells (different sessions).

| geometry | cartridge | dtype | status | eval forward ms | GiB | train forward ms | GiB | train backward ms | GiB | train total ms | GiB |
|---|---|---|---|---|---|---|---|---|---|---|---|
| champion | ManifestoHardLUT | float32 | ok | 0.663 | 0.055 | 17.917 | 6.712 | 7.324 | 3.517 | 25.247 | 6.712 |
| champion | ManifestoHardLUT | bfloat16 | ok | 0.701 | 0.125 | 17.978 | 6.782 | 7.335 | 3.528 | 25.312 | 6.782 |
| champion | ManifestoSoftLUT | float32 | ok | 1.225 | 0.148 | 5.044 | 1.273 | 10.205 | 1.265 | 15.249 | 1.273 |
| champion | ManifestoSoftLUT | bfloat16 | ok | 1.250 | 0.219 | 5.123 | 1.344 | 10.134 | 1.323 | 15.258 | 1.344 |
| champion | FusedManifestoHardLUT | float32 | ok | 0.252 | 0.023 | 0.312 | 0.023 | 1.282 | 0.094 | 1.580 | 0.094 |
| champion | FusedManifestoHardLUT | bfloat16 | ok | 0.193 | 0.012 | 0.252 | 0.012 | 1.149 | 0.094 | 1.401 | 0.094 |
| champion | FusedManifestoSoftLUT | float32 | ok | 0.460 | 0.023 | 0.541 | 0.023 | 1.693 | 0.094 | 2.238 | 0.094 |
| champion | FusedManifestoSoftLUT | bfloat16 | ok | 0.282 | 0.012 | 0.332 | 0.012 | 1.526 | 0.094 | 1.846 | 0.094 |
| champion | ManifestoHardLUT weight=softsign | float32 | ok | 0.670 | 0.055 | 17.957 | 6.743 | 7.655 | 3.579 | 25.611 | 6.743 |
| champion | ManifestoHardLUT weight=softsign | bfloat16 | ok | 0.707 | 0.125 | 17.986 | 6.813 | 7.657 | 3.591 | 25.645 | 6.813 |
| champion | ManifestoSoftLUT weight=softsign | float32 | ok | 1.298 | 0.148 | 5.075 | 1.273 | 10.529 | 1.327 | 15.603 | 1.327 |
| champion | ManifestoSoftLUT weight=softsign | bfloat16 | ok | 1.314 | 0.219 | 5.161 | 1.344 | 10.442 | 1.386 | 15.603 | 1.386 |
| champion | FusedManifestoHardLUT weight=softsign | float32 | ok | 0.266 | 0.023 | 0.341 | 0.023 | 1.589 | 0.094 | 1.929 | 0.094 |
| champion | FusedManifestoHardLUT weight=softsign | bfloat16 | ok | 0.215 | 0.012 | 0.294 | 0.012 | 1.491 | 0.094 | 1.785 | 0.094 |
| champion | FusedManifestoSoftLUT weight=softsign | float32 | ok | 0.477 | 0.023 | 0.575 | 0.023 | 1.869 | 0.094 | 2.436 | 0.094 |
| champion | FusedManifestoSoftLUT weight=softsign | bfloat16 | ok | 0.304 | 0.012 | 0.384 | 0.012 | 1.753 | 0.094 | 2.115 | 0.094 |
| champion | ConfidenceLUT n=1 | float32 | ok | 0.914 | 0.117 | 1.067 | 0.117 | 2.365 | 0.711 | 3.431 | 0.711 |
| champion | ConfidenceLUT n=1 | bfloat16 | ok | 0.942 | 0.188 | 1.083 | 0.199 | 2.374 | 0.746 | 3.458 | 0.746 |
| champion | ConfidenceLUT n=2 | float32 | ok | 1.262 | 0.219 | 1.321 | 0.211 | 4.067 | 0.820 | 5.389 | 0.820 |
| champion | ConfidenceLUT n=2 | bfloat16 | ok | 1.278 | 0.289 | 1.286 | 0.293 | 4.028 | 0.855 | 5.316 | 0.855 |
| champion | FusedConfidenceLUT n=1 | float32 | ok | 0.266 | 0.023 | 0.321 | 0.023 | 1.237 | 0.094 | 1.558 | 0.094 |
| champion | FusedConfidenceLUT n=1 | bfloat16 | ok | 0.204 | 0.012 | 0.271 | 0.012 | 1.141 | 0.094 | 1.412 | 0.094 |
| champion | FusedConfidenceLUT n=2 | float32 | ok | 0.472 | 0.023 | 0.536 | 0.023 | 1.914 | 0.094 | 2.447 | 0.094 |
| champion | FusedConfidenceLUT n=2 | bfloat16 | ok | 0.312 | 0.012 | 0.385 | 0.012 | 1.830 | 0.094 | 2.206 | 0.094 |
| champion | QuantisedConfidenceLUT n=1 | float32 | ok | 1.301 | 0.165 | 1.364 | 0.228 | 5.063 | 0.555 | 6.428 | 0.555 |
| champion | QuantisedConfidenceLUT n=1 | bfloat16 | refused: QuantisedConfidenceLUT does not support low precision | - | - | - | - | - | - | - | - |
| champion | QuantisedConfidenceLUT n=2 | float32 | ok | 2.966 | 0.508 | 2.977 | 0.617 | 9.587 | 1.116 | 12.564 | 1.116 |
| champion | QuantisedConfidenceLUT n=2 | bfloat16 | refused: QuantisedConfidenceLUT does not support low precision | - | - | - | - | - | - | - | - |
| champion | FusedQuantisedConfidenceLUT n=2 | float32 | ok | 2.523 | 0.415 | 2.506 | 0.352 | 9.877 | 0.946 | 12.386 | 0.946 |
| champion | FusedQuantisedConfidenceLUT n=2 | bfloat16 | error: RuntimeError: FusedQuantisedConfidenceLUT: not a case the fused op covers: input dtype torch.bfloat16 (float32 only); table dtype torc | - | - | - | - | - | - | - | - |
| champion | DeployedQuantisedConfidenceLUT | float32 | ok | 2.896 | 0.414 | - | - | - | - | - | - |
| champion | DeployedQuantisedConfidenceLUT | bfloat16 | refused: DeployedQuantisedConfidenceLUT is fp32/fp64 inference only; got input dtype torch.bfloat16 | - | - | - | - | - | - | - | - |
| large | ManifestoHardLUT | float32 | out of memory | 2.481 | 0.219 | - | - | - | - | - | - |
| large | ManifestoHardLUT | bfloat16 | out of memory | 2.606 | 0.359 | - | - | - | - | - | - |
| large | ManifestoSoftLUT | float32 | ok | 5.026 | 0.594 | 20.808 | 5.094 | 40.981 | 4.763 | 61.783 | 5.094 |
| large | ManifestoSoftLUT | bfloat16 | ok | 5.087 | 0.734 | 20.982 | 5.234 | 40.917 | 4.857 | 61.894 | 5.234 |
| large | FusedManifestoHardLUT | float32 | ok | 0.993 | 0.094 | 1.079 | 0.094 | 4.880 | 0.234 | 5.957 | 0.234 |
| large | FusedManifestoHardLUT | bfloat16 | ok | 0.749 | 0.047 | 0.851 | 0.047 | 4.476 | 0.234 | 5.328 | 0.234 |
| large | FusedManifestoSoftLUT | float32 | ok | 1.813 | 0.094 | 2.081 | 0.094 | 6.746 | 0.234 | 8.834 | 0.234 |
| large | FusedManifestoSoftLUT | bfloat16 | ok | 1.117 | 0.047 | 1.228 | 0.047 | 6.453 | 0.234 | 7.683 | 0.234 |
| large | ManifestoHardLUT weight=softsign | float32 | out of memory | 2.566 | 0.219 | - | - | - | - | - | - |
| large | ManifestoHardLUT weight=softsign | bfloat16 | out of memory | 2.693 | 0.359 | - | - | - | - | - | - |
| large | ManifestoSoftLUT weight=softsign | float32 | ok | 5.368 | 0.594 | 21.075 | 5.094 | 42.766 | 5.013 | 63.848 | 5.094 |
| large | ManifestoSoftLUT weight=softsign | bfloat16 | ok | 5.397 | 0.734 | 21.257 | 5.234 | 42.816 | 5.107 | 64.069 | 5.234 |
| large | FusedManifestoHardLUT weight=softsign | float32 | ok | 1.008 | 0.094 | 1.116 | 0.094 | 6.316 | 0.234 | 7.432 | 0.234 |
| large | FusedManifestoHardLUT weight=softsign | bfloat16 | ok | 0.768 | 0.047 | 0.888 | 0.047 | 5.943 | 0.234 | 6.829 | 0.234 |
| large | FusedManifestoSoftLUT weight=softsign | float32 | ok | 1.836 | 0.094 | 2.025 | 0.094 | 7.056 | 0.234 | 9.078 | 0.234 |
| large | FusedManifestoSoftLUT weight=softsign | bfloat16 | ok | 1.181 | 0.047 | 1.267 | 0.047 | 6.888 | 0.234 | 8.157 | 0.234 |
| large | ConfidenceLUT n=1 | float32 | ok | 3.609 | 0.469 | 3.738 | 0.469 | 9.644 | 2.703 | 13.389 | 2.703 |
| large | ConfidenceLUT n=1 | bfloat16 | ok | 3.719 | 0.609 | 3.760 | 0.656 | 9.722 | 2.844 | 13.484 | 2.844 |
| large | ConfidenceLUT n=2 | float32 | ok | 4.838 | 0.875 | 4.665 | 0.844 | 16.164 | 3.141 | 20.823 | 3.141 |
| large | ConfidenceLUT n=2 | bfloat16 | ok | 5.394 | 1.016 | 4.851 | 1.031 | 15.909 | 3.281 | 20.763 | 3.281 |
| large | FusedConfidenceLUT n=1 | float32 | ok | 1.048 | 0.094 | 1.136 | 0.094 | 4.747 | 0.236 | 5.880 | 0.236 |
| large | FusedConfidenceLUT n=1 | bfloat16 | ok | 0.796 | 0.047 | 0.963 | 0.047 | 4.479 | 0.236 | 5.443 | 0.236 |
| large | FusedConfidenceLUT n=2 | float32 | ok | 1.872 | 0.094 | 1.981 | 0.094 | 7.369 | 0.236 | 9.351 | 0.236 |
| large | FusedConfidenceLUT n=2 | bfloat16 | ok | 1.204 | 0.047 | 1.389 | 0.047 | 6.970 | 0.236 | 8.370 | 0.236 |
| large | QuantisedConfidenceLUT n=1 | float32 | ok | 4.796 | 0.520 | 4.881 | 0.770 | 19.954 | 1.932 | 24.836 | 1.932 |
| large | QuantisedConfidenceLUT n=1 | bfloat16 | refused: QuantisedConfidenceLUT does not support low precision | - | - | - | - | - | - | - | - |
| large | QuantisedConfidenceLUT n=2 | float32 | ok | 11.369 | 2.031 | 11.227 | 2.328 | 38.325 | 4.169 | 49.546 | 4.169 |
| large | QuantisedConfidenceLUT n=2 | bfloat16 | refused: QuantisedConfidenceLUT does not support low precision | - | - | - | - | - | - | - | - |
| large | FusedQuantisedConfidenceLUT n=2 | float32 | ok | 9.673 | 1.658 | 9.413 | 1.406 | 39.050 | 3.641 | 48.467 | 3.641 |
| large | FusedQuantisedConfidenceLUT n=2 | bfloat16 | error: RuntimeError: FusedQuantisedConfidenceLUT: not a case the fused op covers: input dtype torch.bfloat16 (float32 only); table dtype torc | - | - | - | - | - | - | - | - |
| large | DeployedQuantisedConfidenceLUT | float32 | ok | 16.871 | 1.656 | - | - | - | - | - | - |
| large | DeployedQuantisedConfidenceLUT | bfloat16 | refused: DeployedQuantisedConfidenceLUT is fp32/fp64 inference only; got input dtype torch.bfloat16 | - | - | - | - | - | - | - | - |
