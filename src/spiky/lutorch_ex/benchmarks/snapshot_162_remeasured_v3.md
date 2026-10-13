# lutorch_ex performance snapshot: #162 re-measured, v3 (the BEFORE of #164, its parent)

**Code.** `c8e48c36`, #162's library code. Materialised from git, nothing changed.

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
| champion | ManifestoHardLUT | float32 | ok | 0.665 | 0.055 | 5.932 | 1.273 | 8.967 | 1.047 | 14.900 | 1.273 |
| champion | ManifestoHardLUT | bfloat16 | ok | 0.699 | 0.125 | 5.999 | 1.344 | 8.926 | 1.082 | 14.928 | 1.344 |
| champion | ManifestoSoftLUT | float32 | ok | 1.226 | 0.148 | 5.046 | 1.273 | 10.202 | 1.265 | 15.250 | 1.273 |
| champion | ManifestoSoftLUT | bfloat16 | ok | 1.250 | 0.219 | 5.123 | 1.344 | 10.135 | 1.323 | 15.258 | 1.344 |
| champion | FusedManifestoHardLUT | float32 | ok | 0.252 | 0.023 | 0.290 | 0.023 | 1.282 | 0.094 | 1.574 | 0.094 |
| champion | FusedManifestoHardLUT | bfloat16 | ok | 0.192 | 0.012 | 0.254 | 0.012 | 1.161 | 0.094 | 1.417 | 0.094 |
| champion | FusedManifestoSoftLUT | float32 | ok | 0.459 | 0.023 | 0.535 | 0.023 | 1.706 | 0.094 | 2.247 | 0.094 |
| champion | FusedManifestoSoftLUT | bfloat16 | ok | 0.282 | 0.012 | 0.350 | 0.012 | 1.586 | 0.094 | 1.936 | 0.094 |
| champion | ManifestoHardLUT weight=softsign | float32 | ok | 0.670 | 0.055 | 5.968 | 1.273 | 9.301 | 1.047 | 15.269 | 1.273 |
| champion | ManifestoHardLUT weight=softsign | bfloat16 | ok | 0.708 | 0.125 | 6.023 | 1.344 | 9.238 | 1.082 | 15.264 | 1.344 |
| champion | ManifestoSoftLUT weight=softsign | float32 | ok | 1.296 | 0.148 | 5.069 | 1.273 | 10.530 | 1.327 | 15.599 | 1.327 |
| champion | ManifestoSoftLUT weight=softsign | bfloat16 | ok | 1.320 | 0.219 | 5.162 | 1.344 | 10.440 | 1.386 | 15.603 | 1.386 |
| champion | FusedManifestoHardLUT weight=softsign | float32 | ok | 0.267 | 0.023 | 0.342 | 0.023 | 1.582 | 0.094 | 1.918 | 0.094 |
| champion | FusedManifestoHardLUT weight=softsign | bfloat16 | ok | 0.215 | 0.012 | 0.296 | 0.012 | 1.494 | 0.094 | 1.789 | 0.094 |
| champion | FusedManifestoSoftLUT weight=softsign | float32 | ok | 0.477 | 0.023 | 0.577 | 0.023 | 1.868 | 0.094 | 2.447 | 0.094 |
| champion | FusedManifestoSoftLUT weight=softsign | bfloat16 | ok | 0.304 | 0.012 | 0.375 | 0.012 | 1.744 | 0.094 | 2.089 | 0.094 |
| champion | ConfidenceLUT n=1 | float32 | ok | 0.912 | 0.117 | 1.075 | 0.117 | 2.365 | 0.711 | 3.438 | 0.711 |
| champion | ConfidenceLUT n=1 | bfloat16 | ok | 0.942 | 0.188 | 1.080 | 0.199 | 2.363 | 0.746 | 3.442 | 0.746 |
| champion | ConfidenceLUT n=2 | float32 | ok | 1.256 | 0.219 | 1.326 | 0.211 | 4.067 | 0.820 | 5.393 | 0.820 |
| champion | ConfidenceLUT n=2 | bfloat16 | ok | 1.280 | 0.289 | 1.287 | 0.293 | 4.027 | 0.855 | 5.314 | 0.855 |
| champion | FusedConfidenceLUT n=1 | float32 | ok | 0.267 | 0.023 | 0.302 | 0.023 | 1.234 | 0.094 | 1.534 | 0.094 |
| champion | FusedConfidenceLUT n=1 | bfloat16 | ok | 0.204 | 0.012 | 0.275 | 0.012 | 1.150 | 0.094 | 1.424 | 0.094 |
| champion | FusedConfidenceLUT n=2 | float32 | ok | 0.472 | 0.023 | 0.539 | 0.023 | 1.933 | 0.094 | 2.470 | 0.094 |
| champion | FusedConfidenceLUT n=2 | bfloat16 | ok | 0.313 | 0.012 | 0.398 | 0.012 | 1.840 | 0.094 | 2.223 | 0.094 |
| champion | QuantisedConfidenceLUT n=1 | float32 | ok | 1.300 | 0.165 | 1.360 | 0.228 | 5.074 | 0.555 | 6.433 | 0.555 |
| champion | QuantisedConfidenceLUT n=1 | bfloat16 | refused: QuantisedConfidenceLUT does not support low precision | - | - | - | - | - | - | - | - |
| champion | QuantisedConfidenceLUT n=2 | float32 | ok | 2.580 | 0.259 | 3.035 | 0.516 | 9.496 | 1.116 | 12.531 | 1.116 |
| champion | QuantisedConfidenceLUT n=2 | bfloat16 | refused: QuantisedConfidenceLUT does not support low precision | - | - | - | - | - | - | - | - |
| champion | FusedQuantisedConfidenceLUT n=2 | float32 | ok | 2.525 | 0.415 | 2.507 | 0.352 | 9.877 | 0.946 | 12.384 | 0.946 |
| champion | FusedQuantisedConfidenceLUT n=2 | bfloat16 | error: RuntimeError: FusedQuantisedConfidenceLUT: not a case the fused op covers: input dtype torch.bfloat16 (float32 only); table dtype torc | - | - | - | - | - | - | - | - |
| champion | DeployedQuantisedConfidenceLUT | float32 | ok | 2.898 | 0.414 | - | - | - | - | - | - |
| champion | DeployedQuantisedConfidenceLUT | bfloat16 | refused: DeployedQuantisedConfidenceLUT is fp32/fp64 inference only; got input dtype torch.bfloat16 | - | - | - | - | - | - | - | - |
| large | ManifestoHardLUT | float32 | ok | 2.477 | 0.219 | 24.526 | 5.094 | 35.922 | 4.188 | 60.449 | 5.094 |
| large | ManifestoHardLUT | bfloat16 | ok | 2.604 | 0.359 | 24.735 | 5.234 | 35.943 | 4.188 | 60.679 | 5.234 |
| large | ManifestoSoftLUT | float32 | ok | 5.038 | 0.594 | 20.812 | 5.094 | 40.968 | 4.763 | 61.781 | 5.094 |
| large | ManifestoSoftLUT | bfloat16 | ok | 5.089 | 0.734 | 20.985 | 5.234 | 40.921 | 4.857 | 61.905 | 5.234 |
| large | FusedManifestoHardLUT | float32 | ok | 0.993 | 0.094 | 1.079 | 0.094 | 4.855 | 0.234 | 5.933 | 0.234 |
| large | FusedManifestoHardLUT | bfloat16 | ok | 0.749 | 0.047 | 0.855 | 0.047 | 4.478 | 0.234 | 5.339 | 0.234 |
| large | FusedManifestoSoftLUT | float32 | ok | 1.812 | 0.094 | 2.063 | 0.094 | 6.700 | 0.234 | 8.747 | 0.234 |
| large | FusedManifestoSoftLUT | bfloat16 | ok | 1.121 | 0.047 | 1.205 | 0.047 | 6.311 | 0.234 | 7.516 | 0.234 |
| large | ManifestoHardLUT weight=softsign | float32 | ok | 2.567 | 0.219 | 24.780 | 5.094 | 37.704 | 4.188 | 62.485 | 5.094 |
| large | ManifestoHardLUT weight=softsign | bfloat16 | ok | 2.692 | 0.359 | 24.986 | 5.234 | 37.777 | 4.188 | 62.765 | 5.234 |
| large | ManifestoSoftLUT weight=softsign | float32 | ok | 5.379 | 0.594 | 21.072 | 5.094 | 42.791 | 5.013 | 63.865 | 5.094 |
| large | ManifestoSoftLUT weight=softsign | bfloat16 | ok | 5.384 | 0.734 | 21.251 | 5.234 | 42.818 | 5.107 | 64.072 | 5.234 |
| large | FusedManifestoHardLUT weight=softsign | float32 | ok | 1.007 | 0.094 | 1.115 | 0.094 | 6.313 | 0.234 | 7.429 | 0.234 |
| large | FusedManifestoHardLUT weight=softsign | bfloat16 | ok | 0.766 | 0.047 | 0.887 | 0.047 | 5.948 | 0.234 | 6.834 | 0.234 |
| large | FusedManifestoSoftLUT weight=softsign | float32 | ok | 1.837 | 0.094 | 2.028 | 0.094 | 7.069 | 0.234 | 9.100 | 0.234 |
| large | FusedManifestoSoftLUT weight=softsign | bfloat16 | ok | 1.182 | 0.047 | 1.255 | 0.047 | 6.780 | 0.234 | 8.043 | 0.234 |
| large | ConfidenceLUT n=1 | float32 | ok | 3.611 | 0.469 | 3.729 | 0.469 | 9.655 | 2.703 | 13.391 | 2.703 |
| large | ConfidenceLUT n=1 | bfloat16 | ok | 3.701 | 0.609 | 3.774 | 0.656 | 9.727 | 2.844 | 13.499 | 2.844 |
| large | ConfidenceLUT n=2 | float32 | ok | 4.839 | 0.875 | 4.696 | 0.844 | 16.175 | 3.141 | 20.878 | 3.141 |
| large | ConfidenceLUT n=2 | bfloat16 | ok | 5.422 | 1.016 | 4.839 | 1.031 | 15.912 | 3.281 | 20.753 | 3.281 |
| large | FusedConfidenceLUT n=1 | float32 | ok | 1.047 | 0.094 | 1.135 | 0.094 | 4.742 | 0.236 | 5.871 | 0.236 |
| large | FusedConfidenceLUT n=1 | bfloat16 | ok | 0.796 | 0.047 | 0.965 | 0.047 | 4.484 | 0.236 | 5.448 | 0.236 |
| large | FusedConfidenceLUT n=2 | float32 | ok | 1.872 | 0.094 | 1.972 | 0.094 | 7.355 | 0.236 | 9.328 | 0.236 |
| large | FusedConfidenceLUT n=2 | bfloat16 | ok | 1.205 | 0.047 | 1.394 | 0.047 | 6.966 | 0.236 | 8.356 | 0.236 |
| large | QuantisedConfidenceLUT n=1 | float32 | ok | 4.793 | 0.520 | 4.879 | 0.770 | 19.862 | 1.932 | 24.747 | 1.932 |
| large | QuantisedConfidenceLUT n=1 | bfloat16 | refused: QuantisedConfidenceLUT does not support low precision | - | - | - | - | - | - | - | - |
| large | QuantisedConfidenceLUT n=2 | float32 | ok | 9.967 | 0.895 | 11.553 | 1.922 | 38.075 | 4.169 | 49.630 | 4.169 |
| large | QuantisedConfidenceLUT n=2 | bfloat16 | refused: QuantisedConfidenceLUT does not support low precision | - | - | - | - | - | - | - | - |
| large | FusedQuantisedConfidenceLUT n=2 | float32 | ok | 9.667 | 1.658 | 9.418 | 1.406 | 39.069 | 3.641 | 48.488 | 3.641 |
| large | FusedQuantisedConfidenceLUT n=2 | bfloat16 | error: RuntimeError: FusedQuantisedConfidenceLUT: not a case the fused op covers: input dtype torch.bfloat16 (float32 only); table dtype torc | - | - | - | - | - | - | - | - |
| large | DeployedQuantisedConfidenceLUT | float32 | ok | 16.872 | 1.656 | - | - | - | - | - | - |
| large | DeployedQuantisedConfidenceLUT | bfloat16 | refused: DeployedQuantisedConfidenceLUT is fp32/fp64 inference only; got input dtype torch.bfloat16 | - | - | - | - | - | - | - | - |
