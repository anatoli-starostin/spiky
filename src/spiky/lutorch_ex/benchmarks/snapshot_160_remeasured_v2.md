# lutorch_ex performance snapshot: #160 re-measured, v2 (the cumulative BEFORE of #164)

**Code.** `a811390e`, #160's library code: the commit snapshot_160.json was measured at. Materialised from git, nothing changed.

**Hardware and software.** RTX 5090 (sm_120, 170 SMs), torch 2.9.1+cu130, Triton 3.5.1, CUDA 13.0, driver 595.84.

**Session (v2).** Measured 2026-10-13 in ONE session together with the other two v2 snapshots: #162's code (c8e48c36) re-measured, #160's code (a811390e) re-measured and #164 HEAD (d73edfad, after the per-table-integer saving), the three trees interleaved cell by cell in a rotating order (each tree first in a third of the cells).
- Each cell ran under each tree's own `python -m spiky.lutorch_ex.bench_snapshot` for that one cell: one fresh process per cell, a Dynamo reset, 10 warm-up + 30 timed iterations, median; peak GiB above the pre-step allocation.
- Champion geometry h16 tph64 nap8 d48 at 8,192 vectors (`champion`) and 32,768 (`large`).

**Conditions** (logged before and after every cell): SM up to 3.07 GHz (unlocked), 32–44 C. **No throttle or power-cap flag at any point** (the intermittent software power cap 0x4 seen in an earlier A/B run did not recur).

**No transient.** Nothing moved beyond its noise floor against #162 or #160 except the cells #164 targets, so no row was re-measured.

Comparisons: `figures_162_remeasured_vs_164_v2/` (primary, #162 being the parent) and `figures_160_vs_164_v2/` (cumulative), both bench_compare.py --same-session. The v1 files (`snapshot_164.json`, `snapshot_164_vs162.json`, `snapshot_16{0,2}_remeasured.json` and their figure directories) are the record of #164 BEFORE the saving and are kept as history.

| geometry | cartridge | dtype | status | eval forward ms | GiB | train forward ms | GiB | train backward ms | GiB | train total ms | GiB |
|---|---|---|---|---|---|---|---|---|---|---|---|
| champion | ManifestoHardLUT | float32 | ok | 0.647 | 0.055 | 17.913 | 6.712 | 7.327 | 3.517 | 25.242 | 6.712 |
| champion | ManifestoHardLUT | bfloat16 | ok | 0.685 | 0.125 | 17.992 | 6.782 | 7.337 | 3.528 | 25.330 | 6.782 |
| champion | ManifestoSoftLUT | float32 | ok | 1.226 | 0.148 | 5.056 | 1.273 | 10.218 | 1.265 | 15.274 | 1.273 |
| champion | ManifestoSoftLUT | bfloat16 | ok | 1.252 | 0.219 | 5.111 | 1.344 | 10.108 | 1.323 | 15.222 | 1.344 |
| champion | FusedManifestoHardLUT | float32 | ok | 0.252 | 0.023 | 0.311 | 0.023 | 1.286 | 0.094 | 1.577 | 0.094 |
| champion | FusedManifestoHardLUT | bfloat16 | ok | 0.193 | 0.012 | 0.254 | 0.012 | 1.158 | 0.094 | 1.412 | 0.094 |
| champion | FusedManifestoSoftLUT | float32 | ok | 0.461 | 0.023 | 0.549 | 0.023 | 1.713 | 0.094 | 2.264 | 0.094 |
| champion | FusedManifestoSoftLUT | bfloat16 | ok | 0.282 | 0.012 | 0.346 | 0.012 | 1.605 | 0.094 | 1.956 | 0.094 |
| champion | ManifestoHardLUT weight=softsign | float32 | ok | 0.671 | 0.055 | 17.958 | 6.743 | 7.660 | 3.579 | 25.619 | 6.743 |
| champion | ManifestoHardLUT weight=softsign | bfloat16 | ok | 0.689 | 0.125 | 18.003 | 6.813 | 7.664 | 3.591 | 25.667 | 6.813 |
| champion | ManifestoSoftLUT weight=softsign | float32 | ok | 1.293 | 0.148 | 5.074 | 1.273 | 10.544 | 1.327 | 15.620 | 1.327 |
| champion | ManifestoSoftLUT weight=softsign | bfloat16 | ok | 1.320 | 0.219 | 5.159 | 1.344 | 10.468 | 1.386 | 15.628 | 1.386 |
| champion | FusedManifestoHardLUT weight=softsign | float32 | ok | 0.267 | 0.023 | 0.341 | 0.023 | 1.593 | 0.094 | 1.916 | 0.094 |
| champion | FusedManifestoHardLUT weight=softsign | bfloat16 | ok | 0.215 | 0.012 | 0.296 | 0.012 | 1.493 | 0.094 | 1.789 | 0.094 |
| champion | FusedManifestoSoftLUT weight=softsign | float32 | ok | 0.481 | 0.023 | 0.579 | 0.023 | 1.899 | 0.094 | 2.465 | 0.094 |
| champion | FusedManifestoSoftLUT weight=softsign | bfloat16 | ok | 0.304 | 0.012 | 0.380 | 0.012 | 1.746 | 0.094 | 2.092 | 0.094 |
| champion | ConfidenceLUT n=1 | float32 | ok | 0.912 | 0.117 | 1.058 | 0.117 | 2.389 | 0.711 | 3.447 | 0.711 |
| champion | ConfidenceLUT n=1 | bfloat16 | ok | 0.943 | 0.188 | 1.074 | 0.199 | 2.386 | 0.746 | 3.461 | 0.746 |
| champion | ConfidenceLUT n=2 | float32 | ok | 1.282 | 0.219 | 1.347 | 0.211 | 4.069 | 0.820 | 5.415 | 0.820 |
| champion | ConfidenceLUT n=2 | bfloat16 | ok | 1.299 | 0.289 | 1.322 | 0.293 | 4.034 | 0.855 | 5.355 | 0.855 |
| champion | FusedConfidenceLUT n=1 | float32 | ok | 0.265 | 0.023 | 0.322 | 0.023 | 1.230 | 0.094 | 1.550 | 0.094 |
| champion | FusedConfidenceLUT n=1 | bfloat16 | ok | 0.204 | 0.012 | 0.270 | 0.012 | 1.147 | 0.094 | 1.415 | 0.094 |
| champion | FusedConfidenceLUT n=2 | float32 | ok | 0.473 | 0.023 | 0.535 | 0.023 | 1.926 | 0.094 | 2.461 | 0.094 |
| champion | FusedConfidenceLUT n=2 | bfloat16 | ok | 0.312 | 0.012 | 0.399 | 0.012 | 1.828 | 0.094 | 2.221 | 0.094 |
| champion | QuantisedConfidenceLUT n=2 | float32 | ok | 2.968 | 0.508 | 2.977 | 0.617 | 9.606 | 1.116 | 12.581 | 1.116 |
| champion | QuantisedConfidenceLUT n=2 | bfloat16 | refused: QuantisedConfidenceLUT does not support low precision | - | - | - | - | - | - | - | - |
| champion | FusedQuantisedConfidenceLUT n=2 | float32 | ok | 2.523 | 0.415 | 2.505 | 0.352 | 9.874 | 0.946 | 12.380 | 0.946 |
| champion | FusedQuantisedConfidenceLUT n=2 | bfloat16 | error: RuntimeError: FusedQuantisedConfidenceLUT: not a case the fused op covers: input dtype torch.bfloat16 (float32 only); table dtype torc | - | - | - | - | - | - | - | - |
| champion | DeployedQuantisedConfidenceLUT | float32 | ok | 2.893 | 0.414 | - | - | - | - | - | - |
| champion | DeployedQuantisedConfidenceLUT | bfloat16 | refused: DeployedQuantisedConfidenceLUT is fp32/fp64 inference only; got input dtype torch.bfloat16 | - | - | - | - | - | - | - | - |
| large | ManifestoHardLUT | float32 | out of memory | 2.479 | 0.219 | - | - | - | - | - | - |
| large | ManifestoHardLUT | bfloat16 | out of memory | 2.676 | 0.359 | - | - | - | - | - | - |
| large | ManifestoSoftLUT | float32 | ok | 5.040 | 0.594 | 20.816 | 5.094 | 40.997 | 4.763 | 61.813 | 5.094 |
| large | ManifestoSoftLUT | bfloat16 | ok | 5.067 | 0.734 | 20.991 | 5.234 | 40.929 | 4.857 | 61.919 | 5.234 |
| large | FusedManifestoHardLUT | float32 | ok | 0.998 | 0.094 | 1.087 | 0.094 | 4.860 | 0.234 | 5.949 | 0.234 |
| large | FusedManifestoHardLUT | bfloat16 | ok | 0.750 | 0.047 | 0.862 | 0.047 | 4.491 | 0.234 | 5.351 | 0.234 |
| large | FusedManifestoSoftLUT | float32 | ok | 1.819 | 0.094 | 2.110 | 0.094 | 6.820 | 0.234 | 8.924 | 0.234 |
| large | FusedManifestoSoftLUT | bfloat16 | ok | 1.118 | 0.047 | 1.221 | 0.047 | 6.418 | 0.234 | 7.640 | 0.234 |
| large | ManifestoHardLUT weight=softsign | float32 | out of memory | 2.567 | 0.219 | - | - | - | - | - | - |
| large | ManifestoHardLUT weight=softsign | bfloat16 | out of memory | 2.690 | 0.359 | - | - | - | - | - | - |
| large | ManifestoSoftLUT weight=softsign | float32 | ok | 5.374 | 0.594 | 21.068 | 5.094 | 42.740 | 5.013 | 63.811 | 5.094 |
| large | ManifestoSoftLUT weight=softsign | bfloat16 | ok | 5.408 | 0.734 | 21.257 | 5.234 | 42.794 | 5.107 | 64.051 | 5.234 |
| large | FusedManifestoHardLUT weight=softsign | float32 | ok | 1.013 | 0.094 | 1.122 | 0.094 | 6.316 | 0.234 | 7.440 | 0.234 |
| large | FusedManifestoHardLUT weight=softsign | bfloat16 | ok | 0.768 | 0.047 | 0.888 | 0.047 | 5.943 | 0.234 | 6.830 | 0.234 |
| large | FusedManifestoSoftLUT weight=softsign | float32 | ok | 1.838 | 0.094 | 2.037 | 0.094 | 7.107 | 0.234 | 9.143 | 0.234 |
| large | FusedManifestoSoftLUT weight=softsign | bfloat16 | ok | 1.174 | 0.047 | 1.254 | 0.047 | 6.831 | 0.234 | 8.088 | 0.234 |
| large | ConfidenceLUT n=1 | float32 | ok | 3.593 | 0.469 | 3.756 | 0.469 | 9.599 | 2.703 | 13.360 | 2.703 |
| large | ConfidenceLUT n=1 | bfloat16 | ok | 3.707 | 0.609 | 3.764 | 0.656 | 9.670 | 2.844 | 13.434 | 2.844 |
| large | ConfidenceLUT n=2 | float32 | ok | 4.988 | 0.875 | 4.834 | 0.844 | 16.100 | 3.141 | 20.924 | 3.141 |
| large | ConfidenceLUT n=2 | bfloat16 | ok | 5.178 | 1.016 | 4.738 | 1.031 | 15.841 | 3.281 | 20.580 | 3.281 |
| large | FusedConfidenceLUT n=1 | float32 | ok | 1.049 | 0.094 | 1.137 | 0.094 | 4.732 | 0.236 | 5.871 | 0.236 |
| large | FusedConfidenceLUT n=1 | bfloat16 | ok | 0.798 | 0.047 | 0.968 | 0.047 | 4.486 | 0.236 | 5.456 | 0.236 |
| large | FusedConfidenceLUT n=2 | float32 | ok | 1.872 | 0.094 | 1.984 | 0.094 | 7.393 | 0.236 | 9.376 | 0.236 |
| large | FusedConfidenceLUT n=2 | bfloat16 | ok | 1.211 | 0.047 | 1.397 | 0.047 | 6.975 | 0.236 | 8.366 | 0.236 |
| large | QuantisedConfidenceLUT n=2 | float32 | ok | 11.388 | 2.031 | 11.219 | 2.328 | 38.338 | 4.169 | 49.555 | 4.169 |
| large | QuantisedConfidenceLUT n=2 | bfloat16 | refused: QuantisedConfidenceLUT does not support low precision | - | - | - | - | - | - | - | - |
| large | FusedQuantisedConfidenceLUT n=2 | float32 | ok | 9.682 | 1.658 | 9.413 | 1.406 | 39.018 | 3.641 | 48.431 | 3.641 |
| large | FusedQuantisedConfidenceLUT n=2 | bfloat16 | error: RuntimeError: FusedQuantisedConfidenceLUT: not a case the fused op covers: input dtype torch.bfloat16 (float32 only); table dtype torc | - | - | - | - | - | - | - | - |
| large | DeployedQuantisedConfidenceLUT | float32 | ok | 16.795 | 1.656 | - | - | - | - | - | - |
| large | DeployedQuantisedConfidenceLUT | bfloat16 | refused: DeployedQuantisedConfidenceLUT is fp32/fp64 inference only; got input dtype torch.bfloat16 | - | - | - | - | - | - | - | - |

> **Note (2026-10-13):** the benchmark modules have since moved: `spiky.lutorch_ex.bench_snapshot` is now `spiky.lutorch_ex.benchmarks.bench_snapshot` (likewise `bench`, `bench_compare`, `bench_fused_confidence`). The command above is recorded as it was run.
