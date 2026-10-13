# lutorch_ex performance snapshot: #162 re-measured, v2 (the BEFORE of #164, its parent)

**Code.** `c8e48c36`, #162's library code: the commit snapshot_162.json was measured at. Materialised from git, nothing changed.

**Hardware and software.** RTX 5090 (sm_120, 170 SMs), torch 2.9.1+cu130, Triton 3.5.1, CUDA 13.0, driver 595.84.

**Session (v2).** Measured 2026-10-13 in ONE session together with the other two v2 snapshots: #162's code (c8e48c36) re-measured, #160's code (a811390e) re-measured and #164 HEAD (d73edfad, after the per-table-integer saving), the three trees interleaved cell by cell in a rotating order (each tree first in a third of the cells).
- Each cell ran under each tree's own `python -m spiky.lutorch_ex.bench_snapshot` for that one cell: one fresh process per cell, a Dynamo reset, 10 warm-up + 30 timed iterations, median; peak GiB above the pre-step allocation.
- Champion geometry h16 tph64 nap8 d48 at 8,192 vectors (`champion`) and 32,768 (`large`).

**Conditions** (logged before and after every cell): SM up to 3.07 GHz (unlocked), 32–44 C. **No throttle or power-cap flag at any point** (the intermittent software power cap 0x4 seen in an earlier A/B run did not recur).

**No transient.** Nothing moved beyond its noise floor against #162 or #160 except the cells #164 targets, so no row was re-measured.

Comparisons: `figures_162_remeasured_vs_164_v2/` (primary, #162 being the parent) and `figures_160_vs_164_v2/` (cumulative), both bench_compare.py --same-session. The v1 files (`snapshot_164.json`, `snapshot_164_vs162.json`, `snapshot_16{0,2}_remeasured.json` and their figure directories) are the record of #164 BEFORE the saving and are kept as history.

| geometry | cartridge | dtype | status | eval forward ms | GiB | train forward ms | GiB | train backward ms | GiB | train total ms | GiB |
|---|---|---|---|---|---|---|---|---|---|---|---|
| champion | ManifestoHardLUT | float32 | ok | 0.649 | 0.055 | 5.948 | 1.273 | 8.979 | 1.047 | 14.925 | 1.273 |
| champion | ManifestoHardLUT | bfloat16 | ok | 0.685 | 0.125 | 6.007 | 1.344 | 8.928 | 1.082 | 14.938 | 1.344 |
| champion | ManifestoSoftLUT | float32 | ok | 1.227 | 0.148 | 5.057 | 1.273 | 10.212 | 1.265 | 15.270 | 1.273 |
| champion | ManifestoSoftLUT | bfloat16 | ok | 1.250 | 0.219 | 5.118 | 1.344 | 10.114 | 1.323 | 15.231 | 1.344 |
| champion | FusedManifestoHardLUT | float32 | ok | 0.252 | 0.023 | 0.313 | 0.023 | 1.264 | 0.094 | 1.566 | 0.094 |
| champion | FusedManifestoHardLUT | bfloat16 | ok | 0.194 | 0.012 | 0.250 | 0.012 | 1.151 | 0.094 | 1.399 | 0.094 |
| champion | FusedManifestoSoftLUT | float32 | ok | 0.463 | 0.023 | 0.547 | 0.023 | 1.726 | 0.094 | 2.271 | 0.094 |
| champion | FusedManifestoSoftLUT | bfloat16 | ok | 0.283 | 0.012 | 0.336 | 0.012 | 1.592 | 0.094 | 1.944 | 0.094 |
| champion | ManifestoHardLUT weight=softsign | float32 | ok | 0.669 | 0.055 | 5.989 | 1.273 | 9.284 | 1.047 | 15.274 | 1.273 |
| champion | ManifestoHardLUT weight=softsign | bfloat16 | ok | 0.689 | 0.125 | 6.027 | 1.344 | 9.237 | 1.082 | 15.265 | 1.344 |
| champion | ManifestoSoftLUT weight=softsign | float32 | ok | 1.291 | 0.148 | 5.086 | 1.273 | 10.544 | 1.327 | 15.630 | 1.327 |
| champion | ManifestoSoftLUT weight=softsign | bfloat16 | ok | 1.320 | 0.219 | 5.155 | 1.344 | 10.466 | 1.386 | 15.620 | 1.386 |
| champion | FusedManifestoHardLUT weight=softsign | float32 | ok | 0.267 | 0.023 | 0.314 | 0.023 | 1.583 | 0.094 | 1.899 | 0.094 |
| champion | FusedManifestoHardLUT weight=softsign | bfloat16 | ok | 0.215 | 0.012 | 0.294 | 0.012 | 1.493 | 0.094 | 1.787 | 0.094 |
| champion | FusedManifestoSoftLUT weight=softsign | float32 | ok | 0.481 | 0.023 | 0.572 | 0.023 | 1.860 | 0.094 | 2.429 | 0.094 |
| champion | FusedManifestoSoftLUT weight=softsign | bfloat16 | ok | 0.304 | 0.012 | 0.388 | 0.012 | 1.785 | 0.094 | 2.153 | 0.094 |
| champion | ConfidenceLUT n=1 | float32 | ok | 0.910 | 0.117 | 1.071 | 0.117 | 2.394 | 0.711 | 3.466 | 0.711 |
| champion | ConfidenceLUT n=1 | bfloat16 | ok | 0.952 | 0.188 | 1.086 | 0.199 | 2.388 | 0.746 | 3.475 | 0.746 |
| champion | ConfidenceLUT n=2 | float32 | ok | 1.279 | 0.219 | 1.347 | 0.211 | 4.066 | 0.820 | 5.412 | 0.820 |
| champion | ConfidenceLUT n=2 | bfloat16 | ok | 1.303 | 0.289 | 1.323 | 0.293 | 4.035 | 0.855 | 5.358 | 0.855 |
| champion | FusedConfidenceLUT n=1 | float32 | ok | 0.266 | 0.023 | 0.295 | 0.023 | 1.224 | 0.094 | 1.522 | 0.094 |
| champion | FusedConfidenceLUT n=1 | bfloat16 | ok | 0.204 | 0.012 | 0.276 | 0.012 | 1.156 | 0.094 | 1.433 | 0.094 |
| champion | FusedConfidenceLUT n=2 | float32 | ok | 0.472 | 0.023 | 0.517 | 0.023 | 1.950 | 0.094 | 2.474 | 0.094 |
| champion | FusedConfidenceLUT n=2 | bfloat16 | ok | 0.312 | 0.012 | 0.385 | 0.012 | 1.831 | 0.094 | 2.199 | 0.094 |
| champion | QuantisedConfidenceLUT n=2 | float32 | ok | 2.579 | 0.259 | 3.037 | 0.516 | 9.492 | 1.116 | 12.530 | 1.116 |
| champion | QuantisedConfidenceLUT n=2 | bfloat16 | refused: QuantisedConfidenceLUT does not support low precision | - | - | - | - | - | - | - | - |
| champion | FusedQuantisedConfidenceLUT n=2 | float32 | ok | 2.523 | 0.415 | 2.503 | 0.352 | 9.872 | 0.946 | 12.377 | 0.946 |
| champion | FusedQuantisedConfidenceLUT n=2 | bfloat16 | error: RuntimeError: FusedQuantisedConfidenceLUT: not a case the fused op covers: input dtype torch.bfloat16 (float32 only); table dtype torc | - | - | - | - | - | - | - | - |
| champion | DeployedQuantisedConfidenceLUT | float32 | ok | 2.897 | 0.414 | - | - | - | - | - | - |
| champion | DeployedQuantisedConfidenceLUT | bfloat16 | refused: DeployedQuantisedConfidenceLUT is fp32/fp64 inference only; got input dtype torch.bfloat16 | - | - | - | - | - | - | - | - |
| large | ManifestoHardLUT | float32 | ok | 2.552 | 0.219 | 24.527 | 5.094 | 35.908 | 4.188 | 60.443 | 5.094 |
| large | ManifestoHardLUT | bfloat16 | ok | 2.676 | 0.359 | 24.729 | 5.234 | 35.923 | 4.188 | 60.654 | 5.234 |
| large | ManifestoSoftLUT | float32 | ok | 5.027 | 0.594 | 20.815 | 5.094 | 41.001 | 4.763 | 61.816 | 5.094 |
| large | ManifestoSoftLUT | bfloat16 | ok | 5.060 | 0.734 | 20.994 | 5.234 | 40.925 | 4.857 | 61.924 | 5.234 |
| large | FusedManifestoHardLUT | float32 | ok | 1.000 | 0.094 | 1.085 | 0.094 | 4.865 | 0.234 | 5.951 | 0.234 |
| large | FusedManifestoHardLUT | bfloat16 | ok | 0.750 | 0.047 | 0.856 | 0.047 | 4.479 | 0.234 | 5.336 | 0.234 |
| large | FusedManifestoSoftLUT | float32 | ok | 1.819 | 0.094 | 2.075 | 0.094 | 6.689 | 0.234 | 8.764 | 0.234 |
| large | FusedManifestoSoftLUT | bfloat16 | ok | 1.102 | 0.047 | 1.221 | 0.047 | 6.422 | 0.234 | 7.641 | 0.234 |
| large | ManifestoHardLUT weight=softsign | float32 | ok | 2.564 | 0.219 | 24.786 | 5.094 | 37.734 | 4.188 | 62.524 | 5.094 |
| large | ManifestoHardLUT weight=softsign | bfloat16 | ok | 2.691 | 0.359 | 24.988 | 5.234 | 37.745 | 4.188 | 62.736 | 5.234 |
| large | ManifestoSoftLUT weight=softsign | float32 | ok | 5.368 | 0.594 | 21.069 | 5.094 | 42.725 | 5.013 | 63.798 | 5.094 |
| large | ManifestoSoftLUT weight=softsign | bfloat16 | ok | 5.424 | 0.734 | 21.257 | 5.234 | 42.824 | 5.107 | 64.084 | 5.234 |
| large | FusedManifestoHardLUT weight=softsign | float32 | ok | 1.010 | 0.094 | 1.119 | 0.094 | 6.316 | 0.234 | 7.434 | 0.234 |
| large | FusedManifestoHardLUT weight=softsign | bfloat16 | ok | 0.768 | 0.047 | 0.886 | 0.047 | 5.942 | 0.234 | 6.827 | 0.234 |
| large | FusedManifestoSoftLUT weight=softsign | float32 | ok | 1.835 | 0.094 | 2.033 | 0.094 | 7.087 | 0.234 | 9.117 | 0.234 |
| large | FusedManifestoSoftLUT weight=softsign | bfloat16 | ok | 1.164 | 0.047 | 1.247 | 0.047 | 6.793 | 0.234 | 8.031 | 0.234 |
| large | ConfidenceLUT n=1 | float32 | ok | 3.614 | 0.469 | 3.735 | 0.469 | 9.591 | 2.703 | 13.336 | 2.703 |
| large | ConfidenceLUT n=1 | bfloat16 | ok | 3.683 | 0.609 | 3.775 | 0.656 | 9.668 | 2.844 | 13.443 | 2.844 |
| large | ConfidenceLUT n=2 | float32 | ok | 4.965 | 0.875 | 4.846 | 0.844 | 16.158 | 3.141 | 21.005 | 3.141 |
| large | ConfidenceLUT n=2 | bfloat16 | ok | 5.182 | 1.016 | 4.733 | 1.031 | 15.834 | 3.281 | 20.567 | 3.281 |
| large | FusedConfidenceLUT n=1 | float32 | ok | 1.049 | 0.094 | 1.136 | 0.094 | 4.745 | 0.236 | 5.883 | 0.236 |
| large | FusedConfidenceLUT n=1 | bfloat16 | ok | 0.796 | 0.047 | 0.959 | 0.047 | 4.484 | 0.236 | 5.440 | 0.236 |
| large | FusedConfidenceLUT n=2 | float32 | ok | 1.873 | 0.094 | 1.965 | 0.094 | 7.388 | 0.236 | 9.393 | 0.236 |
| large | FusedConfidenceLUT n=2 | bfloat16 | ok | 1.214 | 0.047 | 1.371 | 0.047 | 6.961 | 0.236 | 8.359 | 0.236 |
| large | QuantisedConfidenceLUT n=2 | float32 | ok | 9.967 | 0.895 | 11.568 | 1.922 | 38.115 | 4.169 | 49.686 | 4.169 |
| large | QuantisedConfidenceLUT n=2 | bfloat16 | refused: QuantisedConfidenceLUT does not support low precision | - | - | - | - | - | - | - | - |
| large | FusedQuantisedConfidenceLUT n=2 | float32 | ok | 9.682 | 1.658 | 9.412 | 1.406 | 39.021 | 3.641 | 48.433 | 3.641 |
| large | FusedQuantisedConfidenceLUT n=2 | bfloat16 | error: RuntimeError: FusedQuantisedConfidenceLUT: not a case the fused op covers: input dtype torch.bfloat16 (float32 only); table dtype torc | - | - | - | - | - | - | - | - |
| large | DeployedQuantisedConfidenceLUT | float32 | ok | 16.806 | 1.656 | - | - | - | - | - | - |
| large | DeployedQuantisedConfidenceLUT | bfloat16 | refused: DeployedQuantisedConfidenceLUT is fp32/fp64 inference only; got input dtype torch.bfloat16 | - | - | - | - | - | - | - | - |

> **Note (2026-10-13):** the benchmark modules have since moved: `spiky.lutorch_ex.bench_snapshot` is now `spiky.lutorch_ex.benchmarks.bench_snapshot` (likewise `bench`, `bench_compare`, `bench_fused_confidence`). The command above is recorded as it was run.
