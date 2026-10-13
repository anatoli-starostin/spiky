# lutorch_ex performance snapshot: #164 HEAD, v2 (THE SNAPSHOT OF RECORD)

**Code.** `d73edfad`, #164 HEAD, after the per-table-integer saving (ae931703); no uncommitted changes in lutorch_ex. Paired with BOTH snapshot_162_remeasured_v2 and snapshot_160_remeasured_v2 (one measurement of HEAD, same session as both).

**Hardware and software.** RTX 5090 (sm_120, 170 SMs), torch 2.9.1+cu130, Triton 3.5.1, CUDA 13.0, driver 595.84.

**Session (v2).** Measured 2026-10-13 in ONE session together with the other two v2 snapshots: #162's code (c8e48c36) re-measured, #160's code (a811390e) re-measured and #164 HEAD (d73edfad, after the per-table-integer saving), the three trees interleaved cell by cell in a rotating order (each tree first in a third of the cells).
- Each cell ran under each tree's own `python -m spiky.lutorch_ex.bench_snapshot` for that one cell: one fresh process per cell, a Dynamo reset, 10 warm-up + 30 timed iterations, median; peak GiB above the pre-step allocation.
- Champion geometry h16 tph64 nap8 d48 at 8,192 vectors (`champion`) and 32,768 (`large`).

**Conditions** (logged before and after every cell): SM up to 3.07 GHz (unlocked), 32–44 C. **No throttle or power-cap flag at any point** (the intermittent software power cap 0x4 seen in an earlier A/B run did not recur).

**No transient.** Nothing moved beyond its noise floor against #162 or #160 except the cells #164 targets, so no row was re-measured.

Comparisons: `figures_162_remeasured_vs_164_v2/` (primary, #162 being the parent) and `figures_160_vs_164_v2/` (cumulative), both bench_compare.py --same-session. The v1 files (`snapshot_164.json`, `snapshot_164_vs162.json`, `snapshot_16{0,2}_remeasured.json` and their figure directories) are the record of #164 BEFORE the saving and are kept as history.

| geometry | cartridge | dtype | status | eval forward ms | GiB | train forward ms | GiB | train backward ms | GiB | train total ms | GiB |
|---|---|---|---|---|---|---|---|---|---|---|---|
| champion | ManifestoHardLUT | float32 | ok | 0.641 | 0.055 | 5.949 | 1.273 | 8.981 | 1.047 | 14.931 | 1.273 |
| champion | ManifestoHardLUT | bfloat16 | ok | 0.685 | 0.125 | 6.008 | 1.344 | 8.928 | 1.082 | 14.934 | 1.344 |
| champion | ManifestoSoftLUT | float32 | ok | 1.220 | 0.148 | 5.057 | 1.273 | 10.212 | 1.265 | 15.268 | 1.273 |
| champion | ManifestoSoftLUT | bfloat16 | ok | 1.249 | 0.219 | 5.116 | 1.344 | 10.112 | 1.323 | 15.228 | 1.344 |
| champion | FusedManifestoHardLUT | float32 | ok | 0.252 | 0.023 | 0.314 | 0.023 | 1.321 | 0.094 | 1.635 | 0.094 |
| champion | FusedManifestoHardLUT | bfloat16 | ok | 0.192 | 0.012 | 0.249 | 0.012 | 1.194 | 0.094 | 1.434 | 0.094 |
| champion | FusedManifestoSoftLUT | float32 | ok | 0.463 | 0.023 | 0.524 | 0.023 | 1.703 | 0.094 | 2.226 | 0.094 |
| champion | FusedManifestoSoftLUT | bfloat16 | ok | 0.287 | 0.012 | 0.340 | 0.012 | 1.619 | 0.094 | 1.964 | 0.094 |
| champion | ManifestoHardLUT weight=softsign | float32 | ok | 0.663 | 0.055 | 5.974 | 1.273 | 9.292 | 1.047 | 15.268 | 1.273 |
| champion | ManifestoHardLUT weight=softsign | bfloat16 | ok | 0.688 | 0.125 | 6.031 | 1.344 | 9.237 | 1.082 | 15.266 | 1.344 |
| champion | ManifestoSoftLUT weight=softsign | float32 | ok | 1.287 | 0.148 | 5.074 | 1.273 | 10.544 | 1.327 | 15.620 | 1.327 |
| champion | ManifestoSoftLUT weight=softsign | bfloat16 | ok | 1.321 | 0.219 | 5.154 | 1.344 | 10.470 | 1.386 | 15.625 | 1.386 |
| champion | FusedManifestoHardLUT weight=softsign | float32 | ok | 0.270 | 0.023 | 0.311 | 0.023 | 1.593 | 0.094 | 1.900 | 0.094 |
| champion | FusedManifestoHardLUT weight=softsign | bfloat16 | ok | 0.214 | 0.012 | 0.293 | 0.012 | 1.491 | 0.094 | 1.784 | 0.094 |
| champion | FusedManifestoSoftLUT weight=softsign | float32 | ok | 0.480 | 0.023 | 0.567 | 0.023 | 1.833 | 0.094 | 2.400 | 0.094 |
| champion | FusedManifestoSoftLUT weight=softsign | bfloat16 | ok | 0.303 | 0.012 | 0.384 | 0.012 | 1.782 | 0.094 | 2.142 | 0.094 |
| champion | ConfidenceLUT n=1 | float32 | ok | 0.899 | 0.117 | 1.067 | 0.117 | 2.379 | 0.711 | 3.443 | 0.711 |
| champion | ConfidenceLUT n=1 | bfloat16 | ok | 0.943 | 0.188 | 1.072 | 0.199 | 2.363 | 0.746 | 3.434 | 0.746 |
| champion | ConfidenceLUT n=2 | float32 | ok | 1.268 | 0.219 | 1.346 | 0.211 | 4.031 | 0.820 | 5.375 | 0.820 |
| champion | ConfidenceLUT n=2 | bfloat16 | ok | 1.300 | 0.289 | 1.314 | 0.293 | 3.993 | 0.855 | 5.310 | 0.855 |
| champion | FusedConfidenceLUT n=1 | float32 | ok | 0.266 | 0.023 | 0.324 | 0.023 | 1.243 | 0.094 | 1.558 | 0.094 |
| champion | FusedConfidenceLUT n=1 | bfloat16 | ok | 0.213 | 0.012 | 0.284 | 0.012 | 1.150 | 0.094 | 1.430 | 0.094 |
| champion | FusedConfidenceLUT n=2 | float32 | ok | 0.478 | 0.023 | 0.541 | 0.023 | 1.927 | 0.094 | 2.463 | 0.094 |
| champion | FusedConfidenceLUT n=2 | bfloat16 | ok | 0.318 | 0.012 | 0.401 | 0.012 | 1.817 | 0.094 | 2.218 | 0.094 |
| champion | QuantisedConfidenceLUT n=2 | float32 | ok | 2.575 | 0.259 | 3.027 | 0.516 | 9.428 | 1.116 | 12.455 | 1.116 |
| champion | QuantisedConfidenceLUT n=2 | bfloat16 | refused: QuantisedConfidenceLUT does not support low precision | - | - | - | - | - | - | - | - |
| champion | FusedQuantisedConfidenceLUT n=2 | float32 | ok | 0.520 | 0.023 | 0.600 | 0.055 | 2.002 | 0.125 | 2.603 | 0.125 |
| champion | FusedQuantisedConfidenceLUT n=2 | bfloat16 | error: RuntimeError: FusedQuantisedConfidenceLUT: not a case the fused op covers: input dtype torch.bfloat16 (float32 only); table dtype torc | - | - | - | - | - | - | - | - |
| champion | DeployedQuantisedConfidenceLUT | float32 | ok | 0.310 | 0.023 | - | - | - | - | - | - |
| champion | DeployedQuantisedConfidenceLUT | bfloat16 | refused: DeployedQuantisedConfidenceLUT is fp32/fp64 inference only; got input dtype torch.bfloat16 | - | - | - | - | - | - | - | - |
| champion | DeployedQuantisedConfidenceLUT n=1 | float32 | ok | 0.221 | 0.023 | - | - | - | - | - | - |
| champion | DeployedQuantisedConfidenceLUT n=1 | bfloat16 | refused: DeployedQuantisedConfidenceLUT is fp32/fp64 inference only; got input dtype torch.bfloat16 | - | - | - | - | - | - | - | - |
| large | ManifestoHardLUT | float32 | ok | 2.464 | 0.219 | 24.533 | 5.094 | 35.904 | 4.188 | 60.435 | 5.094 |
| large | ManifestoHardLUT | bfloat16 | ok | 2.676 | 0.359 | 24.723 | 5.234 | 35.940 | 4.188 | 60.665 | 5.234 |
| large | ManifestoSoftLUT | float32 | ok | 5.020 | 0.594 | 20.817 | 5.094 | 40.995 | 4.763 | 61.813 | 5.094 |
| large | ManifestoSoftLUT | bfloat16 | ok | 5.067 | 0.734 | 20.993 | 5.234 | 40.913 | 4.857 | 61.905 | 5.234 |
| large | FusedManifestoHardLUT | float32 | ok | 0.997 | 0.094 | 1.081 | 0.094 | 4.930 | 0.234 | 6.011 | 0.234 |
| large | FusedManifestoHardLUT | bfloat16 | ok | 0.743 | 0.047 | 0.845 | 0.047 | 4.596 | 0.234 | 5.444 | 0.234 |
| large | FusedManifestoSoftLUT | float32 | ok | 1.818 | 0.094 | 2.113 | 0.094 | 6.777 | 0.234 | 8.891 | 0.234 |
| large | FusedManifestoSoftLUT | bfloat16 | ok | 1.118 | 0.047 | 1.232 | 0.047 | 6.372 | 0.234 | 7.599 | 0.234 |
| large | ManifestoHardLUT weight=softsign | float32 | ok | 2.557 | 0.219 | 24.788 | 5.094 | 37.722 | 4.188 | 62.518 | 5.094 |
| large | ManifestoHardLUT weight=softsign | bfloat16 | ok | 2.689 | 0.359 | 24.984 | 5.234 | 37.748 | 4.188 | 62.731 | 5.234 |
| large | ManifestoSoftLUT weight=softsign | float32 | ok | 5.375 | 0.594 | 21.078 | 5.094 | 42.724 | 5.013 | 63.799 | 5.094 |
| large | ManifestoSoftLUT weight=softsign | bfloat16 | ok | 5.405 | 0.734 | 21.255 | 5.234 | 42.815 | 5.107 | 64.076 | 5.234 |
| large | FusedManifestoHardLUT weight=softsign | float32 | ok | 1.011 | 0.094 | 1.118 | 0.094 | 6.319 | 0.234 | 7.439 | 0.234 |
| large | FusedManifestoHardLUT weight=softsign | bfloat16 | ok | 0.766 | 0.047 | 0.883 | 0.047 | 5.940 | 0.234 | 6.823 | 0.234 |
| large | FusedManifestoSoftLUT weight=softsign | float32 | ok | 1.836 | 0.094 | 2.024 | 0.094 | 7.053 | 0.234 | 9.076 | 0.234 |
| large | FusedManifestoSoftLUT weight=softsign | bfloat16 | ok | 1.166 | 0.047 | 1.242 | 0.047 | 6.797 | 0.234 | 8.041 | 0.234 |
| large | ConfidenceLUT n=1 | float32 | ok | 3.594 | 0.469 | 3.712 | 0.469 | 9.473 | 2.703 | 13.195 | 2.703 |
| large | ConfidenceLUT n=1 | bfloat16 | ok | 3.705 | 0.609 | 3.837 | 0.656 | 9.572 | 2.844 | 13.399 | 2.844 |
| large | ConfidenceLUT n=2 | float32 | ok | 4.962 | 0.875 | 4.785 | 0.844 | 16.227 | 3.141 | 21.029 | 3.141 |
| large | ConfidenceLUT n=2 | bfloat16 | ok | 5.169 | 1.016 | 4.714 | 1.031 | 15.838 | 3.281 | 20.554 | 3.281 |
| large | FusedConfidenceLUT n=1 | float32 | ok | 1.050 | 0.094 | 1.138 | 0.094 | 4.740 | 0.236 | 5.872 | 0.236 |
| large | FusedConfidenceLUT n=1 | bfloat16 | ok | 0.824 | 0.047 | 0.987 | 0.047 | 4.483 | 0.236 | 5.472 | 0.236 |
| large | FusedConfidenceLUT n=2 | float32 | ok | 1.877 | 0.094 | 1.987 | 0.094 | 7.363 | 0.236 | 9.350 | 0.236 |
| large | FusedConfidenceLUT n=2 | bfloat16 | ok | 1.256 | 0.047 | 1.441 | 0.047 | 6.975 | 0.236 | 8.412 | 0.236 |
| large | QuantisedConfidenceLUT n=2 | float32 | ok | 9.956 | 0.895 | 11.558 | 1.922 | 37.850 | 4.169 | 49.411 | 4.169 |
| large | QuantisedConfidenceLUT n=2 | bfloat16 | refused: QuantisedConfidenceLUT does not support low precision | - | - | - | - | - | - | - | - |
| large | FusedQuantisedConfidenceLUT n=2 | float32 | ok | 1.973 | 0.094 | 2.069 | 0.219 | 7.679 | 0.361 | 9.749 | 0.361 |
| large | FusedQuantisedConfidenceLUT n=2 | bfloat16 | error: RuntimeError: FusedQuantisedConfidenceLUT: not a case the fused op covers: input dtype torch.bfloat16 (float32 only); table dtype torc | - | - | - | - | - | - | - | - |
| large | DeployedQuantisedConfidenceLUT | float32 | ok | 1.250 | 0.094 | - | - | - | - | - | - |
| large | DeployedQuantisedConfidenceLUT | bfloat16 | refused: DeployedQuantisedConfidenceLUT is fp32/fp64 inference only; got input dtype torch.bfloat16 | - | - | - | - | - | - | - | - |
| large | DeployedQuantisedConfidenceLUT n=1 | float32 | ok | 0.891 | 0.094 | - | - | - | - | - | - |
| large | DeployedQuantisedConfidenceLUT n=1 | bfloat16 | refused: DeployedQuantisedConfidenceLUT is fp32/fp64 inference only; got input dtype torch.bfloat16 | - | - | - | - | - | - | - | - |
