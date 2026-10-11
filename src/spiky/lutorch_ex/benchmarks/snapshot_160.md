# lutorch_ex performance snapshot of record (#160)

RTX 5090 (sm_120), torch 2.9.1+cu130, CUDA 13.0, driver 595.84; code `be66a425` (no uncommitted changes in
lutorch_ex); measured 2026-10-11. Champion geometry h16 tph64 nap8 d48 at 8,192 vectors (`champion`) and 32,768
(`large`). Compiled, one fresh process per cell, a Dynamo reset per cell, median of 30 timed iterations after 10
warm-up; peak GiB above the pre-step allocation; clocks not locked. Every class runs its one implementation with its
defaults; `status` says why a cell has no numbers (out of memory, a refused dtype, a case the fused class does not
cover). The IQRs and the environment observations are in `snapshot_160.json`.

Reproduce: `python -m spiky.lutorch_ex.bench_snapshot --geometries champion,large --out snapshot_160.json --md snapshot_160.md`

Accepted cost of #160 (not fixed here; the quantised follow-up PR): `QuantisedConfidenceLUT` is now the pure-torch
reference, without the p2_scalars op (n=2 eval) or the training monolith, which moved to
`FusedQuantisedConfidenceLUT`. Against the #159 snapshot of record (the old default, which used both):
8,192: eval 2.52 -> 2.55 ms, train total 12.43 -> 12.50 ms; 32,768: eval 9.66 -> 10.09 ms (+4.5%), train total
48.56 -> 49.22 ms (+1.4%), train peak 3.64 -> 4.38 GiB (+0.74 GiB). `FusedQuantisedConfidenceLUT` reproduces the old
numbers (32,768: 9.89 ms eval, 49.41 ms / 3.64 GiB train).

| geometry | cartridge | dtype | status | eval forward ms | GiB | train forward ms | GiB | train backward ms | GiB | train total ms | GiB |
|---|---|---|---|---|---|---|---|---|---|---|---|
| champion | ManifestoHardLUT | float32 | ok | 0.664 | 0.055 | 18.225 | 6.712 | 7.521 | 3.517 | 25.838 | 6.712 |
| champion | ManifestoHardLUT | bfloat16 | ok | 0.702 | 0.125 | 18.272 | 6.782 | 7.379 | 3.528 | 25.895 | 6.782 |
| champion | ManifestoSoftLUT | float32 | ok | 1.224 | 0.148 | 5.131 | 1.273 | 10.473 | 1.265 | 15.611 | 1.273 |
| champion | ManifestoSoftLUT | bfloat16 | ok | 1.250 | 0.219 | 5.163 | 1.344 | 10.364 | 1.323 | 15.586 | 1.344 |
| champion | FusedManifestoHardLUT | float32 | ok | 0.254 | 0.023 | 0.288 | 0.023 | 1.260 | 0.094 | 1.560 | 0.094 |
| champion | FusedManifestoHardLUT | bfloat16 | ok | 0.194 | 0.012 | 0.253 | 0.012 | 1.163 | 0.094 | 1.413 | 0.094 |
| champion | FusedManifestoSoftLUT | float32 | ok | 0.465 | 0.023 | 0.528 | 0.023 | 1.642 | 0.094 | 2.174 | 0.094 |
| champion | FusedManifestoSoftLUT | bfloat16 | ok | 0.282 | 0.012 | 0.335 | 0.012 | 1.610 | 0.094 | 1.961 | 0.094 |
| champion | ManifestoHardLUT weight=softsign | float32 | ok | 0.671 | 0.055 | 18.241 | 6.743 | 7.944 | 3.579 | 26.142 | 6.743 |
| champion | ManifestoHardLUT weight=softsign | bfloat16 | ok | 0.708 | 0.125 | 18.290 | 6.813 | 7.970 | 3.591 | 26.173 | 6.813 |
| champion | ManifestoSoftLUT weight=softsign | float32 | ok | 1.294 | 0.148 | 5.089 | 1.273 | 10.781 | 1.327 | 15.884 | 1.327 |
| champion | ManifestoSoftLUT weight=softsign | bfloat16 | ok | 1.317 | 0.219 | 5.192 | 1.344 | 10.597 | 1.386 | 15.869 | 1.386 |
| champion | FusedManifestoHardLUT weight=softsign | float32 | ok | 0.269 | 0.023 | 0.314 | 0.023 | 1.587 | 0.094 | 1.915 | 0.094 |
| champion | FusedManifestoHardLUT weight=softsign | bfloat16 | ok | 0.215 | 0.012 | 0.249 | 0.012 | 1.491 | 0.094 | 1.740 | 0.094 |
| champion | FusedManifestoSoftLUT weight=softsign | float32 | ok | 0.480 | 0.023 | 0.571 | 0.023 | 1.848 | 0.094 | 2.418 | 0.094 |
| champion | FusedManifestoSoftLUT weight=softsign | bfloat16 | ok | 0.305 | 0.012 | 0.381 | 0.012 | 1.720 | 0.094 | 2.093 | 0.094 |
| champion | ConfidenceLUT n=1 | float32 | ok | 0.911 | 0.117 | 1.064 | 0.117 | 2.382 | 0.711 | 3.451 | 0.711 |
| champion | ConfidenceLUT n=1 | bfloat16 | ok | 0.942 | 0.188 | 1.077 | 0.199 | 2.378 | 0.746 | 3.454 | 0.746 |
| champion | ConfidenceLUT n=2 | float32 | ok | 1.250 | 0.219 | 1.334 | 0.211 | 4.082 | 0.820 | 5.416 | 0.820 |
| champion | ConfidenceLUT n=2 | bfloat16 | ok | 1.287 | 0.289 | 1.358 | 0.293 | 4.063 | 0.855 | 5.397 | 0.855 |
| champion | FusedConfidenceLUT n=1 | float32 | ok | 0.267 | 0.023 | 0.322 | 0.023 | 1.241 | 0.094 | 1.547 | 0.094 |
| champion | FusedConfidenceLUT n=1 | bfloat16 | ok | 0.204 | 0.012 | 0.271 | 0.012 | 1.146 | 0.094 | 1.417 | 0.094 |
| champion | FusedConfidenceLUT n=2 | float32 | ok | 0.474 | 0.023 | 0.540 | 0.023 | 1.929 | 0.094 | 2.466 | 0.094 |
| champion | FusedConfidenceLUT n=2 | bfloat16 | ok | 0.316 | 0.012 | 0.382 | 0.012 | 1.786 | 0.094 | 2.165 | 0.094 |
| champion | QuantisedConfidenceLUT n=2 | float32 | ok | 2.550 | 0.297 | 2.688 | 0.509 | 9.804 | 1.095 | 12.497 | 1.095 |
| champion | QuantisedConfidenceLUT n=2 | bfloat16 | refused: QuantisedConfidenceLUT does not support low precision | - | - | - | - | - | - | - | - |
| champion | FusedQuantisedConfidenceLUT n=2 | float32 | ok | 2.524 | 0.415 | 2.499 | 0.352 | 9.873 | 0.946 | 12.525 | 0.946 |
| champion | FusedQuantisedConfidenceLUT n=2 | bfloat16 | error: RuntimeError: FusedQuantisedConfidenceLUT: not a case the fused op covers: input dtype torch.bfloat16 (float32 only); table dtype torc | - | - | - | - | - | - | - | - |
| champion | DeployedQuantisedConfidenceLUT | float32 | ok | 2.662 | 0.203 | - | - | - | - | - | - |
| champion | DeployedQuantisedConfidenceLUT | bfloat16 | refused: DeployedQuantisedConfidenceLUT is fp32/fp64 inference only; got input dtype torch.bfloat16 | - | - | - | - | - | - | - | - |
| large | ManifestoHardLUT | float32 | out of memory | 2.554 | 0.219 | - | - | - | - | - | - |
| large | ManifestoHardLUT | bfloat16 | out of memory | 2.676 | 0.359 | - | - | - | - | - | - |
| large | ManifestoSoftLUT | float32 | ok | 5.038 | 0.594 | 21.156 | 5.094 | 41.808 | 4.763 | 62.979 | 5.094 |
| large | ManifestoSoftLUT | bfloat16 | ok | 5.062 | 0.734 | 21.327 | 5.234 | 41.644 | 4.857 | 62.975 | 5.234 |
| large | FusedManifestoHardLUT | float32 | ok | 0.988 | 0.094 | 1.065 | 0.094 | 4.842 | 0.234 | 5.892 | 0.234 |
| large | FusedManifestoHardLUT | bfloat16 | ok | 0.751 | 0.047 | 0.849 | 0.047 | 4.479 | 0.234 | 5.332 | 0.234 |
| large | FusedManifestoSoftLUT | float32 | ok | 1.814 | 0.094 | 2.066 | 0.094 | 6.902 | 0.234 | 8.941 | 0.234 |
| large | FusedManifestoSoftLUT | bfloat16 | ok | 1.121 | 0.047 | 1.210 | 0.047 | 6.347 | 0.234 | 7.568 | 0.234 |
| large | ManifestoHardLUT weight=softsign | float32 | out of memory | 2.491 | 0.219 | - | - | - | - | - | - |
| large | ManifestoHardLUT weight=softsign | bfloat16 | out of memory | 2.616 | 0.359 | - | - | - | - | - | - |
| large | ManifestoSoftLUT weight=softsign | float32 | ok | 5.343 | 0.594 | 21.445 | 5.094 | 43.502 | 5.013 | 64.949 | 5.094 |
| large | ManifestoSoftLUT weight=softsign | bfloat16 | ok | 5.406 | 0.734 | 21.648 | 5.234 | 43.557 | 5.107 | 65.231 | 5.234 |
| large | FusedManifestoHardLUT weight=softsign | float32 | ok | 1.017 | 0.094 | 1.120 | 0.094 | 6.296 | 0.234 | 7.420 | 0.234 |
| large | FusedManifestoHardLUT weight=softsign | bfloat16 | ok | 0.769 | 0.047 | 0.886 | 0.047 | 5.945 | 0.234 | 6.836 | 0.234 |
| large | FusedManifestoSoftLUT weight=softsign | float32 | ok | 1.847 | 0.094 | 1.978 | 0.094 | 7.017 | 0.234 | 9.010 | 0.234 |
| large | FusedManifestoSoftLUT weight=softsign | bfloat16 | ok | 1.141 | 0.047 | 1.243 | 0.047 | 6.750 | 0.234 | 8.007 | 0.234 |
| large | ConfidenceLUT n=1 | float32 | ok | 3.662 | 0.469 | 3.720 | 0.469 | 9.921 | 2.703 | 13.637 | 2.703 |
| large | ConfidenceLUT n=1 | bfloat16 | ok | 3.772 | 0.609 | 3.809 | 0.656 | 9.966 | 2.844 | 13.745 | 2.844 |
| large | ConfidenceLUT n=2 | float32 | ok | 5.012 | 0.875 | 4.933 | 0.844 | 16.500 | 3.141 | 21.497 | 3.141 |
| large | ConfidenceLUT n=2 | bfloat16 | ok | 4.982 | 1.016 | 6.088 | 1.031 | 16.216 | 3.281 | 22.285 | 3.281 |
| large | FusedConfidenceLUT n=1 | float32 | ok | 1.045 | 0.094 | 1.119 | 0.094 | 4.742 | 0.236 | 5.870 | 0.236 |
| large | FusedConfidenceLUT n=1 | bfloat16 | ok | 0.800 | 0.047 | 0.917 | 0.047 | 4.489 | 0.236 | 5.421 | 0.236 |
| large | FusedConfidenceLUT n=2 | float32 | ok | 1.869 | 0.094 | 1.977 | 0.094 | 7.684 | 0.236 | 9.637 | 0.236 |
| large | FusedConfidenceLUT n=2 | bfloat16 | ok | 1.216 | 0.047 | 1.380 | 0.047 | 6.951 | 0.236 | 8.319 | 0.236 |
| large | QuantisedConfidenceLUT n=2 | float32 | ok | 10.090 | 1.188 | 10.093 | 1.895 | 39.134 | 4.379 | 49.220 | 4.379 |
| large | QuantisedConfidenceLUT n=2 | bfloat16 | refused: QuantisedConfidenceLUT does not support low precision | - | - | - | - | - | - | - | - |
| large | FusedQuantisedConfidenceLUT n=2 | float32 | ok | 9.890 | 1.658 | 9.590 | 1.406 | 39.861 | 3.641 | 49.409 | 3.641 |
| large | FusedQuantisedConfidenceLUT n=2 | bfloat16 | error: RuntimeError: FusedQuantisedConfidenceLUT: not a case the fused op covers: input dtype torch.bfloat16 (float32 only); table dtype torc | - | - | - | - | - | - | - | - |
| large | DeployedQuantisedConfidenceLUT | float32 | ok | 25.295 | 0.812 | - | - | - | - | - | - |
| large | DeployedQuantisedConfidenceLUT | bfloat16 | refused: DeployedQuantisedConfidenceLUT is fp32/fp64 inference only; got input dtype torch.bfloat16 | - | - | - | - | - | - | - | - |
