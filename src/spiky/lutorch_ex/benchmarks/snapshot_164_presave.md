# lutorch_ex performance snapshot: #164 before the per-table-integer saving (9a9f0891)

**Code.** `9a9f0891`, #164 one commit before ae931703. Materialised from git, nothing changed.

**Hardware and software.** RTX 5090 (sm_120, 170 SMs), torch 2.9.1+cu130, Triton 3.5.1, CUDA 13.0, driver 595.84.

**Session.** Measured 2026-10-13 in ONE session (later than the v2 session) together with the other of the pair: #164 before the per-table-integer saving (9a9f0891) and #164 HEAD (d73edfad), interleaved cell by cell, alternating which went first. This pair isolates the saving's own delta, including its training-peak cost.
- Each cell ran under each tree's own `python -m spiky.lutorch_ex.bench_snapshot` for that one cell: one fresh process per cell, a Dynamo reset, 10 warm-up + 30 timed iterations, median; peak GiB above the pre-step allocation.
- Champion geometry h16 tph64 nap8 d48 at 8,192 vectors (`champion`) and 32,768 (`large`).

**Conditions:** SM up to 3.07 GHz (unlocked), 33–44 C. **No throttle or power-cap flag at any point.**

**One transient, re-measured.** FusedManifestoHardLUT·softsign 8k bf16 (code untouched by the saving) came out ×1.18 on train forward: originally eval forward / train forward / train backward / train total 0.214 / 0.247 / 1.489 / 1.737 ms (9a9f0891) vs 0.214 / 0.292 / 1.489 / 1.781 ms (HEAD). Three interleaved re-runs gave 0.292–0.293 ms on BOTH trees (the 9a9f0891 cell's 0.247 was the outlier); the row was re-measured once more on both trees, interleaved, and replaced on both sides.

Comparison: `figures_164_presave_vs_164_v2/` (bench_compare.py --same-session).

| geometry | cartridge | dtype | status | eval forward ms | GiB | train forward ms | GiB | train backward ms | GiB | train total ms | GiB |
|---|---|---|---|---|---|---|---|---|---|---|---|
| champion | ManifestoHardLUT | float32 | ok | 0.639 | 0.055 | 5.952 | 1.273 | 8.975 | 1.047 | 14.931 | 1.273 |
| champion | ManifestoHardLUT | bfloat16 | ok | 0.682 | 0.125 | 5.997 | 1.344 | 8.908 | 1.082 | 14.904 | 1.344 |
| champion | ManifestoSoftLUT | float32 | ok | 1.218 | 0.148 | 5.053 | 1.273 | 10.202 | 1.265 | 15.253 | 1.273 |
| champion | ManifestoSoftLUT | bfloat16 | ok | 1.253 | 0.219 | 5.118 | 1.344 | 10.114 | 1.323 | 15.231 | 1.344 |
| champion | FusedManifestoHardLUT | float32 | ok | 0.254 | 0.023 | 0.319 | 0.023 | 1.306 | 0.094 | 1.605 | 0.094 |
| champion | FusedManifestoHardLUT | bfloat16 | ok | 0.192 | 0.012 | 0.226 | 0.012 | 1.186 | 0.094 | 1.415 | 0.094 |
| champion | FusedManifestoSoftLUT | float32 | ok | 0.463 | 0.023 | 0.554 | 0.023 | 1.705 | 0.094 | 2.259 | 0.094 |
| champion | FusedManifestoSoftLUT | bfloat16 | ok | 0.287 | 0.012 | 0.344 | 0.012 | 1.603 | 0.094 | 1.942 | 0.094 |
| champion | ManifestoHardLUT weight=softsign | float32 | ok | 0.663 | 0.055 | 5.978 | 1.273 | 9.272 | 1.047 | 15.248 | 1.273 |
| champion | ManifestoHardLUT weight=softsign | bfloat16 | ok | 0.707 | 0.125 | 6.023 | 1.344 | 9.221 | 1.082 | 15.247 | 1.344 |
| champion | ManifestoSoftLUT weight=softsign | float32 | ok | 1.290 | 0.148 | 5.072 | 1.273 | 10.552 | 1.327 | 15.627 | 1.327 |
| champion | ManifestoSoftLUT weight=softsign | bfloat16 | ok | 1.318 | 0.219 | 5.158 | 1.344 | 10.431 | 1.386 | 15.592 | 1.386 |
| champion | FusedManifestoHardLUT weight=softsign | float32 | ok | 0.268 | 0.023 | 0.311 | 0.023 | 1.583 | 0.094 | 1.893 | 0.094 |
| champion | FusedManifestoHardLUT weight=softsign | bfloat16 | ok | 0.214 | 0.012 | 0.292 | 0.012 | 1.490 | 0.094 | 1.783 | 0.094 |
| champion | FusedManifestoSoftLUT weight=softsign | float32 | ok | 0.478 | 0.023 | 0.575 | 0.023 | 1.858 | 0.094 | 2.433 | 0.094 |
| champion | FusedManifestoSoftLUT weight=softsign | bfloat16 | ok | 0.310 | 0.012 | 0.394 | 0.012 | 1.790 | 0.094 | 2.185 | 0.094 |
| champion | ConfidenceLUT n=1 | float32 | ok | 0.905 | 0.117 | 1.072 | 0.117 | 2.362 | 0.711 | 3.434 | 0.711 |
| champion | ConfidenceLUT n=1 | bfloat16 | ok | 0.941 | 0.188 | 1.076 | 0.199 | 2.390 | 0.746 | 3.466 | 0.746 |
| champion | ConfidenceLUT n=2 | float32 | ok | 1.263 | 0.219 | 1.309 | 0.211 | 4.061 | 0.820 | 5.374 | 0.820 |
| champion | ConfidenceLUT n=2 | bfloat16 | ok | 1.274 | 0.289 | 1.331 | 0.293 | 4.000 | 0.855 | 5.329 | 0.855 |
| champion | FusedConfidenceLUT n=1 | float32 | ok | 0.267 | 0.023 | 0.331 | 0.023 | 1.240 | 0.094 | 1.560 | 0.094 |
| champion | FusedConfidenceLUT n=1 | bfloat16 | ok | 0.214 | 0.012 | 0.281 | 0.012 | 1.149 | 0.094 | 1.421 | 0.094 |
| champion | FusedConfidenceLUT n=2 | float32 | ok | 0.475 | 0.023 | 0.543 | 0.023 | 1.931 | 0.094 | 2.471 | 0.094 |
| champion | FusedConfidenceLUT n=2 | bfloat16 | ok | 0.317 | 0.012 | 0.410 | 0.012 | 1.817 | 0.094 | 2.227 | 0.094 |
| champion | QuantisedConfidenceLUT n=2 | float32 | ok | 2.574 | 0.259 | 3.036 | 0.516 | 9.421 | 1.116 | 12.457 | 1.116 |
| champion | QuantisedConfidenceLUT n=2 | bfloat16 | refused: QuantisedConfidenceLUT does not support low precision | - | - | - | - | - | - | - | - |
| champion | FusedQuantisedConfidenceLUT n=2 | float32 | ok | 0.518 | 0.023 | 0.591 | 0.023 | 2.088 | 0.094 | 2.679 | 0.094 |
| champion | FusedQuantisedConfidenceLUT n=2 | bfloat16 | error: RuntimeError: FusedQuantisedConfidenceLUT: not a case the fused op covers: input dtype torch.bfloat16 (float32 only); table dtype torc | - | - | - | - | - | - | - | - |
| champion | DeployedQuantisedConfidenceLUT | float32 | ok | 0.310 | 0.023 | - | - | - | - | - | - |
| champion | DeployedQuantisedConfidenceLUT | bfloat16 | refused: DeployedQuantisedConfidenceLUT is fp32/fp64 inference only; got input dtype torch.bfloat16 | - | - | - | - | - | - | - | - |
| champion | DeployedQuantisedConfidenceLUT n=1 | float32 | ok | 0.221 | 0.023 | - | - | - | - | - | - |
| champion | DeployedQuantisedConfidenceLUT n=1 | bfloat16 | refused: DeployedQuantisedConfidenceLUT is fp32/fp64 inference only; got input dtype torch.bfloat16 | - | - | - | - | - | - | - | - |
| large | ManifestoHardLUT | float32 | ok | 2.464 | 0.219 | 24.525 | 5.094 | 35.921 | 4.188 | 60.445 | 5.094 |
| large | ManifestoHardLUT | bfloat16 | ok | 2.603 | 0.359 | 24.728 | 5.234 | 35.909 | 4.188 | 60.641 | 5.234 |
| large | ManifestoSoftLUT | float32 | ok | 5.041 | 0.594 | 20.813 | 5.094 | 40.986 | 4.763 | 61.801 | 5.094 |
| large | ManifestoSoftLUT | bfloat16 | ok | 5.064 | 0.734 | 20.996 | 5.234 | 40.909 | 4.857 | 61.904 | 5.234 |
| large | FusedManifestoHardLUT | float32 | ok | 0.991 | 0.094 | 1.077 | 0.094 | 4.953 | 0.234 | 6.026 | 0.234 |
| large | FusedManifestoHardLUT | bfloat16 | ok | 0.744 | 0.047 | 0.845 | 0.047 | 4.579 | 0.234 | 5.425 | 0.234 |
| large | FusedManifestoSoftLUT | float32 | ok | 1.814 | 0.094 | 2.122 | 0.094 | 6.884 | 0.234 | 9.004 | 0.234 |
| large | FusedManifestoSoftLUT | bfloat16 | ok | 1.114 | 0.047 | 1.241 | 0.047 | 6.419 | 0.234 | 7.660 | 0.234 |
| large | ManifestoHardLUT weight=softsign | float32 | ok | 2.556 | 0.219 | 24.786 | 5.094 | 37.713 | 4.188 | 62.502 | 5.094 |
| large | ManifestoHardLUT weight=softsign | bfloat16 | ok | 2.689 | 0.359 | 24.995 | 5.234 | 37.724 | 4.188 | 62.718 | 5.234 |
| large | ManifestoSoftLUT weight=softsign | float32 | ok | 5.376 | 0.594 | 21.067 | 5.094 | 42.733 | 5.013 | 63.801 | 5.094 |
| large | ManifestoSoftLUT weight=softsign | bfloat16 | ok | 5.408 | 0.734 | 21.251 | 5.234 | 42.812 | 5.107 | 64.068 | 5.234 |
| large | FusedManifestoHardLUT weight=softsign | float32 | ok | 1.004 | 0.094 | 1.110 | 0.094 | 6.345 | 0.234 | 7.456 | 0.234 |
| large | FusedManifestoHardLUT weight=softsign | bfloat16 | ok | 0.764 | 0.047 | 0.884 | 0.047 | 5.934 | 0.234 | 6.818 | 0.234 |
| large | FusedManifestoSoftLUT weight=softsign | float32 | ok | 1.837 | 0.094 | 2.050 | 0.094 | 7.159 | 0.234 | 9.207 | 0.234 |
| large | FusedManifestoSoftLUT weight=softsign | bfloat16 | ok | 1.135 | 0.047 | 1.216 | 0.047 | 6.796 | 0.234 | 8.014 | 0.234 |
| large | ConfidenceLUT n=1 | float32 | ok | 3.598 | 0.469 | 3.708 | 0.469 | 9.495 | 2.703 | 13.199 | 2.703 |
| large | ConfidenceLUT n=1 | bfloat16 | ok | 3.725 | 0.609 | 3.742 | 0.656 | 9.556 | 2.844 | 13.301 | 2.844 |
| large | ConfidenceLUT n=2 | float32 | ok | 4.883 | 0.875 | 4.632 | 0.844 | 16.211 | 3.141 | 20.844 | 3.141 |
| large | ConfidenceLUT n=2 | bfloat16 | ok | 5.061 | 1.016 | 4.733 | 1.031 | 15.777 | 3.281 | 20.512 | 3.281 |
| large | FusedConfidenceLUT n=1 | float32 | ok | 1.043 | 0.094 | 1.136 | 0.094 | 4.760 | 0.236 | 5.899 | 0.236 |
| large | FusedConfidenceLUT n=1 | bfloat16 | ok | 0.823 | 0.047 | 0.984 | 0.047 | 4.478 | 0.236 | 5.462 | 0.236 |
| large | FusedConfidenceLUT n=2 | float32 | ok | 1.876 | 0.094 | 1.979 | 0.094 | 7.406 | 0.236 | 9.387 | 0.236 |
| large | FusedConfidenceLUT n=2 | bfloat16 | ok | 1.233 | 0.047 | 1.429 | 0.047 | 6.966 | 0.236 | 8.390 | 0.236 |
| large | QuantisedConfidenceLUT n=2 | float32 | ok | 9.953 | 0.895 | 11.541 | 1.922 | 37.803 | 4.169 | 49.347 | 4.169 |
| large | QuantisedConfidenceLUT n=2 | bfloat16 | refused: QuantisedConfidenceLUT does not support low precision | - | - | - | - | - | - | - | - |
| large | FusedQuantisedConfidenceLUT n=2 | float32 | ok | 1.926 | 0.094 | 2.049 | 0.094 | 8.140 | 0.236 | 10.189 | 0.236 |
| large | FusedQuantisedConfidenceLUT n=2 | bfloat16 | error: RuntimeError: FusedQuantisedConfidenceLUT: not a case the fused op covers: input dtype torch.bfloat16 (float32 only); table dtype torc | - | - | - | - | - | - | - | - |
| large | DeployedQuantisedConfidenceLUT | float32 | ok | 1.252 | 0.094 | - | - | - | - | - | - |
| large | DeployedQuantisedConfidenceLUT | bfloat16 | refused: DeployedQuantisedConfidenceLUT is fp32/fp64 inference only; got input dtype torch.bfloat16 | - | - | - | - | - | - | - | - |
| large | DeployedQuantisedConfidenceLUT n=1 | float32 | ok | 0.892 | 0.094 | - | - | - | - | - | - |
| large | DeployedQuantisedConfidenceLUT n=1 | bfloat16 | refused: DeployedQuantisedConfidenceLUT is fp32/fp64 inference only; got input dtype torch.bfloat16 | - | - | - | - | - | - | - | - |
