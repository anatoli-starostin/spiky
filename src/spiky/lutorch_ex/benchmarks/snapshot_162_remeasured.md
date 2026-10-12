# lutorch_ex performance snapshot: #162 re-measured (the BEFORE of #164, its parent)

**Code.** `c8e48c36`, #162's library code: the commit snapshot_162.json was measured at (its JSON and .md agree). Materialised from git, nothing changed. Probe: it reproduces its own published numbers (FusedQuantised 8k eval 2.519 vs 2.521 ms, train backward 9.869 vs 9.884; ManifestoHard 8k train backward 8.968 vs 8.971).

**Hardware and software.** RTX 5090 (sm_120, 170 SMs), torch 2.9.1+cu130, Triton 3.5.1, CUDA 13.0, driver 595.84.

**Session.** Measured 2026-10-13 in ONE session together with snapshot_164_vs162 (#164 HEAD), interleaved cell by cell.
- Each cell ran under each tree's own `python -m spiky.lutorch_ex.bench_snapshot` for that one cell: one fresh process per cell, a Dynamo reset, 10 warm-up + 30 timed iterations, median; peak GiB above the pre-step allocation.
- The two trees alternated which went first.
- Champion geometry h16 tph64 nap8 d48 at 8,192 vectors (`champion`) and 32,768 (`large`).

**Conditions** (logged between every cell):
- SM clock ≥ 2.65 GHz on busy cells (unlocked), 34–43 C.
- **No throttle or power-cap flag at any point.**
- Other processes on the card: desktop only, ~180 MiB.

**No transient.** Nothing moved beyond its noise floor except the cells this PR targets, so no row was re-measured.

Comparison: `figures_162_remeasured_vs_164/` (bench_compare.py --same-session). This is #164's primary comparison, #162 being the parent.

| geometry | cartridge | dtype | status | eval forward ms | GiB | train forward ms | GiB | train backward ms | GiB | train total ms | GiB |
|---|---|---|---|---|---|---|---|---|---|---|---|
| champion | ManifestoHardLUT | float32 | ok | 0.665 | 0.055 | 5.933 | 1.273 | 8.971 | 1.047 | 14.909 | 1.273 |
| champion | ManifestoHardLUT | bfloat16 | ok | 0.702 | 0.125 | 5.992 | 1.344 | 8.946 | 1.082 | 14.942 | 1.344 |
| champion | ManifestoSoftLUT | float32 | ok | 1.224 | 0.148 | 5.042 | 1.273 | 10.202 | 1.265 | 15.244 | 1.273 |
| champion | ManifestoSoftLUT | bfloat16 | ok | 1.248 | 0.219 | 5.117 | 1.344 | 10.105 | 1.323 | 15.222 | 1.344 |
| champion | FusedManifestoHardLUT | float32 | ok | 0.253 | 0.023 | 0.329 | 0.023 | 1.325 | 0.094 | 1.654 | 0.094 |
| champion | FusedManifestoHardLUT | bfloat16 | ok | 0.192 | 0.012 | 0.251 | 0.012 | 1.163 | 0.094 | 1.401 | 0.094 |
| champion | FusedManifestoSoftLUT | float32 | ok | 0.461 | 0.023 | 0.549 | 0.023 | 1.708 | 0.094 | 2.258 | 0.094 |
| champion | FusedManifestoSoftLUT | bfloat16 | ok | 0.282 | 0.012 | 0.329 | 0.012 | 1.603 | 0.094 | 1.950 | 0.094 |
| champion | ManifestoHardLUT weight=softsign | float32 | ok | 0.650 | 0.055 | 5.986 | 1.273 | 9.287 | 1.047 | 15.272 | 1.273 |
| champion | ManifestoHardLUT weight=softsign | bfloat16 | ok | 0.689 | 0.125 | 6.019 | 1.344 | 9.245 | 1.082 | 15.261 | 1.344 |
| champion | ManifestoSoftLUT weight=softsign | float32 | ok | 1.295 | 0.148 | 5.075 | 1.273 | 10.530 | 1.327 | 15.606 | 1.327 |
| champion | ManifestoSoftLUT weight=softsign | bfloat16 | ok | 1.314 | 0.219 | 5.155 | 1.344 | 10.435 | 1.386 | 15.590 | 1.386 |
| champion | FusedManifestoHardLUT weight=softsign | float32 | ok | 0.267 | 0.023 | 0.342 | 0.023 | 1.587 | 0.094 | 1.926 | 0.094 |
| champion | FusedManifestoHardLUT weight=softsign | bfloat16 | ok | 0.216 | 0.012 | 0.295 | 0.012 | 1.492 | 0.094 | 1.788 | 0.094 |
| champion | FusedManifestoSoftLUT weight=softsign | float32 | ok | 0.478 | 0.023 | 0.575 | 0.023 | 1.863 | 0.094 | 2.430 | 0.094 |
| champion | FusedManifestoSoftLUT weight=softsign | bfloat16 | ok | 0.305 | 0.012 | 0.384 | 0.012 | 1.761 | 0.094 | 2.119 | 0.094 |
| champion | ConfidenceLUT n=1 | float32 | ok | 0.910 | 0.117 | 1.083 | 0.117 | 2.387 | 0.711 | 3.468 | 0.711 |
| champion | ConfidenceLUT n=1 | bfloat16 | ok | 0.941 | 0.188 | 1.074 | 0.199 | 2.373 | 0.746 | 3.447 | 0.746 |
| champion | ConfidenceLUT n=2 | float32 | ok | 1.316 | 0.219 | 1.360 | 0.211 | 4.102 | 0.820 | 5.459 | 0.820 |
| champion | ConfidenceLUT n=2 | bfloat16 | ok | 1.279 | 0.289 | 1.323 | 0.293 | 4.058 | 0.855 | 5.379 | 0.855 |
| champion | FusedConfidenceLUT n=1 | float32 | ok | 0.265 | 0.023 | 0.319 | 0.023 | 1.237 | 0.094 | 1.538 | 0.094 |
| champion | FusedConfidenceLUT n=1 | bfloat16 | ok | 0.205 | 0.012 | 0.271 | 0.012 | 1.144 | 0.094 | 1.414 | 0.094 |
| champion | FusedConfidenceLUT n=2 | float32 | ok | 0.472 | 0.023 | 0.536 | 0.023 | 1.934 | 0.094 | 2.461 | 0.094 |
| champion | FusedConfidenceLUT n=2 | bfloat16 | ok | 0.314 | 0.012 | 0.396 | 0.012 | 1.838 | 0.094 | 2.220 | 0.094 |
| champion | QuantisedConfidenceLUT n=2 | float32 | ok | 2.581 | 0.259 | 3.034 | 0.516 | 9.485 | 1.116 | 12.519 | 1.116 |
| champion | QuantisedConfidenceLUT n=2 | bfloat16 | refused: QuantisedConfidenceLUT does not support low precision | - | - | - | - | - | - | - | - |
| champion | FusedQuantisedConfidenceLUT n=2 | float32 | ok | 2.523 | 0.415 | 2.505 | 0.352 | 9.885 | 0.946 | 12.388 | 0.946 |
| champion | FusedQuantisedConfidenceLUT n=2 | bfloat16 | error: RuntimeError: FusedQuantisedConfidenceLUT: not a case the fused op covers: input dtype torch.bfloat16 (float32 only); table dtype torc | - | - | - | - | - | - | - | - |
| champion | DeployedQuantisedConfidenceLUT | float32 | ok | 2.881 | 0.414 | - | - | - | - | - | - |
| champion | DeployedQuantisedConfidenceLUT | bfloat16 | refused: DeployedQuantisedConfidenceLUT is fp32/fp64 inference only; got input dtype torch.bfloat16 | - | - | - | - | - | - | - | - |
| large | ManifestoHardLUT | float32 | ok | 2.546 | 0.219 | 24.523 | 5.094 | 35.970 | 4.188 | 60.497 | 5.094 |
| large | ManifestoHardLUT | bfloat16 | ok | 2.681 | 0.359 | 24.733 | 5.234 | 35.965 | 4.188 | 60.703 | 5.234 |
| large | ManifestoSoftLUT | float32 | ok | 5.018 | 0.594 | 20.810 | 5.094 | 40.979 | 4.763 | 61.786 | 5.094 |
| large | ManifestoSoftLUT | bfloat16 | ok | 5.059 | 0.734 | 20.989 | 5.234 | 40.964 | 4.857 | 61.955 | 5.234 |
| large | FusedManifestoHardLUT | float32 | ok | 0.987 | 0.094 | 1.077 | 0.094 | 4.830 | 0.234 | 5.911 | 0.234 |
| large | FusedManifestoHardLUT | bfloat16 | ok | 0.750 | 0.047 | 0.858 | 0.047 | 4.476 | 0.234 | 5.336 | 0.234 |
| large | FusedManifestoSoftLUT | float32 | ok | 1.810 | 0.094 | 2.105 | 0.094 | 6.793 | 0.234 | 8.898 | 0.234 |
| large | FusedManifestoSoftLUT | bfloat16 | ok | 1.121 | 0.047 | 1.210 | 0.047 | 6.342 | 0.234 | 7.551 | 0.234 |
| large | ManifestoHardLUT weight=softsign | float32 | ok | 2.490 | 0.219 | 24.774 | 5.094 | 37.740 | 4.188 | 62.518 | 5.094 |
| large | ManifestoHardLUT weight=softsign | bfloat16 | ok | 2.625 | 0.359 | 25.004 | 5.234 | 37.778 | 4.188 | 62.784 | 5.234 |
| large | ManifestoSoftLUT weight=softsign | float32 | ok | 5.362 | 0.594 | 21.071 | 5.094 | 42.780 | 5.013 | 63.853 | 5.094 |
| large | ManifestoSoftLUT weight=softsign | bfloat16 | ok | 5.415 | 0.734 | 21.239 | 5.234 | 42.806 | 5.107 | 64.047 | 5.234 |
| large | FusedManifestoHardLUT weight=softsign | float32 | ok | 1.003 | 0.094 | 1.108 | 0.094 | 6.337 | 0.234 | 7.444 | 0.234 |
| large | FusedManifestoHardLUT weight=softsign | bfloat16 | ok | 0.769 | 0.047 | 0.886 | 0.047 | 5.942 | 0.234 | 6.828 | 0.234 |
| large | FusedManifestoSoftLUT weight=softsign | float32 | ok | 1.832 | 0.094 | 2.046 | 0.094 | 7.096 | 0.234 | 9.140 | 0.234 |
| large | FusedManifestoSoftLUT weight=softsign | bfloat16 | ok | 1.146 | 0.047 | 1.221 | 0.047 | 6.788 | 0.234 | 8.014 | 0.234 |
| large | ConfidenceLUT n=1 | float32 | ok | 3.625 | 0.469 | 3.737 | 0.469 | 9.638 | 2.703 | 13.370 | 2.703 |
| large | ConfidenceLUT n=1 | bfloat16 | ok | 3.747 | 0.609 | 3.773 | 0.656 | 9.708 | 2.844 | 13.480 | 2.844 |
| large | ConfidenceLUT n=2 | float32 | ok | 5.137 | 0.875 | 5.014 | 0.844 | 16.135 | 3.141 | 21.141 | 3.141 |
| large | ConfidenceLUT n=2 | bfloat16 | ok | 5.313 | 1.016 | 4.638 | 1.031 | 15.861 | 3.281 | 20.495 | 3.281 |
| large | FusedConfidenceLUT n=1 | float32 | ok | 1.043 | 0.094 | 1.142 | 0.094 | 4.766 | 0.236 | 5.904 | 0.236 |
| large | FusedConfidenceLUT n=1 | bfloat16 | ok | 0.803 | 0.047 | 0.965 | 0.047 | 4.476 | 0.236 | 5.441 | 0.236 |
| large | FusedConfidenceLUT n=2 | float32 | ok | 1.867 | 0.094 | 1.977 | 0.094 | 7.418 | 0.236 | 9.409 | 0.236 |
| large | FusedConfidenceLUT n=2 | bfloat16 | ok | 1.218 | 0.047 | 1.393 | 0.047 | 6.957 | 0.236 | 8.335 | 0.236 |
| large | QuantisedConfidenceLUT n=2 | float32 | ok | 9.967 | 0.895 | 11.555 | 1.922 | 38.023 | 4.169 | 49.586 | 4.169 |
| large | QuantisedConfidenceLUT n=2 | bfloat16 | refused: QuantisedConfidenceLUT does not support low precision | - | - | - | - | - | - | - | - |
| large | FusedQuantisedConfidenceLUT n=2 | float32 | ok | 9.678 | 1.658 | 9.407 | 1.406 | 39.068 | 3.641 | 48.476 | 3.641 |
| large | FusedQuantisedConfidenceLUT n=2 | bfloat16 | error: RuntimeError: FusedQuantisedConfidenceLUT: not a case the fused op covers: input dtype torch.bfloat16 (float32 only); table dtype torc | - | - | - | - | - | - | - | - |
| large | DeployedQuantisedConfidenceLUT | float32 | ok | 16.821 | 1.656 | - | - | - | - | - | - |
| large | DeployedQuantisedConfidenceLUT | bfloat16 | refused: DeployedQuantisedConfidenceLUT is fp32/fp64 inference only; got input dtype torch.bfloat16 | - | - | - | - | - | - | - | - |
