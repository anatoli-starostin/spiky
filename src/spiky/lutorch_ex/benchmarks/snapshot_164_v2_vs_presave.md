# lutorch_ex performance snapshot: #164 HEAD, paired with #164 before the saving

**Code.** `d73edfad`, #164 HEAD; no uncommitted changes in lutorch_ex. A second measurement of HEAD (later session); `snapshot_164_v2.json` is the snapshot of record.

**Hardware and software.** RTX 5090 (sm_120, 170 SMs), torch 2.9.1+cu130, Triton 3.5.1, CUDA 13.0, driver 595.84.

**Session.** Measured 2026-10-13 in ONE session (later than the v2 session) together with the other of the pair: #164 before the per-table-integer saving (9a9f0891) and #164 HEAD (d73edfad), interleaved cell by cell, alternating which went first. This pair isolates the saving's own delta, including its training-peak cost.
- Each cell ran under each tree's own `python -m spiky.lutorch_ex.bench_snapshot` for that one cell: one fresh process per cell, a Dynamo reset, 10 warm-up + 30 timed iterations, median; peak GiB above the pre-step allocation.
- Champion geometry h16 tph64 nap8 d48 at 8,192 vectors (`champion`) and 32,768 (`large`).

**Conditions:** SM up to 3.07 GHz (unlocked), 33–44 C. **No throttle or power-cap flag at any point.**

**One transient, re-measured.** FusedManifestoHardLUT·softsign 8k bf16 (code untouched by the saving) came out ×1.18 on train forward: originally eval forward / train forward / train backward / train total 0.214 / 0.247 / 1.489 / 1.737 ms (9a9f0891) vs 0.214 / 0.292 / 1.489 / 1.781 ms (HEAD). Three interleaved re-runs gave 0.292–0.293 ms on BOTH trees (the 9a9f0891 cell's 0.247 was the outlier); the row was re-measured once more on both trees, interleaved, and replaced on both sides.

Comparison: `figures_164_presave_vs_164_v2/` (bench_compare.py --same-session).

| geometry | cartridge | dtype | status | eval forward ms | GiB | train forward ms | GiB | train backward ms | GiB | train total ms | GiB |
|---|---|---|---|---|---|---|---|---|---|---|---|
| champion | ManifestoHardLUT | float32 | ok | 0.659 | 0.055 | 5.946 | 1.273 | 8.981 | 1.047 | 14.930 | 1.273 |
| champion | ManifestoHardLUT | bfloat16 | ok | 0.681 | 0.125 | 5.999 | 1.344 | 8.910 | 1.082 | 14.910 | 1.344 |
| champion | ManifestoSoftLUT | float32 | ok | 1.220 | 0.148 | 5.053 | 1.273 | 10.207 | 1.265 | 15.260 | 1.273 |
| champion | ManifestoSoftLUT | bfloat16 | ok | 1.251 | 0.219 | 5.112 | 1.344 | 10.113 | 1.323 | 15.226 | 1.344 |
| champion | FusedManifestoHardLUT | float32 | ok | 0.251 | 0.023 | 0.302 | 0.023 | 1.286 | 0.094 | 1.594 | 0.094 |
| champion | FusedManifestoHardLUT | bfloat16 | ok | 0.191 | 0.012 | 0.249 | 0.012 | 1.188 | 0.094 | 1.432 | 0.094 |
| champion | FusedManifestoSoftLUT | float32 | ok | 0.464 | 0.023 | 0.551 | 0.023 | 1.700 | 0.094 | 2.251 | 0.094 |
| champion | FusedManifestoSoftLUT | bfloat16 | ok | 0.286 | 0.012 | 0.345 | 0.012 | 1.615 | 0.094 | 1.956 | 0.094 |
| champion | ManifestoHardLUT weight=softsign | float32 | ok | 0.661 | 0.055 | 5.966 | 1.273 | 9.281 | 1.047 | 15.245 | 1.273 |
| champion | ManifestoHardLUT weight=softsign | bfloat16 | ok | 0.707 | 0.125 | 6.024 | 1.344 | 9.223 | 1.082 | 15.248 | 1.344 |
| champion | ManifestoSoftLUT weight=softsign | float32 | ok | 1.288 | 0.148 | 5.079 | 1.273 | 10.550 | 1.327 | 15.631 | 1.327 |
| champion | ManifestoSoftLUT weight=softsign | bfloat16 | ok | 1.317 | 0.219 | 5.156 | 1.344 | 10.436 | 1.386 | 15.593 | 1.386 |
| champion | FusedManifestoHardLUT weight=softsign | float32 | ok | 0.267 | 0.023 | 0.343 | 0.023 | 1.590 | 0.094 | 1.935 | 0.094 |
| champion | FusedManifestoHardLUT weight=softsign | bfloat16 | ok | 0.213 | 0.012 | 0.293 | 0.012 | 1.491 | 0.094 | 1.785 | 0.094 |
| champion | FusedManifestoSoftLUT weight=softsign | float32 | ok | 0.478 | 0.023 | 0.567 | 0.023 | 1.842 | 0.094 | 2.399 | 0.094 |
| champion | FusedManifestoSoftLUT weight=softsign | bfloat16 | ok | 0.310 | 0.012 | 0.392 | 0.012 | 1.786 | 0.094 | 2.157 | 0.094 |
| champion | ConfidenceLUT n=1 | float32 | ok | 0.902 | 0.117 | 1.069 | 0.117 | 2.366 | 0.711 | 3.432 | 0.711 |
| champion | ConfidenceLUT n=1 | bfloat16 | ok | 0.941 | 0.188 | 1.084 | 0.199 | 2.359 | 0.746 | 3.442 | 0.746 |
| champion | ConfidenceLUT n=2 | float32 | ok | 1.269 | 0.219 | 1.303 | 0.211 | 4.070 | 0.820 | 5.372 | 0.820 |
| champion | ConfidenceLUT n=2 | bfloat16 | ok | 1.273 | 0.289 | 1.323 | 0.293 | 3.998 | 0.855 | 5.320 | 0.855 |
| champion | FusedConfidenceLUT n=1 | float32 | ok | 0.266 | 0.023 | 0.301 | 0.023 | 1.215 | 0.094 | 1.515 | 0.094 |
| champion | FusedConfidenceLUT n=1 | bfloat16 | ok | 0.213 | 0.012 | 0.282 | 0.012 | 1.149 | 0.094 | 1.423 | 0.094 |
| champion | FusedConfidenceLUT n=2 | float32 | ok | 0.476 | 0.023 | 0.541 | 0.023 | 1.914 | 0.094 | 2.444 | 0.094 |
| champion | FusedConfidenceLUT n=2 | bfloat16 | ok | 0.318 | 0.012 | 0.408 | 0.012 | 1.821 | 0.094 | 2.229 | 0.094 |
| champion | QuantisedConfidenceLUT n=2 | float32 | ok | 2.573 | 0.259 | 3.029 | 0.516 | 9.423 | 1.116 | 12.453 | 1.116 |
| champion | QuantisedConfidenceLUT n=2 | bfloat16 | refused: QuantisedConfidenceLUT does not support low precision | - | - | - | - | - | - | - | - |
| champion | FusedQuantisedConfidenceLUT n=2 | float32 | ok | 0.519 | 0.023 | 0.598 | 0.055 | 2.001 | 0.125 | 2.598 | 0.125 |
| champion | FusedQuantisedConfidenceLUT n=2 | bfloat16 | error: RuntimeError: FusedQuantisedConfidenceLUT: not a case the fused op covers: input dtype torch.bfloat16 (float32 only); table dtype torc | - | - | - | - | - | - | - | - |
| champion | DeployedQuantisedConfidenceLUT | float32 | ok | 0.310 | 0.023 | - | - | - | - | - | - |
| champion | DeployedQuantisedConfidenceLUT | bfloat16 | refused: DeployedQuantisedConfidenceLUT is fp32/fp64 inference only; got input dtype torch.bfloat16 | - | - | - | - | - | - | - | - |
| champion | DeployedQuantisedConfidenceLUT n=1 | float32 | ok | 0.220 | 0.023 | - | - | - | - | - | - |
| champion | DeployedQuantisedConfidenceLUT n=1 | bfloat16 | refused: DeployedQuantisedConfidenceLUT is fp32/fp64 inference only; got input dtype torch.bfloat16 | - | - | - | - | - | - | - | - |
| large | ManifestoHardLUT | float32 | ok | 2.469 | 0.219 | 24.521 | 5.094 | 35.920 | 4.188 | 60.445 | 5.094 |
| large | ManifestoHardLUT | bfloat16 | ok | 2.674 | 0.359 | 24.721 | 5.234 | 35.908 | 4.188 | 60.636 | 5.234 |
| large | ManifestoSoftLUT | float32 | ok | 5.012 | 0.594 | 20.818 | 5.094 | 40.988 | 4.763 | 61.807 | 5.094 |
| large | ManifestoSoftLUT | bfloat16 | ok | 5.095 | 0.734 | 20.991 | 5.234 | 40.927 | 4.857 | 61.922 | 5.234 |
| large | FusedManifestoHardLUT | float32 | ok | 0.992 | 0.094 | 1.079 | 0.094 | 4.981 | 0.234 | 6.060 | 0.234 |
| large | FusedManifestoHardLUT | bfloat16 | ok | 0.743 | 0.047 | 0.846 | 0.047 | 4.600 | 0.234 | 5.448 | 0.234 |
| large | FusedManifestoSoftLUT | float32 | ok | 1.815 | 0.094 | 2.147 | 0.094 | 6.955 | 0.234 | 9.101 | 0.234 |
| large | FusedManifestoSoftLUT | bfloat16 | ok | 1.114 | 0.047 | 1.239 | 0.047 | 6.417 | 0.234 | 7.654 | 0.234 |
| large | ManifestoHardLUT weight=softsign | float32 | ok | 2.554 | 0.219 | 24.792 | 5.094 | 37.731 | 4.188 | 62.525 | 5.094 |
| large | ManifestoHardLUT weight=softsign | bfloat16 | ok | 2.692 | 0.359 | 24.990 | 5.234 | 37.734 | 4.188 | 62.727 | 5.234 |
| large | ManifestoSoftLUT weight=softsign | float32 | ok | 5.373 | 0.594 | 21.072 | 5.094 | 42.735 | 5.013 | 63.812 | 5.094 |
| large | ManifestoSoftLUT weight=softsign | bfloat16 | ok | 5.390 | 0.734 | 21.250 | 5.234 | 42.793 | 5.107 | 64.046 | 5.234 |
| large | FusedManifestoHardLUT weight=softsign | float32 | ok | 1.005 | 0.094 | 1.113 | 0.094 | 6.347 | 0.234 | 7.459 | 0.234 |
| large | FusedManifestoHardLUT weight=softsign | bfloat16 | ok | 0.765 | 0.047 | 0.881 | 0.047 | 5.936 | 0.234 | 6.817 | 0.234 |
| large | FusedManifestoSoftLUT weight=softsign | float32 | ok | 1.836 | 0.094 | 2.072 | 0.094 | 7.242 | 0.234 | 9.315 | 0.234 |
| large | FusedManifestoSoftLUT weight=softsign | bfloat16 | ok | 1.142 | 0.047 | 1.219 | 0.047 | 6.795 | 0.234 | 8.014 | 0.234 |
| large | ConfidenceLUT n=1 | float32 | ok | 3.612 | 0.469 | 3.726 | 0.469 | 9.500 | 2.703 | 13.229 | 2.703 |
| large | ConfidenceLUT n=1 | bfloat16 | ok | 3.670 | 0.609 | 3.755 | 0.656 | 9.559 | 2.844 | 13.314 | 2.844 |
| large | ConfidenceLUT n=2 | float32 | ok | 4.884 | 0.875 | 4.617 | 0.844 | 16.193 | 3.141 | 20.804 | 3.141 |
| large | ConfidenceLUT n=2 | bfloat16 | ok | 5.055 | 1.016 | 4.727 | 1.031 | 15.781 | 3.281 | 20.511 | 3.281 |
| large | FusedConfidenceLUT n=1 | float32 | ok | 1.044 | 0.094 | 1.146 | 0.094 | 4.748 | 0.236 | 5.896 | 0.236 |
| large | FusedConfidenceLUT n=1 | bfloat16 | ok | 0.824 | 0.047 | 0.984 | 0.047 | 4.480 | 0.236 | 5.467 | 0.236 |
| large | FusedConfidenceLUT n=2 | float32 | ok | 1.874 | 0.094 | 1.981 | 0.094 | 7.407 | 0.236 | 9.389 | 0.236 |
| large | FusedConfidenceLUT n=2 | bfloat16 | ok | 1.256 | 0.047 | 1.446 | 0.047 | 6.977 | 0.236 | 8.415 | 0.236 |
| large | QuantisedConfidenceLUT n=2 | float32 | ok | 9.961 | 0.895 | 11.543 | 1.922 | 37.815 | 4.169 | 49.357 | 4.169 |
| large | QuantisedConfidenceLUT n=2 | bfloat16 | refused: QuantisedConfidenceLUT does not support low precision | - | - | - | - | - | - | - | - |
| large | FusedQuantisedConfidenceLUT n=2 | float32 | ok | 1.932 | 0.094 | 2.064 | 0.219 | 7.693 | 0.361 | 9.755 | 0.361 |
| large | FusedQuantisedConfidenceLUT n=2 | bfloat16 | error: RuntimeError: FusedQuantisedConfidenceLUT: not a case the fused op covers: input dtype torch.bfloat16 (float32 only); table dtype torc | - | - | - | - | - | - | - | - |
| large | DeployedQuantisedConfidenceLUT | float32 | ok | 1.254 | 0.094 | - | - | - | - | - | - |
| large | DeployedQuantisedConfidenceLUT | bfloat16 | refused: DeployedQuantisedConfidenceLUT is fp32/fp64 inference only; got input dtype torch.bfloat16 | - | - | - | - | - | - | - | - |
| large | DeployedQuantisedConfidenceLUT n=1 | float32 | ok | 0.890 | 0.094 | - | - | - | - | - | - |
| large | DeployedQuantisedConfidenceLUT n=1 | bfloat16 | refused: DeployedQuantisedConfidenceLUT is fp32/fp64 inference only; got input dtype torch.bfloat16 | - | - | - | - | - | - | - | - |
