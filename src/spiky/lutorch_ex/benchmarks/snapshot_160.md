# lutorch_ex performance snapshot of record (#160)

**RTX 5090 only** (sm_120), torch 2.9.1+cu130, CUDA 13.0, driver 595.84; the H100 was not re-measured. Code
`a811390e`, the final head of #160 (no uncommitted changes in lutorch_ex); measured 2026-10-12, the whole matrix in one
run. Champion geometry h16 tph64 nap8 d48 at 8,192 vectors (`champion`) and 32,768 (`large`). Compiled, one fresh
process per cell, a Dynamo reset per cell, median of 30 timed iterations after 10 warm-up; peak GiB above the
pre-step allocation; clocks not locked (observed SM clock 2,685-2,700 MHz under load, no throttling; nothing else
heavy on the GPU). Every class runs its one implementation with its defaults; `status` says why a cell has no numbers
(out of memory, a refused dtype, a case the fused class does not cover). The IQRs and the environment observations
are in `snapshot_160.json`.

Reproduce: `python -m spiky.lutorch_ex.bench_snapshot --geometries champion,large --out snapshot_160.json --md snapshot_160.md`

Accepted cost of #160 (not fixed here; the quantised follow-up PR): `QuantisedConfidenceLUT` is now the pure-torch
reference -- without the p2_scalars op (n=2 eval) or the training monolith, which moved to
`FusedQuantisedConfidenceLUT`, and with its exponents in the kernel's operation order (a left fold). Against the #159
snapshot of record (the old default, which used the op and the monolith): 8,192 eval 2.52 -> 2.97 ms, train total
12.43 -> 12.56 ms; 32,768 eval 9.66 -> 11.76 ms, train total 48.56 -> 50.74 ms, eval peak 1.66 -> 2.03 GiB, train peak
3.64 -> 4.17 GiB. `FusedQuantisedConfidenceLUT` keeps the old speed (32,768: eval 9.67 ms, train total 49.85 ms /
3.64 GiB). `DeployedQuantisedConfidenceLUT` eval: 8,192 2.66 -> 2.89 ms; 32,768 25.03 -> 17.18 ms (faster: the
explicit fold changes how Inductor fuses the deployed read).

Run-to-run variation, measured while preparing this snapshot (repeats of the cells that moved > 3% against the
previous version of this file): fused-class cells vary by up to ~5% between sessions at these sub-millisecond to
few-millisecond sizes (clocks unlocked, host launch overhead), with no code change in their path; and the compiled
reference's bf16 cells depend on the kernel configs Inductor picks when it compiles -- `ConfidenceLUT n=2` bf16 at
32,768 measured a training forward of 6.09 ms in the previous version of this file and 4.53-4.89 ms here, with
identical code. Treat single-cell differences of that size as noise, not as regressions or improvements.

| geometry | cartridge | dtype | status | eval forward ms | GiB | train forward ms | GiB | train backward ms | GiB | train total ms | GiB |
|---|---|---|---|---|---|---|---|---|---|---|---|
| champion | ManifestoHardLUT | float32 | ok | 0.665 | 0.055 | 18.331 | 6.712 | 7.476 | 3.517 | 26.066 | 6.712 |
| champion | ManifestoHardLUT | bfloat16 | ok | 0.701 | 0.125 | 18.337 | 6.782 | 7.333 | 3.528 | 25.736 | 6.782 |
| champion | ManifestoSoftLUT | float32 | ok | 1.227 | 0.148 | 5.048 | 1.273 | 10.204 | 1.265 | 15.252 | 1.273 |
| champion | ManifestoSoftLUT | bfloat16 | ok | 1.250 | 0.219 | 5.152 | 1.344 | 10.569 | 1.323 | 15.735 | 1.344 |
| champion | FusedManifestoHardLUT | float32 | ok | 0.254 | 0.023 | 0.292 | 0.023 | 1.276 | 0.094 | 1.565 | 0.094 |
| champion | FusedManifestoHardLUT | bfloat16 | ok | 0.194 | 0.012 | 0.252 | 0.012 | 1.158 | 0.094 | 1.400 | 0.094 |
| champion | FusedManifestoSoftLUT | float32 | ok | 0.462 | 0.023 | 0.555 | 0.023 | 1.730 | 0.094 | 2.284 | 0.094 |
| champion | FusedManifestoSoftLUT | bfloat16 | ok | 0.283 | 0.012 | 0.317 | 0.012 | 1.516 | 0.094 | 1.839 | 0.094 |
| champion | ManifestoHardLUT weight=softsign | float32 | ok | 0.649 | 0.055 | 18.325 | 6.743 | 7.661 | 3.579 | 26.170 | 6.743 |
| champion | ManifestoHardLUT weight=softsign | bfloat16 | ok | 0.688 | 0.125 | 18.001 | 6.813 | 7.664 | 3.591 | 25.661 | 6.813 |
| champion | ManifestoSoftLUT weight=softsign | float32 | ok | 1.291 | 0.148 | 5.097 | 1.273 | 10.912 | 1.327 | 16.033 | 1.327 |
| champion | ManifestoSoftLUT weight=softsign | bfloat16 | ok | 1.316 | 0.219 | 5.161 | 1.344 | 10.602 | 1.386 | 15.813 | 1.386 |
| champion | FusedManifestoHardLUT weight=softsign | float32 | ok | 0.266 | 0.023 | 0.340 | 0.023 | 1.586 | 0.094 | 1.915 | 0.094 |
| champion | FusedManifestoHardLUT weight=softsign | bfloat16 | ok | 0.215 | 0.012 | 0.256 | 0.012 | 1.493 | 0.094 | 1.749 | 0.094 |
| champion | FusedManifestoSoftLUT weight=softsign | float32 | ok | 0.477 | 0.023 | 0.560 | 0.023 | 1.804 | 0.094 | 2.363 | 0.094 |
| champion | FusedManifestoSoftLUT weight=softsign | bfloat16 | ok | 0.312 | 0.012 | 0.339 | 0.012 | 1.698 | 0.094 | 2.034 | 0.094 |
| champion | ConfidenceLUT n=1 | float32 | ok | 0.909 | 0.117 | 1.073 | 0.117 | 2.362 | 0.711 | 3.434 | 0.711 |
| champion | ConfidenceLUT n=1 | bfloat16 | ok | 0.942 | 0.188 | 1.072 | 0.199 | 2.363 | 0.746 | 3.436 | 0.746 |
| champion | ConfidenceLUT n=2 | float32 | ok | 1.254 | 0.219 | 1.327 | 0.211 | 4.078 | 0.820 | 5.406 | 0.820 |
| champion | ConfidenceLUT n=2 | bfloat16 | ok | 1.307 | 0.289 | 1.287 | 0.293 | 4.038 | 0.855 | 5.325 | 0.855 |
| champion | FusedConfidenceLUT n=1 | float32 | ok | 0.267 | 0.023 | 0.321 | 0.023 | 1.228 | 0.094 | 1.542 | 0.094 |
| champion | FusedConfidenceLUT n=1 | bfloat16 | ok | 0.204 | 0.012 | 0.271 | 0.012 | 1.153 | 0.094 | 1.418 | 0.094 |
| champion | FusedConfidenceLUT n=2 | float32 | ok | 0.472 | 0.023 | 0.538 | 0.023 | 1.938 | 0.094 | 2.468 | 0.094 |
| champion | FusedConfidenceLUT n=2 | bfloat16 | ok | 0.315 | 0.012 | 0.382 | 0.012 | 1.817 | 0.094 | 2.211 | 0.094 |
| champion | QuantisedConfidenceLUT n=2 | float32 | ok | 2.967 | 0.508 | 2.978 | 0.617 | 9.587 | 1.116 | 12.565 | 1.116 |
| champion | QuantisedConfidenceLUT n=2 | bfloat16 | refused: QuantisedConfidenceLUT does not support low precision | - | - | - | - | - | - | - | - |
| champion | FusedQuantisedConfidenceLUT n=2 | float32 | ok | 2.527 | 0.415 | 2.519 | 0.352 | 10.303 | 0.946 | 12.803 | 0.946 |
| champion | FusedQuantisedConfidenceLUT n=2 | bfloat16 | error: RuntimeError: FusedQuantisedConfidenceLUT: not a case the fused op covers: input dtype torch.bfloat16 (float32 only); table dtype torc | - | - | - | - | - | - | - | - |
| champion | DeployedQuantisedConfidenceLUT | float32 | ok | 2.888 | 0.414 | - | - | - | - | - | - |
| champion | DeployedQuantisedConfidenceLUT | bfloat16 | refused: DeployedQuantisedConfidenceLUT is fp32/fp64 inference only; got input dtype torch.bfloat16 | - | - | - | - | - | - | - | - |
| large | ManifestoHardLUT | float32 | out of memory | 2.545 | 0.219 | - | - | - | - | - | - |
| large | ManifestoHardLUT | bfloat16 | out of memory | 2.672 | 0.359 | - | - | - | - | - | - |
| large | ManifestoSoftLUT | float32 | ok | 4.947 | 0.594 | 21.269 | 5.094 | 41.879 | 4.763 | 63.181 | 5.094 |
| large | ManifestoSoftLUT | bfloat16 | ok | 5.043 | 0.734 | 21.443 | 5.234 | 41.827 | 4.857 | 63.445 | 5.234 |
| large | FusedManifestoHardLUT | float32 | ok | 0.990 | 0.094 | 1.073 | 0.094 | 4.822 | 0.234 | 5.894 | 0.234 |
| large | FusedManifestoHardLUT | bfloat16 | ok | 0.748 | 0.047 | 0.827 | 0.047 | 4.378 | 0.234 | 5.209 | 0.234 |
| large | FusedManifestoSoftLUT | float32 | ok | 1.809 | 0.094 | 2.084 | 0.094 | 6.721 | 0.234 | 8.802 | 0.234 |
| large | FusedManifestoSoftLUT | bfloat16 | ok | 1.118 | 0.047 | 1.190 | 0.047 | 6.273 | 0.234 | 7.469 | 0.234 |
| large | ManifestoHardLUT weight=softsign | float32 | out of memory | 2.490 | 0.219 | - | - | - | - | - | - |
| large | ManifestoHardLUT weight=softsign | bfloat16 | out of memory | 2.620 | 0.359 | - | - | - | - | - | - |
| large | ManifestoSoftLUT weight=softsign | float32 | ok | 5.353 | 0.594 | 21.529 | 5.094 | 43.663 | 5.013 | 65.469 | 5.094 |
| large | ManifestoSoftLUT weight=softsign | bfloat16 | ok | 5.257 | 0.734 | 21.710 | 5.234 | 43.678 | 5.107 | 65.700 | 5.234 |
| large | FusedManifestoHardLUT weight=softsign | float32 | ok | 1.005 | 0.094 | 1.111 | 0.094 | 6.279 | 0.234 | 7.382 | 0.234 |
| large | FusedManifestoHardLUT weight=softsign | bfloat16 | ok | 0.766 | 0.047 | 0.883 | 0.047 | 5.926 | 0.234 | 6.809 | 0.234 |
| large | FusedManifestoSoftLUT weight=softsign | float32 | ok | 1.839 | 0.094 | 1.967 | 0.094 | 7.449 | 0.234 | 9.411 | 0.234 |
| large | FusedManifestoSoftLUT weight=softsign | bfloat16 | ok | 1.172 | 0.047 | 1.249 | 0.047 | 6.754 | 0.234 | 7.993 | 0.234 |
| large | ConfidenceLUT n=1 | float32 | ok | 3.611 | 0.469 | 3.709 | 0.469 | 10.042 | 2.703 | 13.753 | 2.703 |
| large | ConfidenceLUT n=1 | bfloat16 | ok | 3.820 | 0.609 | 3.761 | 0.656 | 10.093 | 2.844 | 13.895 | 2.844 |
| large | ConfidenceLUT n=2 | float32 | ok | 4.977 | 0.875 | 4.831 | 0.844 | 16.124 | 3.141 | 21.029 | 3.141 |
| large | ConfidenceLUT n=2 | bfloat16 | ok | 5.116 | 1.016 | 4.888 | 1.031 | 16.236 | 3.281 | 21.190 | 3.281 |
| large | FusedConfidenceLUT n=1 | float32 | ok | 1.049 | 0.094 | 1.122 | 0.094 | 4.726 | 0.236 | 5.850 | 0.236 |
| large | FusedConfidenceLUT n=1 | bfloat16 | ok | 0.798 | 0.047 | 0.960 | 0.047 | 4.476 | 0.236 | 5.443 | 0.236 |
| large | FusedConfidenceLUT n=2 | float32 | ok | 1.874 | 0.094 | 1.968 | 0.094 | 7.787 | 0.236 | 9.738 | 0.236 |
| large | FusedConfidenceLUT n=2 | bfloat16 | ok | 1.209 | 0.047 | 1.387 | 0.047 | 6.957 | 0.236 | 8.346 | 0.236 |
| large | QuantisedConfidenceLUT n=2 | float32 | ok | 11.764 | 2.031 | 11.311 | 2.328 | 39.365 | 4.169 | 50.736 | 4.169 |
| large | QuantisedConfidenceLUT n=2 | bfloat16 | refused: QuantisedConfidenceLUT does not support low precision | - | - | - | - | - | - | - | - |
| large | FusedQuantisedConfidenceLUT n=2 | float32 | ok | 9.675 | 1.658 | 9.846 | 1.406 | 39.983 | 3.641 | 49.848 | 3.641 |
| large | FusedQuantisedConfidenceLUT n=2 | bfloat16 | error: RuntimeError: FusedQuantisedConfidenceLUT: not a case the fused op covers: input dtype torch.bfloat16 (float32 only); table dtype torc | - | - | - | - | - | - | - | - |
| large | DeployedQuantisedConfidenceLUT | float32 | ok | 17.184 | 1.656 | - | - | - | - | - | - |
| large | DeployedQuantisedConfidenceLUT | bfloat16 | refused: DeployedQuantisedConfidenceLUT is fp32/fp64 inference only; got input dtype torch.bfloat16 | - | - | - | - | - | - | - | - |
