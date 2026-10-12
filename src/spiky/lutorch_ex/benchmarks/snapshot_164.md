# lutorch_ex performance snapshot of record (#164)

**Hardware and software.** RTX 5090 only (sm_120); the H100 was not re-measured. torch 2.9.1+cu130, Triton 3.5.1, CUDA 13.0, driver 595.84.

**Code.** `41df6f25` (#164's library code after Stage 3); no uncommitted changes in lutorch_ex.

**Run.** Measured 2026-10-12, the whole matrix in one run.
- Champion geometry h16 tph64 nap8 d48, at 8,192 vectors (`champion`) and 32,768 (`large`).
- Compiled. One fresh process per cell, and a Dynamo reset per cell.
- Median of 30 timed iterations after 10 warm-up. Peak GiB is above the pre-step allocation.

**Conditions.**
- Clocks not locked. Observed SM clock 2670 MHz after the champion cells and 2655 MHz after the large ones.
- 32 C at the start, 41 C at the end.
- **No throttling** (reasons 0x0 at every observation).
- Other processes on the card: desktop only, ~180 MiB (140 + 12 + 28 MiB).

**Reading the table.** Every class runs its one implementation with its defaults; `status` says why a cell has no numbers. The IQRs and the environment observations are in `snapshot_164.json`.

Reproduce: `python -m spiky.lutorch_ex.bench_snapshot --geometries champion,large --out snapshot_164.json --md snapshot_164.md`

**What #164 changed against `snapshot_162`.** The rest of the library is unchanged; see `fused_quantised_stage0.md` for the stages.
- **`FusedQuantisedConfidenceLUT`** is one fused CUDA forward (Stage 1) and one hand-written CUDA backward (Stage 4).
  - 32,768 vectors: eval 9.683 → 1.942 ms (peak 1.658 → 0.094 GiB). Train total 48.50 → 10.07 ms (3.641 → 0.236 GiB).
  - 8,192 vectors: eval 2.521 → 0.526 ms. Train total 12.39 → 2.709 ms.
- **`DeployedQuantisedConfidenceLUT`** is one CUDA kernel on the same template, specialised for int8 rows (Stage 3).
  - 32,768 vectors: eval 16.802 → 1.257 ms, peak 1.656 → 0.094 GiB.
  - 8,192 vectors: 2.909 → 0.311 ms.
  - Against FusedConfidenceLUT n=2 in this run: faster than fp32 (1.875 / 0.474). Level with bf16 at 8k (0.314). +3% over bf16 at 32k (1.217).
- **New row: `DeployedQuantisedConfidenceLUT n=1`**, an export of `QuantisedConfidenceLUT(read_top_n=1)`. 0.903 ms at 32,768 vectors, 0.223 ms at 8,192.
- **Run-to-run variation is unchanged.** Cells of untouched classes moved within the noise band: few-ms cells ±6%, ~0.3 ms launch-bound cells up to ±15%. For example, FusedManifestoHardLUT fp32 train forward at 8,192 moved 0.317 → 0.285 ms with no code change in its path.

| geometry | cartridge | dtype | status | eval forward ms | GiB | train forward ms | GiB | train backward ms | GiB | train total ms | GiB |
|---|---|---|---|---|---|---|---|---|---|---|---|
| champion | ManifestoHardLUT | float32 | ok | 0.667 | 0.055 | 5.935 | 1.273 | 8.957 | 1.047 | 14.891 | 1.273 |
| champion | ManifestoHardLUT | bfloat16 | ok | 0.701 | 0.125 | 5.994 | 1.344 | 8.938 | 1.082 | 14.936 | 1.344 |
| champion | ManifestoSoftLUT | float32 | ok | 1.226 | 0.148 | 5.046 | 1.273 | 10.229 | 1.265 | 15.276 | 1.273 |
| champion | ManifestoSoftLUT | bfloat16 | ok | 1.252 | 0.219 | 5.118 | 1.344 | 10.094 | 1.323 | 15.212 | 1.344 |
| champion | FusedManifestoHardLUT | float32 | ok | 0.252 | 0.023 | 0.285 | 0.023 | 1.274 | 0.094 | 1.562 | 0.094 |
| champion | FusedManifestoHardLUT | bfloat16 | ok | 0.192 | 0.012 | 0.255 | 0.012 | 1.160 | 0.094 | 1.415 | 0.094 |
| champion | FusedManifestoSoftLUT | float32 | ok | 0.462 | 0.023 | 0.524 | 0.023 | 1.726 | 0.094 | 2.253 | 0.094 |
| champion | FusedManifestoSoftLUT | bfloat16 | ok | 0.282 | 0.012 | 0.338 | 0.012 | 1.559 | 0.094 | 1.896 | 0.094 |
| champion | ManifestoHardLUT weight=softsign | float32 | ok | 0.670 | 0.055 | 5.962 | 1.273 | 9.286 | 1.047 | 15.249 | 1.273 |
| champion | ManifestoHardLUT weight=softsign | bfloat16 | ok | 0.709 | 0.125 | 6.017 | 1.344 | 9.223 | 1.082 | 15.240 | 1.344 |
| champion | ManifestoSoftLUT weight=softsign | float32 | ok | 1.293 | 0.148 | 5.071 | 1.273 | 10.552 | 1.327 | 15.625 | 1.327 |
| champion | ManifestoSoftLUT weight=softsign | bfloat16 | ok | 1.316 | 0.219 | 5.155 | 1.344 | 10.431 | 1.386 | 15.588 | 1.386 |
| champion | FusedManifestoHardLUT weight=softsign | float32 | ok | 0.268 | 0.023 | 0.343 | 0.023 | 1.598 | 0.094 | 1.935 | 0.094 |
| champion | FusedManifestoHardLUT weight=softsign | bfloat16 | ok | 0.214 | 0.012 | 0.292 | 0.012 | 1.494 | 0.094 | 1.785 | 0.094 |
| champion | FusedManifestoSoftLUT weight=softsign | float32 | ok | 0.479 | 0.023 | 0.572 | 0.023 | 1.875 | 0.094 | 2.448 | 0.094 |
| champion | FusedManifestoSoftLUT weight=softsign | bfloat16 | ok | 0.305 | 0.012 | 0.387 | 0.012 | 1.782 | 0.094 | 2.157 | 0.094 |
| champion | ConfidenceLUT n=1 | float32 | ok | 0.908 | 0.117 | 1.071 | 0.117 | 2.362 | 0.711 | 3.434 | 0.711 |
| champion | ConfidenceLUT n=1 | bfloat16 | ok | 0.943 | 0.188 | 1.093 | 0.199 | 2.364 | 0.746 | 3.455 | 0.746 |
| champion | ConfidenceLUT n=2 | float32 | ok | 1.270 | 0.219 | 1.294 | 0.211 | 4.080 | 0.820 | 5.373 | 0.820 |
| champion | ConfidenceLUT n=2 | bfloat16 | ok | 1.295 | 0.289 | 1.322 | 0.293 | 4.036 | 0.855 | 5.358 | 0.855 |
| champion | FusedConfidenceLUT n=1 | float32 | ok | 0.267 | 0.023 | 0.322 | 0.023 | 1.245 | 0.094 | 1.548 | 0.094 |
| champion | FusedConfidenceLUT n=1 | bfloat16 | ok | 0.204 | 0.012 | 0.270 | 0.012 | 1.150 | 0.094 | 1.413 | 0.094 |
| champion | FusedConfidenceLUT n=2 | float32 | ok | 0.474 | 0.023 | 0.543 | 0.023 | 1.965 | 0.094 | 2.500 | 0.094 |
| champion | FusedConfidenceLUT n=2 | bfloat16 | ok | 0.314 | 0.012 | 0.387 | 0.012 | 1.787 | 0.094 | 2.168 | 0.094 |
| champion | QuantisedConfidenceLUT n=2 | float32 | ok | 2.580 | 0.259 | 3.034 | 0.516 | 9.496 | 1.116 | 12.531 | 1.116 |
| champion | QuantisedConfidenceLUT n=2 | bfloat16 | refused: QuantisedConfidenceLUT does not support low precision | - | - | - | - | - | - | - | - |
| champion | FusedQuantisedConfidenceLUT n=2 | float32 | ok | 0.526 | 0.023 | 0.611 | 0.023 | 2.107 | 0.094 | 2.709 | 0.094 |
| champion | FusedQuantisedConfidenceLUT n=2 | bfloat16 | error: RuntimeError: FusedQuantisedConfidenceLUT: not a case the fused op covers: input dtype torch.bfloat16 (float32 only); table dtype torc | - | - | - | - | - | - | - | - |
| champion | DeployedQuantisedConfidenceLUT | float32 | ok | 0.311 | 0.023 | - | - | - | - | - | - |
| champion | DeployedQuantisedConfidenceLUT | bfloat16 | refused: DeployedQuantisedConfidenceLUT is fp32/fp64 inference only; got input dtype torch.bfloat16 | - | - | - | - | - | - | - | - |
| champion | DeployedQuantisedConfidenceLUT n=1 | float32 | ok | 0.223 | 0.023 | - | - | - | - | - | - |
| champion | DeployedQuantisedConfidenceLUT n=1 | bfloat16 | refused: DeployedQuantisedConfidenceLUT is fp32/fp64 inference only; got input dtype torch.bfloat16 | - | - | - | - | - | - | - | - |
| large | ManifestoHardLUT | float32 | ok | 2.544 | 0.219 | 24.529 | 5.094 | 35.918 | 4.188 | 60.445 | 5.094 |
| large | ManifestoHardLUT | bfloat16 | ok | 2.678 | 0.359 | 24.732 | 5.234 | 35.906 | 4.188 | 60.637 | 5.234 |
| large | ManifestoSoftLUT | float32 | ok | 5.025 | 0.594 | 20.805 | 5.094 | 40.986 | 4.763 | 61.791 | 5.094 |
| large | ManifestoSoftLUT | bfloat16 | ok | 5.061 | 0.734 | 20.988 | 5.234 | 40.984 | 4.857 | 61.979 | 5.234 |
| large | FusedManifestoHardLUT | float32 | ok | 0.990 | 0.094 | 1.074 | 0.094 | 4.822 | 0.234 | 5.896 | 0.234 |
| large | FusedManifestoHardLUT | bfloat16 | ok | 0.749 | 0.047 | 0.847 | 0.047 | 4.495 | 0.234 | 5.342 | 0.234 |
| large | FusedManifestoSoftLUT | float32 | ok | 1.814 | 0.094 | 1.974 | 0.094 | 6.423 | 0.234 | 8.401 | 0.234 |
| large | FusedManifestoSoftLUT | bfloat16 | ok | 1.120 | 0.047 | 1.202 | 0.047 | 6.301 | 0.234 | 7.504 | 0.234 |
| large | ManifestoHardLUT weight=softsign | float32 | ok | 2.567 | 0.219 | 24.785 | 5.094 | 37.746 | 4.188 | 62.530 | 5.094 |
| large | ManifestoHardLUT weight=softsign | bfloat16 | ok | 2.696 | 0.359 | 25.000 | 5.234 | 37.745 | 4.188 | 62.749 | 5.234 |
| large | ManifestoSoftLUT weight=softsign | float32 | ok | 5.379 | 0.594 | 21.074 | 5.094 | 42.784 | 5.013 | 63.858 | 5.094 |
| large | ManifestoSoftLUT weight=softsign | bfloat16 | ok | 5.389 | 0.734 | 21.252 | 5.234 | 42.812 | 5.107 | 64.074 | 5.234 |
| large | FusedManifestoHardLUT weight=softsign | float32 | ok | 1.008 | 0.094 | 1.113 | 0.094 | 6.333 | 0.234 | 7.443 | 0.234 |
| large | FusedManifestoHardLUT weight=softsign | bfloat16 | ok | 0.769 | 0.047 | 0.887 | 0.047 | 5.944 | 0.234 | 6.830 | 0.234 |
| large | FusedManifestoSoftLUT weight=softsign | float32 | ok | 1.837 | 0.094 | 2.024 | 0.094 | 7.062 | 0.234 | 9.086 | 0.234 |
| large | FusedManifestoSoftLUT weight=softsign | bfloat16 | ok | 1.142 | 0.047 | 1.219 | 0.047 | 6.835 | 0.234 | 8.054 | 0.234 |
| large | ConfidenceLUT n=1 | float32 | ok | 3.677 | 0.469 | 3.746 | 0.469 | 9.627 | 2.703 | 13.379 | 2.703 |
| large | ConfidenceLUT n=1 | bfloat16 | ok | 3.724 | 0.609 | 3.782 | 0.656 | 9.633 | 2.844 | 13.413 | 2.844 |
| large | ConfidenceLUT n=2 | float32 | ok | 4.856 | 0.875 | 4.668 | 0.844 | 16.129 | 3.141 | 20.797 | 3.141 |
| large | ConfidenceLUT n=2 | bfloat16 | ok | 5.180 | 1.016 | 4.776 | 1.031 | 15.887 | 3.281 | 20.660 | 3.281 |
| large | FusedConfidenceLUT n=1 | float32 | ok | 1.049 | 0.094 | 1.137 | 0.094 | 4.759 | 0.236 | 5.890 | 0.236 |
| large | FusedConfidenceLUT n=1 | bfloat16 | ok | 0.800 | 0.047 | 0.958 | 0.047 | 4.493 | 0.236 | 5.450 | 0.236 |
| large | FusedConfidenceLUT n=2 | float32 | ok | 1.875 | 0.094 | 1.972 | 0.094 | 7.493 | 0.236 | 9.471 | 0.236 |
| large | FusedConfidenceLUT n=2 | bfloat16 | ok | 1.217 | 0.047 | 1.425 | 0.047 | 7.017 | 0.236 | 8.443 | 0.236 |
| large | QuantisedConfidenceLUT n=2 | float32 | ok | 9.965 | 0.895 | 11.564 | 1.922 | 38.111 | 4.169 | 49.673 | 4.169 |
| large | QuantisedConfidenceLUT n=2 | bfloat16 | refused: QuantisedConfidenceLUT does not support low precision | - | - | - | - | - | - | - | - |
| large | FusedQuantisedConfidenceLUT n=2 | float32 | ok | 1.942 | 0.094 | 2.061 | 0.094 | 8.008 | 0.236 | 10.068 | 0.236 |
| large | FusedQuantisedConfidenceLUT n=2 | bfloat16 | error: RuntimeError: FusedQuantisedConfidenceLUT: not a case the fused op covers: input dtype torch.bfloat16 (float32 only); table dtype torc | - | - | - | - | - | - | - | - |
| large | DeployedQuantisedConfidenceLUT | float32 | ok | 1.257 | 0.094 | - | - | - | - | - | - |
| large | DeployedQuantisedConfidenceLUT | bfloat16 | refused: DeployedQuantisedConfidenceLUT is fp32/fp64 inference only; got input dtype torch.bfloat16 | - | - | - | - | - | - | - | - |
| large | DeployedQuantisedConfidenceLUT n=1 | float32 | ok | 0.903 | 0.094 | - | - | - | - | - | - |
| large | DeployedQuantisedConfidenceLUT n=1 | bfloat16 | refused: DeployedQuantisedConfidenceLUT is fp32/fp64 inference only; got input dtype torch.bfloat16 | - | - | - | - | - | - | - | - |
