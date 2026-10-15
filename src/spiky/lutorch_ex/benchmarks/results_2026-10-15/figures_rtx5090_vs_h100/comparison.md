RTX 5090: NVIDIA GeForce RTX 5090 (sm_120), torch 2.9.1+cu130, driver 595.84, commit f88dd711
H100: NVIDIA H100 80GB HBM3 (sm_90), torch 2.9.1+cu130, driver 580.173.02, commit f88dd711

Wall = median of the per-round medians (ms); GPU = kernel time per step (ms; * = gpu_suspect); peak = train-step peak MiB above the pre-step allocation.

| geometry | cartridge | eval wall RTX 5090 | eval wall H100 | ×eval | train wall RTX 5090 | train wall H100 | ×train | train GPU RTX 5090 | train GPU H100 | train peak MiB RTX 5090 | train peak MiB H100 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| champion | ManifestoHardLUT | 0.675 | 1.174 | 1.74 | 14.964 | 19.551 | 1.31 | 15.260 | 19.756 | 1304.0 | 1304.0 |
| champion | ManifestoSoftLUT | 1.244 | 2.082 | 1.67 | 15.316 | 19.809 | 1.29 | 15.575 | 20.082 | 1304.0 | 1304.0 |
| champion | FusedManifestoHardLUT | 0.281 | 0.447 | 1.59 | 1.545 | 2.072 | 1.34 | 1.503 | 2.029 | 96.0 | 96.0 |
| champion | FusedManifestoSoftLUT | 0.492 | 0.650 | 1.32 | 2.086 | 2.380 | 1.14 | 2.066 | 2.328 | 96.0 | 96.0 |
| champion | ManifestoHardLUT weight=softsign | 0.675 | 1.173 | 1.74 | 15.306 | 19.888 | 1.30 | 15.774 | 20.089 | 1304.0 | 1304.0 |
| champion | ManifestoSoftLUT weight=softsign | 1.318 | 2.128 | 1.62 | 15.663 | 20.175 | 1.29 | 16.066 | 20.414 | 1359.0 | 1359.0 |
| champion | FusedManifestoHardLUT weight=softsign | 0.297 | 0.619 | 2.08 | 1.929 | 3.012 | 1.56 | 1.843 | 2.915 | 96.0 | 96.0 |
| champion | FusedManifestoSoftLUT weight=softsign | 0.509 | 0.706 | 1.39 | 2.321 | 3.066 | 1.32 | 2.246 | 2.971 | 96.0 | 96.0 |
| champion | ConfidenceLUT n=1 | 0.920 | 1.501 | 1.63 | 3.424 | 4.478 | 1.31 | 3.318 | 4.302 | 728.0 | 728.0 |
| champion | ConfidenceLUT n=2 | 1.801 | 1.751 | 0.97 | 5.389 | 6.697 | 1.24 | 5.221 | 6.501 | 840.0 | 840.0 |
| champion | FusedConfidenceLUT n=1 | 0.293 | 0.402 | 1.37 | 1.517 | 1.987 | 1.31 | 1.475 | 1.930 | 96.4 | 96.4 |
| champion | FusedConfidenceLUT n=2 | 0.507 | 0.722 | 1.42 | 2.414 | 3.103 | 1.29 | 2.390 | 3.044 | 96.4 | 96.4 |
| champion | QuantisedConfidenceLUT n=1 | 1.662 | 2.948 | 1.77 | 7.223 | 11.740 | 1.63 | 7.206 | 11.599 | 568.8 | 568.8 |
| champion | QuantisedConfidenceLUT n=2 | 2.585 | 4.484 | 1.73 | 12.489 | 19.123 | 1.53 | 12.574 | 19.192 | 1142.4 | 1142.4 |
| champion | FusedQuantisedConfidenceLUT n=1 | 0.335 | 0.688 | 2.05 | 1.474 | 1.915 | 1.30 | 1.451 | 1.861 | 128.4 | 128.4 |
| champion | FusedQuantisedConfidenceLUT n=2 | 0.565 | 0.952 | 1.69 | 2.497 | 2.744 | 1.10 | 2.444 | 2.680 | 128.4 | 128.4 |
| champion | DeployedQuantisedConfidenceLUT | 0.336 | 0.476 | 1.41 | - | - | - | - | - | - | - |
| champion | DeployedQuantisedConfidenceLUT n=1 | 0.242 | 0.353 | 1.46 | - | - | - | - | - | - | - |
| large | ManifestoHardLUT | 2.625 | 4.476 | 1.71 | 60.713 | 76.061 | 1.25 | 61.908 | 77.598 | 5216.0 | 5216.0 |
| large | ManifestoSoftLUT | 5.080 | 8.069 | 1.59 | 61.970 | 78.708 | 1.27 | 64.000 | 80.393 | 5216.0 | 5216.0 |
| large | FusedManifestoHardLUT | 1.026 | 1.599 | 1.56 | 5.778 | 7.643 | 1.32 | 5.722 | 7.590 | 240.0 | 240.0 |
| large | FusedManifestoSoftLUT | 1.899 | 2.488 | 1.31 | 8.030 | 8.971 | 1.12 | 7.977 | 8.911 | 240.0 | 240.0 |
| large | ManifestoHardLUT weight=softsign | 2.634 | 4.457 | 1.69 | 62.669 | 77.118 | 1.23 | 64.039 | 78.552 | 5216.0 | 5216.0 |
| large | ManifestoSoftLUT weight=softsign | 5.390 | 8.245 | 1.53 | 64.030 | 79.737 | 1.25 | 66.345 | 81.416 | 5216.0 | 5216.0 |
| large | FusedManifestoHardLUT weight=softsign | 1.042 | 2.199 | 2.11 | 7.265 | 11.351 | 1.56 | 7.135 | 11.171 | 240.0 | 240.0 |
| large | FusedManifestoSoftLUT weight=softsign | 1.921 | 2.614 | 1.36 | 8.715 | 11.597 | 1.33 | 8.591 | 11.412 | 240.0 | 240.0 |
| large | ConfidenceLUT n=1 | 3.825 | 5.746 | 1.50 | 13.406 | 16.962 | 1.27 | 13.297 | 16.773 | 2768.0 | 2768.0 |
| large | ConfidenceLUT n=2 | 5.179 | 6.782 | 1.31 | 21.281 | 26.222 | 1.23 | 22.045 | 26.085 | 3216.0 | 3216.0 |
| large | FusedConfidenceLUT n=1 | 1.075 | 1.443 | 1.34 | 5.697 | 7.289 | 1.28 | 5.601 | 7.224 | 241.5 | 241.5 |
| large | FusedConfidenceLUT n=2 | 1.981 | 2.765 | 1.40 | 9.205 | 11.712 | 1.27 | 9.084 | 11.585 | 241.5 | 241.5 |
| large | QuantisedConfidenceLUT n=1 | 6.243 | 11.028 | 1.77 | 27.741 | 44.075 | 1.59 | 28.690 | 44.504 | 2000.0 | 2000.0 |
| large | QuantisedConfidenceLUT n=2 | 9.997 | 17.107 | 1.71 | 49.497 | 74.765 | 1.51 | 51.210 | 76.130 | 4269.4 | 4269.4 |
| large | FusedQuantisedConfidenceLUT n=1 | 1.274 | 2.449 | 1.92 | 5.383 | 6.829 | 1.27 | 5.267 | 6.770 | 369.5 | 369.5 |
| large | FusedQuantisedConfidenceLUT n=2 | 2.131 | 3.558 | 1.67 | 9.072 | 10.197 | 1.12 | 9.028 | 10.081 | 369.5 | 369.5 |
| large | DeployedQuantisedConfidenceLUT | 1.291 | 1.763 | 1.37 | - | - | - | - | - | - | - |
| large | DeployedQuantisedConfidenceLUT n=1 | 0.909 | 1.296 | 1.43 | - | - | - | - | - | - | - |
