RTX 5090: NVIDIA GeForce RTX 5090 (sm_120), torch 2.9.1+cu130, driver 595.84, commit d0136e2090
H100: NVIDIA H100 80GB HBM3 (sm_90), torch 2.9.1+cu130, driver 580.173.02, commit b02cf44008

Wall = median of the per-round medians (ms); GPU = kernel time per step (ms; * = gpu_suspect); peak = train-step peak MiB above the pre-step allocation.

| geometry | cartridge | eval wall RTX 5090 | eval wall H100 | ×eval | train wall RTX 5090 | train wall H100 | ×train | train GPU RTX 5090 | train GPU H100 | train peak MiB RTX 5090 | train peak MiB H100 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| champion | ManifestoHardLUT | 0.675 | 1.179 | 1.75 | 14.961 | 19.567 | 1.31 | 15.150 | 19.785 | 1304.0 | 1304.0 |
| champion | ManifestoSoftLUT | 1.243 | 2.088 | 1.68 | 15.297 | 19.840 | 1.30 | 15.499 | 20.120 | 1304.0 | 1304.0 |
| champion | FusedManifestoHardLUT | 0.282 | 0.447 | 1.58 | 1.635 | 2.214 | 1.35 | 1.753* | 2.171 | 96.0 | 96.0 |
| champion | FusedManifestoSoftLUT | 0.492 | 0.650 | 1.32 | 2.221 | 3.509 | 1.58 | 2.146 | 3.444 | 96.0 | 96.0 |
| champion | ManifestoHardLUT weight=softsign | 0.674 | 1.176 | 1.74 | 15.308 | 19.911 | 1.30 | 15.482 | 20.120 | 1304.0 | 1304.0 |
| champion | ManifestoSoftLUT weight=softsign | 1.318 | 2.132 | 1.62 | 15.672 | 20.123 | 1.28 | 15.862 | 20.382 | 1359.0 | 1359.0 |
| champion | FusedManifestoHardLUT weight=softsign | 0.295 | 0.615 | 2.09 | 1.966 | 3.356 | 1.71 | 1.879 | 3.268 | 96.0 | 96.0 |
| champion | FusedManifestoSoftLUT weight=softsign | 0.509 | 0.702 | 1.38 | 2.384 | 4.313 | 1.81 | 2.287 | 4.217 | 96.0 | 96.0 |
| champion | ConfidenceLUT n=1 | 0.920 | 1.506 | 1.64 | 3.427 | 4.480 | 1.31 | 3.370 | 4.303 | 728.0 | 728.0 |
| champion | ConfidenceLUT n=2 | 1.850 | 1.755 | 0.95 | 5.373 | 6.697 | 1.25 | 5.218 | 6.497 | 840.0 | 840.0 |
| champion | FusedConfidenceLUT n=1 | 0.291 | 0.402 | 1.38 | 1.573 | 2.211 | 1.41 | 1.519 | 2.154 | 96.4 | 96.4 |
| champion | FusedConfidenceLUT n=2 | 0.507 | 0.719 | 1.42 | 2.462 | 4.054 | 1.65 | 2.399 | 3.996 | 96.4 | 96.4 |
| champion | QuantisedConfidenceLUT n=1 | 1.659 | 2.954 | 1.78 | 7.199 | 11.729 | 1.63 | 7.253 | 11.588 | 568.8 | 568.8 |
| champion | QuantisedConfidenceLUT n=2 | 2.584 | 4.491 | 1.74 | 12.507 | 19.135 | 1.53 | 12.591 | 19.215 | 1142.4 | 1142.4 |
| champion | FusedQuantisedConfidenceLUT n=1 | 0.334 | 0.686 | 2.05 | 1.579 | 2.399 | 1.52 | 1.536 | 2.345 | 128.4 | 128.4 |
| champion | FusedQuantisedConfidenceLUT n=2 | 0.563 | 0.948 | 1.68 | 2.545 | 4.112 | 1.62 | 2.457 | 4.051 | 128.4 | 128.4 |
| champion | DeployedQuantisedConfidenceLUT | 0.337 | 0.472 | 1.40 | - | - | - | - | - | - | - |
| champion | DeployedQuantisedConfidenceLUT n=1 | 0.241 | 0.352 | 1.46 | - | - | - | - | - | - | - |
| large | ManifestoHardLUT | 2.627 | 4.485 | 1.71 | 60.614 | 76.037 | 1.25 | 62.080 | 77.599 | 5216.0 | 5216.0 |
| large | ManifestoSoftLUT | 4.996 | 8.095 | 1.62 | 66.996 | 78.803 | 1.18 | 65.893 | 80.503 | 5216.0 | 5216.0 |
| large | FusedManifestoHardLUT | 1.025 | 1.596 | 1.56 | 6.011 | 8.382 | 1.39 | 5.885 | 8.332 | 240.0 | 240.0 |
| large | FusedManifestoSoftLUT | 1.889 | 2.485 | 1.32 | 8.717 | 13.919 | 1.60 | 9.261* | 13.827 | 240.0 | 240.0 |
| large | ManifestoHardLUT weight=softsign | 2.607 | 4.486 | 1.72 | 62.724 | 77.202 | 1.23 | 65.086 | 78.769 | 5216.0 | 5216.0 |
| large | ManifestoSoftLUT weight=softsign | 5.381 | 8.248 | 1.53 | 63.980 | 79.717 | 1.25 | 66.495 | 81.418 | 5216.0 | 5216.0 |
| large | FusedManifestoHardLUT weight=softsign | 1.045 | 2.197 | 2.10 | 7.474 | 12.768 | 1.71 | 7.698 | 12.587 | 240.0 | 240.0 |
| large | FusedManifestoSoftLUT weight=softsign | 1.889 | 2.611 | 1.38 | 9.076 | 16.555 | 1.82 | 8.892 | 16.388 | 240.0 | 240.0 |
| large | ConfidenceLUT n=1 | 3.774 | 5.787 | 1.53 | 13.393 | 16.987 | 1.27 | 14.200* | 16.811 | 2768.0 | 2768.0 |
| large | ConfidenceLUT n=2 | 5.096 | 6.793 | 1.33 | 21.197 | 26.165 | 1.23 | 21.558 | 26.053 | 3216.0 | 3216.0 |
| large | FusedConfidenceLUT n=1 | 1.079 | 1.443 | 1.34 | 5.909 | 7.944 | 1.34 | 5.807 | 7.891 | 241.5 | 241.5 |
| large | FusedConfidenceLUT n=2 | 1.992 | 2.768 | 1.39 | 9.568 | 15.389 | 1.61 | 9.534 | 15.273 | 241.5 | 241.5 |
| large | QuantisedConfidenceLUT n=1 | 6.247 | 11.065 | 1.77 | 27.866 | 44.230 | 1.59 | 29.076 | 44.664 | 2000.0 | 2000.0 |
| large | QuantisedConfidenceLUT n=2 | 9.977 | 17.159 | 1.72 | 49.455 | 74.962 | 1.52 | 51.831 | 76.391 | 4269.4 | 4269.4 |
| large | FusedQuantisedConfidenceLUT n=1 | 1.266 | 2.452 | 1.94 | 5.794 | 8.779 | 1.52 | 5.673 | 8.710 | 369.5 | 369.5 |
| large | FusedQuantisedConfidenceLUT n=2 | 2.117 | 3.555 | 1.68 | 9.662 | 15.621 | 1.62 | 9.474 | 15.498 | 369.5 | 369.5 |
| large | DeployedQuantisedConfidenceLUT | 1.294 | 1.763 | 1.36 | - | - | - | - | - | - | - |
| large | DeployedQuantisedConfidenceLUT n=1 | 0.909 | 1.294 | 1.42 | - | - | - | - | - | - | - |
