# Frozen top-k hyperplane truncation sweep on trained exp024

Model: exp024 (anchor-init, single-stream + LayerNorm + Linear unembedder, HyperplaneMHL backbone).
For each backbone HyperplaneMHL row hyperplane_weight[t,i,:] (dim 384), KEEP ONLY the top-k
coordinates by |value| (real signed values retained), zero the rest. hyperplane_bias UNCHANGED,
tables (weights) UNCHANGED. All 18 sites (qk/v/out x6 layers). No retrain, frozen eval,
val bpb through the same harness as the exp024 same-protocol baseline (eval_steps=20).
All loads 0 missing / 0 unexpected, 18 sites each. bit-flip = fraction of address bits where
sign(<w_k,x>+b) != sign(<w_full,x>+b) on a batch of real val activations, averaged per site.

| k | val_bpb | Delta vs 1.2150 | flip qk | flip v | flip out |
|---|---|---|---|---|---|
| full (untouched) | 1.2150 | - | - | - | - |
| 2 | 2.6201 | +1.4051 | 0.1550 | 0.1436 | 0.1466 |
| 3 | 2.4543 | +1.2393 | 0.1517 | 0.1406 | 0.1424 |
| 4 | 2.3929 | +1.1779 | 0.1489 | 0.1385 | 0.1385 |
| 5 | 2.2859 | +1.0709 | 0.1461 | 0.1357 | 0.1355 |
| 10 | 2.0110 | +0.7960 | 0.1351 | 0.1262 | 0.1240 |
| 20 | 1.7079 | +0.4929 | 0.1174 | 0.1112 | 0.1072 |
| 30 | 1.5377 | +0.3227 | 0.1041 | 0.0991 | 0.0954 |

Reference: FastMHL argmax/argmin 2-coord swap (collapsed to x[a]-x[b]) = 2.669.
This top-2 keeps real weights + bias -> 2.6201, marginally better (as expected).

Trend (gap = bpb - 1.2150): 2->1.405, 5->1.071, 10->0.796, 20->0.493, 30->0.323.
The gap decays roughly exponentially, ~halving every ~15-18 kept coords, and the per-coord
improvement is itself slowing (k10->20 ~0.048/coord in ln-gap, k20->30 ~0.042/coord).
Extrapolating that decay, the model wouldn't get within ~1% of the 1.2150 baseline until
roughly k ~ 70-110 kept coords -- a fifth to a third of the 384-dim input. i.e. the trained
anchor-init hyperplanes are genuinely DENSE, not low-rank/sparse: useful signal is spread
across many tens of coordinates, so a hard top-k truncation recovers only very slowly.
