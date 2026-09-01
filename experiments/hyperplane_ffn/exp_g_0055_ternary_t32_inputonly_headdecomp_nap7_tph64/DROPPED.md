# exp_g_0055 — launched, aborted at step 3,200, superseded

Was exp_g_0053 with one field changed: the target-density hinge asking for 32 of each
hyperplane's 384 components non-zero instead of 192 (`lut_target_nonzero_frac`
0.5 -> 0.0833333), penalty left ON at lambda=100. Everything else identical, so it
would have been the high-sparsity rung above exp_g_0053.

**Dropped because the question it was asked did not need a retrain.** The point was to
find out whether high sparsity finally makes the sparse inference kernel beat the dense
GEMM. Density is density — the kernel's cost depends on how many non-zeros there are,
not on how they were produced — so the same question is answered instantly by taking
exp_g_0053's trained weights and hard-truncating each hyperplane to its top-k
coordinates by pre-ternarization magnitude. That sweep (k = 192, 96, 64, 48, 32, 16, 8)
cost minutes instead of an hour and covered seven densities instead of one. It found no
crossover at any density: the sparse kernel only reaches parity at k=8 (2.1% non-zero),
and once the feature-major transpose it requires is counted it is still 1.14x the dense
bf16 stage — at a point where quality has collapsed by +0.348 bpb.

**The one result worth keeping from the 3,200 steps it did run** is a third confirmation
of the target-undershoot bias. The hinge compares against the SOFT surrogate
mean|tanh(w/2T)|, not the hard non-zero count, and the realized hard density lands
consistently below the request:

| run | asked | realized | under by |
|---|---|---|---|
| exp_g_0042 | 64 | 47.9 | 25% |
| exp_g_0053 | 192 | 173.9 | 9.4% |
| **exp_g_0055** | **32** | **26.7 by step 3,200** | **17%** |

`metrics.csv` and `ternary_drift.csv` are kept as the record of that. There is no
`summary.json`, no `loss.png` and no checkpoint — the run was killed before it wrote
any of them.

**The number 0055 is left as a gap.** Its question was answered by the truncation sweep,
which is why no retrain at any other sparsity target is planned.
