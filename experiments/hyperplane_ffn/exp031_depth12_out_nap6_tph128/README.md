# exp031 - trade nap for tph at out_proj (exp030, out nap7/tph64 -> nap6/tph128)

Exact clone of **exp030** (12 layers, single-stream + Linear head, anchor_pairs init, hard, low tph
qk nap4/tph32, v nap6/tph32) with ONE change: **out_proj nap7/tph64 -> nap6/tph128** - one fewer
address bit, double the tables per head.

## Key: out_proj TABLE budget is held fixed
Rows/head = 2^nap * tph: nap7/tph64 = 128*64 = 8192; nap6/tph128 = 64*128 = 8192. Identical.
So out_proj table params are UNCHANGED; only the affine hyperplane params shift (nap6/tph128 has
more table-sets to address -> slightly MORE hyperplane params: +123,200/layer). This isolates the
*shape* of the out_proj addressing (more shallow tables vs fewer deep ones) at fixed table budget.

## Params - 89,502,024 (formula)
- exp030 (out nap7/tph64): 88,023,624  (matches measured 88,023,624)
- **exp031 (out nap6/tph128): 89,502,024**  (+1,478,400 vs exp030, -187,049,436 vs exp024 276,551,460)

## Baselines
exp024 6L full-tph **1.2034** (276.5M); exp026 6L low-tph **1.3735** (56.6M);
exp030 12L low-tph **~1.3150** (88.0M). Same seed/data.
