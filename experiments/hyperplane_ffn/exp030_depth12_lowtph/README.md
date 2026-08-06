# exp030 - depth-for-tph trade (exp026 at 12 layers)

Exact clone of **exp026** (single-stream + Linear head, anchor_pairs init, hard forward, low tph
qk32/v32/out64) with ONE change: **num_layers 6 -> 12** (double the depth). Trades depth against
table width: does a deeper-but-narrow stack beat exp026's shallow-narrow one, and how close does it
get to exp024's full-width 6-layer model?

## Only change vs exp026
`num_layers`: 6 -> 12. Everything else byte-identical (train.py unchanged - reads num_layers + tph
from config). Per-layer block seeds stay collision-free (qk 0..11, v 200..211, out 400..411).

## Params - 88,023,624 (formula, verified vs exp024/exp026)
Per-layer stack (low tph, incl. norms) = 5,238,086; fixed (tok_emb 12,582,912 + Linear head
12,582,912 + ln_final) = 25,166,592. total(nl) = 25,166,592 + 5,238,086*nl.
- exp026 (6L): 56,595,108  (matches measured 56,595,108)
- **exp030 (12L): 88,023,624**  (+31,428,516 vs exp026, -188,527,836 vs exp024 276,551,460)

## Baselines
exp024 full-width 6L **1.2034** (276.5M); exp026 low-tph 6L **1.3735** (56.6M). Same seed/data.
