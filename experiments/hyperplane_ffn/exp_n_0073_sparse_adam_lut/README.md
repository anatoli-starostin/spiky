# exp_n_0073 — sparse/lazy Adam for the LUT tables (does dense AdamW mishandle sparse table grads?)

**Hypothesis.** The LUT table cells receive *embedding_bag-style sparse* gradients (a
cell only gets gradient on steps a token routes to it). Dense AdamW decays a cell's
moments (β1,β2) every step regardless, and (if weight-decayed) shrinks it every step —
so tail cells could suffer "stale momentum" / spurious shrink. Does a lazy/sparse Adam
that advances a cell's moments **only on steps it's actually hit** help val_bpb?

**Base recipe** = exp_n_0052 (6L/384/H8/d48/tph64/nap6 batched hard LUT, tied dense,
AdamW lr3e-4 wd0.1 cosine 10% warmup, betas (0.9,0.95), total batch 24576 tokens).
Prototype length 3000 steps. Only this folder; wraps FastMultiHeadLut /
CompressionMultiHeadLUT, does not modify shared src.

## Part A — diagnostic (`DIAG=1`, on the baseline run)
Each step, compute the set of LUT cells that receive a nonzero gradient (= are selected
by the real batch, via the hard routing indices) and track per-cell the **gap in
optimizer steps between successive hits**. Reports: gap distribution
(mean/median/p90/p99/max) across cells, frequent-vs-tail deciles, distinct-cells-hit
per step vs total (196,608), the implied Adam moment decay `β^gap` (β1=0.9; β2 at both
the recipe's 0.95 and the task's 0.999), and the hypothetical wd shrink between hits.
Note: exp_n_0052 already places LUT tables in the **wd=0** group, so the actual recipe
applies **no** weight decay to table cells (shrink = 0); the hypothetical wd=0.1 figure
is reported for completeness.

## Part B — A/B (equal token budget)
`OPT_MODE=baseline`: all params incl. LUT tables under dense AdamW (exactly 0052).
`OPT_MODE=sparse`: LUT table params under **LazyAdamTables** — a lazy/sparse Adam whose
per-row moments update ONLY on steps that row is selected (frozen between hits, per-row
bias-correction `1-β^cnt`); all other params under the normal AdamW. LR/schedule/betas
matched. (SparseAdam proper needs sparse grads, which the LUT's index_add backward does
not emit, so the equivalent hit-gated update is implemented directly — no shared-src edit.)

Reports val_bpb trajectories for both and whether sparse-Adam moves val_bpb
toward/below the tied-dense **1.196646** vs the dense-AdamW baseline at equal budget.
Smoke (`SMOKE_STEPS`) validated both modes + the diagnostic.
