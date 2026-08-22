# exp_n_0075 — Issue #108: per-cell hit-gap distribution + do rare cells underperform?

Diagnostic (not a training run). **Loads the trained exp_n_0052 checkpoint** (16k,
near-converged → routing ~stationary) and runs a **WINDOW=500** real steps at the
cosine-end LR (3e-5, ~frozen), measuring per LUT cell (196,608 = 6 blocks × 512 tables
× 64 rows):

1. **Hit-gap distribution.** #steps hit, hit-fraction, and gaps between consecutive
   hits (median/p90/p99/max). Overall: what fraction of cells are every-step / frequent
   / rare / very-rare / never.
2. **Do rare cells underperform?** Convergence proxy = per-cell **residual gradient PER
   TOKEN** = Σ(row grad-norm) / Σ(tokens routed). Normalizing by token count is
   essential: embedding_bag is mode=sum, so a frequent cell's raw grad is bigger just
   from summing more tokens — dividing by token count removes that confound. Cells are
   bucketed by hit-fraction and the per-token residual is compared across buckets; if
   the rare buckets have systematically larger per-token residual, rare cells are
   worse-fit and a per-cell-hit optimizer might be justified.

Cell indexing matches exp_n_0072/0073 (`d = zf[:,anchor_a]−zf[:,anchor_b]`, 6-bit
sign-pack → cell in [0,64); global = block*32768 + table*64 + idx). Wraps modules only;
does not modify `fast_multi_head_lut.py` / `compression_mhl.py`. Outputs
`cell_hit_gap_diag.json` + `cell_hit_gap_diag.png`.
