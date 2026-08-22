# exp_n_0076 — cell-level (direction-preserving) momentum for the LUT tables

**Distinction.** exp_n_0074 raised β1 on the table params under *element-wise* Adam
(per-scalar m,v — each of a cell's 48 dims rescaled independently, which distorts the
cell's gradient *direction*) and found higher β1 monotonically hurts. This experiment
implements a **cell-level** optimizer that treats each table cell (one [48] row) as a unit:

- first moment `m`: per-cell **vector** [.,48], EMA of the gradient vector (β1);
- second moment `v`: per-cell **scalar** [.], EMA of the cell's gradient mean-square (β2);
- update `= lr · m̂ / (√v̂ + eps)`, the **same scalar** √v̂ scaling the whole 48-dim
  cell → the gradient **direction is preserved** (per-cell LAMB/LARS-style normalization).

Only the LUT table params (`ffn.lut_batched.weights`) use this; all other params stay on
AdamW (0.9,0.95). Table wd=0 (recipe). It's dense (all cells updated every step — matches
exp_n_0075's finding that ~99.8% of cells are hit every step). No shared-src edits.

**Base recipe** = exp_n_0052. **Runs** at equal 73.7M-token budget (3000 steps @1× batch):
- `elementwise_b0.9` — baseline (table Adam β1=0.9, β2=0.95) = the exp_n_0074 reference.
- `cell_b0.9`, `cell_b0.95`, `cell_b0.98` — cell-level momentum at rising β1.

**Question.** Does direction-preserving cell-level momentum beat element-wise Adam at
fixed tokens, and does it tolerate higher β1 better than the element-wise sweep (which
degraded monotonically)? Reports final val_bpb, val_bpb-vs-tokens, max grad-norm
(instability), and a verdict. Smoke validated both modes.
