# exp_n_0074 — table-param momentum (β1) vs the bigger-batch benefit

**Hypothesis.** The LUT-vs-dense gap at 1× batch is a gradient-**variance** problem:
each table row's gradient is averaged over only a small slice of the 24,576 tokens/step,
so it's much noisier than a dense weight's. 1.5× batch is known to close the gap toward
~1.19. Can raising **β1 on the LUT table params only** (longer temporal averaging of the
noisy per-cell gradient) recover some of that gain at **fixed 1× batch and fixed token
budget** — i.e. buy the bigger-batch effect without more tokens?

**Base recipe** = exp_n_0052 (6L/384/H8/d48/tph64/nap6 batched hard LUT, tied dense,
AdamW lr3e-4 wd0.1 cosine, betas (0.9,0.95)). Only this folder; wraps modules, no
shared-src edit.

**Setup.** The LUT table params (`ffn.lut_batched.weights`, the [tables,rows,dim] cells)
go in their OWN AdamW param group with betas `(B1_TABLE, 0.95)`, wd=0 (as recipe). All
other params (attention, head, LayerNorms, compress/decompress, and the LUT temps) stay
in the normal groups at (0.9,0.95). LR/schedule/wd matched exactly. Standard AdamW
per-group betas — no custom optimizer.

**Runs (equal 73.7M-token budget = 3000 steps @1×):**
- `b1_0.9` (baseline, β1=0.9 @1×), `b1_0.98`, `b1_0.99`, `b1_0.999` — all 1× batch, 3000 steps.
- `batch1.5x` — β1=0.9, batch ×1.5 (DBS 72, 36,864 tok/step), 2000 steps → same total tokens. The target line.

Effective averaging horizon ≈ 1/(1−β1): 0.9→10, 0.98→50, 0.99→100, 0.999→1000 steps.

**Report.** val_bpb vs **tokens** (so different batch sizes compare at equal budget) for
all runs; how much of the (baseline→1.5×-batch) bpb gain each β1 recovers; any high-β1
instability (grad-norm / divergence). Prototype length first. Smoke validated 1× and
1.5× paths.
