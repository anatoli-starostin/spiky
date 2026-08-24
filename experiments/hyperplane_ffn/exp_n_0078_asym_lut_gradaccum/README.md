# exp_n_0078 — asymmetric per-step grad accumulation: is the LUT gradient-starved?

**Hypothesis under test.** The LUT-vs-dense val_bpb gap is (partly) a *weak-LUT-gradient*
problem: at 1× batch the LUT table/routing params get too few / too noisy gradients per
step, and the observed 1.5×-batch win (exp_n_0046 → dense parity 1.196862) is really the
LUT finally getting a denser gradient — not a whole-model bigger-batch effect.

**Clean isolation.** Each step uses **72 distinct sequences** from a single loader, split
into **A = 48** and **B = 24**:
- **Pass A** (full model): every param's `.grad` = gradient of the mean-over-A loss.
- **Pass B** (non-LUT params frozen via `requires_grad=False`): only the **LUT params**
  accumulate B's gradient; non-LUT `.grad` is left exactly as A produced it.
- Assemble the token-weighted mean over the full 72 seqs for the LUT only:
  `LUT.grad = (2/3)·gA + (1/3)·gB`, `nonLUT.grad = gA`. One AdamW step.

So **non-LUT params see 48 seq/step (identical to the 1× baseline)** while the **LUT params
see the mean-over-72 gradient** (denser, lower-variance) — with **no whole-model
bigger-batch confound**. LUT params := everything under a `.ffn.` CompressionMultiHeadLUT:
compression proj, table weights (`lut_batched.weights`), learnable temps, decompression
proj (anchors/powers/offset are buffers, excluded). No shared-src edits — routing is pure
`.grad` manipulation + transient `requires_grad` toggling on the stock model.

**Recipe** = exp_n_0052 (depth 6 / 384 / 6-head attn; LUT H8/d48/tph64/nap6, batched hard,
learnable temps; tied dense; lr 3e-4 cosine, warmup 10%, wd 0.1, betas (0.9,0.95),
**16000 steps**). Same #optimizer-steps as the 1× baseline, so non-LUT token exposure is
identical to it.

**Gradient-routing verified** (`GRADCHECK=1`): over 3 steps, non-LUT `max|grad−gA| = 0`
(bitwise), LUT `max|grad−((2/3)gA+(1/3)gB)| = 7.5e-9` (fp32), and A vs B gradients differ
(spread ~2e-2) confirming 72 distinct seqs, no A/B overlap.

**References / win condition.** 1× end-to-end-LUT (exp_n_0052) val_bpb ≈ **1.2286**;
1.5×-batch dense parity (exp_n_0046) = **1.196862**. If this asymmetric run moves toward
**~1.19**, the LUT was gradient-starved and denser LUT-only gradient recovers most of the
1.5×-batch gain. If it stays near 1.2286, the gap is *not* a LUT-gradient-density problem
(consistent with exp_n_0072–0077, which already ruled out the optimizer/density angle).

Outputs: `metrics.csv`, `summary.json`, `loss.png`, `checkpoint.pt`.
