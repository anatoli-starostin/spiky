# exp_n_0072 — mixup "shadow-batch" gradient densification for the LUT model

**Idea.** At a FIXED real batch / real-token budget, add an equal-size *shadow* batch
each step built by convex-interpolating existing real points (mixup), and fold its
loss into `total = real_loss + λ·shadow_loss`. The interpolated points route to
intermediate LUT cells and near-boundary anchors, densifying gradient **without adding
real tokens**. Question: does this move val_bpb toward / below the tied-dense
**1.196646** at equal real-token budget (vs the e2e-LUT **1.2285517**)?

**Base recipe** = exp_n_0052 (CompressionMultiHeadLUT FFN; 6L / d384 / 6 heads /
seq512; LUT H8, d48, tph64, nap6, batched hard forward, learnable temps; tied dense;
AdamW lr3e-4 wd0.1 cosine 10% warmup; total batch 24576 tokens). Prototype length:
**6000 steps** (shorter than 0052's 16k — enough to see the trend).

**Interpolation level = token EMBEDDING (manifold mixup).** Chosen as the single
cleanest interception that flows through the *entire* existing forward (attention + all
6 blocks' LUTs), producing genuinely interpolated hidden states at every LUT. Per step:
pair the batch with a random permutation, draw a per-sequence `α ~ Beta(a,a)` (a=0.2,
mass near the endpoints so mixing stays mild and doesn't blunt the sharp routing), form
`e_mix = α·e + (1−α)·e[perm]`, run the transformer from `e_mix`, and use the mixup CE
identity `shadow_loss = mean_s[ α_s·CE(logits_mix[s], y[s]) + (1−α_s)·CE(logits_mix[s], y[perm][s]) ]`.
Both `α` (Beta param `mixup_alpha`) and `λ` (`mixup_lambda`) are config knobs.

**A/B at equal real-token budget.** `RUN_TAG=baseline` (real only) vs `RUN_TAG=mixup`
(real + shadow), SAME n_steps and SAME real batch each step → identical real-token
budget; the shadow batch only adds gradient (and ~2× compute/step). Each writes
`metrics_<tag>.csv` + `summary_<tag>.json`; when both exist a combined `compare_bpb.png`
is produced.

**Instrumentation** (every `routing_probe_every` steps, no_grad probe):
- `val_bpb` (primary).
- `off_dist_frac`: of distinct (block,table,cell) routed by the shadow batch, the
  fraction NOT routed by the real batch that step.
- `cov_real_frac` / `cov_union_frac`: distinct LUT cells receiving gradient, real-only
  vs real+shadow (union), out of the 6·512·64 = 196,608 total cells.

**Constraints.** Only touches this folder; does NOT modify `fast_multi_head_lut.py` /
`compression_mhl.py`. Smoke (`SMOKE_STEPS`) validated both paths + instrumentation.
