# exp_n_0071 — DEPTH sweep of MLP-approximates-LUT (does composition break the floor?)

> **HEAD-0 RESULT (head0_depth_sweep.py): depth PARTLY breaks the floor but hits a new one ~0.79 — it does NOT
> reach 1.0.** Fitting deep pre-norm residual MLPs (depth {2,3,4,6,8} × width {16,32,64}, cosine LR, 10k steps)
> to block-0 head 0 (48→48): depth 2 tops out at R² **0.713** (≈ the exp_n_0067 2-layer floor 0.689), but
> **depth 3 jumps to R² ~0.79** (best **0.7884** at depth3×width32) — so composition/depth IS a real missing
> ingredient (+0.08 R² from one extra hidden layer). BUT beyond depth 3 it **plateaus then degrades** (depth 8 →
> 0.74–0.77, overfitting at up to 57M params). So a deep MLP captures ~79% of the head's output variance vs ~69%
> for a shallow one, but the last ~21% is still irreducible: the hard hyperplane-routing head is **partly
> compositional (depth helps) yet still genuinely non-smooth (floors at ~0.79, not 1.0)**. Refines the exp_n_0067
> conclusion. (The general all-block depth sweep in depth_approx.py remains held.)

> **STATUS: code-before-run (smoke passed; full sweep running).** exp_n_0067 showed a **2-layer** GELU MLP's
> approximation of the trained LUT FLOORS even at 64× width (block0 R²≈0.93, block5 ≈0.88, single FastMHL head
> ≈0.69). But the LUT's hard routing is piecewise-CONSTANT (hyperplane-bounded cells, discontinuous) and a
> CONJUNCTION of 6 sign tests — compositional/logical, which shallow nets need exponential width for but DEPTH
> represents efficiently. **Does depth break through the floor width couldn't?**

## Setup (same targets as exp_n_0067)
Targets = exp_n_0052's trained LUTs (checkpoint loads exact): **block 0** (easiest, 384→384), **block 5**
(hardest, 384→384), and a **single FastMHL head** of block 0 (48→48, isolated via the batched head-0 slice).
Real activations streamed as inputs, frozen LUT outputs as targets, 20% val holdout.

## Approximators: deep pre-norm residual MLPs
`depth` = number of Linear layers on the main path. depth 2 = the exp_n_0067 baseline
(`Linear→GELU→Linear`). depth ≥ 3 = `Linear(in,H)` → **(depth−2) pre-norm residual blocks** `x + Linear(GELU(LN(x)))`
→ `LN → Linear(H,out)` — residual + LayerNorm so deep nets optimize cleanly. Sweep **depth ∈ {2, 3, 4, 6, 8}** at
fixed width **H = 8×dim**, plus one **corner** (depth 6 × width 16×). Trained with AdamW + cosine LR (5% warmup),
grad-clip, `fit_steps` to convergence.

## The question
Does depth drive the approximation error **→ 0** (the LUT IS representable by a deep-enough MLP — it just needed
composition/depth, not width), or does it **STILL floor** (the discrete routing is genuinely outside the smooth-MLP
class regardless of depth)? Especially the **single FastMHL head** (purest hard routing): if depth lifts its R²
from 0.69 toward ~1.0, composition was the missing ingredient; if it stays stuck, the non-smoothness is
fundamental. Headline curve: R² (and val MSE) vs depth for block0, block5, head, compared to the 2-layer numbers.

## Smoke
Checkpoint loads exact (missing=0); block0/block5/head0 collected; a depth-6 residual net trains and optimizes
(val MSE + R² computed at each depth). No shared-module edits.
