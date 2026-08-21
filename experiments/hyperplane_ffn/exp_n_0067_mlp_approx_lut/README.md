# exp_n_0067 — MLP-approximates-LUT ablation (the function-class gap, from the other side)

> **STATUS: code-before-run (smoke passed; queued after the exp_n_0056 λ2/4/8 extension).** Question: how well
> can a standard 2-layer widening GELU MLP approximate a TRAINED CompressionMHL, as a function of MLP hidden
> width? If even wide MLPs can't fit the LUT, the discrete hyperplane routing is a fundamentally different
> (spiky/non-smooth) function than a smooth GELU MLP represents.

## Targets (trained LUTs from exp_n_0052, val_bpb 1.2285517)
Load exp_n_0052's checkpoint (the 6 trained CompressionMHL blocks; loads exact, missing=0). Study:
- **block 0** (easiest — lowest in-model MSE) and **block 5** (hardest — highest), each the whole 384→384 CMHL.
- a **single FastMHL head** of block 0 (its 48→48 mapping), isolated via the batched FastMHL's head-0 slice
  (block-diagonal routing → head 0's output depends only on head 0's input).

## Method
Stream real activations through the frozen exp_n_0052 model; inputs = each block's FFN-input `x` (and the head's
compressed 48-slice), targets = the frozen LUT's outputs on those inputs (`n_collect` token-vectors, 20% held
out for val). Fit 2-layer GELU MLPs `Linear(in, w·dim) → GELU → Linear(w·dim, out)` for **w ∈ {4, 8, 16, 32, 64}**
(dim = 384 whole block, 48 head), MSE to convergence (Adam). Per width record train/val MSE, R² (=1−MSE/var),
MLP param count, and mark where MLP params cross the LUT's own count (~1.87M/block, ~197k/head). Plot val MSE vs
width and vs params for block0, block5, head.

## Headline question
Does the MLP approximation MSE **→ 0** as width grows (a smooth MLP CAN represent the discrete LUT — the LUT is
just a peculiar-but-smooth function), or does it **floor** at a non-zero value regardless of width (the LUT's
hard hyperplane routing is genuinely non-smooth / outside the GELU-MLP function class)? The latter would explain
why LUT slots underperform dense: they compute a fundamentally different kind of function.

## Smoke
Checkpoint loads exact (missing=0); block0/block5/head0 activations collected; MLPs fit at each width with
val MSE + R² reported; LUT param counts (1,868,546 block / 196,608 head) computed. No shared-module edits.
