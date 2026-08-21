# exp_n_0067 — MLP-approximates-LUT ablation (the function-class gap, from the other side)

> **RESULT: a smooth GELU MLP CANNOT fully fit the trained LUT — approximation MSE FLOORS regardless of width.**
> Fitting 2-layer MLPs (w = 4…64×) to exp_n_0052's trained LUTs: **block 0** plateaus at R²≈0.93 (val MSE ~1.5e-4),
> **block 5** at R²≈0.88, and the **single FastMHL head** at only **R²≈0.69** — and going from 4× to 64× width
> (up to ~10× the LUT's own params) does NOT lower the error; for the blocks it even *degrades* (overfits). The
> raw FastMHL head (hard sign-test routing, no smoothing compress/decompress Linears) is the LEAST
> MLP-representable. So the LUT's discrete hyperplane routing is a genuinely non-smooth function **outside the
> smooth GELU-MLP function class** — the unifying explanation for why (from the other side) a dense FFN can't be
> made cheaply LUT-representable either (exp_n_0055/0056). The two function classes are fundamentally different.
> See `mlp_approx_curves.png`.

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
