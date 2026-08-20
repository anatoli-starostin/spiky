# exp_n_0056 — LUT-representability-constrained dense training

> **STATUS: code-before-run (smoke passed; queued behind exp_n_0055 on the H100).** Train a vanilla dense
> transformer from scratch on the LM objective, but constrain every FFN to stay reproducible by a LUT of fixed
> (exp_n_0052) capacity. The dense FFN gives a smooth, fully-differentiable optimization path while being
> regularized to stay LUT-friendly; the deployable artifact is the swap-in model (FFNs → co-trained LUTs).

## Co-training (per block b, per step)
`x_b = ln2(x)` fed to the FFN; `ffn_b(x_b)` = dense FFN output:
```
loss_lut_b = MSE(lut_b(x_b), ffn_b(x_b).detach())   # trains the LUT only (FFN detached)
loss_reg_b = MSE(ffn_b(x_b), lut_b(x_b).detach())   # pulls FFN/upstream toward the LUT (LUT detached)
total      = CE_LM + Σ_b loss_lut_b + λ · Σ_b loss_reg_b
```
The two detaches keep the roles clean — verified in the smoke: `loss_lut` grads reach the LUT only (FFN/x get
none); `loss_reg` grads reach the FFN + upstream only (LUT gets none). `λ` ramps linearly 0 → `lambda_reg_target`
(0.1) over `lambda_ramp_frac` of training (start ~0 so the FFN trains freely early, tighten later).

## Subsampled LUT losses (keeps it fast)
The dense LM forward/backward runs at the full `device_bs 48 × seq 512` (24,576 token-vectors) for CE. But both
LUT-side losses are computed on a random **`lut_batch_tokens` = 4096**-token subsample of each step's batch (same
indices for both LUT losses per block) — so the heavy STE-surrogate compute stays ~6× lighter and the regularizer
is a stochastic estimate on those positions.

## Config
Dense side = the standard rung (16k steps, device_bs 48, grad_accum 1, warmup 1600, seed 1, clean val 245,760,
tied unembedder), the architecture that reaches ~1.19. LUT side = **exp_n_0052's CompressionMHL hyperparameters
1:1** (H8/d48/tph64/nap6, joint=false, batched_multi_head_input=true, hard-forward/soft-backward,
learnable_temps=true, noise 1e-3, seed 1000+block). Dense deployable 23,209,728 params; 6 co-trained LUTs
11,211,276. No shared-module edits.

## Metrics / eval
Each eval logs BOTH (a) **dense-own val_bpb** (real FFNs) and (b) **swap-in val_bpb** (all 6 FFNs replaced by
their co-trained LUTs — the deployable number), plus per-block FFN↔LUT MSE. Compared head-to-head vs dense
1.196646 and end-to-end LUT exp_n_0052 1.2285517. Plots: dense-own + swap-in bpb vs step, and per-block MSE vs
step.

## Smoke (30 steps, full model size)
Grad routing verified (both detaches correct); λ ramps 0→0.1; both bpb evals compute (dense-own 2.560, swap-in
2.588 at step 30 — swap-in tracks dense-own within +0.029 even untrained); per-block MSE tracked; subsample
routing works.
