# exp_n_0056 — LUT-representability-constrained dense training

> **BASELINE RESULT (4096 / λ0.1 / annealed, 16k, 0.67 h): dense-own bpb 1.1963480, swap-in bpb 1.3578040.**
> The representability constraint at λ0.1 costs the dense model **nothing** — dense-own 1.19635 ≈ the vanilla
> dense baseline 1.196646 (even a hair better). But the swap-in (deployable LUT) is **+0.1612 vs dense** and
> **+0.1293 WORSE than end-to-end LUT exp_n_0052 (1.2285517)** — at λ0.1 the FFN is NOT pulled hard enough toward
> the LUT to be representable (per-block MSE b0 0.0024 → b5 0.0203, growing with depth; swap-in↔dense-own gap
> +0.161). This directly motivates the **high-λ** sweep (λ 0.5, 1.0): pull the FFN harder so swap-in approaches
> dense-own while dense-own stays near 1.19 (the win condition). This is the base run of the overnight sweep
> (batch size, λ ladder, anneal-vs-constant, asymmetric batches, FFN width) — see exp_n_0057…exp_n_0066 and
> `exp_n_0056_night_log.txt`. Reloadable dense+6-LUT checkpoint saved (checkpoint_final.pt).

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
