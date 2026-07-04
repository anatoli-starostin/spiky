---
name: lut-prenorm-is-magnitude-calibration
description: "Pre-LUT norm (ln_pre/ln_post) does ONLY per-token magnitude calibration; learnable affine is inert; any per-token scale (std/RMS/MAD/mean-abs) is equivalent. mean-abs is cheapest drop-in. exp470-475, 2026-05-21."
metadata: 
  node_type: memory
  type: project
  originSessionId: fa43f8ba-4262-4d5d-ab44-b1dfbc584286
---

# Pre-LUT norm = pure magnitude calibration; affine inert (2026-05-21, exp470–475)

Full 2×2 (L1/L2 × center/no-center) + affine ablations on exp453 (LION 0.9/0.95, bs=16, 8K, 1.4967). Only ln_pre/ln_post (the norms feeding the LUTs) were changed; q_norm/k_norm/ln_final stay full LayerNorm.

## The matrix — all ≈ exp453 within noise
| | center | no center |
|---|---|---|
| L2 | exp472 std (LN no-affine) → **= exp453 exactly** | exp474 RMS → ≈ exp453 |
| L1 | exp473 MAD `(x-μ)/mean(|x-μ|)` → ≈ exp453 | **exp475 mean-abs `x/(mean(|x|)+eps)` → 1.4962** |

Ablations proving the divisor (not the affine) is load-bearing:
- exp470 (center + **scalar** γ, no divisor): **+0.05**
- exp471 (center + **per-channel** γ, no divisor): **+0.07** (per-channel γ inert — exp471≈exp470)

## Conclusions
1. **The pre-LUT norm does exactly one thing: per-token magnitude calibration** — keep `|d| = |x_a−x_b| ~ soft temperature T` so the soft-backward softmax is sharp (not uniform/mushy, not saturated). Without the divisor, raw tok_emb is tiny (`|d|~0.05 ≪ T=0.5`) → diffuse gradient → +0.05–0.07 deficit. It's a TRAINING-dynamics effect (gradient sharpness), not a selection effect — the divisor is sign-preserving so it never changes which row is picked.
2. **The learnable affine (γ, β) is INERT** — exp472 (LayerNorm with γ,β frozen at init=1,0) = exp453 exactly. Reason: LUT weights don't participate in selection (selection = sign of input-difference over fixed anchors), and their output magnitude is normed away downstream → per-channel γ/β buy nothing. Per-channel γ specifically changes selection in theory but the model gains ~0 from it.
3. **Any per-token positive scale is equivalent**: std, RMS (=√(σ²+μ²), ~more aggressive by √(1+(μ/σ)²)), MAD (≈0.8σ), mean-abs. For the LUT, μ cancels in differences (centering is a no-op) and the scale constant is absorbed by the learnable T.
4. **mean-abs `x/(mean(|x|)+eps)` is the cheapest correct choice** — no params, no centering, no square, no sqrt → most matmul-free-friendly. **Adopt exp475 as the new baseline recipe** (= exp453 within noise, simpler/cheaper norm; exp475=1.4962 vs exp453=1.4967 is inside ±0.003 noise → a TIE/cost-win, NOT a bpb SOTA).

## WD corollary (not run)
Weight decay on LUT weights is redundant here: LUT output is normed away downstream and selection is weight-independent → WD acts only as effective-LR (van-Laarhoven WD+norm result), which `lut_lr` already tunes. Skip LUT WD. (WD DOES make sense on the unembedder — ln_final normalizes its input so weight scale = logit temperature.)

See [[lut-optimizer-sweep]], [[lut-scatter-specialization-sota]].
