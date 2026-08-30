# Ternary target-192 hinge — full 16k (exp_n_0134) — results (task 3183de07)

Full-anneal (n_steps = lr_schedule_steps = 16000) run of gpustar's **exp_g_0044** recipe: ternary
hyperplane routing + per-head output decompress (inner_out 48), target-density hinge at 192 non-zeros,
λ=100. Everything else identical to exp_g_0044 (normalize_weights, T=max_entropy 0.392065, divisor
sqrt_expected_nonzero 16, trainable_bias, random init, n_heads 4, nap 8, tph 128, bs12/ga4, eff batch
24,576, same seed/data). 76,373,004 params. exp_g_0044 itself stopped this schedule at 4,000 steps
(1.34841, early-trajectory); this run completes the anneal.

## Result — final val_bpb 1.18943
Converged (15600→1.1900, 15800→1.1895, 16000→**1.18943**).

| baseline | bpb | Δ (ternary − baseline) |
|---|---|---|
| exp_n_0121 — full-anneal LUT anchor (nap8/tph128 CompressionMHL) | 1.19146 | **−0.00203** |
| vanilla 4× MLP FFN (exp073) | 1.19665 | **−0.00722** |

**The ternary sparse routing BEATS the dense-cell LUT anchor at full anneal**, not just matches it — the
4k reading (+0.0020 vs the no-penalty ternary exp_g_0037) does not carry to 16k; over the full schedule
the target-192 hinge lands −0.0020 *below* exp_n_0121. Sparsity here is better than free.

## Final routing state (ternary_drift.csv, step 16000)
- frac_zero **0.5516**, **172.17 non-zeros/hyperplane** (of 384) — the hinge held near the 192 target region.
- surrogate 0.4954 (< 0.5 target) → **hinge released, penalty exactly 0** (it did its job early and let go).
- bias learned: b absmean 0.0154 (from 0 at init) — the off-centre dead-zone threshold is being used.
- T drifted 0.30–0.59 (task gradient only, penalty has T detached), score/T 0.4583.
- churn down to 323K routing changes/eval (from 2.57M early) — routing stabilized.

## Takeaway
A ternary-routed FFN with a target-density hinge (≈55% of routing components zero, ~172 non-zeros per
hyperplane) reaches **1.18943 at full 16k — below both the dense-cell LUT anchor (exp_n_0121) and the
vanilla 4× MLP FFN** — while being multiply-free at inference (adds/subtracts + a compare against −b) and
2-bit-per-entry at deployment. The full-anneal verdict the exp_g_0044 board flagged as unestablished:
the sparsity is not a cost here, it is a small win.
