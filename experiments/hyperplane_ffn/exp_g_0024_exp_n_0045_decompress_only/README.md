# exp_g_0024 — decompress-only FFN LUT (STOPPED EARLY)

`inner_in_dim=-1, inner_out_dim=48, nap=7, tph=2048, n_heads=1` — 91,742,220 params,
device_bs 12 / grad_accum 4 / eval_steps 40 (tokens/step 24,576, val tokens 245,760).

## STOPPED EARLY at step 11,000 / 16,000 (2026-08-26)

Terminated by decision to free the RTX 5090 for exp_g_0025, the clean one-variable
compress ablation. No `summary.json`; `metrics.csv` holds the partial curve
(55 evals, steps 200..11,000) and is kept.

Why it was not worth the remaining 5,000 steps: exp_g_0024 was **above (worse than)
exp_n_0043 at all 54 aligned eval steps**, and the gap was flat, not closing.

```
latest aligned step 10,800   exp_g_0024 1.254151   exp_n_0043 1.224953   +0.029198
last 8 aligned evals         +0.028088 -> +0.029198   (drift +0.001110, wobble +/-0.0009)
all 54 aligned points        mean |delta| 0.029891
                             min +0.007492 @ step 200   max +0.048655 @ step 2,200
```

Holding that ~+0.029 offset to 16,000 would land near 1.232 — worse than
exp_n_0043's 1.2029199 and ~0.034 off the exp_n_0045 anchor of 1.1977670.

## Caveat on reading it as evidence

exp_g_0024 is **not** a clean ablation. On top of dropping the compress projection
it also changed `n_heads` 8 → 1 and `tph` 256 → 2048, and it is the smaller model
(91,742,220 vs 93,403,488, −1.66M), so capacity and head/table geometry are
confounded with the structural change. Removing exactly that confound is what
exp_g_0025 was built to do.
