# Matrix-menu LUT research: standing plan and gates

Standing mandate (Anatoli, tasks 381fc666 / 9e02988b / 892efffc / 285cb426): lowest FFN-distillation FVU with LUT
modules under CHEAP inference (storage AND MACs vs FFN). One GPU job at a time; commit locally, never push.

## Gates for long (16K) runs
1. **No 16K run for ANY menu config without per-cell bias** (`lut_menu_per_cell_bias: true`). (285cb426)
   The M128 / tph16 result (mean FVU 0.1713, beats Light) is explicitly **not** promoted to 16K.
2. At most ONE menu config + its matched Light control (same horizon, identically rescaled schedule) at a time.
   (892efffc)
3. Promotion needs, measured at 4K (with the fixes in force):
   (A) beats the matched Light control at equal/cheaper inference, OR ties within noise while materially cheaper
       (>= 2x on one axis, not worse on the other), OR behind on the endpoint but passes (B);
   (B) last-1K relative FVU improvement mean >= +1.0% AND >= 4 of 6 layers >= +0.5% (calibrated: every Light 4K run
       <= +0.94% mean; the 16K cosine plateau is ~0%).
   State config, the (A)/(B) numbers and the reason in Slack BEFORE launching. Anatoli decides the 16K launch.
4. Long runs: warmup 10% of the horizon, cosine to 0.1x at the final step (harness default via --steps); report
   learning curves (every 1K) for menu AND Light, and train vs val MSE (16K ~ 1.85 passes). 32K only if the
   promoted 16K run is still falling at its end (same (B) rule on its last quarter).

## Noise floor (measured)
menu H8 tph16 M64, seed 2 vs seed 1 (torch seed + lut_base_seed): per-layer |d| <= 0.0026, mean +0.0005.

## Cost accounting
Hard-read inference MACs/FFN = 0.25 (compress+decompress, H8 d48) + tph*H*atom/FFN (+ bias bag, negligible),
atom = d_in*d_out (dense) or r*(d_in+d_out) (menu_rank). Independent of M. Storage = menu (H*M*atom floats)
+ indices (6-8 bits/cell) (+ per-cell bias: n_tables*256*d_out floats).
