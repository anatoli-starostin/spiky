# exp_g_0203 — 48K learnable-tau top-2 blend — analysis

**Config:** 48K fresh fork of exp_g_0195 (Light + margin + top-2 blend, learnable per-layer
tau init 0.5, no z_norm, NAP8/tph128/H4, seed 1, optimized code). ONLY change vs 0195:
n_steps 16000→48000 (cosine LR + warmup auto-stretched to 4800). 67,351,686 params.
Corrected eval bs48×100 skip-12. Verified by reload: **1.132300 == summary, delta 0.00**,
strict load (0 missing / 0 unexpected). Trained 3.6h, EXIT 0.

## Result (σ = 0.00335)

| vs | value | Δ | σ |
|---|---|---|---|
| **0203 final** | **1.132300** | — | — |
| 0203 best | 1.131621 | −0.000679 | −0.20σ |
| **vanilla@48K corrected (0177)** | 1.115420 | **+0.016880** | **+5.04σ** |
| 0195@16K (same arch, 16K) | 1.160637 | −0.028337 | −8.46σ |

**Headline (and it reverses the 16K read): the learnable-tau blend does NOT beat dense at
48K — it is +5.04σ *behind* vanilla@48K.** The arch itself keeps improving a lot from 16K→48K
(−8.46σ), but dense improves *faster*. The 16K "at/under parity" (0195 was −0.35σ vs vanilla
seed2) was **a crossover artifact**, not a durable win.

## The crossover — matched-step vs vanilla@48K (0177)

| phase | steps | 0203 vs vanilla | reading |
|---|---|---|---|
| warmup, soft blend | 500–4000 | +18.6σ → +1.2σ | blend mixes the neighbour heavily while tables untrained — starts far behind |
| **crossover + brief lead** | 4500–7500 | −0.9σ → −0.4σ | draws even at ~4500, best lead −0.86σ at 5500 |
| parity band | 8000–17000 | ±0.5σ wobble | tracks dense within noise |
| **dense pulls away** | 17000–48000 | +0.4σ → **+5.0σ** | monotone: +1σ@22k, +2σ@29k, +3σ@34k, +4σ@40k, +5σ@48k |

The blend matches dense through the mid-regime and briefly leads near ~5k, but from ~17k on
dense keeps descending faster while the blend flattens. By 48k the blend is ~1.3% relative
behind. **Whatever the top-2 blend buys early, it is a fixed-size head start that dense erases
and overtakes given enough steps** — the routing gate does not add long-horizon capacity here.

## Tau trajectory: 0195@16K → 0203@48K

| layer | 0195@16K | 0203@48K | Δ_m | 0203/Δ_m | nearest 2^k |
|---|---|---|---|---|---|
| L0 | 0.1583 | 0.0687 | 0.0331 | 2.08× | 2^-4 (1.10×) |
| L1 | 0.2924 | 0.2030 | 0.0724 | 2.80× | 2^-2 (0.81×) |
| L2 | 0.3337 | 0.2312 | 0.0780 | 2.96× | 2^-2 (0.92×) |
| L3 | 0.3804 | 0.2527 | 0.0816 | 3.10× | 2^-2 (1.01×) |
| L4 | 0.4646 | 0.3274 | 0.0919 | 3.56× | 2^-2 (1.31×) |
| L5 | 0.6689 | 0.5111 | 0.1079 | 4.74× | 2^-1 (1.02×) |

tau **descended further with more steps** — from ~4–6× Δ_m at 16K to ~2–4.7× Δ_m at 48K —
so the longer run does migrate tau toward the measured margin gap, but it never reaches it
(still 2–5× above). Per-layer order still matches Δ_m (sharpest where margins smallest).
Incidentally the 48K taus land **closer to powers of two** than 0195's (L1–L3, L5 within
0.81–1.02× of the nearest 2^k) — a minor plus for shift-only-tau, though the quant study
already showed pow2 tau is cheap regardless.

## Takeaways

1. **The top-2 blend is not a 48K win.** At the horizon that matters it is +5σ behind dense.
   The 16K parity did not survive extension — it was the two trajectories crossing.
2. **The blend's benefit is front-loaded and bounded.** It helps most while tables are
   under-trained (a smoother read-out during warmup/early training) and stops helping once
   dense catches up. This is consistent with 0195's own trajectory (a lead that was still
   "widening slightly" at 16K but on a decelerating curve).
3. **Direction for the thread:** a fixed soft-blend does not close the gap to dense at length.
   If the blend is to matter long-horizon it needs to *add capacity that compounds* (e.g. more
   candidates n>2, or a learned per-token routing that keeps paying off), not just soften the
   argmax. The quant study's deployment verdict (int16/int8 tables free; multiply-lean, not
   multiply-free) stands independently of this — it is about the read-out cost, not the gap.

Refs (corrected protocol): vanilla@48K 1.115420 (0177); 0195@16K 1.160637; 0193@16K control
1.172852; σ = 0.00335 (vanilla@16K seed spread).
