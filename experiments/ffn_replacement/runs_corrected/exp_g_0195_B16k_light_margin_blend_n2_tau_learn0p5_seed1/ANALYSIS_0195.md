# exp_g_0195 — learnable-tau top-2 blend (16K) — analysis

**Config:** Light + margin score + top-2 blend (read_top_n=2), **learnable per-layer tau, init 0.5 flat**, no z_norm, NAP8/tph128/H4, seed 1. Optimized code path (topk→min + default-on submodule torch.compile). 67,351,686 params. Corrected eval (bs48×100 skip-12). Verified by checkpoint reload: 1.160637 == summary, delta 0.00e+00, strict load (0 missing/unexpected).

## Result (σ = 0.00335, the vanilla@16K seed spread)

| vs | value | Δ | σ |
|---|---|---|---|
| **0195 final** | **1.160637** | — | — |
| 0193 (n=1 control, margin no-znorm) | 1.172852 | **−0.01222** | **−3.65σ** |
| 0194 (frozen-tau blend @ Δ_m) | 1.182259 | −0.02162 | −6.45σ |
| vanilla@16K seed1 (0135) | 1.165147 | −0.00451 | −1.35σ |
| vanilla@16K seed2 (0176) | 1.161798 | −0.00116 | −0.35σ |

**Headline:** the learnable-tau top-2 blend is **−3.65σ better than the n=1 control (0193)** and **below both vanilla@16K seeds** — i.e. at/under dense parity at 16K, and the best Light arm to date. It also crushes the frozen-tau blend (0194) by −6.45σ: **tuning the blend temperature, not fixing it at the measured margin gap, is what makes the blend help.**

## Trajectory vs 0193 (matched step, σ)
0195 starts WORSE (softer blend mixes the neighbour heavily while tables are untrained): +3.7σ at step 500, crossing 0193 at ~step 2500, then monotonically pulling ahead — −2σ by ~6k, −3σ by ~11k, **−3.65σ at 16k** (still widening slightly at the end). So the blend's benefit emerges as tables+tau co-adapt.

## Tau trajectory (init 0.5 → final) vs measured Δ_m
| layer | final tau | Δ_m | final/Δ_m | 2/tau | nearest 2^k |
|---|---|---|---|---|---|
| L0 | 0.1583 | 0.0331 | 4.78× | 12.64 | 2^-3=0.125 (1.27×) |
| L1 | 0.2924 | 0.0724 | 4.04× | 6.84 | 2^-2=0.25 (1.17×) |
| L2 | 0.3337 | 0.0780 | 4.28× | 5.99 | 2^-2=0.25 (1.33×) |
| L3 | 0.3804 | 0.0816 | 4.66× | 5.26 | 2^-1=0.5 (0.76×) |
| L4 | 0.4646 | 0.0919 | 5.06× | 4.30 | 2^-1=0.5 (0.93×) |
| L5 | 0.6689 | 0.1079 | 6.20× | 2.99 | 2^-1=0.5 (1.34×) |

**tau DESCENDED from 0.5 toward Δ_m at every layer — fastest at L0 (smallest Δ_m), slowest at L5 — but settled ~4–6× ABOVE the measured margin gap, not at it.** The per-layer ORDER matches Δ_m (sharpest tau where margins are smallest), but the learned optimum is a much SOFTER blend than the "matched-Δ_m" prior. That directly explains why 0194 (frozen at Δ_m) underperformed: Δ_m is too sharp; the routing wants ~4–6× softer.

**Power-of-two proximity (for shift-only deploy):** none are tight — tau is 1.17–1.34× off the nearest 2^k (L3/L4 closest, ~0.76–0.93× of 0.5). 2/tau spans 3.0–12.6 (L4≈4.3, L5≈3.0). A shift-only routing weight would need to round tau to powers of two; the eval-only PTQ study (step 3b) will measure the bpb cost of that rounding.
