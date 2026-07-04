---
name: lut-optimizer-sweep
description: "LUT-param optimizer sweep on exp428 arch (2026-05-20): LION β=(0.9,0.95) BEATS AdamW (1.4967 vs 1.4983, −0.0016) at HALF optimizer memory — new bs=16 LUT-LM best. β2 sweet spot: 0.99 bad, 0.95 beats Adam, 0.9=Signum≈Adam. signSGD/Signum 1.5023 (+0.004). SGD+mom 1.5319."
metadata:
  node_type: memory
  type: project
  originSessionId: fa43f8ba-4262-4d5d-ab44-b1dfbc584286
---

# LUT-param optimizer sweep — LION β=(0.9,0.95) BEATS AdamW at half memory (2026-05-20)

**Headline: LION with β₁=0.9, β₂=0.95 on the LUT param group beats AdamW (1.4967 vs exp428 1.4983, −0.0016) AND uses half the optimizer-state memory (1 momentum tensor vs Adam's m+v). New bs=16 LUT-LM best at this arch.** The moderate-memory two-timescale split is the sweet spot.

Setup: swapped ONLY the LUT param group's optimizer (non-LUT stay AdamW); all else = exp428 (NAP=6, bs=16, 8K, 89.4M, lr schedule cosine+10% warmup). Lion + SignSGD classes inline in exp447+ train.py; `lut_optimizer` config selects adamw|lion|signsgd|sgd, `lut_betas`/`lut_momentum` set betas. lr for sign-family set to 2e-4 (matched to exp446's realized Adam lr ~2.2e-4; SGD set from grad RMS).

## Leaderboard (final @8K, vs AdamW exp428 = 1.498293)
| optimizer | lut_lr | final bpb | Δ | opt mem |
|---|---|---|---|---|
| **LION β=(0.9,0.95)** (exp453) | 2e-4 | **1.496708** | **−0.0016** | ½ |
| AdamW (exp428) | 1e-3 | 1.498293 | — | 1× (m,v) |
| signSGD/Signum β=0.9 (exp448) | 2e-4 | 1.502308 | +0.004 | ½ |
| SGD+mom (exp451) | 0.1 | 1.531944 | +0.034 | ½ |
| LION β=(0.9,0.9)≡Signum (exp452) | 2e-4 | == exp448 (killed, confirmed) | +0.004 | ½ |
| LION β=(0.9,0.99) (exp447/449) | 2e-4/3e-4 | ~+0.05 (killed) | +0.05 | ½ |

## β₂ axis (β₁=0.9 fixed) — there IS a sweet spot
| β₂ | result | mechanism |
|---|---|---|
| 0.99 | ~+0.05 (bad) | stored momentum too slow → signs stale directions |
| **0.95** | **−0.0016 (beats Adam)** | moderate memory accumulates sparse-row signal; β₁ blend keeps step fresh |
| 0.90 | +0.004 (≡ Signum) | no extra memory beyond the β₁ EMA |

## Key findings
1. **LION β=(0.9,0.95) is the best LUT optimizer found** — beats AdamW by 0.0016 at half optimizer memory. Below AdamW at EVERY eval from step 600→8000 (consistent trajectory, not a final fluke), which makes the small final margin credible despite being within single-eval noise (~±0.003). Caveat: one seed, 8K/bs=16 only.
2. **The β₂ two-timescale split is load-bearing and non-monotone.** β₂=0.99 hurts (stale), β₂=0.9 collapses to Signum (≈Adam), β₂=0.95 beats Adam. For per-row-sparse LUT grads, a moderate-memory stored momentum accumulates each row's sparse hits while the β₁=0.9 sign-blend keeps the direction fresh — something neither Signum (no memory) nor AdamW (√v̂ damping) does.
3. **Adam's per-coordinate √v̂ adaptivity is NOT the lever on LUTs.** Signum (no √v̂) already ≈ Adam; the win comes from momentum structure, not magnitude scaling. Consistent with exp446 (Adam already ~sign-like, factor 0.22) and exp445 (exp428→exp444 gap is functional, not optimization).
4. **LION(β₁=β₂)≡Signum exactly** (LION signs c=β₁·m+(1−β₁)·g; with β₁=β₂ this equals the updated m, same recursion as Signum's buffer — exp452 reproduced exp448 at every eval). LION's value over Signum is ONLY the β₁≠β₂ split.
5. **lr for sign/SGD must come from grad scale, not Adam's value.** LION/signSGD ≈ Adam's realized lr (2e-4). SGD needs lr from grad RMS (2.1e-3 → lr0.1 works, lr0.01 stalls +0.21). SGD+mom also showed strong decay-phase catch-up (+0.068 peak → +0.034 final).

## Follow-ups
- Confirm exp453 with a 2nd seed (de-risk the −0.0016 margin) before adopting.
- Refine β₂ (0.93 / 0.97) to find the optimum; possibly sweep β₁ too.
- Validate at longer horizon / other scale. If it holds, **adopt LION β=(0.9,0.95) as the default LUT optimizer** — better than Adam at half the memory.

Code: Lion/SignSGD + `lut_optimizer`/`lut_betas` config in nanochat_exps/exp453_lion_b2_0p95/train.py. See [[effective-lr-probe-exp446]], [[soft-wgrad-neutral-exp445]], [[hard-forward-is-the-goal]].
