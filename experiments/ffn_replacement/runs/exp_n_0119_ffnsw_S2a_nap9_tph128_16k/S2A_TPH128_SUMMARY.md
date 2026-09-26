# S2a nap9, tph 256→128 — full 16k (exp_n_0119) — results (task 4238493e)

exp_n_0118 (the nap9 winner, 1.17460) with ONE knob change: **tph 256 → 128**. train.py byte-identical;
full 16k schedule identical (lr 3e-4, warmup 1600, cosine/16000, bs12/ga4, effective batch 24,576).

## Cost envelope (predicted == measured)
| | total params | FFN FLOPs | vBW | ×FLOP / ×vBW vs dense |
|---|---|---|---|---|
| 0118 tph256 | 180.60M | 2.286M | 2.375M | 6.19 / 5.96 |
| **0119 tph128** | **105.10M** | **2.028M** | **2.081M** | **6.98 / 6.80** |

Halving tph **halves the table params** (151M→75.5M table rows; total 180.60M→105.10M) and halves the
*selected-row* vBW component — but **total decode-regime vBW drops only ~12%** (2.375→2.081M; the
compress/decompress weight bytes dominate) and **FFN-FLOP only ~11%** (only the routing term scales with
tph; the two matmuls are unchanged). So this is essentially a **param/table-size cut at a nearly-unchanged
FLOP/vBW envelope**.

## Quality
**final val_bpb 1.18386** — still beats both dense targets:
- vs exp_n_0084 (dense-V FFN) 1.19866 → **−0.01480**
- vs exp_n_0045 (tied) 1.1977 → **−0.01384**
- vs exp_n_0118 (tph256) 1.17460 → **+0.00926** (the cost of halving the table)

**62% of nap9's win over dense-V is retained at 58% of the params** (0118 kept −0.0241 vs 0084; 0119 keeps −0.0148).

## Step-aligned gap 0119 − 0118 — a FLAT penalty
The cost of halving tph is a *nearly constant parallel offset* across the whole run, not a growing or
shrinking gap:

| window | mean gap (0119 − 0118) |
|---|---|
| early (≤4000) | +0.00988 |
| mid (4200–10000) | +0.00909 |
| late (10200–16000) | +0.00941 |
| final @16000 | +0.00926 |

(Small early wiggle: gap peaks ~+0.015 around step 400–2200, dips to ~+0.003–0.006 around 2800–4400 —
the same transient region seen in 0118 vs 0084 — then settles flat at ≈+0.0094.) Contrast: nap9's edge
*over dense-V* grew with training, but the *tph256-vs-tph128* penalty is flat — table count buys a
roughly step-independent quality offset.

## Takeaway
tph128 is a favorable operating point: **~42% fewer params than tph256, still −0.015 bpb below dense-V and
−0.014 below tied**, at nearly the same FLOP/vBW. But tph256 is meaningfully better (+0.009) if the table
budget is available — the win scales with table count, just sub-linearly. Natural sweep: tph ∈ {160, 192}
to trace the params-vs-bpb curve between these two points.

See `tph128_vs_0118_stepaligned.png`.
