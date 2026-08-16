# FFN-slot LUT sweep (exp036, exp042, exp043–exp069)

FFN slot per block: `x = x + gamma·Linear(384→384)(h) + CompressionMultiHeadLUT{n_heads, inner_in, inner_out, tph, nap}(h)`, `h = ln2(x)`.
CompressionMHL fixed for all: `joint_head_compression=False` (independent per-head), `forward_mode="hard"`, fp32, near-zero table init, per-layer seed 1000+idx, decompress zero-init.
`inner_in=-1` → no compress (LUT reads x); `inner_out=-1` → no decompress (LUT emits 384, heads summed). `gamma=1` adds a plain `Linear(384→384)` skip.

Shared trainer: `train.py` (config-driven; `ffn_type`, `gamma`, `tie_unembedder`, `lut_*`). Training config identical across the sweep: MinimalGPT+RoPE d384/6L/6H/seq512, device_bs 48, total_bs 24576, **n_steps 4096**, lr 3e-4, wd 0.1, warmup 0.1, eval_every 200, seed 1, same data. Runs param-matched to the vanilla baseline of their sweep.

## Sweep A — UNTIED unembedder (target total 35,792,640; reference = exp032 dense FFN **1.39371**)

| run | exp | n_heads | inner_in/out | nap | gamma | tph | total params | Δ% | val_bpb |
|-----|-----|--------:|:------------:|----:|:-----:|----:|-------------:|----:|--------:|
| A1  | exp036 | 1 | 64/64 | 6 | 0 | 276 | 35,795,328 | +0.0075 | 1.40699 |
| A2  | exp043 | 2 | 64/64 | 6 | 0 | 132 | 35,795,712 | +0.0086 | 1.40735 |
| A3  | exp042 | 3 | 64/64 | 6 | 0 | 84  | 35,796,096 | +0.0097 | 1.39577 |
| A4  | exp044 | 4 | 64/64 | 6 | 0 | 60  | 35,796,480 | +0.0107 | 1.39811 |
| A5  | exp045 | 6 | 64/64 | 6 | 0 | 36  | 35,797,248 | +0.0129 | 1.39063 |
| A6  | exp046 | 3 | 64/64 | 5 | 0 | 168 | 35,796,096 | +0.0097 | 1.39219 |
| A7  | exp047 | 3 | 64/64 | 7 | 0 | 42  | 35,796,096 | +0.0097 | 1.40603 |
| A8  | exp048 | 3 | 32/32 | 6 | 0 | 180 | 35,795,520 | +0.0080 | 1.40369 |
| A9  | exp049 | 3 | 96/96 | 6 | 0 | 52  | 35,796,672 | +0.0113 | 1.39356 |
| A10 | exp050 | 3 | 128/128 | 6 | 0 | 36 | 35,797,248 | +0.0129 | 1.39698 |
| A11 | exp051 | 3 | -1/64 | 6 | 0 | 90  | 35,794,944 | +0.0064 | 1.42388 |
| A12 | exp052 | 3 | 64/-1 | 6 | 0 | 15  | 35,793,792 | +0.0032 | 1.44419 |
| A13 | exp053 | 3 | -1/-1 | 6 | 0 | 16  | 35,792,640 | +0.0000 | 1.48079 |
| A14 | exp054 | 3 | 64/64 | 6 | 1 | 72  | 35,798,400 | +0.0161 | 1.42022 |

## Sweep B — TIED unembedder (target total 23,209,728; reference = exp055 tied vanilla dense FFN)

| run | exp | n_heads | inner_in/out | nap | gamma | tph | total params | Δ% | val_bpb |
|-----|-----|--------:|:------------:|----:|:-----:|----:|-------------:|----:|--------:|
| B0  | exp055 | — dense FFN (tied vanilla baseline) | | | | — | 23,209,728 | +0.0000 | 1.35543 |
| B1  | exp056 | 1 | 64/64 | 6 | 0 | 276 | 23,212,416 | +0.0116 | 1.39896 |
| B2  | exp057 | 2 | 64/64 | 6 | 0 | 132 | 23,212,800 | +0.0132 | 1.39154 |
| B3  | exp058 | 3 | 64/64 | 6 | 0 | 84  | 23,213,184 | +0.0149 | 1.38371 |
| B4  | exp059 | 4 | 64/64 | 6 | 0 | 60  | 23,213,568 | +0.0165 | 1.37955 |
| B5  | exp060 | 6 | 64/64 | 6 | 0 | 36  | 23,214,336 | +0.0198 | 1.38111 |
| B6  | exp061 | 3 | 64/64 | 5 | 0 | 168 | 23,213,184 | +0.0149 | 1.38394 |
| B7  | exp062 | 3 | 64/64 | 7 | 0 | 42  | 23,213,184 | +0.0149 | 1.38607 |
| B8  | exp063 | 3 | 32/32 | 6 | 0 | 180 | 23,212,608 | +0.0124 | 1.38582 |
| B9  | exp064 | 3 | 96/96 | 6 | 0 | 52  | 23,213,760 | +0.0174 | 1.38402 |
| B10 | exp065 | 3 | 128/128 | 6 | 0 | 36 | 23,214,336 | +0.0198 | 1.38780 |
| B11 | exp066 | 3 | -1/64 | 6 | 0 | 90  | 23,212,032 | +0.0099 | 1.40742 |
| B12 | exp067 | 3 | 64/-1 | 6 | 0 | 15  | 23,210,880 | +0.0050 | 1.41168 |
| B13 | exp068 | 3 | -1/-1 | 6 | 0 | 16  | 23,209,728 | +0.0000 | 1.45407 |
| B14 | exp069 | 3 | 64/64 | 6 | 1 | 72  | 23,215,488 | +0.0248 | 1.39622 |

## Synthesis (all 27 done; 4096 steps)

**Best per sweep:**
- Sweep A (untied): **A5 = 6 heads, 64/64, nap6, g0 → 1.39063**, which BEATS the untied dense FFN (exp032, 1.39371) by −0.00308. Two more configs also beat it: A6 (nap5) 1.39219 and A9 (96/96) 1.39356.
- Sweep B (tied): the **tied dense FFN baseline (B0, 1.35543) beats every tied LUT variant**; best LUT is B4 (4 heads) 1.37955 (+0.024 behind B0).

**Levers (both sweeps agree):**
- **Head count = the dominant lever.** More independent heads → better, saturating around 4–6: A 1h/2h ≈ 1.407, 3h 1.396, 4h 1.398, 6h **1.391**; B 1h 1.399 → 4h **1.380** → 6h 1.381. (1h≈2h weak; the gain kicks in at 3h+.)
- **Projections are essential.** Dropping compress (-1/64) or decompress (64/-1) hurts badly, and the pure-LUT slot (-1/-1) is the worst by far (A13 1.481, B13 1.454). The compress→LUT→decompress bottleneck is doing real work.
- **nap:** nap5 ≥ nap6 > nap7 (A6 1.392 < A3 1.396 < A7 1.406). Slight preference for fewer, wider tables.
- **inner width:** 64 and 96 best; 32 and 128 slightly worse. Mild, non-monotonic (interacts with tph at fixed budget).
- **gamma (parallel Linear) HURTS:** A14 1.420 vs A3 1.396; B14 1.396 vs B3 1.384. Spending budget on a parallel Linear steals it from the tables and loses.

**Tying flips the LUT-vs-dense verdict.** Untied: several LUT slots beat the dense FFN. Tied: the dense FFN pulls decisively ahead — because tying helps the dense baseline far more (untied dense 1.39371 → tied dense 1.35543, −0.038) than it helps the LUT slot (untied A5 1.391 → tied B5 1.381, −0.010). So "LUT beats dense" is specific to the untied setting at 4096 steps.

_(All runs step-limited at 4096, best=final. exp070 re-runs the Sweep-A champion (A5) at 16000 steps vs exp002's dense trajectory to test a late crossover.)_

## Follow-ups — full 16k budget & Lion-for-LUT (exp070/072/073/074)

| exp | config | steps | opt(LUT) | val_bpb | ref |
|-----|--------|------:|---------|--------:|-----|
| exp002 | untied dense FFN | 16k | — | 1.20144 | untied dense ref |
| exp070 (AdamW, git 7ff2e5b2) | untied CompressionMHL 6h | 16k | AdamW | 1.23103 | +0.0296 vs exp002 |
| exp070 (Lion, current) | untied CompressionMHL 6h | 16k | **Lion** | 1.23289 | +0.0314 vs exp002; **+0.0019 vs its own AdamW** |
| exp073 | tied dense FFN | 16k | — | **1.19665** | tied dense ref (beats untied dense 1.20144) |
| exp074 | tied CompressionMHL 6h | 16k | Lion | 1.23472 | +0.0381 vs exp073 |
| exp045 / exp072 | untied CompressionMHL 6h | 4k | AdamW / Lion | 1.39063 / **1.38767** | 4k: Lion −0.0030 vs AdamW |

**Findings:** (1) At 4k, Lion on the LUT tables beats AdamW by −0.0030 (exp072<exp045). (2) That edge does NOT survive to 16k — exp070-Lion (1.23289) is marginally worse than exp070-AdamW (1.23103); Lion ≈ AdamW at full budget. (3) The dense FFN wins decisively at 16k in BOTH untied (+0.031) and tied (+0.038) settings, at every checkpoint — no crossover. (4) Tying helps the dense model at 16k too (tied dense 1.19665 < untied dense 1.20144). So the LUT FFN slot's competitiveness is a short-schedule (4k) phenomenon; at the full budget the dense GELU FFN is clearly ahead regardless of LUT optimizer or tying.
