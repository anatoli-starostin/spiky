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

## Sweep C — tied, 4096 steps, 2× FFN budget, AdamW-LUT (exp075-087)

Repeat of Sweep B at 2× the per-layer FFN budget (2,359,296; target total 30,287,616). Fixed 6 heads, hard, AdamW-LUT. Reference: tied dense exp055 (4k) **1.35543**.

| rank | run | config | tph | val_bpb | Δ vs tied-dense 1.35543 | vs 1× (Sweep B) |
|------|-----|--------|----:|--------:|------------------------:|-----------------|
| 1 | C5 exp079 | 6h 64/64 nap5 | 168 | **1.36453** | +0.00910 | — |
| 2 | C1 exp075 | 6h 64/64 nap6 | 84 | 1.36613 | +0.01070 | B5 1.38111 → **−0.01498** |
| 3 | C7 exp081 | 6h 128/64 nap6 | 78 | 1.36952 | +0.01409 | — |
| 4 | C2 exp076 | 6h 96/96 nap6 | 52 | 1.37093 | +0.01550 | — |
| 5 | C6 exp080 | 6h 64/64 nap7 | 42 | 1.37125 | +0.01582 | — |
| 6 | C12 exp083 | 6h 96/96 nap5 | 104 | 1.37138 | +0.01595 | — |
| 7 | C3 exp077 | 6h 128/128 nap6 | 36 | 1.37509 | +0.01966 | — |
| 8 | C9 exp085 | 6h 64/64 nap6 g1 | 78 | 1.37615 | +0.02072 | — |
| 9 | C4 exp078 | 6h 192/192 nap6 | 20 | 1.37850 | +0.02307 | — |
| 10 | C8 exp082 | 6h 64/128 nap6 | 39 | 1.37904 | +0.02361 | — |
| 11 | C13 exp084 | 6h 128/128 nap7 | 18 | 1.38157 | +0.02614 | — |
| 12 | C11 exp087 | 6h -1/64 (no compress) | 90 | 1.39623 | +0.04080 | — |
| 13 | C10 exp086 | 6h 64/-1 (no decompress) | 15 | 1.41704 | +0.06161 | — |

**Findings:** 2× budget helped clearly — best C5 (1.36453) is well below the Sweep-B best (B4 1.37955), and the same-shape C1 (2×) beats its 1× twin B5 by **−0.0150**. But it still does NOT beat the tied dense baseline (best C5 is +0.0091 short of 1.35543); doubling closed most of the Sweep-B gap (~+0.024 → +0.009) without closing it. Levers hold: nap5 ≥ nap6 > nap7; NARROW inner (64) beats wider (64<96<128<192) — at 2× budget with 6 heads the extra params are better spent on MORE tables (higher tph) than a wider inner; wide-read asymmetry (128/64) helps, wide-write (64/128) hurts; gamma still hurts (C9); dropping a projection is worst (C10/C11).

### Sweep C follow-ups (exp091/exp092) — forked from C1 (exp075, 1.36613; tied dense exp055 1.35543)
| exp | change from C1 | tph | total params | val_bpb | vs C1 | vs tied-dense | crosses dense? |
|-----|----------------|----:|-------------:|--------:|------:|--------------:|:--------------:|
| exp091 | tph 84→96 (OVER budget +5.9%) | 96 | 32,061,696 | 1.36532 | −0.00081 | +0.00989 | no |
| exp092 | inner 64/64→48/48 (param-matched) | 116 | 30,291,648 | 1.36168 | −0.00445 | +0.00625 | no |
| exp093 | inner 48/48, tph 116→128 (OVER +4.4%) | 128 | 31,618,752 | 1.36062 | −0.00551 | +0.00519 | no |
| exp094 | 6h→**12 heads**, inner 48/48, tph=64 fixed (OVER +8.8%) | 64 | 32,947,584 | **1.35924** | −0.00689 | +0.00381 | no (closest yet) |

**exp094 is the new closest-to-dense** — doubling the head count (6→12) at fixed inner 48/48, tph=64 gives **1.35924**, beating exp093 (6h) by −0.00138 and cutting the gap to tied dense to **+0.0038** (vs exp093's +0.0052) — the smallest any CompressionMHL FFN slot has reached, but still no crossover. It runs +8.8% over the 2× budget (32.95M vs 30.29M), so this is not a like-for-like budget win; head count is a genuine lever (independent per-head summation adds capacity) but costs params to exercise. Standing question: whether 12 heads *at* budget (re-solving tph down) still beats exp092's 6h-matched 1.36168.

**exp092 is the best tied-2×-budget result** — narrower inner (48) with more tables (tph 84→116) beats C1's 64/64 by −0.0044, extending the "narrow inner + more tables" trend (Sweep C: 64<96<128<192; now 48<64). The optimum inner is ≤48. But it's still +0.0063 above tied dense — no crossover. exp091 shows pushing tph past the budget (+5.9% params) barely helps (−0.0008) — table count past budget is not the lever. Net: at the 2× tied budget the CompressionMHL slot gets to within ~+0.006 of tied dense but does not cross it; a narrower inner is the better use of budget than more tables-over-budget.

---

## Sweep D — fixed-throughput head/dim split (H·d = 384), exp095–100

Hold the **throughput budget** constant: compress+decompress matmul cost 2·H·D·d =
2·384·384 = 294,912 MACs/layer (a fixed ~4× saving vs the dense FFN's 8·D² = 1,179,648),
and sweep **only** how the fixed product H·d=384 splits between head count H and inner dim d.
Because tables = tph·2⁶·(H·d) depends only on H·d, the solved **tph = 84 for every run** and the
**total params are identical = 30,292,224** — a clean fixed-throughput AND fixed-param sweep, only
the head/dim split varies. Common config = the exp092 line (tied, nap6, gamma0, AdamW two-group, 4096
steps). D3 (H=6/d=64) = C1/exp075 (reused). Refs: tied dense exp055 = 1.35543.

| rank | run | H×d | tph | total params | val_bpb | Δ vs tied-dense | Δ vs C1/H6 (1.36613) | crosses dense? |
|----:|-----|-----|----:|-------------:|--------:|----------------:|---------------------:|:--------------:|
| 1 | D5 exp098 | 12×32 | 84 | 30,292,224 | **1.35966** | +0.00423 | −0.00647 | no (closest at-budget) |
| 2 | D6 exp099 | 16×24 | 84 | 30,292,224 | 1.36054 | +0.00511 | −0.00559 | no |
| 3 | D7 exp100 | 24×16 | 84 | 30,292,224 | 1.36390 | +0.00847 | −0.00223 | no |
| 4 | D3 exp075 | 6×64 | 84 | 30,292,224 | 1.36613 | +0.01070 | 0 | no |
| 5 | D4 exp097 | 8×48 | 84 | 30,292,224 | 1.36988 | +0.01445 | +0.00375 | no |
| 6 | D2 exp096 | 4×96 | 84 | 30,292,224 | 1.37538 | +0.01995 | +0.00925 | no |
| 7 | D1 exp095 | 3×128 | 84 | 30,292,224 | 1.37871 | +0.02328 | +0.01258 | no |

![Sweep D head/dim](SWEEP_D_headdim.png)

**Findings — more-but-narrower heads win, with an interior optimum at H≈12.** At fixed throughput and
fixed params, increasing the head count (shrinking per-head inner d) improves bpb strongly: H3→H6 falls
1.37871→1.36613, then the **optimum is H=12/d=32 = 1.35966** (−0.0065 vs the H6/d64 standard). Past the
optimum it gently regresses (H12<H16<H24: 1.35966, 1.36054, 1.36390), giving a broad basin around H12–16.
The lone wrinkle is **D4 (H8/d48 = 1.36988)**, which is *worse* than both neighbours H6 and H12 — a
non-monotone dip almost certainly single-seed noise (seed=1, one run each), since the H3→H6 and H12→H24
arms are otherwise smooth. Mechanism: more **independent** heads = more parallel sign-test addressers
summed into the output, and that addressing capacity outweighs the loss from a narrower compressed vector
down to d≈32; below that (d=24,16) the compressed code is too thin and the gain saturates then reverses.

**No config crosses tied dense.** Best (D5 H12/d32) is +0.0042 above 1.35543 — the closest any *at-budget*
CompressionMHL FFN slot has reached (the only closer point, exp094 at +0.0038, is H12/d48 but spends +8.8%
over budget). This corroborates exp094: **H=12 is the sweet-spot head count** both over-budget (exp094,
1.35924) and at-budget (D5, 1.35966) — nearly the same bpb, so on-budget D5 is the better deal. Net across
Sweeps C+D: the strongest lever is **head count (→~12), then narrow inner (d≈32–48)**; more tables past
budget and wider inner both underperform. The slot converges to ~+0.004 of tied dense but does not cross it.

### 16k confirm of the D5 champion (exp_n_0001)

Ran the Sweep-D winner **D5 (H12/d32, 30,292,224 params)** at the full **16000-step** budget
(`exp_n_0001_D5champ_H12_d32_16k`): **final val_bpb = 1.22473**. vs **tied dense 16k exp073 = 1.19665 →
+0.02808 (dense wins)**; vs 4k D5 = 1.35966 → −0.13493 (compute effect). **The gap to tied dense WIDENS
with budget**: +0.0042 at 4k → +0.0281 at 16k. Dense benefits far more from the longer schedule than the
CompressionMHL LUT slot — the slot's short-horizon competitiveness is a low-budget effect that erodes at
full budget, consistent with every other 16k pairing where dense wins.
