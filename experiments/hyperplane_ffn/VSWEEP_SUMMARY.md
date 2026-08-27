# v_lut overnight sweep — results & conclusions

**Question:** can a routed V (CompressionMHL v_lut replacing the dense Linear V in attention) match or beat the dense Linear V? At what param/FLOP/bandwidth cost?

**Setup:** each run is a self-contained ~3500-step training on the same cosine LR schedule, scored against an **adjusted-exp_n_0084 DENSE-V anchor** trained under identical settings (val_bpb 1.3979). nap=6 fixed, inner_in==inner_out, AdamW-standard (no LR sweep), FFN slot untouched, batched path, shared lutorch unmodified. "matched" = v_lut table-init noise re-tuned so step-0 V RMS hits a target (fixed init-B noise 0.363 over-scales wide configs).

## Ranked results (val_bpb @3500, lower=better)
| rank | config | total params | v_lut params | val_bpb | vs dense (1.3979) |
|---|---|---|---|---|---|
| — | **DENSE-V anchor** | 67.35M | 0 | **1.3979** | ref |
| 1 | in256-match (Vrms0.5) | 105.80M | 39.33M | **1.4045** | **+0.0066** |
| 2 | in192-match (Vrms0.5) | 95.97M | 29.50M | 1.4134 | +0.0155 |
| 3 | in128-match (Vrms0.5) | 86.13M | 19.67M | 1.4178 | +0.0199 |
| 4 | in96-match (Vrms0.39) | 81.22M | 14.75M | 1.4208 | +0.0229 |
| 5 | in48/out48 (Vrms0.54) | 73.84M | 7.38M | 1.4215 | +0.0236 |
| 6 | BIG-C H4/in48/tph352 | 93.31M | 26.84M | 1.4236 | +0.0257 |
| 7 | in48-match (Vrms0.39) | 73.84M | 7.38M | 1.4255 | +0.0276 |
| 8 | BIG-A H8/in48/tph176 | 94.19M | 27.73M | 1.4258 | +0.0279 |
| 9 | in64/out64 (Vrms0.63) | 76.30M | 9.83M | 1.4274 | +0.0295 |
| 10 | H8 in24/out24 | 73.84M | 7.38M | 1.4332 | +0.0353 |
| 11 | v_lut baseline in24/out24 | 70.16M | 3.69M | 1.4364 | +0.0385 |

## Findings
1. **Inner-dim is the one effective lever.** At the optimal init V-scale it closes the gap monotonically and *accelerating*: in48 +0.0236 → in128 +0.0199 → in192 +0.0155 → **in256 +0.0066**. Extrapolates to MATCH dense around inner ~290–320.
2. **Init V-scale has an optimum ≈0.5.** Matching to dense's 0.388 is too low (in48-match 0.39 = +0.0276 vs in48 0.54 = +0.0236); the fixed init-B noise 0.363 over-scales wide configs (Vrms→1.09) and hurts — the earlier "big configs plateau" was partly this confound.
3. **Dead ends:** routing heads HURT (H8 +0.0353 vs baseline +0.0385, and H8 drags BIG-A); tph / free-bandwidth is flat (BIG-C tph352 +0.0257 ≥ tph88 +0.0236); over-budget via heads/tph plateaus.
4. FFN LUT init untouched throughout (std 0.000577); no shared-lib edits; no duplicate config keys; every run passed a real fwd+bwd smoke (grad finite, no OOM/NaN).

## Answers
- **Cheapest routed-V that MATCHES dense-V:** none at ≤~1.5× params within 3500 steps — but the gap is small and shrinking. Practical near-match: **in256 (105.8M, 1.57× dense) at +0.0066 (0.5%)**. Cheapest "decent" config: in48/out48 (73.8M, +0.0236).
- **Does routed-V BEAT dense-V at any budget?** Not within tested budgets (best is +0.0066 behind at 1.57× params). BUT the accelerating close means it very likely **matches, and could beat, dense around inner ~300 (~1.7–1.8× params)** — cost is real: wide inner = larger compress/decompress matmuls (FLOPs), gather stays cheap.

## Recommendation
- Run **in288/in320 at Vrms~0.5** to confirm crossing dense, then a **full 16k-step finale** of the crossing config — the short-run schedule may also understate routed-V (it appears to learn V slightly slower).
- Best single lever: widen inner_in=inner_out with init V-scale ≈0.5; do NOT add routing heads or tables.
