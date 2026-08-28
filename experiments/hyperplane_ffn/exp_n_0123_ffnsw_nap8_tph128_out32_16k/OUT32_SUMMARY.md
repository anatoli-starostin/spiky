# inner_out 48→32 on nap8/tph128 — full 16k (exp_n_0123) — the multi-axis knob (task 4247a858)

exp_n_0121 (nap8/tph128 in48/out48, 1.19145) with ONE knob change: **inner_out 48→32** (asymmetric
in48/out32). train.py byte-identical; full 16k, effective batch 24,576, identical schedule.

## inner_out cuts ALL axes (unlike inner_in) — predicted == measured
| | params | FFN FLOPs | vBW | ×FLOP / ×vBW |
|---|---|---|---|---|
| 0121 in48/out48 | 67.35M | 2.015M | 2.081M | 7.02 / 6.80 |
| **0123 in48/out32** | **54.62M** | **1.671M** | **1.687M** | 8.47 / 8.39 |

inner_out (eff_out) is the LUT's table output width, so narrowing it cuts everything at once: table
params 37.7M→25.2M, decompress matmul + weights, routing sum, and selected-row vBW. Params −19%,
FLOP −17%, vBW −19%. (Contrast: inner_in only touches the compress matmul, leaving the table intact.)

## Quality — still beats dense-V at 19% fewer params
**final val_bpb 1.19749:**
- vs exp_n_0121 in48/out48 (67.35M): **+0.00604** (cost of the narrower table-width, −12.73M params)
- vs exp_n_0084 dense-V (67.35M): **−0.00117** — still beats dense-V, at 54.62M = **19% fewer params**
- vs exp_n_0045 tied (1.1977): −0.00021 (dead-even with the tied baseline)

## Step-aligned vs 0121 — the penalty SHRINKS with training
Gap 0123−0121: early(≤4000) +0.01343 → mid +0.00689 → late +0.00598. The narrower table-output hurts
early fitting most and partly recovers late — the *opposite* of the tph/nap gaps (which grew). Table
output *width* buys early-fit expressivity; table *count* (tph) and cells (nap) keep paying off late.

## Updated params↔bpb Pareto frontier (all beat dense-V)
| params | config | final bpb | vs dense-V |
|---|---|---|---|
| 54.62M | in48/out32 (0123) | 1.19749 | −0.00117 |
| 67.35M | nap8/tph128 (0121) | 1.19145 | −0.00721 |
| 105.10M | nap9/tph128 (0119) | 1.18386 | −0.01480 |
| 180.60M | nap9/tph256 (0118) | 1.17460 | −0.02406 |

(nap9/tph64 (0120, 1.19859) remains dominated.) 0123 extends the frontier to the cheapest param point yet
that still beats dense-V.

## Knob map (now fully characterized on this line)
- **nap** (cells): pure param/quality; zero vBW, ~zero FLOP. Cheapest bpb-per-param, pays off late.
- **tph** (tables): param + vBW; ~1.9× costlier per param than nap; pays off late.
- **inner_in** (compress): FLOP + vBW only, ~no param change (table untouched).
- **inner_out** (table width): cuts params + FLOP + vBW together — the "shrink everything" knob; costs
  early-fit quality that partly recovers late.

## Takeaway
inner_out is the lever when you want to shrink *compute and bandwidth and params* together; it took the
routed FFN down to 54.62M while still edging dense-V. For pure param-vs-bpb the tph/nap axes are more
efficient. Next frontier points: in48/out40, or in48/out32 at a higher tph to trade table-width for
table-count.

See `out32_frontier.png`.
