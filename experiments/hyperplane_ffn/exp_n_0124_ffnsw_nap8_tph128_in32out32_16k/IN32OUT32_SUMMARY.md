# nap8/tph128 in32/out32 — full 16k (exp_n_0124) — the compute floor / dense-V crossover (task 0a0e21a2)

exp_n_0123 (in48/out32, 1.19749) with ONE knob change: **inner_in 48→32**, so both inner dims = 32.
train.py byte-identical; full 16k, effective batch 24,576, identical schedule.

## Envelope — cheapest FLOP/vBW on the line (predicted == measured)
| | params | FFN FLOPs | vBW | ×FLOP / ×vBW |
|---|---|---|---|---|
| 0121 in48/out48 | 67.35M | 2.015M | 2.081M | 7.02 / 6.80 |
| 0123 in48/out32 | 54.62M | 1.671M | 1.687M | 8.47 / 8.39 |
| **0124 in32/out32** | **54.47M** | **1.376M** | **1.392M** | **10.29 / 10.17** |

inner_in only touches the compress matmul/weights → −18% FLOP and −18% vBW vs 0123 at ~no param change
(~148K fewer; table unchanged at 25.2M). First config on this line past **10× vs dense on both axes**.

## Quality — dips just below dense-V (the crossover)
**final val_bpb 1.19973:**
- vs 0123 in48/out32 (54.62M): **+0.00224** (cost of inner_in 48→32)
- vs 0121 in48/out48 (67.35M): +0.00828
- vs exp_n_0084 dense-V (67.35M): **+0.00107** — now just *below* dense-V
- vs exp_n_0045 tied (1.1977): +0.00203

So the routed FFN beats dense-V down to 0123 (in48/out32) and finally slips under at 0124 (in32/out32) —
the dense-V-quality **crossover** on this line is right around 54M / narrow-both-inner.

## compress vs decompress dim — inner_in is the cheaper compute knob
Each dim taken 48→32 from its baseline:
- **inner_out** (0121→0123): +0.00604 bpb — cuts params 12.73M + FLOP 17% + vBW 19%
- **inner_in** (0123→0124): +0.00224 bpb — cuts FLOP 18% + vBW 18%, ~0 params

inner_in costs ~1/3 the bpb of inner_out for a similar FLOP/vBW saving, because it doesn't remove table
capacity (params) — quality lives in the table. **To cut compute/bandwidth, narrow inner_in before inner_out.**

## Step-aligned vs 0123 — compress width helps only LATE
Gap 0124−0123: early(≤4000) **−0.00262** (0124 actually *ahead* early) → mid +0.00016 → late +0.00223.
The narrower compress trains slightly faster early (mild regularization / easier optimization) but the
extra compress capacity (in48) pays off late — same "capacity pays late" theme as nap/tph, and the mirror
of inner_out (which hurt early, recovered late).

## Frontier position — two different frontiers
- **params↔bpb:** 0123 (54.62M, 1.19749) **dominates** 0124 (54.47M, 1.19973) — 0124 saves only 0.15M
  params for +0.0022 bpb, so it is off the param frontier.
- **FLOP/vBW↔bpb:** 0124 (1.376M/1.392M) is the **cheapest-compute point** (>10× dense), extending that
  frontier below 0123 — useful only in a compute/bandwidth-bound regime, and it costs dipping ~0.001 under
  dense-V.

## Takeaway
inner_in is a pure compute/bandwidth lever (cheapest bpb-per-FLOP/vBW), not a param lever — reach for it
only when compute/bandwidth is the binding constraint. On this line the routed FFN's dense-V-quality
break-even sits around in48/out32 (0123, 54.62M, still −0.0012 ahead); pushing to in32/out32 buys the
cheapest compute but crosses just under dense-V.

See `in32out32_frontier.png`.
