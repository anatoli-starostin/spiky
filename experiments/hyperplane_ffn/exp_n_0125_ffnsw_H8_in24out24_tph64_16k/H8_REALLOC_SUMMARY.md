# H8 in24/out24 tph64 — full 16k (exp_n_0125) — head-count reallocation is a loss (task 08b218bb)

exp_n_0124 (H4 in32/out32 nap8 tph128, 1.19973) reallocated to **H8 in24 out24 nap8 tph64** — more
routing heads, narrower per-head inner, fewer tables/head, with **total tables n_heads·tph = 8·64 = 512
held constant** (== 0124's 4·128). train.py byte-identical; full 16k, effective batch 24,576.

## Envelope — a reallocation, not a shrink (predicted == measured)
| | params | FFN FLOPs | vBW | ×FLOP / ×vBW |
|---|---|---|---|---|
| 0124 H4 in32/out32 | 54.47M | 1.376M | 1.392M | 10.29 / 10.17 |
| **0125 H8 in24/out24** | **48.48M** | **1.942M** | **1.933M** | 7.29 / 7.32 |

8 heads × 24-wide inner = wider total compress/decompress (H·in = 192 vs 128) → **+41% FLOP / +39% vBW**;
but narrower eff_out=24 shrinks the table (25.2M→18.9M) → **−11% params** (48.48M, smallest on the line).

## Result — H8-narrow LOSES to H4-wider
**final val_bpb 1.20332:**
- vs 0124 H4 in32/out32 (54.47M): **+0.00359** — worse, *despite +41% FLOP*
- vs 0084 dense-V (67.35M): +0.00466 (below dense-V)
- vs 0045 tied (1.1977): +0.00562

Head-to-head at fixed 512 total tables, 0125 is worse on **bpb AND FLOP AND vBW**; it only wins on params
(−11%). **Fewer, wider-inner, wider-output heads beat more, narrower ones.**

## Step-aligned vs 0124 — more heads help EARLY, per-head capacity wins LATE
Gap 0125−0124: early(≤4000) **−0.00328** (0125 actually *ahead* early) → mid +0.00251 → late +0.00377.
More routing heads give more paths and fit faster early, but the narrower per-head inner / table-output
/ fewer-tph lose capacity that matters late — so it ends behind. Same "capacity pays late" theme seen for
nap, tph, and inner_in; **head count is an early-convergence lever, not a late-quality one.**

## Frontier position — dominated
0125 (48.48M, 1.20332) is the cheapest-param point but is worse bpb than 0124 (only 6M more params) AND
burns 41% more compute — off every useful frontier. The params↔bpb frontier stays: 54.62M in48/out32
(0123, still beats dense-V) → 67.35M nap8/tph128 (0121) → 105.10M nap9/tph128 (0119) → 180.60M
nap9/tph256 (0118).

## Takeaway
At a fixed table budget, do **not** split into more/narrower routing heads — keep fewer, wider heads
(wider inner, wider table output, more tph/head). Head count buys early-convergence speed but costs
late-training quality and, here, extra compute. The winning FFN-LUT recipe remains H4 with the capacity
in inner-width / table-width / tph, and nap for cheap late-paying param capacity.

See `h8_realloc_frontier.png`.
