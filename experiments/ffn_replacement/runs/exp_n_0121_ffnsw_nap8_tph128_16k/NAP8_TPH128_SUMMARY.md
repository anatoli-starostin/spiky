# nap8/tph128 — full 16k (exp_n_0121) — cells vs tables (task c3fed5fc)

exp_n_0119 (nap9/tph128) with ONE knob change: **nap 9 → 8** (cells 512→256). Isolates the CELL axis at
fixed tph=128. train.py byte-identical; full 16k, effective batch 24,576, identical schedule.

## Envelope (predicted == measured) — nap is the pure cell/param knob
| | params | FFN FLOPs | vBW | ×FLOP / ×vBW |
|---|---|---|---|---|
| 0119 nap9/tph128 | 105.10M | 2.028M | 2.081M | 6.98 / 6.80 |
| **0121 nap8/tph128** | **67.35M** | **2.015M** | **2.081M** | 7.02 / 6.80 |

nap8 halves cells → table 75.5M→37.7M → total 105.10M→67.35M, but **vBW is identical** (2.081M) and
FFN-FLOP barely moves (−0.6%): nap spends only cells (params), at zero bandwidth / ~zero FLOP.

## Result — at iso-params, MORE TABLES beats MORE CELLS, and BEATS dense-V
**final val_bpb 1.19145.** The 67.35M iso-param triple (all same 37.7M table budget):
| config | params | final bpb | vs dense-V |
|---|---|---|---|
| **0121 nap8/tph128** (256 cells × 128 tables) | 67.35M | **1.19145** | **−0.00721** ✅ |
| 0120 nap9/tph64 (512 cells × 64 tables) | 67.35M | 1.19859 | −0.00007 (tie) |
| 0084 dense-V | 67.35M | 1.19866 | 0 |

At the SAME param budget, allocating the table as **more tables / fewer cells** (nap8/tph128) beats
**fewer tables / more cells** (nap9/tph64) by **−0.00714**, and beats dense-V by −0.0072 (and tied 1.1977
by −0.0063). So the routed FFN **does** beat dense-V at iso-params — you just have to spend the budget on
tables, not cells.

## Tables are ~1.9× more valuable per param than cells
Removing the same 37.7M params from 0119 (105.10M→67.35M) two ways:
- via CELLS (nap 9→8, →0121): **+0.00760**
- via TABLES (tph 128→64, →0120): **+0.01473**  → tables cost **~1.9×** more to remove.

This *revises* the prior task's read: the 67.35M "cliff" (0120 = dead heat with dense) was a
**mis-allocation**, not a fundamental iso-param ceiling. Reallocating to more tables recovers a real
−0.0072 win over dense-V at the identical budget.

## Updated params↔bpb frontier (best routed per param)
- 67.35M → nap8/tph128 **1.19145** (beats dense-V; nap9/tph64 is dominated / off-frontier)
- 105.10M → nap9/tph128 1.18386
- 180.60M → nap9/tph256 1.17460

Step-aligned 0121−0119 gap grows with training (early +0.0028 → late +0.0076) — extra cells (nap9) help
more late, same as tables, but ~half the per-param value.

## Takeaway
tph (table count) is the more efficient table-budget knob than nap (cells) — ~1.9× the bpb-per-param.
Spend on tables first. The routed FFN beats dense-V even at dense-V's own param budget when allocated
well. Next: nap8/tph192 or nap8/tph96 to refine the frontier; a tables-vs-cells iso-param sweep at
larger budgets.

See `nap8_frontier.png`.
