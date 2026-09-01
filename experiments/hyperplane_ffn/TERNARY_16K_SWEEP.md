# Ternary FFN — 16K star / one-axis-at-a-time sweep (for the paper)

**Status: COMPLETE** — the 10-run star sweep plus the two follow-on points (0149 H8-no-penalty, 0150 H1/out192).

**Standout: exp_n_0149 (H8 + no-penalty) = 1.18203** — stacks the two axis-winners (Axis-A H8 × Axis-C no-penalty)
and ties the sweep-best B4/tph256 (1.18187), confirming the gains combine. exp_n_0150 (H1/out192) = 1.20011
extends the head axis to its single-head endpoint — the A axis is monotone H1(1.20011) < H2 < H4 < H8, no
turn-over (the CompressionMHL FFN head line, by contrast, is U-shaped with H2 best / H1 worst).

## Setup
One-axis-at-a-time sweep around **exp_n_0134** (ternary target-192 full-16k, **val_bpb 1.18943**), the
full-anneal completion of exp_g_0044. Baseline held constant unless the axis varies it: ternary hyperplane
routing on the full 384-d input + per-head output decompress, **n_heads=4, inner_out=48, nap=8 (256 cells),
tph=128, target nnz=192, λ=100**, normalize_weights, random init, T=max_entropy (0.392065), derived divisor
D=16, trainable bias. All runs: n_steps=16000, lr_schedule_steps=16000, warmup 1600, lr 3e-4, cosine→10%,
AdamW (0.9,0.95), grad-clip 1.0, effective batch 24,576, same seed/data, UNTIED unembedder. Single H100 →
sequential. Derived constants verified at every launch (T=0.392065, D=16, equal-thirds init, score/T≈1.995).

## Results (sorted by val_bpb; Δ vs exp_n_0134 = 1.18943)
| run | axis | H | inner_out | nap(cells) | tph | target nnz | params | val_bpb | Δ vs 0134 |
|---|---|---|---|---|---|---|---|---|---|
| **0147** | **B4** | 4 | 48 | 8(256) | **256** | 192 | 123.59M | **1.18187** | **−0.00756** ⭐ best |
| 0145 | B2 | 4 | 48 | **10(1024)** | 128 | 192 | 191.98M | 1.18296 | −0.00647 |
| 0143 | A2 | **8** | 24 | 8(256) | 128 | 192 | 85.84M | 1.18483 | −0.00460 |
| 0141 | C3 | 4 | 48 | 8(256) | 128 | — (λ=0, no penalty) | 76.37M | 1.18757 | −0.00186 |
| 0140 | C2 | 4 | 48 | 8(256) | 128 | 256 | 76.37M | 1.18851 | −0.00092 |
| **0134** | baseline | 4 | 48 | 8(256) | 128 | 192 | 76.37M | 1.18943 | 0 |
| 0142 | A1 | 2 | 96 | 8(256) | 128 | 192 | 71.64M | 1.19583 | +0.00640 |
| 0144 | B1 | 4 | 48 | 6(64) | 128 | 192 | 45.70M | 1.19694 | +0.00751 |
| 0139 | C1 | 4 | 48 | 8(256) | 128 | 128 | 76.37M | 1.20460 | +0.01517 |
| 0146 | B3 | 4 | 48 | 8(256) | 64 | 192 | 52.77M | 1.20600 | +0.01657 |
| 0148 | C4 | 4 | 48 | 8(256) | 128 | 64 | 76.37M | 1.20942 | +0.01999 |
| 0149 | A2b | 8 | 24 | 8(256) | 128 | — (no penalty) | 85.84M | 1.18203 | −0.00740 |
| 0150 | A0 | 1 | 192 | 8(256) | 128 | 192 | 69.27M | 1.20011 | +0.01068 |

**Best of the sweep: B4 (tph256) = 1.18187**, −0.0076 vs baseline. The top three (tph256, nap10, H8) are
all "more capacity" configs, all beating baseline by −0.005…−0.008.

## Axis findings (all three axes complete)
### Axis A — n_heads × inner_out (H·out=192 fixed): MONOTONE, more/narrower heads win
H2/out96 (1.19583) > H4/out48 (1.18943) > **H8/out24 (1.18483)**. More routing heads (narrower per-head
output) is strictly better — the ternary FFN routes on the **full 384-d input per head**, so more heads =
more parallel routing diversity. **OPPOSITE of the CompressionMHL FFN head line** (H2/d96 best there, H1
worst), where heads route in a compressed 48-d space so over-splitting hurts. (H1/out192 = 0150 will add
the single-head endpoint; H16 untested but the trend says try it.)

### Axis B — nap / tph (LUT addressing): MONOTONE both ways, more capacity → better
- cells (vary nap, tph=128): nap6 (1.19694) > nap8 (1.18943) > **nap10 (1.18296)** — more cells → better.
- tables (vary tph, nap=8): tph64 (1.20600) > tph128 (1.18943) > **tph256 (1.18187)** — more tables → better.
Both point the same way: bigger LUT (deeper cells *and* more tables) keeps improving bpb; tph256 is the best
single point. (Cost: params climb — 123.6M at tph256, 192.0M at nap10.)

### Axis C — target nnz (sparsity): MONOTONE, denser → better; no-penalty best
target=64 (1.20942) → 128 (1.20460) → 192 (1.18943) → 256 (1.18851) → **no penalty (1.18757)**. The
target-density hinge is a **pure quality cost at full 16K**, growing as the target tightens. **This reframes
the 4K "sparsity is free" reading (exp_g_0044): at full anneal it is not free.** The hinge's value is the
deployment-time density it buys (~172 nnz/hp at target-192 vs ~269 unregularised), a compute/quality
tradeoff — but on bpb alone, denser wins.

## Cross-axis & next steps
Every axis rewards *more capacity / less-constrained routing*: more tables, more cells, more heads, no
sparsity penalty. The three axis-winners (tph256, nap10, H8) and the no-penalty setting are all independent
gains — **stacking them (e.g. H8 + tph256/nap10 + no-penalty) is the obvious next config** and likely well
below 1.18. exp_n_0149 (H8 + no-penalty) is the first stacked test (running now). Note the density/compute
tradeoff is orthogonal to bpb; if deployment sparsity matters, the hinge is the knob, at a known bpb cost.

## Comparators (full-anneal 16K)
exp_n_0134 ternary target-192 = 1.18943; untied vanilla 4× MLP (exp_n_0135) = 1.20144; CompressionMHL FFN
anchor (exp_n_0121) = 1.19146. The sweep-best B4 (1.18187) beats all three by a clear margin.
