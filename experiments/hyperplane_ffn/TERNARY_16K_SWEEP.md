# Ternary FFN — 16K star / one-axis-at-a-time sweep (PARTIAL — for the paper)

**⚠️ PARTIAL RESULTS.** 7 of 10 runs complete; **B2 (nap10), B3 (tph64), B4 (tph256) still pending**
(B2 in-flight ~81%). This file will be updated when the B axis finishes. Do not treat the B-axis
conclusions as final.

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
| **0143** | **A2** | **8** | **24** | 8(256) | 128 | 192 | 85.84M | **1.18483** | **−0.00460** ⭐ best |
| 0141 | C3 | 4 | 48 | 8(256) | 128 | — (λ=0, no penalty) | 76.37M | 1.18757 | −0.00186 |
| 0140 | C2 | 4 | 48 | 8(256) | 128 | 256 | 76.37M | 1.18851 | −0.00092 |
| **0134** | baseline | 4 | 48 | 8(256) | 128 | 192 | 76.37M | 1.18943 | 0 |
| 0142 | A1 | 2 | 96 | 8(256) | 128 | 192 | 71.64M | 1.19583 | +0.00640 |
| 0144 | B1 | 4 | 48 | 6(64) | 128 | 192 | 45.70M | 1.19694 | +0.00751 |
| 0139 | C1 | 4 | 48 | 8(256) | 128 | 128 | 76.37M | 1.20460 | +0.01517 |
| 0148 | C4 | 4 | 48 | 8(256) | 128 | 64 | 76.37M | 1.20942 | +0.01999 |
| 0145 | B2 (pending) | 4 | 48 | 10(1024) | 128 | 192 | 191.98M | *~1.1934 @ step 12.8k (running)* | — |
| 0146 | B3 (queued) | 4 | 48 | 8(256) | 64 | 192 | 52.77M | queued | — |
| 0147 | B4 (queued) | 4 | 48 | 8(256) | 256 | 192 | 123.59M | queued | — |

**Best so far: A2 (H8/out24) = 1.18483**, −0.0046 vs baseline.

## Axis findings
### Axis A — n_heads × inner_out (product fixed at H·out=192): MONOTONE, more/narrower heads win
H2/out96 (1.19583) > H4/out48 (1.18943) > **H8/out24 (1.18483)**. More routing heads (narrower per-head
output) is strictly better. Because the ternary FFN routes on the **full 384-d input per head**, more heads
= more parallel routing diversity. **This is the OPPOSITE of the CompressionMHL FFN head line** (there H2/d96
was the sweet spot and H1 the worst), where each head routes in a compressed 48-d space so over-splitting
hurts. Different mechanisms, opposite optima. (H16 untested — the trend suggests it's worth trying.)

### Axis C — target nnz (sparsity): MONOTONE, denser → better; no-penalty best
target=64 (1.20942) → 128 (1.20460) → 192 (1.18943) → 256 (1.18851) → **no penalty (1.18757)**. The
target-density hinge is a **pure quality cost at full 16K**, growing as the target tightens. **This reframes
the 4K "sparsity is free" reading (exp_g_0044): at full anneal it is not free** — pure ternary (no hinge)
is the best on bpb, and every step of enforced sparsity costs more. (The value of the hinge is the
deployment-time density it buys, e.g. ~172 nnz/hp at target-192 vs ~269 unregularised — a compute/quality
tradeoff, not a bpb win.)

### Axis B — nap / tph (LUT addressing): PARTIAL
nap6/tph128 (64 cells, 1.19694) is worse than baseline (fewer cells hurts). **nap10 (1024 cells), tph64,
tph256 pending** — will show whether more cells / more-or-fewer tables continue monotonically. (Early
in-flight B2/nap10 is tracking ~1.1934 at step 12.8k, likely to land near or below baseline.)

## Cross-axis note
A2 (H8, still *with* the target-192 hinge) beats C3 (H4, *no* penalty). So the head-count gain (Axis A)
outweighs the sparsity-penalty cost (Axis C). The untested combination **H8 + no-penalty** is the most
likely overall optimum and is the natural next run.

## Comparators (full-anneal 16K)
exp_n_0134 ternary target-192 = 1.18943; untied vanilla 4× MLP (exp_n_0135) = 1.20144; CompressionMHL FFN
anchor (exp_n_0121) = 1.19146. The sweep-best A2 (1.18483) beats all three.
