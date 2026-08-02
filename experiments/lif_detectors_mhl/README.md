# LIFDetectorsMHL — LIF-detector front-end for HyperplaneMultiHeadLUT

`LIFDetectorsMHL` (module: `spiky.lutorch.lif_detectors_mhl`) is a drop-in replacement for the **index
front-end** of `HyperplaneMultiHeadLUT`: it keeps the same multi-head-LUT skeleton (`n_tables =
n_heads·tables_per_head` tables, each with `n_anchor_pairs` index bits selecting one of `2**n_anchor_pairs`
rows of `n_outputs` values, rows summed within a head) but produces each table's index bits with **combined
LIF detectors over latency-coded inputs** instead of affine hyperplane sign-tests, and uses
**straight-through hard addressing** so training and inference use the exact same discrete lookup.

## Combined LIF detector (per detector)

Inputs are latency-coded `t_i = clamp(c − α·x_i, 0, T)` (default `c=16, α=3, T=32`); arrivals `a_i = t_i + d_i`.

```
V_self = Σ_i  w_i · exp(-ReLU(r - a_i)/τ_s) · sigmoid((r - a_i)/ε)          # magnitude / value-range channel
V_pair = Σ_{i≠j} P_ij · exp(-ReLU(a_j - a_i)/τ_p) · sigmoid((a_j - a_i)/ε)  # order / contrast channel
V      = V_self + V_pair
bit    = sigmoid((V - θ) / temp_bit)
```

Per detector trainable: `d, w ∈ R^N`, readout `r`, off-diagonal `P ∈ R^{N×N}` (self-pairs masked), `τ_s,
τ_p` (softplus>0), `θ`; global `temp_bit` (exp>0). The pair channel is initialised near zero (`0.01`) so each
detector starts as a pure value/range unit and grows contrast structure only where it helps. The two
channels are complementary — self=magnitude half-spaces, pair=order/contrast — and the optimiser picks the
per-detector mix.

## Straight-through hard addressing (mirrors the teacher)

The teacher (`HyperplaneMultiHeadLUT`, `forward_mode="hard"`) does a **hard forward** — affine sign-pack
argmax address → `F.embedding_bag(mode='sum')` reads exactly one row per table — with a **soft backward**
(full-K softmax surrogate over the `2**nap` rows). We mirror it with an **output-level straight-through**:

```
prow_soft = Π_k [b_soft if code-bit=1 else 1-b_soft]   # b_soft = sigmoid((V-θ)/temp_bit); NO detach
prow_hard = one-hot at the packed argmax address        # non-differentiable
y_soft = prow_soft @ table ;  y_hard = prow_hard @ table
y = y_soft + (y_hard - y_soft).detach()                 # forward = hard single cell; gradient via y_soft
```

The forward **value** is the hard single cell (a single row read, exactly like the teacher, up to fp
rounding), while the gradient flows entirely through `y_soft`. Bit packing is MSB-first and identical to the
teacher: `addr = Σ_k bit_k·2**(nap−1−k)` (for nap=6, `[32,16,8,4,2,1]`).

**Backward parity is now EXACT.** A product of independent per-bit Bernoullis over the `2**nap` outcomes *is*
the softmax over those cells, so the `y_soft` backward is the **exact full-K softmax over all `2**nap`
cells** — matching `HyperplaneMultiHeadLUT`'s hard-forward / soft-backward exactly (no longer a
"selected-cell only" approximation; gradient reaches the bits of non-selected cells too). Consequence: the
soft training objective and the hard/argmax inference objective **coincide by construction** — no soft-blend
"escape hatch". (An earlier fully-soft-addressing prototype scored soft R² 0.68 but collapsed to hard R²
−0.64 because training exploited row-blending; the straight-through version fixes this.)

## Current distillation result (real Walker2d LUT)

Teacher: the real trained int4 Walker2d LUT-SAC actor (`walker2d_lut_actor_int4.npz`; input_dim=17,
n_heads=1, tables_per_head=32, n_anchor_pairs=6, n_outputs used=6 action means), frozen. Input: standardized
Walker2d obs (`walker_dataset_stats.json`), `x ≈ N(0,1)^17`. Objective: MSE on the 6 action means. Tables
warm-started from the oracle's dequant weights and trained; detector bank + `temp_bit` from scratch. 6000
Adam steps, batch 256, `ε` annealed 2.0→0.3 (~29 min CPU).

| metric (held-out 4096) | value |
|---|---|
| **hard/argmax inference R²** | **0.61** |
| hard RMSE | 0.68 = **7.4% of action range** |
| soft-blend reference R² | 0.51 (now *worse* than hard — escape hatch closed) |
| ST-forward vs hard | identical (metrics coincide) |

Reproduce: `PYTHONPATH=../../src python distill_walker2d.py` (this dir; assets included).

## Discrete-addressing error analysis

Per-bit distillation of the real front-end reaches ~97% agreement per hyperplane bit. But an address is 6
bits, so a table's address is fully correct with probability ≈ `0.97^6 ≈ 0.83` — i.e. on average **~5 of the
32 tables per sample are mis-addressed**, and a single wrong bit jumps to a *different* 64-row cell whose
value is unrelated, injecting error into the summed action. This compounding is why hard R² sits at ~0.61
rather than near the hand-built structural pure-spnet conversion (~0.5% error / R²≈1.0): the structural
version copies the exact hyperplanes as weights, whereas this is a from-scratch *learned* detector bank.
Likely gains: longer training / larger detector budget, a selection-temperature schedule, or a per-bit
auxiliary loss to sharpen the addressing.

## Next step — gpustar trains Walker2d from scratch with LIFDetectorsMHL

`LIFDetectorsMHL` mirrors the `HyperplaneMultiHeadLUT` constructor/`forward(x)->(B, n_heads, n_outputs)`
signature, so it substitutes into the Walker2d LUT-SAC training code with minimal changes: build it with the
same `(input_dim, n_heads, n_outputs, n_anchor_pairs, tables_per_head)` and train the actor end-to-end (no
oracle) with `mode="st"`. Because addressing is straight-through hard, the trained actor runs at inference in
the same discrete regime it was trained in.
