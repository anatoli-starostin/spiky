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

Trainable per **detector**: `d, w ∈ R^N`, off-diagonal `P ∈ R^{N×N}` (self-pairs masked). Trainable per
**table** (one value per LUT, broadcast across its `nap` detectors): readout `r`, time constants `τ_s, τ_p`
(softplus>0), threshold `θ`. Plus one global `temp_bit` (exp>0). (`r/τ_s/τ_p/θ` were per-detector originally;
sharing them per-table cut their freedom from `n_tables·nap` to `n_tables` — 75,073→74,433 params — with
equal-or-better fidelity, since `P` at 55,488 dominates the parameter mass.) The pair channel is initialised
near zero (`0.01`) so each detector starts as a pure value/range unit and grows contrast structure only where
it helps. The two channels are complementary — self=magnitude half-spaces, pair=order/contrast.

## Straight-through hard addressing (mirrors the teacher)

The teacher (`HyperplaneMultiHeadLUT`, `forward_mode="hard"`) does a **hard forward** — affine sign-pack
argmax address → `F.embedding_bag(mode='sum')` reads exactly one row per table — with a **soft backward**
(full-K softmax surrogate over the `2**nap` rows). We mirror it with FastMHL's **decoupled straight-through**
— the two gradients are routed separately:

```
prow_hard = Π_k [hard_bit or 1-hard_bit]   # one-hot at the packed argmax address (non-differentiable bits)
prow_soft = Π_k [b_soft or 1-b_soft]        # b_soft = sigmoid((V-θ)/temp_bit) = softmax over the 2**nap cells
y_hard = prow_hard @ table                  # WEIGHT gradient -> only the selected row per table updates
y_addr = prow_soft @ table.detach()         # ADDRESS/detector gradient -> full-K softmax; table detached (no weight grad)
y = y_hard + y_addr - y_addr.detach()       # forward VALUE == hard single cell; address grad injected, its value cancelled
```

The forward **value** is the hard single cell (a single row read, exactly like the teacher). The **table**
gradient follows the hard forward — only the argmax-selected row per table updates (like `embedding_bag` + a
1-row scatter) — while the **detector/address** gradient follows the full-K softmax over all `2**nap` cells.
This matches `FastMHL` / `HyperplaneMultiHeadLUT` hard-mode (hard weight scatter + full-K softmax address
backward). Bit packing is MSB-first and identical to the teacher: `addr = Σ_k bit_k·2**(nap−1−k)`
(for nap=6, `[32,16,8,4,2,1]`).

**Why `table.detach()` in the address term is the crux:** without it (the earlier non-decoupled variant
`y = y_soft + (y_hard − y_soft).detach()` with `y_soft = prow_soft @ table`) the soft blend smears the weight
gradient across ALL `2**nap` rows, the table absorbs the objective via a near-uniform blend, the bits never
sharpen (stuck ~0.5, temp_bit ~1.0), and hard-inference collapses to R²≈0.14. Decoupling lets the selected
row track the target while the detectors sharpen (temp_bit shrinks, per-bit logits grow), recovering the
bit-level track. (An even earlier fully-soft-addressing prototype scored soft R² 0.68 but collapsed to hard
R² −0.64 for the same row-blending reason.)

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

Reproduce: `PYTHONPATH=../../src python distill_walker2d.py [--steps 6000] [--save PATH]` (this dir; assets
included). `--save` writes the trained student (`state_dict` + `config` + `metrics`) to a checkpoint;
training uses a constant LR (a cosine anneal was evaluated and dropped — it scored slightly worse).

## Discrete-addressing error analysis

Per-bit distillation of the real front-end reaches ~97% agreement per hyperplane bit. But an address is 6
bits, so a table's address is fully correct with probability ≈ `0.97^6 ≈ 0.83` — i.e. on average **~5 of the
32 tables per sample are mis-addressed**, and a single wrong bit jumps to a *different* 64-row cell whose
value is unrelated, injecting error into the summed action. This compounding is why hard R² sits at ~0.61
rather than near the hand-built structural pure-spnet conversion (~0.5% error / R²≈1.0): the structural
version copies the exact hyperplanes as weights, whereas this is a from-scratch *learned* detector bank.
Likely gains: longer training / larger detector budget, a selection-temperature schedule, or a per-bit
auxiliary loss to sharpen the addressing.

## Variant: PureLIFDetectorsMHL (pure time-to-first-spike)

A leaner sibling (`spiky/src/spiky/lutorch/pure_lif_detectors_mhl.py`, tests
`test_pure_lif_detectors_mhl.py`, harness `distill_walker2d_ttfs.py`) with **no ordered-pair `P` block**.
Each detector is a single LIF cell read out by its **time of first spike**:

- arrival `a_i = latency(x_i) + delay[d,i]`; membrane at the k-th (ascending-sorted) arrival
  `V_k = Σ_{j: a_j≤a_k} w[d,j]·exp(-(a_k − a_j)/τ)` (O(N²); exponents ≤0 so stable);
- hard first-spike time `t* =` earliest arrival with `V_k ≥ θ_mem` (fixed 1.0), else `t_window`; smooth
  first-success gives the soft `t*`;
- **flipped detection** `s = L − t*` (an *early* spike = detected): `bit = 1[t* < L]`,
  `soft_bit = sigmoid(s/temp_bit)`.

Reuses the **same decoupled hard-forward/soft-backward straight-through + LUT decode** as `LIFDetectorsMHL`
(winning-cell-only table gradient; full-K softmax address gradient with `table.detach()`).

**Params: 19,104** (~26% of the P-equipped model's 74,433) — `delay (192,17)`, `w (192,17)`, `L (192,)`
per-detector; per-LUT `tau_raw/log_T_cross/log_temp_bit (32,)`; `table (32,64,6)`.

**Distillation (same recipe, constant LR, 6000 steps):** hard-inference **R² 0.402, RMSE 0.840 = 9.19% of
range** — vs the per-table LIF (with `P`) **R² 0.646 / 7.07%**. So TTFS reproduces the sharp int4-LUT
*worse* (−0.24 R²) at ~4× fewer params; the dropped ordered-pair channel was carrying real fidelity (bits
still sharpen hard, `temp_bit` → ~0.065). Distilling *against a sharp LUT* structurally favors the richer
`P` model; a from-scratch RL actor may trade differently.

**Backward audit (float64, all 8 checks pass):** hard-forward invariant exact (`max|y_st−y_hard|=1.4e-17`);
table gradient = winning cell only (address-only path gives `max|table.grad|=0`); detector gradient = the
full-weight soft first-spike surrogate (all inputs incl. off-winner arrivals, plus per-LUT `τ/T_cross/temp`);
`gradcheck` of the soft first-spike and the soft-mode forward pass; no NaN/Inf; correct flipped-sign.

## Variant: BucketLIFDetectorsMHL (single neuron per table, time-bucket address)

The leanest sibling (`spiky/src/spiky/lutorch/lif_multi_head_lut.py` (n_det=1), tests
`test_lif_multi_head_lut.py`, harness `distill_walker2d_bucket.py`). Instead of `nap` bit-detectors +
`P`, each table has **exactly one LIF neuron**, and the table row is **which time bucket its first spike
lands in** (default `M=16` buckets):

- first-spike front-end identical to `PureLIFDetectorsMHL` (sorted-arrival causal membrane, hard/soft
  first-spike time);
- **trainable, strictly-increasing** per-LUT bucket boundaries via `boundaries = beta_base +
  cumsum(softplus(beta_raw))` (shape `(n_tables, M−1)`);
- soft address is a **partition of unity** built from boundary sigmoids `S_k = sigmoid((t−b_k)/T_bkt)`
  (`g_0=1−S_0`, `g_m=S_{m−1}−S_m`, `g_last=S_{M−2}`, `Σ_m g_m = 1`); hard address is
  `m* = (t ≥ boundaries).sum()` (searchsorted). `t_window` sits above all initial boundaries, so a
  non-crossing neuron (`t*=t_window`) folds into the **last** bucket automatically.

Reuses the **same decoupled straight-through LUT decode** (winning-bucket-only table gradient via `y_hard`;
full soft-partition address gradient via `y_addr = g @ table.detach()`; forward == hard).

**Weights — bounded excitatory (default).** The single per-table synapse vector is `w = w_max·sigmoid(w_raw)`
(`0 ≤ w ≤ w_max`, `w_max = 2`), with a **hot init** `w_raw ~ N(-2.2, 0.5)` so neurons spike at step 0
instead of ~97% collapsing into the no-spike bucket. This excitatory-only bounded form beat free-signed
weights (ablation below) and is the current default.

**Params: 4,768** at the Walker2d config (`M=16`, `n_tables=32`) — `delay/w_raw (32,17)`, per-LUT
`tau_raw/log_T_cross/log_T_bkt/beta_base (32,)`, `beta_raw (32,15)`, `table (32,16,6)`.

**Distillation** (frozen int4 LUT teacher, constant LR, 6000 steps; the 16-row table is learned from scratch
since 16 buckets ≠ the oracle's 64 rows). With the default bounded-excitatory hot init, **M=16 / T=32 reaches
hard R² ≈ 0.418** (RMSE 0.828, 9.06% of range) — the best bucket-variant result, beating free-signed weights
(0.387) and edging past `PureLIFDetectorsMHL` (0.402); still below the `P`-equipped model (0.64). The trained
bank is **inherently sparse**: ~64% of synapses (`w < 0.05·w_max`) prune to exactly zero with ~no accuracy
loss (pruned R² 0.417), and each neuron effectively integrates ~6 of its 17 inputs.

**What moved the needle (ablations):**
- *Weight-init temperature is the real lever.* The original cold signed init (`0.2·randn`) left ~94% of samples
  in the no-spike bucket at step 0, wasting early training just learning to spike. Draining that collapse via a
  hotter init lifts fidelity: signed `w=2.0` → R² 0.317→0.387; bounded excitatory hot → 0.418.
- *Bounded excitatory > free-signed* (0.418 vs 0.387) at equal params — and the bound never even saturates
  (no `w` reaches `w_max`); the gain is the excitatory constraint + hot init, not the ceiling.
- *Capacity: more tables ≫ more buckets* (measured under the earlier cold init): 4× buckets → +0.008 R²
  (M=64/T=32), 4× tables → +0.070 (M=16/T=128, 19,072 params). Front-end neuron count is the lever, not
  bucket resolution.
- *A bistable double-well weight penalty `λ·mean(w·(w_max−w))` was tried and NOT adopted:* it only pushed
  weights toward 0 (never toward `w_max` — one-sided), cost R² (−0.045 at λ=0.15), and didn't improve the
  already-clean pruning.

Harness CLI knobs (uncommitted `.pt` checkpoints, gitignored): `--buckets M` sets the bucket count,
`--tables T` overrides the **student**'s table count (the oracle keeps its own 32), e.g.
`python distill_walker2d_bucket.py --buckets 16 --tables 128 --save bucket_lif_ttfs_student_t128.pt`.

## Next step — gpustar trains Walker2d from scratch with LIFDetectorsMHL

`LIFDetectorsMHL` mirrors the `HyperplaneMultiHeadLUT` constructor/`forward(x)->(B, n_heads, n_outputs)`
signature, so it substitutes into the Walker2d LUT-SAC training code with minimal changes: build it with the
same `(input_dim, n_heads, n_outputs, n_anchor_pairs, tables_per_head)` and train the actor end-to-end (no
oracle) with `mode="st"`. Because addressing is straight-through hard, the trained actor runs at inference in
the same discrete regime it was trained in.
