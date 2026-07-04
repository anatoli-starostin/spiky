# Spiky Library — Summary

**Repository:** github.com/anatoli-starostin/spiky
**Author:** Anatoly Starostin
**Purpose:** CUDA-enabled, PyTorch-compatible library implementing differentiable lookup tables for spike polychronisation, inspired by the Spiking Manifesto (Izhikevich, 2025).

The library spans three levels of abstraction, from biological spiking simulation to differentiable lookup tables to permutation-based neural network layers.

---

## 1. SpNet — Spiking Neural Network Simulation

`src/spiky/spnet/spnet.py`

Implements an Izhikevich spiking neural network with STDP learning. This is the biological foundation — not differentiable, not used for training via backprop, but provides the conceptual grounding for the entire project.

### Neuron Model (Izhikevich)
- Membrane voltage: v' = cf_2·v² + cf_1·v + cf_0 - u + I
- Recovery variable: u' = a(b·v - u)
- When v ≥ threshold: spike, reset v → c, u → u + d
- Supports excitatory and inhibitory neuron types with different parameters

### Synaptic Dynamics
- Variable delays (0–255 timesteps) — critical for polychronisation
- STDP learning: exponential decay traces updated per spike
- Long-term potentiation (LTP) and depression (LTD)
- Weight bounds and decay

### SpikeCore (SpikingNet class)
- `add_connections()`: Build network from ChunkOfConnections (sparse connection format)
- `compile()`: Finalise structure, initialise CUDA kernels
- `process_ticks()`: Simulate network with optional STDP
- Spatial neuron groups with SIMD-aligned storage

### Polychronisation Connection
The variable delays are what makes polychronisation possible: specific input spike timing patterns activate specific downstream neurons. The number of such polychronous groups can exceed the number of synapses — this is the "factorial capacity" that the Spiking Manifesto emphasises.

---

## 2. LUTorch — Differentiable Lookup Tables

The core contribution. Abstracts away biological neuron dynamics and focuses on the essential computation: **pairwise comparisons → binary index → table lookup**.

### AnchorPairsLookup (`anchor_pairs_lookup.py`)

The fundamental operation: input vector x → binary table index.

For each table with nap anchor pairs (a_k, b_k):
1. delta_k = x[a_k] - x[b_k]
2. bit_k = 1 if delta_k > cmp_eps, else 0
3. index j = Σ bit_k × 2^k ∈ {0, ..., 2^nap - 1}

**Surrogate gradients:** sign() has zero gradient everywhere. Solved via uncertainty functions:
- INVERSE_L1: U(delta) = 0.5 / (1 + |delta|)
- INVERSE_QUADRATIC: U(delta) = 0.5 / (1 + delta²)

Large uncertainty near decision boundaries (|delta| ≈ 0), vanishing for confident comparisons. Each table contributes equal-magnitude, opposite-sign gradients to its two anchor neurons — self-regularising with zero-mean gradients.

**n_alternatives:** Identifies the closest alternative index (flip the least-confident bit) for smooth interpolation between adjacent table entries.

**Memory:** Recompute-in-backward mode saves ~264 MB per block by recomputing indices from small anchor buffers instead of caching.

### LProjection (`l_projection.py`)

Gathers weight vectors from tables using lookup indices.

- Weight shape: [n_tables, 2^nap, n_outputs]
- **Non-smooth mode:** Direct index → sparse, fast, piecewise-constant
- **Smooth mode:** Blend main entry (weight 1-ΣU) with alternatives (weighted by normalised uncertainties) — gradient flow through continuous routing

Optional: weight column normalisation, output calibration.

### MultiHeadLut (`multi_head_lut.py`)

Main user-facing module combining AnchorPairsLookup + LProjection:

```
Input [B, input_dim]
  → AnchorPairsLookup → indices [B, n_tables]
  → LProjection → [B, n_tables, n_outputs]
  → Sum over tables_per_head → [B, n_heads, n_outputs]
```

Key parameters:
- `n_anchor_pairs` (nap): Bits per index → table size = 2^nap
- `tables_per_head` (tph): Tables summed per head
- `n_buckets`: Effective table size multiplier (for positional encoding)
- `anchor_sampling_policy`: Strategy for selecting comparison pairs

Fused `MultiHeadLutFunction` uses `F.embedding_bag` to avoid materialising the full [B, n_tables, n_outputs] intermediate.

### Anchor Sampling Policies (`lut_helpers.py`)

Controls which input dimensions each table compares:

| Policy | Description |
|--------|-------------|
| BALANCED | Random permutations, balanced dimension coverage |
| CONNECTED | Pairs share consecutive indices |
| DISCONNECTED | No shared indices within a table |
| HIERARCHICAL | Multi-scale with exponential distances |
| MULTISCALE | All integer distances |
| CONV2D | 3×3 dilated kernel on 2D grid |
| FULL_COVERAGE | All unique pairs from upper triangle |
| DISCONNECTED_FULL_COVERAGE | Greedy resampling for complete coverage |

FULL_COVERAGE is key for small input_dim (e.g. 32): covers all C(d,2) possible comparisons.

---

## 3. Ranking Tools — Permutation-Based Representations

`src/spiky/lutorch/ranking_tools.py`

Implements permutation/rank-based representations for neural computation. The bridge between biological spike ordering and practical differentiable layers.

### Core Idea
Represent input features through **relative ordering** (which coordinate is bigger) rather than magnitudes. A rank projection maps x ∈ R^d to M binary comparisons {x[a_i] > x[b_i]} — capturing the permutation structure.

### RankProjection
Linear projection on top of rank features:
- Sample M pairs from upper triangle of input_dim
- Project: `soft: d/(temp + |d|) ∈ (-1, 1)` or `hard: sign(d) ∈ {-1, 1}` with STE
- Apply learned linear layer on rank features → output

Replaces dense linear layers with permutation-based sparse computation.

### RankAttention
Rank-based attention replacing standard scaled dot-product:
- Q and K projected through pairwise comparisons instead of linear projections
- Rank features used as Q/K for standard SDPA
- Temperature controls sharpness of soft comparisons

### RankWTAAttention
Winner-Take-All variant:
- Materialises score matrix: scores = RQ · RK^T
- WTA selection of winning key per query
- Differentiable via uncertainty-weighted gradient carriers

### PairVoting
Each comparison pair has a learnable output vector:
- Rank feature {-1, +1} selects which direction to vote
- Sum of votes → output
- No cross-pair mixing; each pair votes independently

### PositionalPermutation
Fixed random permutation per sequence position — non-learnable positional encoding:
- Permutes input dimensions before processing
- Changes which dimensions get compared for different positions

### LearnedSoftPermutations
Learnable soft permutations via softmax over assignment matrices:
- Parameters: scores [n_perms, n, n] → softmax → doubly-stochastic matrix
- Hard/soft modes via argmax + STE

### Utility: add_rank_preserving_noise()
Adds noise bounded by minimum gap between sorted elements — preserves relative ordering while perturbing values. Useful for data augmentation that respects permutation structure.

---

## 4. HyperLUT — Differentiable LUT Replacement

`src/spiky/lutorch/hyper_lut.py`

Replaces the discrete table lookup with a continuous MLP approximation:

1. Each head samples its own n_pairs from upper triangle
2. Pairwise comparisons: sign(x[a] - x[b]) → {0, 1} (hard forward, soft backward via STE)
3. Per-head fc1: [n_pairs → hidden_dim] via block-diagonal einsum
4. Optional LayerNorm
5. ReLU
6. fc2: [hidden_dim → n_outputs]

**HyperLUTBackbone:** Extractable shared backbone (pairs + fc1 + activation). Multiple output heads share a backbone, paying for comparisons and fc1 once.

Key advantage: MLP sees all M comparisons simultaneously (vs MultiHeadLut where each table sees only its nap pairs independently). Avoids 2^nap exponential table size.

---

## 5. Attention Modules

`src/spiky/lutorch/lut_attention.py`

### LUTAttention (V1)
Cross-attention using MultiHeadLut with n_outputs=1 for scoring.
- Pair representation: c1·x_i + c2·x_j (linear combination) or [x_i, x_j] (concatenation)
- Softmax over keys → attention weights
- Supports causal masking and relative positional buckets

### LUTAttentionV2
Pairwise attention with explicit relative positional encoding.
- Concatenates [x_i, x_j, RelPE[dist(i,j)]] → MultiHeadLut
- Scatter-add outputs by query position → [B, T, H, O]
- Self-excitement: y = f × mean(|f|) — amplifies strongly-activated pairs
- Fused CUDA kernel for non-smooth, n_alternatives=1

### LUTAttentionV3
Raw attention score generator — best-performing variant.
- Same pair concatenation as V2
- Returns dense [B, T, T, H, O] with -inf masking
- Used with standard softmax + matmul: softmax(scores) @ V
- Separates "what to attend to" (LUT) from "how to combine" (softmax@V)

---

## 6. WTA Modules

`src/spiky/lutorch/wta_lookup.py`, `multi_head_wta.py`

Winner-Take-All: alternative indexing via topk selection instead of pairwise comparisons.
- Finds argmax (winner) + runner-ups per channel
- Returns deltas = winner_value - runner_up_value
- Uncertainty-weighted gradient propagation
- Used in RankWTAAttention

---

## 7. Spatial / 2D Wrappers

### ProjectionLUT
Applies MultiHeadLut to 2D [B, H, W] inputs via patch extraction with unfold.

### Conv2DLut
4D convolution-style [B, C, H, W] → patches → shared MultiHeadLut.

---

## 8. Native CUDA Backend

`native/lutorch/lutorch.cu`

Performance-critical CUDA kernels:
- `anchor_pairs_lookup_forward_na1_kernel`
- `anchor_pairs_lookup_backward_all_kernel`
- `lprojection_backward_na1_carriers_kernel`
- `lprojection_backward_na1_nonsmooth_weights_kernel`
- `lut_attn_fwd_na1` / `lut_attn_bwd_na1` (fused LUTAttentionV2)

Managed by `LUTorchManager` singleton, controlled by `_USE_LUTORCH_CUSTOM_CUDA_KERNELS` flag.

Also: `native/common/misc.cpp` and `native/spiky/misc/misc.cpp` for CUDA context management (with CUDA 13 compatibility).

---

## Unified Conceptual Thread

All three layers leverage **comparison-based representations**:

| Level | What is compared | Representation | Training |
|-------|-----------------|----------------|----------|
| SpNet | Spike timings | Polychronous groups | STDP (biological) |
| RankingTools | Input dimensions | Permutation features | STE + backprop |
| MultiHeadLut | Anchor pairs | Binary table indices | Surrogate gradients |
| HyperLUT | Anchor pairs | Binary features → MLP | STE + backprop |

The information is always in the **relative ordering** — which element is bigger than which — not in absolute magnitudes. This gives factorial encoding capacity (n!) vs linear capacity of magnitude-based representations.
