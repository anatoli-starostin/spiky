# LUT Transformer Experiments — Structure & Logic

## Overarching Goal

Take a small vanilla transformer (~5M params) trained on FineWeb byte-level data and achieve comparable validation loss using a model based on **permutation embeddings** (relative ordering) instead of **magnitude embeddings** (real-valued vectors), replacing matrix multiplications with lookup tables or equivalent permutation-aware operations.

Success criteria: comparable loss, comparable parameter count, or at least lower bandwidth than the vanilla model.

---

## 1. Experimental Setup

- **Dataset:** Subset of FineWeb (standard web text dataset)
- **Tokenisation:** Byte-level, VOCAB_SIZE = 257 (256 bytes + BOS token)
- **Context:** CONTEXT_SIZE = 32 tokens
- **Hardware:** NVIDIA H100 80GB
- **Metrics:**
  - Validation loss (cross-entropy, on 10k held-out test regions)
  - Number of weights (parameter count)
  - Virtual bandwidth (memory access per forward pass)
- **Training:** Adam optimiser, cosine schedule with warmup (10% warmup, decay to 10%), batch sizes 16–256 depending on experiment
- **Transparency:** All hyperparameters, architectures, and training details fully specified

## 2. Vanilla Transformer Baseline

Standard transformer: d_model=256, 4 heads, 6 layers, FFN 4×, no dropout.

| Exp | lr | Val Loss | Params |
|-----|-----|----------|--------|
| exp001 | 1e-4 | 1.612 | 4.87M |
| **exp002** | **1e-3** | **1.356** | **4.87M** |
| exp003 | 1e-3 + wd | 1.355 | 4.87M |

**Target: val_loss = 1.356** (exp002, 100K steps). Weight decay negligible effect.

## 3. The Ideas Behind LUT Transformers

### What are we trying to do?

Vanilla transformer operates on magnitude-based high-dimensional embeddings (d=256) using matrix multiplications. We replace:
- **Embeddings:** Low-dimensional permutation embeddings (d=32, information in relative ordering)
- **Operations:** Lookup tables instead of matrix multiplications

### Data flow similarity

We are NOT reinventing sequence modelling. The transformer structure is kept intact:
- Sequences of token embeddings
- Layers with residual connections
- Attention scores + softmax
- Unembedder returning logits over vocabulary

The ONLY change is the **basic language** (permutations vs magnitudes) and the **basic mechanism** (LUT lookup vs matmul).

### Positional embeddings in LUT transformers

A non-trivial challenge with low-dimensional permutation embeddings:
- Simple addition with the input doesn't work well at small dimensions (destroys ordering information)
- Approaches tried: positional buckets, relative positional embeddings, fixed positional embeddings
- Summation vs concatenation trade-off (concatenation preserves ordering but increases input_dim)
- **Key finding (exp009–011):** Concat PE > additive PE > no PE. Dedicating separate dimensions to position prevents routing interference.

### nap and tph — the two fundamental LUT parameters

- **nap** (n_anchor_pairs): Bits per table index → table size = 2^nap. Controls expressiveness per table.
- **tph** (tables_per_head): Number of tables summed per head. Controls capacity.
- Different anchor pair sampling policies (BALANCED, FULL_COVERAGE, etc.)

**nap/tph tradeoff (exp059–063, ~5M params budget):**

| nap | tph | entries/table | val_loss |
|-----|-----|---------------|----------|
| 7   | 32  | 128           | 1.827    |
| 5   | 128 | 32            | 1.729    |
| **4** | **256** | **16**    | **1.706** |
| 3   | 512 | 8             | 1.706    |
| 2   | 1024| 4             | 1.719    |
| 1   | 2048| 2             | 1.965    |

**Key finding:** Many small tables (low nap, high tph) work better than few big tables in terms of loss. nap=3–4 is the sweet spot. But big tables are much faster due to lower bandwidth — fewer table lookups needed.

**Anchor sampling policies (exp064–069):**

| Policy | Val Loss |
|--------|----------|
| random | 1.706 |
| connected | 1.721 |
| disconnected | 1.687 |
| **full_coverage** | **1.684** |

**FULL_COVERAGE wins:** Ensuring all unique pairs are covered across tables gives consistent improvement.

## 4. [PLACEHOLDER: LUT Mechanism Details]

*Detailed description of anchor pair comparison, binary indexing, table lookup, surrogate gradients. Referenced from Spiking Manifesto.*

## 5. Simple LUT Transformer

Two closely related models with the same core idea: replace all projections with LUTs.

### LUT Transformer Baseline (exp004)
- Uses **positional buckets** for relative positional encoding
- LUTAttention (V1) for attention computation
- 191M params → val_loss = 1.463

### LUTTransformerV3 (exp173–184)
- Uses **explicit relative positional embeddings** (RelPE)
- LUTAttentionV3: generates raw scores [B, T, T, H, 1], applies standard softmax + matmul
- Separate value LUT and out-projection LUT

**V3 evolution:**

| Exp | Config | Params | Val Loss |
|-----|--------|--------|----------|
| exp173 | nap=4, tph=256, d_v=16 | 3.26M | 1.549 |
| exp177 | all nap=6, op_tph=512 | 9.65M | 1.446 |
| **exp184** | **v_tph=256, op_nap=5, op_tph=768, 100K** | **6.51M** | **1.441** |

**Key findings from V3 sweeps:**
- **Out_proj is the most important component** — dominates every budget allocation sweep
- Exclusion sets (prohibiting within-group pairs): didn't help

### The main computational problem

To compute attention scores with LUTs, one must materialise ALL ⟨i, j⟩ embedding pairs and apply the LUT to each pair. For context T, that's O(T²) LUT calls with concatenated input [x_i, x_j, RelPE].

This is computationally heavy on GPU. However, with specialised hardware (only comparators + table reads), this particular operation can be strongly optimised. For research purposes this makes it hard to run many experiments due to wall-clock time.

### LUTAttentionV2 and Self-Excitement (exp161–164)

An intermediate architecture where LUT directly produces per-pair outputs (no separate V/softmax):
- Self-excitement: y = f × mean(|f|) — amplifies strongly-activated pairs
- val_loss = 1.519 at 3.2M params — self-excitement gave ~0.1 improvement at zero parameter cost

## 6. Ranking Attention LUT Transformer

**The most promising branch of this research.**

### The Key Idea: Kendall Tau via Cosine Similarity

Ranking attention projects input through pairwise comparisons into a binary feature space. In this space:
- Each input is represented by its **rank signature** — a vector of pairwise comparison results
- **Kendall tau correlation** between two inputs' orderings is proportional to the **cosine similarity** of their rank signatures
- Standard scaled dot-product attention on rank features computes permutation-based similarity

This allows using efficient SDPA while operating on permutation features.

### Architecture (LUTRankAttnV2, exp197+)

Per block:
- Q, K projections: LUT on (input + positional embedding), output d_qk per head
- V projection: LUT on input, output d_v per head
- Attention: standard SDPA(Q, K, V) with causal mask
- Out-projection: LUT mapping H×d_v → E
- Optional FFN: LUT mapping E → E
- Residual + LayerNorm

### Results

**Ranking Attention evolution:**

| Exp | Description | Val Loss | Params | Steps |
|-----|-------------|----------|--------|-------|
| exp020 | First RankAttention | 1.568 | 46.2M | 50K |
| exp024 | Smooth LUT + smooth RankAttn | 1.437 | 46.2M | 50K |
| exp084 | d_v=16 redesign | 1.585 | 7.55M | 50K |
| exp197 | RankAttn V2, clean design | 1.505 | 6.3M | 50K |
| exp198 | + LayerNorm + op_tph=768 | 1.472 | 7.9M | 100K |
| exp200 | Q/K nap=5 | 1.506 | 12.6M | 100K |
| exp207 | Optimal: tiny Q/K, nap6 OP, FFN | **1.408** | 17.7M | 100K |
| exp208 | + pos_dim=32, nap6 V | 1.426 | 8.3M | 100K |
| exp209 | d_qk=8 variant | 1.444 | 8.1M | 50K |

**Best ranking attention result: exp207, val_loss = 1.408, 17.7M params.**
Gap to vanilla: **0.052** — very close.

**Key finding from exp207:** Tiny Q/K (nap=3, tph=64), heavy out_proj (nap=6, tph=768) and heavy FFN (nap=5, tph=768) with GELU unembedder.

### Scaling with nap (more entries per table)

| Exp | nap | tph | Val Loss | Params |
|-----|-----|-----|----------|--------|
| exp216 | 5 | 128 | 1.513 | 4.0M |
| exp218 | 5 | 256 | 1.507 | 7.9M |
| exp225 | **10** | **128** | **1.381** | **125.9M** |
| exp226 | 10 | 128, 100K | 1.381 | 125.9M |
| exp232 | 12 | 128 | 1.405 | ~250M |

**exp225/226 (nap=10): val_loss = 1.381 — BEATS vanilla baseline (1.356 gap = 0.025)!**
But at enormous parameter cost: 125.9M (26× vanilla). nap=12 is worse — overfitting.

### The remaining problem: number of weights

The best results are very close to (or approach) vanilla, but at much higher parameter counts. The weight tables are the bottleneck:
- exp207 (1.408): 17.7M params (3.6× vanilla)
- exp225 (1.381): 125.9M params (26× vanilla)

This motivates HyperLUT.

## 7. HyperLUT — Overcoming the Weight Problem

### Origin

Proposed by **Vyacheslav Kluchnikov** who studied the field and suggested a different approach.

### The Idea

Instead of allocating many LUT tables, map directly from binary comparison space to output using classic deep learning:
- **Input:** M ≤ N×(N-1)/2 binary comparison features (hard forward, soft backward via STE)
- **Architecture:** 2-layer MLP with wide hidden layer and ReLU
- **Output:** Per-head output vector

### Why This is Interesting

1. **Still permutation-based:** By design, the only thing HyperLUT sees is the order of elements — binary comparison results
2. **Exploits GPU matmul:** The MLP uses matrix multiplications that are fast on modern hardware
3. **Best of both worlds:** Permutation semantics with efficient GPU execution
4. **Better understanding:** HyperLUTs help understand the limits of permutational embeddings

### Results So Far

| Exp | Description | Val Loss | Params | Steps |
|-----|-------------|----------|--------|-------|
| exp193 | HyperLUT V3 softmax, hid=64 | 1.657 | 2.3M | 25K |
| exp194 | HyperLUT, wider, rational | 1.611 | 4.3M | 25K |
| exp199 | HyperLUT + RankAttn | 1.513 | 4.3M | 100K |
| exp227 | HyperLUT transformer | 1.462 | ~15M | 50K |

### Current Status

Preliminary results promising but no HyperLUT model yet on par with vanilla baseline. The best HyperLUT result (1.462) is behind the best LUT result (1.381) but uses far fewer parameters.

### Future Direction: HyperLUT → LUT Distillation

An interesting open question: can we **distil trained HyperLUTs into standard LUT tables** without training or with minimal calibration?

If so, the workflow becomes:
1. Train with HyperLUTs (fast, differentiable, GPU-friendly)
2. Convert to LUT tables (low-bandwidth inference on specialised hardware)

This would be an algorithm to find the optimal set of LUTs that substitutes a given HyperLUT.

### Broader Perspective

HyperLUTs are one approach to trainable mappings between permutations. Other approaches may exist:
- Reservoir computing with classical spiking networks inside
- Other algebraic structures on the permutation group
- This is an **open problem**

## 8. Larger Contexts and Scaling

**[PLACEHOLDER — Work in progress]**

The most important part of the study, not yet ready. Plan:
- Test best architectures from sections 5–7 with larger context sizes
- Final goal: train LUT transformer within Andrey Karpathy's **nanoGPT** project
- Evaluate scaling behaviour: how does the gap to vanilla change with context length?
- Bandwidth advantages should become more pronounced at scale

---

## Master Results Table

| Approach | Best Exp | Val Loss | Params | Gap to Vanilla |
|----------|----------|----------|--------|----------------|
| **Vanilla baseline** | exp002 | **1.356** | **4.87M** | — |
| Simple LUT (V3 softmax) | exp184 | 1.441 | 6.5M | 0.085 |
| LUTAttentionV2 + SE | exp164 | 1.519 | 3.2M | 0.163 |
| RankAttn optimal | exp207 | 1.408 | 17.7M | 0.052 |
| RankAttn + pos32 | exp208 | 1.426 | 8.3M | 0.070 |
| RankAttn nap=10 | exp225 | **1.381** | 125.9M | **0.025** |
| Uniform LUT | exp216 | 1.513 | 4.0M | 0.157 |
| HyperLUT + RankAttn | exp199 | 1.513 | 4.3M | 0.157 |
| HyperLUT transformer | exp227 | 1.462 | ~15M | 0.106 |

## Key Takeaways

1. **Permutation-based transformers can match vanilla** — exp225 (1.381) is within 0.025 of vanilla (1.356)
2. **But at high parameter cost** — the weight tables carry 26× more parameters
3. **Architecture matters more than scale** — V3 with softmax, then RankAttention, were the biggest leaps
4. **nap/tph tradeoff is fundamental** — many small tables for loss, fewer big tables for bandwidth
5. **Out_proj and FFN dominate budget** — Q/K can be tiny (nap=3, tph=64)
6. **RankAttention is the key innovation** — enables SDPA on permutation features
7. **HyperLUTs bridge GPU efficiency and permutation semantics** — promising but still behind LUTs in loss
8. **Scaling to larger contexts is the critical next step**
