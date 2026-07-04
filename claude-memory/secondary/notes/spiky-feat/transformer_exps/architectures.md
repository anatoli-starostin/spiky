# Architecture Diagrams

## 1. Vanilla Baseline (exp002)

Standard causal transformer with SDPA.

```
tokens → Embedding(257, 256) + PositionalEmb(32, 256)
       ↓
  ┌─────────────────── × 6 layers ───────────────────┐
  │                                                    │
  │  x → QKV Linear(256 → 768) → split Q, K, V       │
  │       Q, K, V: [B, 4 heads, T, 64]                │
  │       ScaledDotProductAttention(Q, K, V, causal)   │
  │       → concat heads → Linear(256 → 256)          │
  │       → x + LayerNorm(attn_out)                    │
  │                                                    │
  │  x → Linear(256 → 1024) → ReLU → Linear(1024→256) │
  │       → x + LayerNorm(ffn_out)                     │
  │                                                    │
  └────────────────────────────────────────────────────┘
       ↓
  Linear(256 → 257) → logits

Parameters: 4.87M
```

---

## 2. LUT Baseline (exp004)

First LUT transformer. Uses LUTAttention V1 with positional buckets.

```
tokens → Embedding(257, 32) + PositionalEmb(32, 32) → cat → [B, T, 64]
       ↓
  ┌─────────────────── × 6 layers ───────────────────┐
  │                                                    │
  │  Score LUT (LUTAttention V1):                      │
  │    For each pair (i,j):                            │
  │      combined = PairProcessing(x_i, x_j)          │
  │      MultiHeadLut(combined) → [H, 1]              │
  │    → attention_scores [B, T, T, H]                 │
  │    → softmax → attention_weights                   │
  │                                                    │
  │  Value LUT:                                        │
  │    MultiHeadLut(x) → [B*T, H, E//H]               │
  │                                                    │
  │  attn_weights @ V → attn_out                       │
  │  → x + Dropout(attn_out)                           │
  │                                                    │
  │  FFN LUT:                                          │
  │    MultiHeadLut(x) → [B*T, 1, E]                  │
  │  → x + Dropout(ffn_out)                            │
  │                                                    │
  └────────────────────────────────────────────────────┘
       ↓
  x / ||x|| → dot product with unembedding → logits

Parameters: 191M (nap=10, large tables)
```

---

## 3. RankAttention (exp020)

Replace Q@K dot product with pairwise rank features.

```
tokens → Embedding + PositionalEmb → [B, T, 64]
       ↓
  ┌─────────────────── × 6 layers ───────────────────┐
  │                                                    │
  │  RankAttention:                                    │
  │    For each token x:                               │
  │      rank_q = RankProjection_Q(x)                  │
  │        → sample M pairs from d_qk dims             │
  │        → sign(x[a_i] - x[b_i]) or soft version    │
  │        → Linear(M → d_qk)                          │
  │      rank_k = RankProjection_K(x)                  │
  │    ScaledDotProductAttention(rank_q, rank_k, V)    │
  │                                                    │
  │  Value: standard linear or LUT                     │
  │  FFN: standard or LUT                              │
  │                                                    │
  └────────────────────────────────────────────────────┘
       ↓
  logits

Parameters: 46M (exp020)
```

---

## 4. Minimalistic LUT Transformer (exp048–059)

Compact all-LUT transformer exploring nap/tph tradeoff.

```
tokens → Embedding(257, 32) + PositionalEmb → [B, T, 32]
       ↓
  ┌─────────────────── × 6 layers ───────────────────┐
  │                                                    │
  │  Score LUT (LUTAttention V1):                      │
  │    MultiHeadLut(pair_features)                     │
  │    input_dim=32, H=4, n_outputs=1                  │
  │    nap=4, tph=256 (sweet spot)                     │
  │    → softmax → attn_weights                        │
  │                                                    │
  │  Value LUT:                                        │
  │    MultiHeadLut(x)                                 │
  │    input_dim=32, H=4, n_outputs=8                  │
  │    nap=4, tph=256                                  │
  │    → [B*T, H, 8]                                   │
  │                                                    │
  │  attn_weights @ V → reshape → [B, T, 32]          │
  │  → x + attn_out                                    │
  │                                                    │
  │  FFN LUT:                                          │
  │    MultiHeadLut(x) → [B*T, 1, 32]                 │
  │  → x + ffn_out                                     │
  │                                                    │
  └────────────────────────────────────────────────────┘
       ↓
  Linear(32 → 257) → logits

Parameters: ~5M (varies with nap/tph)
Key: tph × 2^nap = const budget; nap=4, tph=256 optimal
```

---

## 5. Self-Excitement — LUTAttentionV2 (exp163–164)

V2 replaces pairwise softmax attention with scatter-sum + self-excitement.

```
tokens → Embedding(257, 32) + learned RelPE [T, 16]
       ↓
  ┌─────────────────── × 6 layers ───────────────────┐
  │                                                    │
  │  LUTAttentionV2:                                   │
  │    For each valid pair (i,j), causal:              │
  │      features = cat(x_i, x_j, RelPE[|i-j|])       │
  │      MultiHeadLut(features) → [H, O]              │
  │                                                    │
  │    Self-excitement (per pair, per head):            │
  │      s = mean(|output|)                            │
  │      output = output × s                           │
  │                                                    │
  │    scatter_add by query position → [B, T, H, O]   │
  │                                                    │
  │  → reshape [B, T, E]                               │
  │  → x + LayerNorm(attn_out)                         │
  │                                                    │
  └────────────────────────────────────────────────────┘
       ↓
  Linear(32, 128) → ReLU → Linear(128, 257) → logits

Parameters: 3.19M
```

---

## 6. Quantisation-Aware Training (exp145–146)

Same architecture as V2 model with added quantisation regularisation.

```
Standard V2 architecture (same as above)
       +
  Quantisation loss: λ × mean(sin²(π × w × D))

  D = 256 (grid resolution)
  λ ramps from 0 to λ_max after start_fraction of training

  Effect: pushes each weight toward nearest multiple of 1/D
  → weights can be stored as 8-bit integers (log2(256) = 8 bits)
  → loss penalty minimal (1.509 vs ~1.52 without qreg at 100K)
```

---

## 7. V3 Architecture — Softmax LUT Attention (exp173–184)

Classic transformer attention pattern with all projections replaced by LUTs.

```
tokens → Embedding(257, 32) + learned RelPE [T, 16]
       ↓
  ┌─────────────────── × 6 layers ───────────────────┐
  │                                                    │
  │  Score LUT (LUTAttentionV3):                       │
  │    For each valid pair (i,j):                      │
  │      features = cat(x_i, x_j, RelPE[dist])        │
  │      MultiHeadLut(features) → [H, 1]              │
  │    → dense [B, T, T, H, 1]                        │
  │    → squeeze → permute → [B, H, T, T]             │
  │    → softmax(dim=-1)                               │
  │           ↓                                        │
  │  Value LUT (per-token):                            │
  │    MultiHeadLut(x) → [B*T, H, d_v]                │
  │    → reshape → [B, H, T, d_v]                     │
  │           ↓                                        │
  │  softmax(scores) @ V → [B, H, T, d_v]             │
  │    → permute → [B, T, H×d_v]                      │
  │    → LayerNorm(H×d_v)                              │
  │           ↓                                        │
  │  Out-proj LUT (per-token):                         │
  │    MultiHeadLut(attn_out) → [B*T, 1, E]           │
  │    → reshape → [B, T, E]                           │
  │           ↓                                        │
  │  x + LayerNorm(proj_out)                           │
  │                                                    │
  └────────────────────────────────────────────────────┘
       ↓
  Linear(32 → 257) → logits

exp184 config:
  Score: nap=6, tph=128, H=4, n_outputs=1
  Value: nap=4, tph=256, H=4, n_outputs=d_v=16
  OutProj: nap=5, tph=768, H=1, n_outputs=32
  Parameters: 6.51M
  Best val_loss: 1.4411 @ 100K steps
```

---

## 8. HyperLUT (exp193–195)

Replaces MultiHeadLut with comparison-based MLP. Sees all pairs at once.

```
HyperLUT module (replaces MultiHeadLut):

  input x [B, input_dim]
       ↓
  Pairwise comparisons:
    For M sampled pairs (a_i, b_i):
      hard: (x[a_i] > x[b_i]) → {0, 1}        ← forward
      soft: sigmoid(d/t) or 0.5+0.5·d/(t+|d|)  ← backward (STE)
    → features [B, M]
       ↓
  Linear(M → H × hidden_dim)
    → reshape [B, H, hidden_dim]
    → GELU
       ↓
  Linear(hidden_dim → n_outputs)   (per head)
    → output [B, H, n_outputs]


Transformer block (same structure as V3):

  Score HyperLUT → softmax → Value HyperLUT → matmul
    → LayerNorm → OutProj HyperLUT → residual + LayerNorm

exp195 config:
  Score: 1024 pairs, hidden=32/head, H=4
  Value: 496 pairs (all), hidden=64/head, H=4
  OutProj: 2016 pairs (all), hidden=128, H=1
  Parameters: 3.15M

Key advantage over MultiHeadLut:
  MultiHeadLut: each table sees nap pairs independently
  HyperLUT: MLP sees ALL M pairs at once → cross-pair interactions
  → avoids 2^nap exponential table size
  → fully differentiable (no uncertainty derivative approximation)
```
