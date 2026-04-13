# LUT-Based Transformers: Research Report

## Overview

This report covers our exploration of replacing standard linear projections in transformers with Lookup Table (LUT) based modules. The goal: build a transformer where all learned operations are LUT lookups — piecewise-constant functions that compare pairs of input features to produce outputs. Such models could eventually run on hardware that only needs comparators and table reads, not multiply-accumulate units.

We use a byte-level language model on FineWeb text (vocab=257, context=32) as our benchmark throughout.

**Vanilla baseline** (exp002): d_model=256, 4 heads, 6 layers, FFN 4×, 4.87M params → **val_loss = 1.356** at 100K steps.

---

## 1. LUT Baseline (exp004)


### Vanilla Baseline Architecture (exp002)

```mermaid
flowchart TD
    subgraph input["Input"]
        direction LR
        tokens["tokens
[B, T]"]
        emb["TokenEmbedding
Embedding 257 to 256"]
        pos["pos_emb buffer
[T, 256]"]
        tokens --> emb
        emb -->|"+"| add_pe["x + pos_emb
[B, T, 256]"]
        pos --> add_pe
    end

    subgraph block["SDPABlock  x6"]
        direction TB
        x_in(( x ))

        subgraph attn["Self-Attention"]
            direction TB
            qkv["qkv
Linear 256 to 768
fused QKV"]
            sdpa["SDPA
d_head=64, h=4
causal"]
            op["out_proj
Linear 256 to 256"]
            ln1["LayerNorm"]

            qkv --> sdpa
            sdpa --> op
            op --> ln1
        end

        r1(( + ))

        subgraph ffn["Feed-Forward"]
            direction TB
            fc1["Linear 256 to 1024"]
            relu["ReLU"]
            fc2["Linear 1024 to 256"]
            ln2["LayerNorm"]
            fc1 --> relu --> fc2 --> ln2
        end

        r2(( + ))

        x_in --> qkv
        x_in --> r1
        ln1 -->|"[B,T,256]"| r1
        r1 --> fc1
        r1 --> r2
        ln2 -->|"[B,T,256]"| r2
    end

    subgraph output["Output"]
        direction LR
        unemb["unembedder
Linear 256 to 257"]
        logits["logits [B,T,257]"]
        unemb --> logits
    end

    add_pe -->|"x [B,T,256]"| x_in
    r2 -->|"x [B,T,256]"| unemb
```

The first LUT transformer used LUTAttention V1 with learnable pairwise comparisons for attention scores, LUT-based value projection, and LUT-based FFN. The architecture was direct — each LUT module compares anchor pairs of input features, forms a binary index, and looks up a weight vector.

- **Architecture:** embedding_dim=64, 4 heads, 6 layers, LUTAttention + value LUT + FFN LUT
- **Parameters:** 191M (very large — each LUT table had 2^10 = 1024 entries)
- **Result:** val_loss = **1.463** at 100K steps


### LUT Baseline Architecture (exp004)

```mermaid
flowchart TD
    subgraph input["Input"]
        direction LR
        tokens["tokens
[B, T]"]
        emb["TokenEmbedder
Embedding 257 to 32"]
        pos["pos_emb buffer
[1, T, 32]"]
        tokens --> emb
        emb -->|"cat"| add_pe["concat(tok_emb, pos_emb)
[B, T, 64]"]
        pos --> add_pe
    end

    subgraph block["LUTBlock  x6"]
        direction TB
        x_in(( x ))

        subgraph attn["Self-Attention  (LUTAttention V1)"]
            direction TB
            pair["PairProcessing
cat(x_i, x_j)
[B, T, T, 64]"]
            al["attn_lut
MultiHeadLut
input=64, H=4
nap=10, tph=96
→ scores [B, T, T, H]"]
            sm["softmax
temp=0.25, causal"]
            vl["value_lut
MultiHeadLut
input=64, H=4
nap=10, tph=96
→ [B*T, H, 16]"]
            attn_out["softmax(scores) @ V
[B, T, 64]"]

            pair --> al --> sm
            x_in --> vl
            sm & vl --> attn_out
        end

        r1(( + ))

        ffn["ffn
MultiHeadLut
input=64, H=1
nap=12, tph=96
→ [B, T, 64]"]

        r2(( + ))

        x_in --> pair
        x_in --> r1
        attn_out -->|"[B,T,64]"| r1
        r1 --> ffn
        r1 --> r2
        ffn -->|"[B,T,64]"| r2
    end

    subgraph output["Output"]
        direction LR
        norm["normalize
x / (||x|| + ε)"]
        unemb["unembedder
dot product with
Embedding 257×64"]
        logits["logits [B,T,257]"]
        norm --> unemb --> logits
    end

    add_pe -->|"x [B,T,64]"| x_in
    r2 -->|"x [B,T,64]"| norm
```

Despite 40× more parameters than vanilla, the LUT baseline was slightly worse. The massive table sizes were wasteful — most entries rarely accessed.

---

## 2. Ranking Attention (exp020, exp058)

We explored RankAttention — replacing dot-product attention with pairwise rank features. Instead of Q@K, attention scores come from comparing pairs of features within Q and K vectors.

- **exp020** (RankAttention, 46M params): val_loss = **1.568** — worse than LUT baseline
- **exp058** (RankProjection for all projections, 0.6M params): val_loss = **1.953** — too few params

RankAttention struggled because the rank features lose magnitude information. However, the pairwise comparison idea would later resurface in HyperLUT.

---

## 3. Minimalistic Transformer with nap/tph Tradeoff (exp048–059)


### Minimalistic LUT Transformer (exp059)

```mermaid
flowchart TD
    subgraph input["Input"]
        direction LR
        tokens["tokens
[B, T]"]
        emb["TokenEmbedder
Embedding 257 to 32"]
        pos["pos_emb buffer
[1, T, 32]"]
        tokens --> emb
        emb -->|"+"| add_pe["x + pos_emb
[B, T, 32]"]
        pos --> add_pe
    end

    subgraph block["LUTBlock  x6"]
        direction TB
        x_in(( x ))

        subgraph attn["Self-Attention"]
            direction TB
            ql["q_lut
tph=256, nap=4"]
            kl["k_lut
tph=256, nap=4"]
            vl["v_lut
tph=256, nap=4"]
            ra["RankAttention
d_qk=8, M=28
binary rank features
causal SDPA"]
            op["out_proj
tph=256, nap=4"]

            x_in --> ql & kl & vl
            ql & kl & vl --> ra
            ra -->|"[B*T, 32]"| op
        end

        r1(( + ))
        ffn["ffn
tph=256, nap=4"]
        r2(( + ))

        x_in --> r1
        op -->|"[B,T,32]"| r1
        r1 --> ffn
        r1 --> r2
        ffn -->|"[B,T,32]"| r2
    end

    subgraph output["Output"]
        direction LR
        unemb["unembedder
tph=256, nap=4"]
        logits["logits [B,T,257]"]
        unemb --> logits
    end

    add_pe -->|"x [B,T,32]"| x_in
    r2 -->|"x [B,T,32]"| unemb
```

We discovered that the nap (anchor pairs per table) vs tph (tables per head) tradeoff is critical. Fixing total budget (tph × 2^nap = const at 5M params):

| nap | tph | entries/table | val_loss |
|-----|-----|---------------|----------|
| 8   | 16  | 256           | 1.942    |
| 6   | 64  | 64            | 1.759    |
| 5   | 128 | 32            | 1.729    |
| **4** | **256** | **16**    | **1.706** |
| 3   | 512 | 8             | 1.706    |
| 2   | 1024| 4             | 1.719    |

**Key finding:** More tables with fewer entries per table beats fewer tables with more entries, down to nap=3–4. This led to our coverage formula: `tph ≈ input_dim × (input_dim - 1) / nap`.

---

## 4. Self-Excitement (exp163–164)

LUTAttentionV2 introduced self-excitement: `y_o = f_o × mean(|f|)`. Each pair's contribution is weighted by the overall activation magnitude — pairs with stronger signals dominate.

- **exp161** (V2, no SE): val_loss = **1.626** (3.16M)
- **exp163** (V2 + self-excitement): val_loss = **1.533** (3.16M)
- **exp164** (V2 + SE + recompute + MLP unembedder): val_loss = **1.519** (3.19M)

Self-excitement gave a ~0.1 improvement at zero parameter cost.


### LUTAttentionV2 with Self-Excitement (exp163)

```mermaid
flowchart TD
    subgraph input["Input"]
        direction LR
        tokens["tokens
[B, T]"]
        emb["TokenEmbedder
Embedding 257 to 32"]
        rel_pe["rel_pe (learned)
[T, 16]"]
        tokens --> emb
    end

    subgraph block["LUTBlock  x6  (LUTAttentionV2)"]
        direction TB
        x_in(( x ))

        subgraph attn["Self-Attention with Self-Excitement"]
            direction TB
            pairs["enumerate pairs (i,j)
causal, include_diagonal
cat(x_i, x_j, RelPE[|i-j|])
[B, M, 80]"]
            lut["MultiHeadLut
input=80, H=1, O=32
nap=4, tph=1024
→ f [B, M, H, O]"]
            se["Self-Excitement
scale = mean(|f|) per pair
y = f × scale
[B, M, H, O]"]
            sc["scatter_add by query pos i
→ [B, T, H, O]"]

            pairs --> lut --> se --> sc
        end

        ln["LayerNorm(32)"]
        r1(( + ))

        x_in --> pairs
        x_in --> r1
        sc -->|"[B,T,32]"| ln --> r1
    end

    subgraph output["Output"]
        direction LR
        unemb["unembedder
Linear 32 to 257"]
        logits["logits [B,T,257]"]
        unemb --> logits
    end

    emb -->|"x [B,T,32]"| x_in
    rel_pe -->|"[T,16]"| pairs
    r1 -->|"x [B,T,32]"| unemb
```

---

## 5. Weights Quantisation (exp145–146)


### Quantisation-Aware Architecture (exp145)

```mermaid
flowchart TD
    subgraph input["Input"]
        direction LR
        tokens["tokens
[B, T]"]
        emb["TokenEmbedder
Embedding 257 to 32"]
        pos["pos_emb buffer
[1, T, 16]"]
        tokens --> emb
    end

    subgraph block["LUTBlock  x6"]
        direction TB
        x_in(( x ))
        cat["concat(x, pos)
[B*T, 48]"]

        subgraph attn["Self-Attention  (RankAttention)"]
            direction TB
            ql["q_lut
MultiHeadLut
input=48, H=4
nap=4, tph=264
→ d_qk=16"]
            kl["k_lut
MultiHeadLut
input=48, H=4
nap=4, tph=264
→ d_qk=16"]
            vl["v_lut
MultiHeadLut
input=48, H=4
nap=4, tph=264
→ d_v=16"]
            ra["RankAttention
causal SDPA
→ [B, T, H, d_v]"]
            op["out_proj
MultiHeadLut
input=H×d_v=64, H=1
nap=4, tph=480
→ [B, T, 32]"]

            cat --> ql & kl & vl
            ql & kl & vl --> ra
            ra -->|"[B*T, 64]"| op
        end

        r1(( + ))

        x_in --> cat
        x_in --> r1
        op -->|"[B,T,32]"| r1
    end

    subgraph qreg["Quantization Regularization"]
        direction LR
        qloss["L_quant = λ × mean(sin²(π·w·D))
D=64, λ ramps 0→0.01
starting at 20% of training"]
    end

    subgraph output["Output"]
        direction LR
        unemb["unembedder
Linear 32 to 257"]
        logits["logits [B,T,257]"]
        unemb --> logits
    end

    emb -->|"x [B,T,32]"| x_in
    pos -->|"[B,T,16]"| cat
    r1 -->|"x [B,T,32]"| unemb
    block -.->|"applied to all LUT weights"| qloss
```

We tested quantisation-aware training using a sinusoidal regularisation term: `λ × mean(sin²(πwD))` which pushes weights toward multiples of 1/D.

- **exp145** (50K steps, quant_reg): val_loss = **1.605** (6.36M)
- **exp146** (100K steps, quant_reg): val_loss = **1.509** (6.36M)

The quantisation penalty didn't significantly hurt training — the model learned to place weights at grid points while maintaining good loss. This suggests LUT weights can be represented with low bit-width (log2(D) bits per weight).

---

## 6. V3 Architecture — Classic Attention with LUT Projections (exp173–184)


### V3 Softmax Architecture (exp184)

```mermaid
flowchart TD
    subgraph input["Input"]
        direction LR
        tokens["tokens
[B, T]"]
        emb["TokenEmbedder
Embedding 257 to 32"]
        rel_pe["rel_pe (learned)
[T, 16]"]
        tokens --> emb
    end

    subgraph block["LUTBlock  x6  (LUTAttentionV3)"]
        direction TB
        x_in(( x ))

        subgraph score["Score LUT  (LUTAttentionV3)"]
            direction TB
            spairs["enumerate pairs (i,j)
cat(x_i, x_j, RelPE[|i-j|])
[B, M, 80]"]
            slut["score_lut
MultiHeadLut
input=80, H=4, O=1
nap=6, tph=128
→ [B, T, T, H, 1]"]
            ssm["softmax (causal)
→ attn_weights [B, H, T, T]"]
            spairs --> slut --> ssm
        end

        subgraph value["Value LUT  (per token)"]
            direction TB
            vlut["value_lut
MultiHeadLut
input=32, H=4, O=16
nap=4, tph=256
→ [B*T, H, d_v=16]"]
        end

        matmul["softmax(scores) @ V
→ [B, T, H×d_v=64]"]
        attn_norm["LayerNorm(64)"]

        subgraph proj["Out-Projection LUT"]
            direction TB
            oplut["out_proj
MultiHeadLut
input=64, H=1, O=32
nap=5, tph=768
→ [B, T, 32]"]
        end

        norm["LayerNorm(32)"]
        r1(( + ))

        x_in --> spairs
        x_in --> vlut
        x_in --> r1
        ssm & vlut --> matmul
        matmul --> attn_norm --> oplut
        oplut -->|"[B,T,32]"| norm --> r1
    end

    subgraph output["Output"]
        direction LR
        unemb["unembedder
Linear 32 to 257"]
        logits["logits [B,T,257]"]
        unemb --> logits
    end

    emb -->|"x [B,T,32]"| x_in
    rel_pe -->|"[T,16]"| spairs
    r1 -->|"x [B,T,32]"| unemb
```

The breakthrough came from LUTAttentionV3: generating raw attention scores with LUTs, then applying standard softmax + matmul. This separates "what to attend to" (LUT scores) from "how to combine" (softmax@V).

**Architecture per block:**
```
x → [Score LUT] → attention scores [B, T, T, H, 1]
x → [Value LUT] → values [B, T, H, d_v]
     softmax(scores) @ values → [B, T, H, d_v]
     LayerNorm → [Out-proj LUT] → [B, T, E]
     residual + LayerNorm
```

### Evolution of V3 results:

| Exp | Config | Params | Val Loss |
|-----|--------|--------|----------|
| exp173 | nap=4, tph=256, d_v=16 | 3.26M | 1.549 |
| exp177 | all nap=6, op_tph=512 | 9.65M | 1.446 |
| exp181 | v_tph=256, op_nap=5, op_tph=768 | 6.51M | 1.458 |
| **exp184** | **same as exp181, 100K** | **6.51M** | **1.441** |

**Key findings from sweeps:**
- **Out_proj is the most important component** — dominates every budget allocation sweep
- **nap > tph for expressiveness** — more entries per table beats more tables
- **Pairs processor:** Concat (default) slightly better than SignedDiff at long training
- **Exclusion sets** (prohibiting within-group pairs): didn't help

### Gap to vanilla:

| Model | Params | Val Loss |
|-------|--------|----------|
| Vanilla (exp002) | 4.87M | **1.356** |
| Best LUT V3 (exp184) | 6.51M | **1.441** |
| Gap | | **0.085** |

### Analysis of the gap:

Gradient analysis revealed balanced |grad/weight| ratios across components (~10⁻⁴ on log scale) — no gradient starvation. The gap appears to be representational: LUT piecewise-constant functions have limited resolution compared to dense linear projections.

Attention entropy analysis showed vanilla achieves much sharper attention (entropy 0.06–0.49) vs LUT (0.68–1.17) — the LUT score function can't produce well-separated scores.

---

## 7. HyperLUT (exp193–195)


### HyperLUT Architecture (exp193)

```mermaid
flowchart TD
    subgraph input["Input"]
        direction LR
        tokens["tokens
[B, T]"]
        emb["TokenEmbedder
Embedding 257 to 32"]
        rel_pe["rel_pe (learned)
[T, 16]"]
        tokens --> emb
    end

    subgraph hlut["HyperLUT (replaces MultiHeadLut)"]
        direction LR
        hl_pairs["M pairwise comparisons
x_a vs x_b → {0,1} (STE)
[N, M]"]
        hl_lin1["Linear(M → H×hid)
GELU"]
        hl_lin2["Linear(hid → n_out)
→ [N, H, O]"]
        hl_pairs --> hl_lin1 --> hl_lin2
    end

    subgraph block["HyperBlock  x6"]
        direction TB
        x_in(( x ))

        subgraph score["Score HyperLUT  (LUTAttentionV3)"]
            direction TB
            spairs["enumerate pairs (i,j)
cat(x_i, x_j, RelPE[|i-j|])
[B, M, 80]"]
            sh["score HyperLUT
input=80, H=4, O=1
500 pairs, hid=64
→ [B, T, T, H, 1]"]
            ssm["softmax (causal)
→ attn_weights [B, H, T, T]"]
            spairs --> sh --> ssm
        end

        subgraph value["Value HyperLUT  (per token)"]
            direction TB
            vh["value HyperLUT
input=32, H=4, O=16
496 pairs, hid=64
→ [B*T, H, d_v=16]"]
        end

        matmul["softmax(scores) @ V
→ [B, T, H×d_v=64]"]
        attn_norm["LayerNorm(64)"]

        subgraph proj["Out-Proj HyperLUT"]
            direction TB
            oph["out_proj HyperLUT
input=64, H=1, O=32
2016 pairs, hid=64
→ [B, T, 32]"]
        end

        norm["LayerNorm(32)"]
        r1(( + ))

        x_in --> spairs
        x_in --> vh
        x_in --> r1
        ssm & vh --> matmul
        matmul --> attn_norm --> oph
        oph -->|"[B,T,32]"| norm --> r1
    end

    subgraph output["Output"]
        direction LR
        unemb["unembedder
Linear 32 to 257"]
        logits["logits [B,T,257]"]
        unemb --> logits
    end

    emb -->|"x [B,T,32]"| x_in
    rel_pe -->|"[T,16]"| spairs
    r1 -->|"x [B,T,32]"| unemb
```

HyperLUT replaces the lookup table mechanism entirely. Instead of:
- Compare nap pairs → binary index → table lookup → sum over tables

HyperLUT does:
- Compare M pairs → {0,1} vector → Linear → GELU → Linear → output

The MLP sees **all M comparisons simultaneously** (vs LUT where each table only sees its nap pairs independently). This avoids the 2^nap exponential table size while allowing cross-comparison interactions.

### Early results:

| Exp | Config | Params | Val Loss | Steps |
|-----|--------|--------|----------|-------|
| exp193 | hid=64 uniform, sigmoid | 2.34M | 1.657 | 25K |
| exp194 | hid=32/64/256, rational | 4.32M | 1.611 | 25K |
| exp195 | pairs=1024/496/2016, hid=32/64/128 | 3.15M | running | 50K |

HyperLUT is promising: 2.34M params reaching 1.657 at only 25K steps (vs MultiHeadLut needing 6.5M for similar early convergence). The fully differentiable backward (sigmoid STE vs uncertainty derivative) could help with the convergence plateau we observed in standard LUTs.

**Open questions:**
- Will HyperLUT close the gap to vanilla at longer training?
- Temperature tuning: 0.1 works better than 1.0 — sharper comparisons help
- Sigmoid vs rational soft function: both viable, need controlled comparison

---

## Summary & Next Steps

### Progress timeline:

| Milestone | Val Loss | Params | Approach |
|-----------|----------|--------|----------|
| LUT baseline (exp004) | 1.463 | 191M | Brute-force LUTs |
| Minimalistic (exp059) | 1.706 | 5.0M | nap/tph optimisation |
| Self-excitement (exp164) | 1.519 | 3.2M | V2 + SE |
| V3 softmax (exp184) | **1.441** | 6.5M | Classic attention with LUT projections |
| Vanilla baseline | **1.356** | 4.9M | Standard transformer |

### Key insights:
1. **Architecture matters more than scale** — V3 (softmax attention) was the biggest single improvement
2. **nap > tph** — expressiveness per table trumps table count
3. **Out_proj dominates** — the output mixing layer needs the most capacity
4. **The 0.085 gap to vanilla** is likely representational, not optimisational
5. **HyperLUT** is a promising new direction that avoids the exponential table size problem
