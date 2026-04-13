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
