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
tph=128, nap=8"]
            kl["k_lut
tph=128, nap=8"]
            vl["v_lut
tph=128, nap=8"]
            ra["RankAttention
d_head=8, smooth=True
normalize_deltas=True
causal SDPA"]
            op["out_proj
tph=128, nap=8"]

            x_in --> ql & kl & vl
            ql & kl & vl --> ra
            ra -->|"[B*T, 32]"| op
        end

        r1(( + ))
        ffn1["ffn1
tph=128, nap=8"]
        ri(( + ))
        ffn2["ffn2
tph=128, nap=8"]
        r2(( + ))

        x_in --> r1
        op -->|"[B,T,32]"| r1
        r1 --> ffn1
        r1 --> ri
        ffn1 --> ri
        ri --> ffn2
        r1 --> r2
        ffn1 --> r2
        ffn2 --> r2
    end

    subgraph output["Output"]
        direction LR
        unemb["unembedder
tph=128, nap=8"]
        logits["logits [B,T,257]"]
        unemb --> logits
    end

    add_pe -->|"x [B,T,32]"| x_in
    r2 -->|"x [B,T,32]"| unemb
```
