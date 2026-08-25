# exp101 — single-stream attention with SEPARATE q/k/v as CompressionMultiHeadLUT

Clone of **exp024** (single-stream, plain-Linear unembedder) with two changes to the
attention front-end:

1. **Split the joint `qk_lut` into separate `q_lut` and `k_lut`** (exp024 packed Q+K into
   one LUT with `n_outputs=2*d_qk` and sliced it).
2. **Swap the attention LUTs from HyperplaneMultiHeadLUT → CompressionMultiHeadLUT.**

Attention sites (E=384, H=6, d_qk=d_v=64):
```
q_lut = k_lut = v_lut = CompressionMultiHeadLUT(
    input_dim=384, output_dim=64, inner_in_dim=48, inner_out_dim=-1,
    nap=6, tph=32, n_heads=6, multihead_output=True,
    forward_mode="hard", use_bf16=True)            # -> [N, 6, 64]  (heads SEPARATE)
out_proj = CompressionMultiHeadLUT(
    input_dim=384, output_dim=384, inner_in_dim=48, inner_out_dim=48,
    nap=6, tph=32, n_heads=8, forward_mode="hard", use_bf16=True)   # -> [N, 384]
```

### `multihead_output` — external subclass, NO shared-src edit
Stock `CompressionMultiHeadLUT` collapses the head axis (sums when `inner_out_dim=-1`, or
Linear-mixes via decompress otherwise), so it cannot emit the 6 independent per-head q/k/v
vectors attention needs. Rather than edit the shared
`src/spiky/lutorch/compression_mhl.py` (standing rule: extend shared lutorch modules by
subclassing/wrapping externally only), this experiment adds a **local subclass**
`CompressionMultiHeadLUTMH` in `mh_compression.py` with a keyword-only
`multihead_output: bool = False`. When True it returns `[N, n_heads, eff_out]` **without
collapsing heads** (batched path returns `y`; loop path `torch.stack(parts, dim=1)`).
Validated: requires `inner_out_dim == -1` (decompress mixes heads → ValueError otherwise),
not supported with `joint_head_compression`, requires a compress projection. Default False
delegates to `super().forward` → stock behavior bit-for-bit unchanged.

`test_multihead_output.py` asserts: (a) default path bit-for-bit identical to stock,
(b) `multihead_output=True` → `[N,6,64]`, (c) ValueError when `inner_out_dim != -1`,
(d) stock collapsed output == multihead output summed over heads (max|Δ|=0). All pass.

### Everything else = exp024
Single residual stream E=384 (no residual_lut / emb_resid / D-stream), 6 layers, LayerNorm
everywhere (incl. per-head q/k LayerNorm and `ln_final`), plain untied `Linear(384→32768)`
unembedder, RoPE 1e4, causal softmax SDPA. Optimizer: Lion on LUT table weights
(`lut_lr 2e-4`, ndim-3 → Lion), AdamW on the rest (compress/decompress + unembed decay
wd 0.1; biases/temps/tok_emb no-wd). 16000 steps, 24,576 tok/step. Total params **47.8M**.

Smoke-verified (`SMOKE=1`): q/v_lut → [N,6,64], out_proj → [N,384], logits → [B,T,32768],
optimizer grouping builds cleanly (lut=18.87M → Lion, decay=16.34M, tok_emb=12.58M,
nodecay=16.7K, hyperplane=0). **Not yet trained.**

Baselines to beat: exp024 single-stream Hyperplane **1.2034**; exp010 dual-stream **1.1940**.

Note: `use_bf16=True` here is FastMHL's bf16-autocast flag; table STORAGE stays fp32
(`weight_dtype` default). For bf16 table storage (HBM bandwidth), also pass
`weight_dtype=torch.bfloat16`.
