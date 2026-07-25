# exp024 — exp023 (single-stream + Linear) with anchor-pairs hyperplane init

Exact clone of **exp023** (single-stream champion-lineage + plain Linear unembedder,
the control) with ONE change: `hyperplane_init="anchor_pairs"` for all backbone LUTs
(qk/v/out_proj `HyperplaneMultiHeadLUT`), instead of exp023's `random`/scale 0.05.

This is the **anchor-vs-random init A/B at the single-stream + Linear architecture** —
the same comparison exp010 (anchor) vs exp012 (random) made at dual-stream scale, but
here for the cheaper single-stream + Linear model.

## Everything else = exp023 (byte-identical)
Single residual stream (no dual stream, no residual LUTs); LayerNorm everywhere incl.
`ln_final` before the unembedder; plain `nn.Linear(384 → 32768, bias=False)` unembedder;
backbone 6L / E384 / 6h×d_qk=d_v=64 / ctx512 / vocab32768 / RoPE1e4 / untied tok_emb;
LUT qk nap4/tph256, v nap6/tph256, out_proj nap7/tph512 (hard); seed 42; 16000 steps,
24,576 tokens/step, LR over 16000 (warmup 1600, cosine to 0.1× floor), Lion lut_lr 2e-4 /
AdamW adam_lr 3e-4 (hyperplane w/b no-wd by name, Linear unembedder wd 0.1, tok_emb/norms
wd 0), grad clip 1.0, eval every 500.

## Params (smoke-verified) — Total 276,551,460 (identical to exp023; init doesn't change counts)
LUT tables 207,618,048 (Lion) · hyperplanes 43,760,640 (AdamW no-wd, qk/v/out) · Linear
unembedder 12,582,912 (AdamW wd 0.1) · tok_emb 12,582,912 · norms 6,948. Groups sum exactly
→ nothing unrouted. Step-1 loss 10.5769 (≈ exp023's 10.5681; Linear-unembedder init).

## Baselines
exp010 dual-stream **1.1940**; exp023 single-stream + Linear **random-init** (the direct
anchor-vs-random comparison). Same seed/data.
