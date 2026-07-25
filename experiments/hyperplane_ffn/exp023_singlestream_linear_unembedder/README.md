# exp023 — single-stream champion lineage with a PLAIN Linear unembedder (the CONTROL)

Exact clone of **exp022** with ONE change: the `HyperplaneCodeLUT` unembedder is replaced
by a plain `nn.Linear(E=384 → V=32768, bias=False)` — the standard linear unembedder (same
as exp010's).

**Purpose — the control that isolates the single-stream reduction FROM the unembedder:**
- vs **exp010** (dual-stream + Linear, **1.1940**) → isolates the cost of removing the dual
  stream / residual LUTs (both are single-stream vs dual-stream, both Linear unembedder).
- vs **exp019 / exp020 / exp022** (single-stream + code-LUT) → isolates the cost of the
  code-LUT unembedder (all single-stream, code-LUT vs Linear).

## Everything else = exp022 (byte-identical)
Single residual stream (no dual stream, no residual LUTs); LayerNorm everywhere incl.
`ln_final` before the unembedder; `hyperplane_init="random"` scale 0.05 for all backbone
LUTs (qk/v/out `HyperplaneMultiHeadLUT`); backbone 6L / E384 / 6h×d_qk=d_v=64 / ctx512 /
vocab32768 / RoPE1e4 / untied tok_emb; LUT qk nap4/tph256, v nap6/tph256, out_proj nap7/tph512
(hard); seed 42; 16000 steps, 24,576 tokens/step, LR over 16000 (warmup 1600, cosine to 0.1×
floor), Lion lut_lr 2e-4 / AdamW adam_lr 3e-4 (hyperplane w/b no-wd by name, Linear unembedder
wd 0.1, tok_emb/norms wd 0), grad clip 1.0, eval every 500. (`code_unemb_*` config keys are
vestigial/unused with the Linear unembedder.)

## Params (smoke-verified) — Total 276,551,460
- Linear unembedder 12,582,912 (AdamW wd 0.1) = 384·32768.
- LUT tables 207,618,048 (Lion, qk/v/out). Hyperplanes 43,760,640 (AdamW no-wd, qk/v/out only —
  no code-LUT hyperplanes). tok_emb 12,582,912 · norms 6,948.
- Groups sum exactly to 276,551,460 → nothing unrouted.
- Step-1 loss 10.5681 — the Linear-unembedder init behavior (random Linear weights → logits not
  perfectly uniform, ≈ exp010's own 10.5613; a bit above ln(32768)=10.397).

## Baselines
exp010 dual-stream Linear **1.1940**; exp019/020/022 single-stream code-LUT. Same seed/data.
