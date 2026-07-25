# exp022 — exp020 (single-stream random-init code-LUT) with unembedder T 16 → 64

Exact copy of **exp020** with ONE change: the `HyperplaneCodeLUT` unembedder's
`n_tables` **T = 64** (was 16). `nap=15` → V=2^15=32768, `input_dim` E=384, V unchanged.
Tests whether **more voting** recovers quality at champion E=384 scale.

Everything else byte-identical to exp020: single residual stream (no dual stream, no
residual LUTs), LayerNorm everywhere incl. `ln_final` before the code-LUT,
`hyperplane_init="random"` scale 0.05 for **all** LUTs (qk/v/out HyperplaneMHL sites AND
the code-LUT hyperplanes), `w_cell` const 0.02, backbone 6L / E384 / 6h×d_qk=d_v=64 /
ctx512 / vocab32768 / RoPE1e4 / untied tok_emb; LUT qk nap4/tph256, v nap6/tph256,
out_proj nap7/tph512 (hard); seed 42; 16000 steps, 24,576 tokens/step, LR over 16000
(warmup 1600, cosine to 0.1× floor), Lion lut_lr 2e-4 / AdamW adam_lr 3e-4 (hyperplane w/b
no-wd by name, w_cell wd 0.1, tok_emb/norms wd 0), grad clip 1.0, eval every 500.

## Memory
The code-LUT loops/accumulates over the T tables with **gradient checkpointing**, so it
never materializes [N, T, V] and peak activation memory stays ~O([N, V]) even at T=64.
Smoke-verified: 5 steps at T=64, no OOM.

## Params (smoke-verified) — Total 266,435,300
- Unembedder (T=64) = 2,466,752 = hyperplanes T·nap·(E+1)=369,600 + w_cell T·V=2,097,152
  (up from T=16's 616,688).
- LUT tables 207,618,048 (Lion, qk/v/out — same as exp020).
- Hyperplanes 44,130,240 (AdamW no-wd) = qk/v/out 43,760,640 + code-LUT 369,600.
- w_cell 2,097,152 (AdamW wd 0.1) · tok_emb 12,582,912 · norms 6,948.
- Groups sum exactly to 266,435,300 → nothing unrouted. Step-1 loss 10.4542 ≈ ln(32768).

## Baselines
exp010 dual-stream Linear **1.1940**; exp020 T=16 single-stream random-init (same-step
cross-reference). Same seed/data.
