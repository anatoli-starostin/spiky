# exp021 — champion exp010 with ONLY the unembedder swapped (Linear → HyperplaneCodeLUT)

Clean one-variable isolation of the code-LUT unembedder at champion scale. Byte-identical
to **exp010** (E=384 dual-stream champion, best_val_bpb **1.1940**) EXCEPT the final
`nn.Linear(D=384 → V=32768)` unembedder is replaced by
`HyperplaneCodeLUT(input_dim=D=384, nap=15 → V=2^15=32768, T=16, anchor-pairs init)`.

**Kept from exp010 (unchanged):** DUAL stream (E-stream + D-stream); BOTH residual LUTs
(per-layer `residual_lut` ×6 + `emb_resid_lut`); **MeanAbsNorm** pre-norms (NOT LayerNorm —
this is the pure Linear→code-LUT delta vs exp010); `ln_final(D)` LayerNorm feeding the
unembedder (the code-LUT reads `ln_final(x_resid)`, the D-stream, dim 384); LUT sites qk
nap4/tph256, v nap6/tph256, out_proj nap7/tph512, residual nap6/tph256, emb_resid nap6/tph256
(HyperplaneMHL, anchor/hard); seed 42; 16k recipe; eval_every 200.

## Params (smoke-verified) — Total 312,760,354
Exactly exp010's 324,726,578 − 12,582,912 (old Linear unembedder) + 616,688 (code-LUT) = 312,760,354.
- LUT tables 251,658,240 (Lion) — IDENTICAL to exp010 (all sites kept).
- Hyperplanes 47,992,560 (AdamW no-wd) = exp010's 47,900,160 + code-LUT hyperplane_weight/bias 92,400.
- w_cell 524,288 (AdamW wd 0.1) — the old Linear-unembedder slot.
- tok_emb 12,582,912 · norms/temps 2,354 (= exp010's — confirms MeanAbsNorm parameter-free norms kept).
- Groups sum exactly to the total → NO unembedder params unrouted. Step-1 loss 10.4046 ≈ ln(32768).

## Purpose / baselines
Isolate whether the code-LUT unembedder underperforms at scale INDEPENDENT of the single-stream
reduction. Baselines: exp010 dual-stream Linear **1.1940**; single-stream code-LUT exp019 (anchor)
/ exp020 (random) for cross-reference. Same seed/data.
