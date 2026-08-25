# exp_g_0018 — small-E + concatenation readout (clone of exp103)

Tracking issue: **#108**. Run on **gpustar (RTX 5090)**.

Clone of `exp103_singlestream_compression_manyheads` with **three** changes:

1. **`E` 384 → 64.**
2. **Inter-layer residual summation → pure concatenation, no residual.**
3. **Asymmetric head dims:** q/k keep `d_qk = 64`; v drops to `d_v = 16`.

Everything else is exp103: 6 layers, 48 routing mini-heads summed in groups of 8 into 6 attention
heads, per-site LUT config (`qk nap4/tph4`, `v nap6/tph8`, `out nap6/tph128`, `inner_in 48`),
RoPE 1e4, causal SDPA, LayerNorm everywhere, Lion on LUT tables / AdamW elsewhere, seq 512,
vocab 32768, 24,576 tok/step (48 sequences), 16,000 steps.

`mh_compression.py` is copied **byte-identical** from exp103 (`cmp` clean) — the shared
`src/spiky/lutorch/` is **not** modified.

## The concat readout

```
y_0 = tok_emb(tokens)                    [B,T,64]
y_l = block_l(y_{l-1})     l = 1..6      each block reads the PREVIOUS block's output
x   = concat(y_1 … y_6)                  [B,T,384]   <- concat_dim = 6 × 64
    -> LayerNorm(384) -> Linear(384, 32768)          <- UNTIED unembedder
```

The token embedding is **not** concatenated — only the 6 block outputs, so `concat_dim` is exactly
`N_LAYERS × E`.

**This removes the residual path entirely.** exp103's blocks have no FFN, so `x_next = x + out_e`
was its *only* skip connection; replacing summation with concatenation means 6 stacked attention
blocks with no skip. That is the literal, confirmed reading of the design — and it is the main
trainability risk. `layer_combine: "residual_sum"` is retained in the code (restores `x + out_e`
while still concatenating at the end) as a one-flag ablation to separate *"small E + concat
readout"* from *"no residual"* if this arm trains badly.

**A convenient coincidence:** `concat_dim = 384` is exactly exp103's `E`, so the unembedder
`Linear(384, 32768)` is *identical in shape* to exp103's despite the stream being 6× narrower.
Untied is guaranteed structurally — `tok_emb` is `[32768, 64]`, the unembedder is `[32768, 384]`.

## Asymmetric attention

`d_qk = 64`, `d_v = 16`: scores from the 64-dim q·k, values 16-dim, so the attention output
interior is `H·d_v = 6 × 16 = 96` and `out_proj` maps `96 → 64`. Verified directly that SDPA
supports this (output width follows `v`):

```
q(2,6,8,64)  k(2,6,8,64)  v(2,6,8,16)  ->  out(2,6,8,16)
```

## Params — 41,343,152 total (exp103: 73,584,432, −44%)

| site | total | per layer |
|---|--:|--:|
| out_proj (6 layers) | 19,245,708 | 3,207,618 |
| unembedder | 12,582,912 | — |
| v_lut (6 layers) | 3,257,868 | 542,978 |
| tok_emb | 2,097,152 | — |
| q_lut (6 layers) | 2,078,220 | 346,370 |
| k_lut (6 layers) | 2,078,220 | 346,370 |
| block LayerNorms | 2,304 | — |
| ln_final | 768 | — |
| **TOTAL** | **41,343,152** | |

Forward check: `logits (2, 512, 32768)`, finite, init loss **10.5567** against
`ln(32768) = 10.3972`.

## Two structural notes for whoever reads the result

**`out_proj` dominates at 46.6% and the attention interior is not its lever.** It is **98% LUT
tables** (3,145,728 of 3,207,618 per layer = `8 heads × tph 128 × 2^6 rows × inner_out 48`), and
table size is independent of input width. Shrinking the interior 384 → 96 saved only 664K.
`out_tph` 128 → 64 would free ~9.4M.

**q/k are now majority compress-projection, not tables.** Per layer q_lut = 149,760 compress +
196,608 tables, where the compress is `Linear(64 → 48×48 = 2304)` — a 36× *expansion* feeding a
per-head "compression" of `64 → 48` that barely compresses. At exp103's `E = 384` that same
projection genuinely compressed 8×. If the design intent is per-head compression rather than
projection, `qk_inner_in` / `v_inner_in` want to scale down with `E`; left at 48 here to keep the
change set to the three items above.

## Baselines

exp024 single-stream Hyperplane **1.2034**; exp010 dual-stream **1.1940**; exp103 in progress.

## Status

Built, smoke-tested, committed before launch. 16,000-step run launched on the 5090.
