# exp105_densecat — DenseNet-style dense-concat refactor of exp_g_0021

Clone of gpustar's **exp_g_0021_smallE_concat_readout_noutdecomp** (small E=64, 48 routing
mini-heads grouped 8→1 into 6 attn heads, readout LUT between two LayerNorms, out_proj
no-decompress) refactored into a **DENSE-CONCAT** architecture: residual SUM replaced by
residual CONCAT (DenseNet-style feature reuse).

## Core change: dense-concat carry-forward (NO residual add)
- Running stream starts = token embedding (E=64).
- Each layer L reads the **full running stream** (dim (L+1)·E), produces exactly E, and its
  output is **concatenated** onto the stream (`x = cat([x, out])`) — the concat IS the
  carry-forward. There is NO per-layer additive skip (`x + out` removed; block returns
  `out_e` only). Verified via real-forward hooks:
  - layer input dims = **[64, 128, 192, 256, 320, 384]** = E, 2E, 3E, 4E, 5E, 6E.
- After all 6 layers the stream = emb + all 6 outputs = **(N_LAYERS+1)·E = 7·64 = 448**.

## q/k/v built PER-LAYER with growing input_dim
Each layer's q/k/v CompressionMHL uses `input_dim = (L+1)·E`; the compress projection
compresses the (larger) per-layer input to the same inner_in=48. inner/nap/tph/heads
unchanged from 0021 (q/k inner48/nap4/tph4, v inner48/nap6/tph8, 48 routing heads). out_proj
unchanged (input H·d_v=96 fixed, output E=64, inner_in48/inner_out=-1/nap6/tph128/heads8).
So q/k/v param cost GROWS with layer index (compress widens), out_proj is flat.

## Readout on the final stream
Readout reads the **final (N_LAYERS+1)·E = 448** running stream (emb + all layer outputs —
the natural choice, since the running stream literally ends at 7E; 6E=384 would exclude the
embedding). It then OUTPUTS 384 (the vanilla/reference unembedder dim), so:
`ln_final(448) → readout_lut CompressionMHL(input 448 → output 384, inner48/nap6/tph64/heads8)
→ ln_readout(384) → unembedder Linear(384→32768)`. (readout in≠out: compress 448→8·48=384,
decompress 8·48=384→384.)

## Per-concat LayerNorms (addendum): PRE-norm convention
exp_g_0021 uses **PRE-norm** (`ln_pre` applied at the block's start, before q/k/v). So the
per-concat norm is each layer's `ln_pre`, sized to the grown width — **LayerNorm(E),
LayerNorm(2E), …, LayerNorm(6E)** feeding layers 0..5, plus `ln_final = LayerNorm(7E=448)`
on the final concat before readout. No separate/duplicate norms; consistent pre-norm.

## Verified params (build + fwd/bwd smoke; NOT launched)
TOTAL **56,018,226** (readout output 448→384 shrinks the unembedder input, saving 2,121,920:
unembedder 14,680,064→12,582,912 = -2,097,152, readout decompress -24,640, ln_readout -128).
Per-site q/k/v/out_proj totals unchanged (q_lut 4,290,060 · k_lut 4,290,060 · v_lut 5,469,708
· out_proj 25,389,324); readout_lut 1,893,122 · unembedder 12,582,912 (384→32768) · tok_emb
2,097,152. Per-layer q grows 346,370 (L0) → 1,083,650 (L5).
No duplicate config keys; effective tph q/k=4, v=8, out=128.

Everything else = exp_g_0021: E=64, d_qk=64, d_v=16, H=6, 6 layers, seq 512, vocab 32768,
24,576 tok/step, 16k steps, Lion-on-LUT / AdamW. Shared lutorch NOT modified.
