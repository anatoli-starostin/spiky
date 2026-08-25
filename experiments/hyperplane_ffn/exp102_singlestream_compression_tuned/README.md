# exp102 — single-stream CompressionMHL attention, PER-SITE tuned

Clone of **exp101_singlestream_compression_sep_qk** with per-site LUT tuning (everything
else identical: single residual stream, separate q/k/v via the local
`CompressionMultiHeadLUTMH` multihead_output subclass, out_proj CompressionMHL, Lion-on-
LUT-tables / AdamW split, hard forward, 16k steps, batch 48 seq / 24,576 tok/step, seed).
Shared lutorch NOT modified.

**Per-site LUT config (the only change vs exp101):**
| site | n_heads | inner_in | nap | tph | inner_out | output |
|------|---------|----------|-----|-----|-----------|--------|
| q_lut, k_lut | 6 | 48 | 4 | 32 | −1 (no decompress) | [N,6,64] |
| v_lut | 6 | 48 | 6 | 64 | −1 | [N,6,64] |
| out_proj | 8 | 48 | 6 | 128 | 48 | [N,384] |

(vs exp101: all q/k/v were nap6/tph32; out_proj was tph32. exp102 shrinks q/k routing
(nap4/tph32) and grows v (tph64) and out_proj (tph128) — informed by the out_proj sweep
where tph128 was best.)

**Verified param counts (build-smoke, forward+backward OK):**
- q_lut = 1,844,940 · k_lut = 1,844,940 · v_lut = 10,102,476 · out_proj = 20,648,460
  (each summed over the 6 layers)
- rest (attn-free: tok_emb + untied unembed + LayerNorms) = 25,172,736
- **TOTAL = 59,613,552** (~59.6M; exp101 was 47.8M — the extra is v tph64 + out tph128 tables)

Per-site params live under distinct config keys (`qk_nap/qk_tph`, `v_nap/v_tph`,
`out_nap/out_tph`, …). NOTE: the cloned config carried stale exp024 Hyperplane-era keys
`v_tph`/`out_tph` — duplicate JSON keys silently overrode the new values until removed;
fixed here.

**Baselines:** exp101 (all-tph32) 16k final val_bpb **1.30213**; exp024 single-stream
Hyperplane **1.2034**; exp073 23M dense **1.19665**.

Outputs: metrics.csv, summary.json, loss.png, checkpoint.pt.
