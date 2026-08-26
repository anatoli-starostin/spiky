# exp_n_0085_vlut_C_nap6 — STOPPED EARLY at step 5200 / 16000

Routed-V experiment: the dense Linear V projection in attention is replaced by a
CompressionMHL v_lut (config "C", nap6). q/k stay dense; out_proj, FFN slot
(n_heads=4/inner48/nap7/tph256), and the untied unembedder are all kept.
v_lut = CompressionMultiHeadLUT(384→384, n_heads=4 routing, inner_in=24,
inner_out=24, nap=6, tph=8, hard, batched); its 384 output is reshaped to 6
attention heads × 64 for SDPA. See config.json `_arch_note` for full detail.
Total params 67,207,128.

## Why stopped early (SIGTERM at step 5200)
It tracked **~+0.115 bpb BEHIND exp_n_0084** (dense Linear V) at matched eval steps
throughout early training — e.g. @step 4400: 0085 = 1.4514 vs 0084 = 1.33633
(Δ +0.115); @step 5200 (last): 0085 = 1.42416. The routed low-bandwidth V
(nap6, tph8, inner_out24) was not competitive with the dense Linear V early, and
completing the remaining ~1.5 h was not worth it. Stopped cleanly (SIGTERM to the
training pid), GPU released, partial curve preserved here.

## Successor
Superseded by **exp_n_0086_vlut_5x_tph88**, which enlarges the v_lut ~5× in params
via tph 8→88 (nap=6 and inner_out=24 UNCHANGED) — testing whether more table
capacity (at low bandwidth) closes the gap to the dense Linear V.

## Partial artifacts
- `metrics.csv` — val_bpb curve through step 5200 (last partial val_bpb 1.42416).
- `loss.png` — the partial curve vs exp_n_0084 and the 1.1977 tied baseline.
- No `summary.json` (never reached 16k); `checkpoint.pt` / `run.log` are gitignored.
