# exp_n_0001 — 16k confirm of Sweep-D champion D5 (exp098, H12/d32)

Full 16000-step run of the Sweep-D winner **D5 = exp098** (CompressionMHL FFN slot, tied).
Clone of exp098's exact config/architecture with **ONLY n_steps 4096 → 16000** changed;
everything else identical.

- CompressionMHL FFN slot: n_heads=12, inner_in=inner_out=32 (H·d=384), nap=6, tph=84,
  joint=False, gamma=0, tied embeddings, AdamW two-group (LUT tables no-wd; compress/decompress
  Linears wd=0.1). depth=6, n_embd=384, vocab=32768, seq_len=512, device_bs=48, total_bs=24576,
  eval_every=200, lr=3e-4. **Total params = 30,292,224** (identical to exp098, smoke-confirmed).
- `_n_` prefix avoids merge collisions with gpustar's `_g_` runs.

Purpose: does the 4k-champion split hold up at the full budget? Compare final val_bpb to:
(a) 4k D5 = 1.35966, (b) tied dense 16k exp073 = 1.19665, (c) tied dense 4k exp055 = 1.35543.
