# exp072 — Sweep-A champion (6h) with LUT tables on LION (4k steps)

Exact clone of the Sweep-A champion **exp045 / exp070** (CompressionMultiHeadLUT FFN slot:
n_heads=6, inner 64/64, nap6, gamma0, independent per-head, hard; untied; d384/6L/6H;
device_bs 48, total_bs 24576, lr 3e-4, wd 0.1, warmup 0.1, seed 1; **4096 steps**), with ONE
change: the **FastMHL LUT-table weights are optimized by Lion instead of AdamW**.

## Optimizer (hybrid; reuses the exp006–017 / examples/lutgpt pattern verbatim)
Two optimizers, both under the shared warmup(0.1)+cosine(→0.1×) schedule:
- **Lion** on the FastMHL LUT tables — lr **2e-4**, betas **(0.9, 0.95)**, weight_decay 0.
  (Same Lion impl and config the earlier hyperplane_ffn experiments used; β2=0.95 is the
  documented sweet spot, effective lut_lr ≈ 2e-4.) → **5,308,416** params.
- **AdamW** on everything else (compress/decompress linears, attention, embeddings, norms,
  lm_head) exactly as before: 2-D → wd 0.1, 1-D → no wd, lr 3e-4, betas (0.9,0.95). →
  **30,488,832** params.
Total **35,797,248** (== exp045/exp070, param-matched to exp032/exp002 within +0.013%).

Only the optimizer of the LUT tables differs from exp045 (which put them in the AdamW
no-weight-decay group). Everything else — architecture, data, schedule, seed — is identical.

## Comparison (fully-annealed 4096-step runs only)
Compare against the runs whose LR schedule is also fully annealed by step 4096:
- **exp032** (4k dense-FFN yardstick) = 1.39371
- **exp045** (the AdamW-LUT champion, same config, 4096 steps) = 1.39063
(NOT exp070's 16k trajectory at step 4096 — its cosine is only ~25% annealed there, so it is
not a valid same-schedule comparison.)

## Status
Launched under the owner's GO (GPU free). Own Slack progress bar. Outputs: metrics.csv,
summary.json, loss.png, checkpoint.pt (gitignored).
