# exp070 — Sweep-A champion at FULL 16k steps (LUT-vs-dense crossover test)

The Sweep A champion FFN slot — **exp045 = CompressionMultiHeadLUT, n_heads=6, inner_in=inner_out=64,
nap=6, gamma=0, independent per-head, hard** — trained at the **full exp002 budget: n_steps=16000**
(vs 4096 in the sweep). Everything else is byte-identical to exp002 / exp045: untied unembedder,
MinimalGPT+RoPE d384/6L/6H/seq512, device_bs 48, total_bs 24576, lr 3e-4, wd 0.1, warmup 0.1,
eval_every 200, seed 1, fp32, same data.

**Params:** 35,797,248 (same as exp045; param-matched to exp032/exp002 within +0.013%).

## Purpose — crossover
At 4096 steps exp045 (1.3906) already edges past the 4k dense-FFN point (exp032, 1.39371). This
run tests whether the LUT-based slot holds up at the FULL budget vs the dense-FFN reference
**exp002 (untied vanilla dense FFN, 16000 steps, final val_bpb 1.20144)**. The comparison logs
val_bpb at every eval (steps 200, 400, …, 16000) for both, and identifies whether/where a
crossover occurs (LUT ahead early → dense pulls ahead late, or the reverse).

## Scheduling
Queued to launch on the free H100 **after** the exp043–069 sweep finishes (≤4-concurrent cap
respected; no oversubscription). Its own Slack progress bar. Est. ~1–1.5 h solo.

Outputs: `metrics.csv` (val_bpb per eval), `summary.json`, `loss.png`, `checkpoint.pt` (gitignored).
