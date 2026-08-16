# exp073 — TIED vanilla dense-FFN baseline at 16k steps

The long-run **tied dense** baseline (we only had exp055 at 4096 steps). Standard dense FFN
slot (Linear 384→1536 → GELU → Linear 1536→384), **tied unembedder**
(`lm_head.weight = tok_emb.weight`, same storage — verified), trained at the full exp002
budget (16000 steps). Otherwise identical to exp055/exp002: MinimalGPT+RoPE d384/6L/6H/seq512,
device_bs 48, total_bs 24576, lr 3e-4, wd 0.1, warmup 0.1, seed 1, eval_every 200.

**Params: 23,209,728** = tied floor 16,131,840 + 6×dense-FFN 1,179,648. (Tying dedups the
12.58M head; confirmed by the reduced count + `head.weight is tok_emb.weight`.)

Matched-pair partner: **exp074** (tied LUT 6-head, same 16k budget). This is the tied-16k
reference the LUT run is measured against.

Launched after exp072 freed the GPU (own progress bar). Outputs: metrics.csv, summary.json,
loss.png, checkpoint.pt (gitignored).
