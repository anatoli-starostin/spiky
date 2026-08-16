# exp074 — TIED CompressionMHL 6-head LUT slot at 16k steps

The long-run **tied LUT** run: CompressionMultiHeadLUT FFN slot (n_heads=6, inner 64/64,
nap6, gamma0, independent per-head, hard, no inner-residual), **tied unembedder**
(`lm_head.weight = tok_emb.weight`, same storage — verified), at the full 16000-step budget.
Clone of exp060 (B5, tied LUT 6h, tph=36) with n_steps 4096→16000; otherwise identical to
exp073/exp002: d384/6L/6H/seq512, device_bs 48, total_bs 24576, lr 3e-4, wd 0.1, warmup 0.1,
seed 1, eval_every 200.

**Params: 23,214,336** = tied floor 16,131,840 + 6×LUT-slot 1,180,416 (compress 147,840 +
FastMHL 6·36·64·64=884,736 + decompress 147,840). Param-matched to the tied dense slot
(1,179,648/layer) within +768/layer (+0.02% total). Tying confirmed by the count +
`head.weight is tok_emb.weight`.

Matched-pair partner: **exp073** (tied dense, same 16k budget). The long-run tied
dense-vs-LUT comparison — does the tied-dense advantage seen at 4096 steps (exp055 1.35543 vs
best tied LUT exp059 1.37955) hold, widen, or narrow at the full 16k budget?

Launched after exp072 freed the GPU (own progress bar). Outputs: metrics.csv, summary.json,
loss.png, checkpoint.pt (gitignored).
