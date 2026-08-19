# exp_n_0047 — H8/d48/tph64 hard, standard batch (48), 24000 steps — same-total-tokens control for exp_n_0046

Clone of **exp_n_0046** with three changes: `device_batch_size` 72→48, `total_batch_size` 36864→24576,
`n_steps` 16000→24000. Everything else identical (H8/d48/tph64 hard forward, tied, nap6, seq_len 512, depth 6,
n_embd 384, n_head 6, seed 1, bf16, AdamW lr 3e-4 betas 0.9/0.95 eps 1e-8, 0033 grouping).

**This is the same-total-tokens CONTROL for exp_n_0046:** both see **589,824,000 total training tokens**
(0047 = 48·512·24000; 0046 = 72·512·16000) — 0047 gets there with **standard batch + more steps**, 0046 with
**bigger batch + fewer steps**. Isolates "more tokens via bigger batch (0046)" vs "more tokens via more steps
(0047)" against the exp_g_0006 baseline (1.228335, which used 24,576×16000 = 393,216,000 tokens). If 0046 and
0047 land close, the gain (if any) is from total tokens, not batch shape.

**Resolved batch:** device_bs 48 × seq 512 = **24,576 tokens/step**, `grad_accum = 24576 // 24576 = 1`,
effective batch 24,576. **Warmup:** `lr_warmup_fraction=0.1` is a FRACTION of n_steps, so it scales to
**2400 warmup steps** over 24000 (cosine decay to the 0.1 floor over the full 24k) — schedule stretches
correctly. **Eval:** `eval_steps=10` at device_bs 48 → **245,760 val tokens** — matches the exp_g_0006/exp_n_0044
baseline exactly (val_bpb directly comparable, unlike 0046's 368,640 at device_bs 72).

**Params = 27,343,200 (SMOKE-confirmed)** — batch/steps are zero-param changes; identical to exp_n_0046.

Config diff vs exp_n_0046: exactly `device_batch_size`, `total_batch_size`, `n_steps`, `exp_name`. Not launched
(built + SMOKE only); serial after exp_n_0046.
