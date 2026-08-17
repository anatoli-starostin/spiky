# exp_n_0005 — H12, inner d=32, tph=128 (H·d=384), 16k

Clone of **exp_n_0004** with **n_heads 8→12** and inner **d 48→32** (H·d stays **384**,
fixed-throughput line), tph=128. Everything else identical: nap=6, tied, gamma0, AdamW-LUT
two-group, 16000 steps, standard config (device_bs 48, total_bs 24576, eval_every 200, depth 6,
n_embd 384, vocab 32768, seq_len 512, lr 3e-4).

**Params (SMOKE-confirmed) = 36,780,288 — identical to exp_n_0003 and exp_n_0004**, because tables =
H·tph·2⁶·d = tph·2⁶·(H·d) = 128·2⁶·384 = 3,145,728 depends only on H·d and tph (not the split), and
compress/decompress depend only on H·d: compress 12·32·385 = 147,840, decompress 12·32·384+384 = 147,840
→ per-layer 3,441,408 → total 36,780,288 (+21.4% over the 2× budget).

This is the **split-optimum candidate**: the tph=128/16k family is now H6/d64 (exp_n_0003) < H8/d48
(exp_n_0004) < **H12/d32 (this run?)** — more-but-narrower heads have been winning; H12 is the expected top
of this fixed-params/fixed-throughput line, mirroring the 4k Sweep-D optimum.

Compare to: exp_n_0004 H8/d48 = 1.21738, exp_n_0003 H6/d64 = 1.21994, exp_n_0002 H12/d64 = 1.20823,
exp_n_0001 D5 H12/d32/tph84 = 1.22473, tied dense 16k exp073 = 1.19665.
