# exp_n_0004 — H8, inner d=48, tph=128 (H·d=384), 16k

Clone of **exp_n_0003** with **n_heads 6→8** and inner **d 64→48** (H·d stays **384**, on the
fixed-throughput line), tph=128. Everything else identical: nap=6, tied, gamma0, AdamW-LUT
two-group, 16000 steps, standard config (device_bs 48, total_bs 24576, eval_every 200, depth 6,
n_embd 384, vocab 32768, seq_len 512, lr 3e-4).

**Queued to run AFTER exp_n_0003 finishes — NOT concurrent on the H100.**

**Params (SMOKE-confirmed):** identical to exp_n_0003 = **36,780,288**, because both fix H·d=384 and
tph=128, and every term depends only on those: compress 8·48·385 = 147,840, decompress 8·48·384+384 =
147,840, tables 8·128·2⁶·48 = 3,145,728 (= 128·2⁶·384) → per-layer 3,441,408 → total 36,780,288
(+21.4% over the 2× budget). So exp_n_0003 (H6/d64) vs exp_n_0004 (H8/d48) is a pure **head/dim split**
comparison at fixed params, fixed throughput, tph=128, 16k.

Purpose: at 16k with tph=128, does the H8/d48 split beat H6/d64 (exp_n_0003) and the H12 variants?
Compare to (a) exp_n_0003 H6/d64, (b) exp_n_0002 H12/d64 = 1.20823, (c) exp_n_0001 D5 H12/d32 = 1.22473,
(d) tied dense 16k exp073 = 1.19665.
