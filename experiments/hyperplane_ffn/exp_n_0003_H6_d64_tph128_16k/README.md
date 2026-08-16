# exp_n_0003 — H6, inner d=64, tph=128 (H·d=384), 16k

Clone of **exp_n_0002** with **n_heads 12→6**, inner **d=64** (so H·d back to **384** — on the
fixed-throughput line), and **tph 84→128**. Everything else identical: nap=6, tied, gamma0,
AdamW-LUT two-group, 16000 steps, standard config (device_bs 48, total_bs 24576, depth 6,
n_embd 384, vocab 32768, seq_len 512).

**Params (SMOKE-confirmed):**
- compress = 6·64·385 = 147,840
- decompress = 6·64·384 + 384 = 147,840
- tables = 6·128·2⁶·64 = 3,145,728  (out_width = inner_out = 64)
- per-layer = 3,441,408 → total = 16,131,840 + 6·3,441,408 = **36,780,288**
- vs exp_n_0002 (H12/d64/tph84 = 44,450,304): **−7,670,016**
- vs exp_n_0001 D5 (H12/d32/tph84 = 30,292,224): **+6,488,064**
- vs 2× budget (30,287,616): **+6,492,672 (+21.4% over)**; vs tied dense (23,209,728): +13,570,560

**Throughput:** H·d=384 restores projection cost 2·384·384 = 294,912 MACs/layer (~4× cheaper than
dense's 1,179,648) — back on the fixed-throughput line. Params still over budget because tph=128 at
out-width 64 is a lot of tables.

Purpose: at 16k, does fewer/wider heads (H6/d64) with many tables (tph128) beat the H12 variants?
Compare to (a) exp_n_0002 H12/d64 = 1.20823, (b) exp_n_0001 D5 H12/d32 = 1.22473, (c) tied dense 16k = 1.19665.
