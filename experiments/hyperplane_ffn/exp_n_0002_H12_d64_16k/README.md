# exp_n_0002 — H12, inner d=64 (H·d=768), 16k

Clone of **exp_n_0001** (D5 champion, H12/d32, 16k) with the inner dim raised **d=32 → 64**
(n_heads stays 12). Everything else identical: nap=6, tph=84, tied, gamma0, AdamW-LUT
two-group, 16000 steps, standard config (device_bs 48, total_bs 24576, depth 6, n_embd 384,
vocab 32768, seq_len 512).

**Breaks the fixed-throughput constraint (intended).** d=64 doubles H·d 384→768, so the
compress+decompress projection cost doubles (2·H·D·d = 2·768·384 = 589,824 MACs/layer, vs
294,912 before — now only ~2× cheaper than the dense FFN's 1,179,648, not ~4×). With tph held
at 84 the param count rises well above the 2× budget:
- compress = 12·64·385 = 295,680
- decompress = 12·64·384 + 384 = 295,296
- tables = 12·84·2⁶·64 = 4,128,768
- per-layer = 4,719,744 → total = 16,131,840 + 6·4,719,744 = **44,450,304**
- vs exp_n_0001 (30,292,224): **+14,158,080 (+46.7%)**; vs 2× budget (30,287,616): **+46.8% over**

Purpose: does a wider per-head compressed vector (d=64) at H=12 beat the narrow-inner champion
at the full 16k budget? Compare to (a) exp_n_0001 D5 16k = 1.22473, (b) tied dense 16k exp073 = 1.19665.
