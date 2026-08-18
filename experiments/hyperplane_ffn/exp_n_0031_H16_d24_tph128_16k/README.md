# exp_n_0031 — H16, inner d=24, tph=128, nap6, tied, 16k

Same as **exp_n_0030** (H=16, inner d=24, nap6, tied, vanilla backbone, 16k) but **tph 64 → 128** —
isolates the tables-per-head effect at the H16/d24 point. H·d = 16·24 = 384 → still fixed-throughput,
~4× cheaper FFN matmul than dense (2·384·384 = 294,912 vs 8·384² = 1,179,648).

**Params (SMOKE-confirmed):** tables 16·128·2⁶·24 = 3,145,728 (double exp_n_0030's 1,572,864) → per-layer
147,840 + 147,840 + 3,145,728 = 3,441,408 → total = 16,131,840 + 6·3,441,408 = **36,780,288**
(≈ 1.585× dense's 23,209,728; same total as the other tph128/H·d=384 points exp_n_0004/0005).

Runs 16k directly, serial after exp_n_0030. Compares: (a) vs exp_n_0030 (H16/d24/tph64 = 27.34M) to
isolate tph 64→128; (b) vs exp_n_0004 (H8/d48/tph128) and exp_n_0005 (H12/d32/tph128) to extend the
tph128 head/dim split family to H16; (c) vs tied dense 16k (1.19665).
