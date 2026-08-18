# exp_n_0034 — H16/d24, nap5, tph128, learnable_temps, tied, 16k

Clone of **exp_n_0033** (H16/d24, tied, vanilla, std0.02 compress, learnable_temps=True, 16k) with the
routing/table granularity changed: **nap 6→5** (2⁵=32 clusters/table instead of 64) and **tph 64→128**
(twice the tables/head). Trades routing RESOLUTION for table MULTIPLICITY.

**Fixed budget:** the product 2^nap · tph = 32·128 = 4096 = 64·64 (exp_n_0033), so the table tensor size —
and thus the param count and FLOPs — stay the same. tables/layer = 16·128·2⁵·24 = 1,572,864 (identical to
exp_n_0033). H·d = 16·24 = 384 → ~4× cheaper FFN matmul than dense (unchanged). This isolates **finer
routing (nap6, more clusters) vs more table-mixing (nap5 + 2× tables) at fixed budget**.

**Params = 27,343,296 (SMOKE-confirmed)** = 27,343,104 + 192 learnable temp scalars (16 heads × 2 × 6
layers) = 1.178× tied dense (23,209,728). Same as exp_n_0033.

**learnable_temps = True** (config lut_learnable_temps=True; also now the FastMHL primitive default). Runs
16k, serial after exp_n_0033. Compare to exp_n_0033 (nap6/tph64, 1.229xx expected) and exp_n_0030
(fixed-temp nap6/tph64, 1.22936) and tied dense (1.19665).
