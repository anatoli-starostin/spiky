# exp_n_0046 — H8/d48/tph64, hard forward, tied, **1.5× batch**, 16k

Single-slot CompressionMHL, **H8 / d48 / tph64 / nap6, hard forward, tied, learnable_temps=true**, plain AdamW
(0033 grouping: lr 3e-4, betas 0.9/0.95, eps 1e-8, 2-D decay wd 0.1 / LUT+temps+1-D nodecay wd 0), seq_len 512,
depth 6, n_embd 384, n_head 6, seed 1, bf16, 16k steps. **The only change vs a standard H8/d48/tph64 hard run is
a 1.5× larger batch:** device_batch_size 48→72 and total_batch_size 24576→36864, i.e. **36,864 tokens/step**
(grad_accum stays 1). Tests whether 1.5× more total training tokens (denser per-cluster surrogate gradients per
step) moves the tph64 hard config's final val_bpb.

**Params = 27,343,200 (SMOKE-confirmed)** — batch is a zero-param change; identical to the H8/d48/tph64 shape
(= exp_n_0044's count). LUT tables 9,437,184 (1× exp_n_0033's budget). = 1.178× tied dense.

> **NOTE / spec discrepancy (flagged for confirmation before launch):** the task said to "clone exp_n_0033
> (H8/d48/tph64)", but **exp_n_0033 is actually H16/d24/tph64 (27,343,296 params)** — NOT H8/d48. The required
> param target (27,343,200), the name (`H8_d48`), and the explicit "H8/d48/tph64" description all point to
> **H8/d48/tph64**, which is exp_n_0044's shape (there is no nebius *hard* H8/d48/tph64 baseline — that config
> is gpustar's exp_g_0006). So this was built as **H8/d48/tph64 hard** by cloning exp_n_0044 (the real H8/d48/tph64
> config) and setting `lut_forward_mode=hard` + the 1.5× batch. Param count 27,343,200 matches the target exactly.
> Consequently the "diff vs exp_n_0033 = exactly the batch field" check does NOT hold (0046 differs from the real
> 0033 in H, d, and batch); and the batch change is TWO config fields (device + total batch), not one.

Not launched (built + SMOKE only); serial after exp_n_0045.
