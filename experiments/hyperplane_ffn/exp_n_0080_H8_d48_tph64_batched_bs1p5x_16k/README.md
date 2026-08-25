# exp_n_0080 — 1.5×-batch clone of exp_n_0052 (batched LUT path)

Exact clone of **exp_n_0052_H8_d48_tph64_batched_control_16k** (the 1× batched-path LUT
control) with the effective batch scaled to **1.5×**, mirroring how exp_n_0046 scaled its
base:

| | base (exp_n_0052) | 1.5× (this / exp_n_0046) |
|---|---|---|
| device_batch_size | 48 | **72** |
| total_batch_size (tokens/step) | 24,576 | **36,864** |
| grad_accum | 1 | 1 (one 72-seq micro-batch) |
| n_steps | 16,000 | 16,000 (unchanged) |

Everything else identical to exp_n_0052: depth 6 / n_embd 384 / 6-head attn / seq 512,
LUT H8 / d48 / tph64 / nap6, **batched path** (`lut_batched_multi_head_input=true`), hard
forward, learnable temps, tied dense, lr 3e-4 cosine (10% warmup), wd 0.1, seed 1.
Params 27,343,116.

**vs exp_n_0046:** exp_n_0046 is the same H8/d48/tph64 model at 1.5× batch but on the
per-head **loop** path (no `lut_batched_multi_head_input`). This run is the **batched-path**
1.5× analog (forward-equivalent to the loop path, faster). Reference: exp_n_0046 final
val_bpb **1.196862**; 1× batched control exp_n_0052 = **1.228552**. So this tests whether
the batched 1.5× reproduces exp_n_0046's dense-parity ~1.197.

Outputs: metrics.csv, summary.json, loss.png, checkpoint.pt.
