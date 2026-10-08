# Stage B(c): value-embedding ablation (d24_1xh100 @ 5568)

Self-measured unablated baseline val bpb = **0.720741** (val shard_06542, split_tokens=20971520, eval_steps=1280, DBS=8, FA3/SSSL, bf16).

| VE layer | val bpb | Δ vs baseline |
|---|---|---|
| 23 | 0.727369 | +0.006628 |
| 15 | 0.725197 | +0.004456 |
| 21 | 0.725087 | +0.004346 |
| 19 | 0.724938 | +0.004197 |
| 17 | 0.723582 | +0.002841 |
| 13 | 0.722936 | +0.002195 |
| 11 | 0.722655 | +0.001914 |
| 9 | 0.722012 | +0.001271 |
| 1 | 0.721691 | +0.000950 |
| 3 | 0.721684 | +0.000943 |
| 7 | 0.721526 | +0.000785 |
| 5 | 0.721507 | +0.000766 |
| **all 12** | 0.858214 | +0.137473 |

Peak reserved GPU memory (this process, capped at fraction 0.22): 16.42 GiB.
