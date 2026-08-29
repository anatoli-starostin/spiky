# SUPERSEDED — HyperplaneMHL inside the sandwich (not the intended architecture)

**Stopped at step 800 of 4,000. Superseded by
`exp_g_0029_exp_n_0121_hyperplane_pure_4k`. Kept as a record, not a result.**

## What this build did, and why it was wrong

The intent was to use HyperplaneMHL as a **direct, pure replacement** for the FFN LUT:
no compress Linear, no decompress Linear, every table cell storing a full 384-dim
output.

This build instead kept exp_n_0121's CompressionMHL **sandwich** —
`compress Linear(384 → 4×48)` → per-head LUT over 48 dims → `decompress Linear(4×48 → 384)`
— and replaced only the inner FastMHL filling with HyperplaneMHL. The tell is in the
numbers: table params stayed at **37,748,736** and the total rose only **+1.79%**
(67,351,692 → 68,555,952), all of it hyperplane `w`/`b`. The intended change grows the
tables ~8× because the cell width goes 48 → 384.

| | table params | total | vs exp_n_0121 |
|---|---|---|---|
| exp_n_0121 (FastMHL in sandwich) | 37,748,736 | 67,351,692 | — |
| **this build** (Hyperplane in sandwich) | 37,748,736 | 68,555,952 | +1.79% |
| exp_g_0029 pure (Hyperplane, no sandwich) | **301,989,888** | **340,166,412** | +405% |

## What it did produce

800 steps before it was stopped, on exp_n_0121's held 16k LR schedule:

```
step 200   val_bpb 2.617427
step 400   val_bpb 2.137053
step 600   val_bpb 1.946554
step 800   val_bpb 1.847736
```

These track exp_n_0121's own first evals closely, which is expected: with
`hyperplane_init="anchor_pairs"` the model starts bit-for-bit as the FastMHL the
parent started from, and 800 steps is early for the learned hyperplanes to have
diverged much.

That makes this a partial answer to a *different*, still-open question — **does
learning the hyperplanes help while keeping the compression sandwich?** — which is a
much cheaper variant than the pure one (+1.8% params vs +405%). It was not run to
completion and no conclusion should be drawn from 800 steps.

The code here is intact and correct *for what it does*: see
`local_hyperplane_compression.py`. Re-running it to 4,000 steps would answer that
question properly if it is ever worth asking.
