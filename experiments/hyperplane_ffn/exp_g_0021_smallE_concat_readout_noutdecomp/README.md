# exp_g_0021 — concat readout + readout LUT, out_proj decompress REMOVED

Tracking issue: **#108**. Run on **gpustar (RTX 5090)**. Commit `4b65d350` (code), results here.

`exp_g_0020_smallE_concat_readoutlut` with **one** change: `out_inner_out` 48 → −1, removing
out_proj's decompress stage. Config-only — `train.py` is byte-identical to exp_g_0020's.

## Result

**final = best = `1.3406161999734172`** · 49,356,082 params · **1.877 h** · 16,000 steps.

Converged: the last five evals read 1.342177, 1.342091, 1.341470, 1.341588, **1.340616** — flat to
~1e-3 over the final 800 steps.

| reference | val_bpb | Δ 0021 |
|---|--:|--:|
| exp010 dual-stream | 1.1940 | **+0.146616** |
| exp024 single-stream | 1.2034 | **+0.137216** |

**This whole small-E line is well behind the E=384 baselines** — +0.137 against exp024 is not a
near miss, it is a different regime. Shrinking E 384 → 64 and reading out by concatenation costs
far more than the readout LUT and the out_proj simplification give back.

## What the ablation chain actually shows

| arm | change vs previous | matched-step outcome |
|---|---|---|
| exp_g_0018 | small E + concat, **no residual** | — |
| exp_g_0019 | **+ residual** | residual wins: ahead at 16/16 evals after the step-2,200 crossover |
| exp_g_0020 | **+ readout LUT** | (only 8 evals before it was stopped) |
| **exp_g_0021** | **− out_proj decompress** | see below |

**vs exp_g_0019** (41 common evals, 200 … 8,200): **0021 ahead at 41/41**, −0.024302 at the last
common step, mean −0.032 over the first third and −0.022 over the last. **vs exp_g_0018**
(26 evals): ahead at **26/26**, −0.025586 at step 5,200.

So the 0019 → 0021 step is a solid ~0.024 improvement, sustained across every matched eval.
**But that comparison spans two changes** — the readout LUT *and* the removed decompress — so it
does not attribute the gain to either one.

**The isolated decompress question is still unanswered.** exp_g_0021 vs exp_g_0020 is the
single-variable pair, and exp_g_0020 was stopped at step 1,600, leaving only 8 common evals that
end in a tie (+0.000174, after 0021 had led by up to −0.0079 mid-range). At 10% of the budget on
one seed, that settles nothing — and this line has reversed late repeatedly (exp_g_0018 led for
2,200 steps then lost permanently; exp_g_0016b led until step 6,000 then reversed). **Running
exp_g_0020 to 16k is what would actually answer it.**

## Parameter context

Removing the decompress made the model **bigger**: 43,212,466 → **49,356,082** (+6,143,616,
+14.2%). With `inner_out_dim = −1` the LUT's effective output width becomes `output_dim`, so every
out_proj table cell widens 48 → 64:

```
out_proj, per layer          exp_g_0020    exp_g_0021       delta
  tables (8 × 128 × 2^6 × W)  3,145,728     4,194,304   +1,048,576   W: 48 → 64
  compress Linear(96 → 384)      37,248        37,248            0
  decompress Linear(384 → 64)    24,640             0      -24,640
  TOTAL                       3,207,618     4,231,554   +1,023,936
```

The removed Linear saves 24,640; the cell widening costs 1,048,576 — **43× more**. A simplification
of the data path, an increase in parameters. out_proj is now **51.4%** of the model.

Memory went the other way: peak GPU **13,488 → 11,144 MiB (−17%)**, because the wide
`[N, 8·48=384]` concatenated intermediate leaves the forward/backward graph.

## Verified by execution, not construction

`out_proj.has_decompress = False`, `decompress` is `nn.Identity` holding **zero** parameters, and
running it gives `out_proj(4, 96) → (4, 64)` — the 8 heads are summed via the batched path's
`y.sum(dim=1)` rather than concatenated and projected.

Incidentally, q/k/v were **already** decompress-free (`inner_out_dim = −1` from the start, emitting
at `output_dim` directly). The readout LUT is now the **only** decompress left in the model.

## Status

Complete. Results committed. Single seed.
