# exp_g_0048 — function-emitting cells, 24 gaussians per cell (HALTED at step 2,800)

**Halted on request at step 2,800 of 4,000 for trailing. Last eval
`val_bpb = 1.5671`.** No summary.json and no checkpoint: both are written only at
the end of training.

| | val bpb @ 2,800 | params |
|---|---|---|
| **exp_g_0048** | **1.5671** | 85,337,868 |
| exp_n_0136 raw FastMHL | 1.4405 | 330,704,652 → **+0.1267** |
| exp_n_0121 compression anchor | 1.4341 | 67,351,692 → **+0.1330** |

## The idea

A raw FastMHL cell stores one learnable value per output dimension — 384 numbers to
describe one 384-long curve. Here a cell instead stores 24 gaussian bumps over the
output-index axis, `(mean, log_sigma, signed amp)` each, and its output vector is those
bumps evaluated at every index:

    W[c, i] = sum_k amp[c,k] * exp(-0.5 * ((i - mu[c,k]) / sigma[c,k])^2)

72 params per cell against 384 — 5.33x fewer, 85.3M total against the raw fork's 330.7M.
`weights` becomes a synthesised tensor instead of a stored Parameter, via a property on a
local subclass; `fast_multi_head_lut.py` is not modified.

Crucially this does **not** commute with the gather-sum — each cell has its own mu and
sigma, so its curve shape cannot be pulled out of the sum over gathered cells. That is what
makes it different in kind from a learned decompress, and also why the table must be
materialised every forward.

## Why it was stopped

It was 0.1267 behind the raw fork and 0.1330 behind
the anchor at step 2,800. For scale, the largest late reversal anywhere on this board is
exp_g_0039's, worth 0.0377 between its best and final position — an order of magnitude
smaller than this gap. It was not a case where the ordering was plausibly going to flip.

## Cost, and the fix that made it viable

Eager it ran at 5.46 s/step (24.3 h for 16k). `torch.compile` on the synthesis brought that
to 1.525 s/step and cut peak VRAM from 21.42 to 12.20 GiB — 3.58x faster and 9.2 GiB
lighter, no change to the maths. The synthesis is a long elementwise chain over a
`[chunk, K, 384]` intermediate that eager runs as six separate kernels; inductor fuses it
into one that never materialises it. `fe_compile` is on by default because of this.

Profiling also showed the synthesis runs exactly twice per slot per fwd+bwd — once forward,
once as the checkpoint recompute — which is 48 passes per optimizer step across 6 slots and
4 grad-accum micro-batches. Dropping the checkpoint OOMs and would only buy 1.22x.

## Caveat

4,000 steps on the FULL 16,000-step LR schedule — the schedule is anchored to 16,000 and
the run is simply cut, so it ends at **94% of peak LR** having traversed ~6% of the cosine.
Comparable to every other short run on this board and to the first 4k of any 16k run of the
same recipe. No full-anneal conclusion follows from it.
