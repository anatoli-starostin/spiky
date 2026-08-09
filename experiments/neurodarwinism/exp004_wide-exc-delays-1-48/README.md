# exp004 — excitatory delays widened to 1–48 (96 metas)

**Hypothesis.** The readout is time-to-first-spike, so the only way the network can express a
*rank* is by separating spike times. exp002 measured only 3.42 distinct first-spike ticks per
state across 6 outputs. If the delay range is what limits temporal separation, widening it from
1–20 to **1–48** should give the network more room to spread its outputs and cut the tie rate.

**Setup.** As exp002's architecture (800 exc / 200 inh, ~97k synapses) but at **K=32**, with
`--d-max 48` — **96 synapse metas** instead of 40, one per (delay, sign). 300 rounds.
`config.json` is reconstructed, but **d_max = 48 is measured**: the checkpoint's delays span
[1, 48] over 48 distinct values.

## Result — the extra range was allocated and then left empty

| | |
|---|---|
| rounds | 300 / 300 |
| peak corrected tau-b | **+0.3995** at round 281 |
| final EWMA best / mean | +0.3571 / +0.3540 |
| **held-out corrected tau-b** | **+0.2761 ± 0.0321** (member 31, 80,781 synapses) — recovered post-hoc, see below |
| pool collapsed at round | 63 |
| synapses per net | 97,094 → 80,857 |
| wall | 2153 s |

Peak +0.3995 at K=32 is above the K=32 baseline's +0.3498 and below K=128's +0.4308 — so
widening the delays did help somewhat. But the checkpoint shows it did **not** help for the
reason the hypothesis proposed.

**The load did not move outward.** The |w|-weighted mean delay is **5.18 ticks** — against
**5.20** in exp002 at d_max 20. Quadrupling the available range moved the centre of mass by
0.02 ticks. The raw (unweighted) mean delay dutifully doubles to 23.0, because synapses are
*allocated* uniformly across the range, but the ones carrying weight stay at the short end.

**And it made the sparsity worse.** The fraction of excitatory synapses carrying non-negligible
weight fell from 23.2 % (exp002) to **12.0 %** — the lowest of any run — with **88.0 % dead**
and only 10.9 % at the ceiling. Spreading the same synapse budget over 2.4× as many delay bins
halves the number that end up mattering.

**The tie rate barely moved.** 0.2023 here vs 0.2187 in exp002, with 3.71 distinct first-spike
ticks per state vs 3.42. Teacher agreement is the chapter's best at **0.6324** (0.7928
excluding ties), and corrected tau-b **0.3859** — but the mechanism under test contributed
almost none of that.

## The held-out number, and why the original run could not produce it

All 300 rounds completed. The held-out evaluation then died:

```
RuntimeError: CUDA runtime API error cudaErrorMemoryAllocation
  at native/spiky/misc/misc.cpp:56
  in build_pool -> sp.add_connections   (steady_state.py:731)
```

and the supervisor, doing exactly what it should, resumed from the checkpoint and hit the
identical failure **40 times in a row** before giving up (`supervisor.log` records every
attempt).

**It was never out of memory.** Re-running the same evaluation standalone, with 29 GB of the
GPU free, reproduced the failure exactly — which is what exposed the real cause.
`steady_state.py:816` is

```python
hb = build_pool([genomes[best_i]], dev, seed=1)      # no stdp_lr, no w_max
```

and `build_pool` picks its meta bank off `stdp_lr`. With `stdp_lr == 0` it builds
`harness.delay_metas()`, whose `d_max` is a **default argument bound at import** from
`harness.D_MAX = 20` — so it ignores `--d-max` entirely — and it applies no inhibitory bank
offset. This run's genome carries meta indices up to **95** (d_max 48 → `stage2_metas`' 96-meta
plastic + frozen banks). The engine indexed 96 metas' worth of synapses into a 20-meta network,
dereferenced out of range, and asked the allocator for a nonsense size. An addressing bug
wearing an OOM's clothes — which is also why more memory never helped and 40 restarts all died
the same way.

Recovered with `src/eval_heldout.py`, which rebinds `D_MAX`/`N_DELAY_METAS` the way
`steady_state.main()` does and passes `--stdp-lr` so the bank matches the genome:

```sh
python ../src/eval_heldout.py --ckpt <ck_delay148.npz> --d-max 48 --stdp-lr 0.01 --repeats 10
```

| | mean ± std over 10 builds |
|---|---|
| **held-out corrected tau-b** | **+0.2761 ± 0.0321** (range +0.2225 … +0.3170) |
| raw tau-b | +0.3225 ± 0.0477 |
| own null | +0.0464 ± 0.0245 |
| tie rate | 0.1654 ± 0.0223 |
| distinct first-spike ticks / state | 4.04 ± 0.20 |
| silent outputs | 0.00 % |

**Two caveats, both load-bearing.** The ± is not sampling noise on the held-out set — it is
**build noise**: constructing the same genome ten times gives ten different networks, spread
0.032 in corrected tau. And this is the **post-#92 engine**; every held-out number recorded
during the original runs came from the buggy one and does not reproduce. See
[the chapter README](../README.md#a-warning-about-every-held-out-number-here) before comparing
this figure with exp002's or exp003's. `heldout_eval.json` holds all ten draws.

Against its own final training EWMA (+0.3571) the held-out mean is **0.081 lower** — the widest
train/held-out gap of any run in the chapter (exp002's is 0.002). The 96-meta configuration
generalises worse, not just no better.

## Reading

Delay range is not the constraint. The network does not *use* long delays even when it has
them; it uses the short end and lets the rest go dead. Any future attempt at temporal
separation should act on the readout — the length of the readout window, a graded rather than
first-spike decode, or an evolvable per-output threshold — not on the delay budget.
