# exp_profiling — where the Walker2d bucket-LIF SAC run spends its time, and 1.45× back

Reference config: exp_c32b (16 buckets × 32 tables), single seed, solo on the GPU.

**Result: 330.6 → 227.5 ms/iter, a measured 1.45× on the profiling harness and a projected
1.56× on a full 10,000-iteration run (≈59 → ≈38 min/seed). Every parameter bit-identical.**

## Phase 1 — the breakdown

260 iterations, updates from 60, one eval. `block_until_ready` at every stage boundary.

| stage | total s | ms/iter | % of stages | calls |
|---|---:|---:|---:|---:|
| update | 25.99 | 99.96 | 39.8% | 6,400 |
| eval | 19.45 | 74.82 | 29.8% | 1 |
| rollout | 13.82 | 53.14 | 21.2% | 260 |
| coverage | 3.80 | 14.61 | 5.8% | 6,400 |
| buf_write | 1.62 | 6.22 | 2.5% | 260 |
| roll_host | 0.63 | 2.41 | 1.0% | 260 |
| ckpt | 0.00 | 0.02 | 0.0% | 1 |
| **stages** | **65.31** | **251.2** | 100% | |
| **wall** | **88.07** | **338.7** | | **25.8% unaccounted = Python dispatch** |

One-time compile: rollout 11.7 s, update 4.5 s, eval 40.6 s — **56.7 s total**, unchanged by
any optimisation below and irrelevant beyond the first few hundred iterations.

### A caveat about that table, because it misled me

`block_until_ready` at a stage boundary bills that stage for draining everything JAX had
already queued. The rollout stage reads 53 ms/iter here, but the **same jitted rollout
measured standalone costs 8.9 ms** — the other 44 ms was the previous iteration's update
block finishing. Stage attribution is an upper bound per stage; the shares are indicative,
not exact. All headline speedups below are therefore measured **uninstrumented**
(`PROFILE_NOSYNC=1`), which runs the loop exactly as production does.

I also initially read the rollout's apparent cost as dispatch-bound and was wrong.
`rollout_scaling.py` settles it: scan lengths 1→32 cost 8.89→295.97 ms, i.e. **33.3× for
32× the work with ~0 fixed overhead**. The rollout is genuinely compute-bound at ~9 ms per
64-env MJX step, and there is nothing to reclaim there without changing the algorithm.

### The real problem

The inner `for _ in range(32)` loop dispatched ~8 host operations per update — a key split,
five separate buffer gathers, the jitted update, and **a device→host sync plus a numpy
`bincount` for row coverage**. That is ~256 host round-trips per training iteration to
perform 32 updates.

## Phase 2 — changes, each measured

Uninstrumented, 260 iters, warmup 60, min-of-run:

| # | change | wall s | ms/iter | Δ vs previous | share of total win |
|---|---|---:|---:|---:|---:|
| 0 | baseline | 85.95 | 330.6 | — | — |
| 1 | + on-device coverage (scatter-add, no host sync) | 66.87 | 257.2 | **−19.08 s** | **71%** |
| 2 | + jitted buffer insert | 65.58 | 252.2 | −1.29 s | 5% |
| 3 | + fused 32-update `lax.scan` (one dispatch/iter) | **59.14** | **227.5** | −6.44 s | 24% |

**Total 85.95 → 59.14 s = 1.453×.** Python overhead fell from 25.8% of wall to 0.6%.

The dominant win is not the fused scan — it is deleting the per-update device→host sync.
Pulling `rows` to the host 32× per iteration cost more than everything else put together,
because each sync stalls the pipeline until the GPU drains.

### What was tried and rejected

- **Longer jitted rollout / `lax.scan` over env steps.** Would amortise nothing (rollout is
  compute-bound, measured above) and changes the data-per-update ratio, so it changes the
  learned result. Not applied.
- **Cheaper/less frequent eval.** Eval is ~16% of a projected full run. Halving its
  frequency (500 → 1,000) would save ~8% and does not touch the learned parameters, but it
  halves the monitoring resolution that this chapter has repeatedly needed (the c32b seed-0
  dip was only diagnosable because evals were dense). **Offered, not applied** — it is a
  monitoring trade-off, not a free win.
- **`donate_argnums` on the replay buffer.** The jitted insert already lets XLA reuse the
  buffers; explicit donation gave no further measurable gain and makes the caller's `buf`
  invalid on reuse, which is a footgun for a 2%-of-total stage.

## Phase 3 — equivalence

> ### ⚠️ CORRECTION (2026-08-04): "bit-exact" was config-specific. I over-claimed.
>
> The check below passes bit-exactly at **16 buckets × 32 tables**, the config it was
> written for. It does **not** hold in general. Re-run at 32 × 64 (exp_c37's shape) the
> same harness reports differences of **~1e-3 to 1e-5 relative** on the critic and actor —
> while the **RNG key stays bit-exact**.
>
> The cause is not a logic error. The control flow, operation order and random stream are
> identical (the exact-key result proves it, and the sampled batch indices were checked
> separately and agree for every buffer size). What differs is that XLA compiles a
> standalone-jitted `update` and the *same code inlined inside a `lax.scan`* into different
> HLO, and is free to reassociate floating-point reductions differently between them. At
> 16×32 the two happened to fuse identically; at 32×64 they do not.
>
> Measured in the real trainer at 32×64: **~1e-5 relative after ONE update block**, growing
> to ~1e-2 after 12 iterations through ordinary RL chaos.
>
> **The honest claim is therefore: semantically identical, not bitwise reproducible.** It is
> the same class of difference as changing an XLA flag or a GPU model. Against this
> chapter's seed-to-seed sd of ~1,200 it is scientifically irrelevant — but two runs across
> the change cannot be diffed bit-for-bit, and anyone reproducing a published number should
> use the trainer that produced it.

`check_equivalence.py` runs the original Python loop and the fused scan from an identical
start on an identical RNG stream, 6 iterations × 32 updates, and compares **every** tensor:

```
EXACT  actor.{beta_base, beta_raw, delay, log_T_bkt, log_T_cross, table, tau_raw, w_raw}
EXACT  critic.q1.{w1,b1,w2,b2,w3,b3}   target.q1.{...}
EXACT  log_alpha      row_coverage      rng key
ALL BIT-EXACT — the fused block is the same computation.
```

Bit-exact, not "close": the scan body performs the same operations in the same order on the
same key stream. The RNG key and the coverage counts are compared too, since a diverged key
would silently change every future batch. One thing deliberately verified rather than
assumed: `size` is a concrete Python int in the loop and a *traced* argument in the fused
block, so `jax.random.randint(key, shape, 0, size)` could in principle differ — it does not.

## Projected full run

10,000 iterations, warmup 500, eval every 500 (20 evals), single seed, solo GPU:

| | baseline | optimised |
|---|---:|---:|
| rollout (10,000 × 8.9 ms) | 89 s | 89 s |
| loop + updates (9,500 iters) | 3,088 s | 1,815 s |
| eval (20 × 18.6 s) | 372 s | 372 s |
| **total** | **≈3,548 s = 59 min** | **≈2,275 s = 38 min** |

**≈1.56× on a full run.** The projection is higher than the harness's 1.45× because eval is
a smaller share at eval-every-500 than at the harness's eval-every-200, so the optimised
part is a larger fraction. The 59-min baseline projection is consistent with exp_c32b's
observed 113 min/seed under 3-way concurrency.

Remaining budget after the change: updates 80%, eval 16%, rollout 4%. Further gains need an
algorithmic change (batch size, update ratio, env count), not more engineering.

## Files

| file | what |
|---|---|
| `profile_run.py` | baseline loop, stage-instrumented; `--dev-coverage` / `--jit-insert` ablations; `PROFILE_NOSYNC=1` for clean wall timing |
| `profile_opt.py` | the fused-scan loop |
| `rollout_scaling.py` | scan-length sweep proving the rollout is compute-bound |
| `check_equivalence.py` | bit-exactness of the fused block vs the Python loop |
| `profile_*.json`, `abl_*.json`, `rollout_scaling.json` | raw numbers |

Nothing committed. nucstar's torch branch untouched. The change is **not** yet applied to
the c32b/c33/c34/c35/c36 trainers — those are the runs already reported, and I did not want
to alter the code that produced published numbers without a go-ahead.
