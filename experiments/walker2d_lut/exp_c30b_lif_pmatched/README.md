# exp_c30b — a cheaper-P LIF detector front-end for the Walker2d SAC actor

> ## ⚠️ CORRECTION (2026-08-03): this experiment was param-matched to the WRONG NUMBER.
>
> It was built to hit 49,152, described throughout as "the hyperplane baseline's budget".
> **49,152 is not that budget.** It is exp_c29's *table-only* count for its nap6/tph64
> cells (`tph * 2**nap * 12 = 64*64*12`); exp_c29's own totals were 56,064–70,912. The
> anchor this experiment actually compares against — exp_c18, 4308.0 ± 500.1 — is
> **nap6/tph32**: table 24,576, hyperplane `w`/`b` 3,456, **total 28,032**.
>
> So exp_c30b is **1.72× the baseline, not 0.98×**, and the original headline ("no
> measurable difference at 98% of the baseline's parameters") was wrong. The *returns*
> below are unaffected and were independently re-verified; only the parameter framing was.
>
> Every model in this comparison carries the **same 24,576-entry table**, so the totals are
> dominated by a component none of them changes. The real comparison is the index
> front-end, and by that measure this experiment cut exp_c30's front-end by 62% and is
> still **6.8× the hyperplane's**. exp_c31 (PureLIF) is the variant that gets close, at
> 1.97×.

exp_c30 showed the LIF-detector front-end drives the actor about as well as the hyperplane
sign-tests — but at a front-end of 62,785 against 3,456, so it said nothing per parameter.
This experiment cuts `P` and asks how much of that cost was necessary.

**Result: no measurable difference, at 38% of exp_c30's front-end cost.**

| | CPU-ref 100 ep | front-end | table | total | vs exp_c30b |
|---|---:|---:|---:|---:|---|
| exp_c18 hyperplane, nap6/tph32, 6 seeds | 4308.0 ± 500.1 | 3,456 | 24,576 | 28,032 | −221.2 (Welch se 607.6, \|t\| 0.36) |
| exp_c30 LIF, dense P, 3 seeds | 3931.3 ± 585.8 | 62,785 | 24,576 | 87,361 | +155.5 (Welch se 664.8, \|t\| 0.23) |
| **exp_c30b LIF, factorised P, 3 seeds** | **4086.8 ± 991.2** | **23,617** | 24,576 | **48,193** | — |

Per seed: 3478.1 (97/100 full) · **5230.6** (100/100, 4.234 m/s) · 3551.6 (56/100).

## The reduction

    P[m] = Pu[m] @ Pv[m].T  +  Pb[m] @ 1.T          Pu, Pv: (N, 2)   Pb: (N,)

A rank-2 factorisation plus a rank-1 term whose right factor is pinned to the all-ones
vector. `Pb[m, i]` is not filler — it is a per-**source**-channel weight applied to all of
detector *m*'s outgoing comparisons: "does channel *i*'s arrival order matter to this
detector at all", independent of which channel it is compared against. So the reduced `P`
is a structured rank-3 matrix with one factor frozen.

**48,193 actor params** — 23,617 detectors + 24,576 table. That is **44.8% below** exp_c30.
It was aimed at 49,152 and lands 1.95% under it; see the correction above for why 49,152
was the wrong target.

### Why this shape

Budget for `P` was taken as 49,152 − 31,873 = 17,279. The clean alternatives all miss that
target or cost too much:

| option | P params | total | vs 49,152 | pair-channel cost |
|---|---:|---:|---:|---|
| per-detector rank 2 | 13,056 | 44,929 | −8.6% | 2× |
| per-detector rank 3 | 19,584 | 51,457 | +4.7% | 3× |
| per-table-shared V, rank 4 | 15,232 | 47,105 | −4.2% | 4× |
| pair channel on 9 of 17 channels | 15,552 | 47,425 | −3.5% | 0.28× |
| shared CP dictionary, C=76 | 17,176 | 49,049 | −0.2% | **76× — infeasible** |
| **rank 2 + source bias** | **16,320** | **48,193** | **−1.95%** | **2×** |

The CP dictionary lands closest on paper and cannot be used: its shared basis does not
fold into the per-detector contraction, so the cost scales with C rather than with the
rank. Dropping 8 of 17 channels reaches −3.5% but removes those channels from ordered
comparison **entirely**, and exp_c29 spent a whole wave establishing that per-channel
liveness is exactly where these models fail. Landing *under* rather than over is
deliberate: a result that holds at 98% of the baseline's budget is the conservative claim.

## Verification

There is no torch reference for this architecture — the factorisation is ours. So
`check_lowrank.py` uses **exp_c30's dense module as the oracle** (itself parity-checked
against torch 13/13, table gradient bit-identical): materialise the `P` this factorisation
represents, hand it to the dense implementation, require agreement. That is stronger than
a shape or gradient-flow check, because the factorised forward never builds `P` — a
transposed index or a misplaced off-diagonal mask in the contraction is invisible to those
checks but cannot survive comparison against the dense path.

9/9 pass. `forward hard` and `address` **bit-identical** (rel 0.0), membrane 4.47e-08,
`st == hard` identity 1.19e-07, gradients reach all 11 parameters with none dead, table
gradient still a hard scatter, and the dense-equivalent `P` init std lands at **0.01000**
— matching the torch reference, so the pair channel still starts near zero and each
detector still begins as a pure value/range unit.

## Two things worth carrying forward

**Fewer parameters, more compute.** 78.7–79.6 min per seed against exp_c30's 45.5–46.0 —
about 73% *slower* while carrying 45% fewer parameters. The dense pair channel is one
fused elementwise multiply-and-sum over the (B, M, N, N) gate tensor; the factorised one
is two einsums over that same tensor plus a bias reduction. Parameter count and wall clock
move in opposite directions here, and anything that quotes one as a proxy for the other
will be wrong.

**The terminal dip is now 6 for 6.** Every seed in exp_c30 and exp_c30b peaks before the
end and gives return back over the final stretch of the eps anneal — here 3520→3231,
5484→5230, 3993→3685. The checkpoint is the *final* actor, so the quoted numbers pay that
cost in all six runs. Stopping the anneal at ~0.6, or holding eps once return plateaus,
remains the cheapest untested improvement in this line.

Row coverage again reaches ~100% of all 2,048 rows, as in exp_c30. Whatever limits this
actor, it is not an under-addressed table.

## Files

| file | what |
|---|---|
| `jax_lif_lowrank.py` | the factorised model — everything except `P` is exp_c30 verbatim |
| `check_lowrank.py` | 9 checks against exp_c30's dense (torch-verified) oracle |
| `lif_sac_lr.py` | exp_c30's trainer, repointed at the factorised module |
| `eval_lif_cpu.py` | 100-episode deterministic CPU reference — **the only number quoted** |
| `run_sweep_lr.sh` / `collect.py` / `plot_c30b.py` / `slack_bar_lr.py` | sweep, table, figure, bar |

## Reproduce

```bash
python check_lowrank.py              # must print ALL CHECKS OK first
nohup ./run_sweep_lr.sh > run_sweep_lr.log 2>&1 &
python collect.py                    # mjx venv
MPLCONFIGDIR=/tmp/mplcfg python plot_c30b.py   # spiky venv (matplotlib)
```

SAC recipe, eps schedule, determinism, eval convention: all identical to exp_c30. Only the
`P` parameterisation differs.
