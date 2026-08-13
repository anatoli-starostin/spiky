# exp_c29 — paused 2026-08-02 ~17:32 UTC for a machine shutdown

Everything is stopped and the GPU is free. This file is the whole state; nothing else
needs to be remembered to pick the work back up.

---

## Read this first: interrupted cells restart from zero, and that costs nothing but time

The trainer checkpoints the **actor only**. Verified contents of a mid-run checkpoint:

```
b, const_set, constants, log_T_sel, log_T_soft, n_heads, obs_dim, pair_a, pair_b, tph, w, weights
```

No critic, no target critic, no Adam state, no replay buffer, no `log_alpha` — and
`const_lut_sac.py` has **no `--resume` flag** (`grep -c resume` → 0). So a cell that was
stopped at iteration 2,000 cannot continue from 2,000. It has to be re-run from 0.

**That loses no science, only wall clock.** Determinism is on
(`XLA_FLAGS=--xla_gpu_deterministic_ops=true`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`), which
exp_c17 showed makes a fixed seed reproduce **bit-for-bit** (0 of 28,034 checkpoint
elements differing). Re-running an interrupted cell retraces exactly the same trajectory
and lands on exactly the same number. The cost of this pause is **~9.5 minutes × 3 cells**
of recomputation, and nothing else.

The three partial checkpoints are left on disk (they are at iteration 2,000, MJX return
in the low hundreds — a policy that cannot walk yet). They will be **overwritten** on
relaunch, which is correct. Do not evaluate them; they are not results.

---

## Wave 1 — balanced, nap6/tph64 — **COMPLETE**

Nothing to redo. `SWEEP_DONE` is written; `results.json`, all `*_cpueval.json`, all
`*_bitusage.json`, and `c29_c29_contrast.png` are on disk.

100-episode deterministic CPU reference:

| arm | s0 | s1 | s2 | mean |
|---|---:|---:|---:|---|
| none (17-dim, magnitude-blind) | 3964.3 | 4423.7 | 4175.9 | **4188.0 ± 229.9** |
| grid (33-dim, 16 thresholds) | 4550.0 | 4372.1 | 4389.5 | **4437.2 ± 98.1** |

Paired `grid − none`: **+585.7 / −51.6 / +213.6 → +249.2 ± 320.1** (se 185), 2 of 3
seeds favouring grid. **Not a demonstrated effect.**

Salvaged single seeds from the dropped arms (seed 0): `random` **4092.0**,
`clumped` **3849.4**.

**The caveat that matters more than the headline.** `grid` and `random` at seed 0 share
*bit-identical wiring* and differ only in where the sixteen constants sit — by half a bin
width on average (0.167 standardised, 4% of the range). They came out **458 apart**,
nearly twice the grid-vs-none effect. The configuration's response to a deliberately
negligible perturbation is larger than the effect being measured.

**Mechanism (points against the hypothesis).** Adding thresholds roughly *halved* the
addressed partition: rows reached per table **24.4–32.9** for grid vs **51.4–55.7** for
none, with 11–19% of threshold bits permanently stuck on visited states (after the
const-const repair already removed the 23% dead by construction). Threshold comparators
also carry less information: mean binary entropy **0.450–0.577** bits vs **0.794–0.942**
for pairwise ones. Grid is nominally ahead *despite* worse addressing. The one consistent
pattern in grid's favour is reliability: full-length episodes 100/93/98 vs 37/100/96.

**Constant usage — the actionable finding.** Entropy per constant is sharply
**bell-shaped**: peak **0.86 bits at c10 (+0.009)**, falling to **0.16–0.19 bits** at the
extremes, and **38% of the bits wired to c0 (−2.53) are dead** versus ~0% for c9–c11.
About **six of sixteen constants do nearly all the work**.

> **Next time, place the constants by QUANTILE MASS (equal probability between adjacent
> levels) rather than by EQUAL SPACING in value.** The range rule was already corrected
> once before launch — the pooled percentile range put 7 of 16 constants in dead
> territory, and switching to the median per-channel range cut dead threshold bits from
> 66% to 8% — but the residual bell shape says that fix did not go far enough.

---

## Wave 2 — canonical_full_coverage, nap6/tph64 — **INTERRUPTED**

Started 14:21:35Z, stopped ~17:32Z. Launches 3 of 6.

| cell | state at stop | action on resume |
|---|---|---|
| none s0 | **running, iteration 2,000/10,000** (MJX 346.3, row-cov 94.8%, 9.2 min in) | re-run from 0 |
| grid s0 | **running, iteration 2,000/10,000** (MJX 241.8, row-cov 71.4%, 9.4 min in) | re-run from 0 |
| none s1 | **running, iteration 2,000/10,000** (MJX 603.0, row-cov 94.3%, 9.7 min in) | re-run from 0 |
| grid s1 | not started | run |
| none s2 | not started | run |
| grid s2 | not started | run |

`SWEEP_DONE_CANONICAL` does **not** exist, so relaunching `run_sweep_canonical.sh` starts
the whole wave over — which is what is wanted, since none of the six finished.

---

## Wave 3 — nap5/tph128, balanced — **QUEUED, NEVER STARTED**

No cell ran. Param-matched to wave 1 (128 × 2^5 × 12 = 49,152 learnable), 640 comparators
vs 384, 2,816 active values/step vs 1,536.

---

## Current design

**Arms: `none` + `grid`, seeds 0/1/2 → 6 runs per wave.** `random` and `clumped` were
dropped part-way through wave 1 (see the header note in `run_sweep.sh` for exactly what
wave 1 had launched by then).

**Open decision, not yet actioned:** running **`random` at all three seeds** as a design
sanity check. It is the cheapest decisive test available — ~2 h, 3 runs — because it asks
whether this configuration can resolve a 249-point effect *at all*, given that grid and
random differ by 458 at seed 0 while differing by half a bin width in design. If they
agree across seeds, the seed-0 gap was a fluke and +249 deserves more seeds; if they
disagree, the instrument cannot see the effect and the question needs a different design.
I would run this **before** waves 2 and 3, not after.

Also outstanding, from an earlier exchange: `nap8/tph16` (the other param-matched
direction — 128 comparators, 448 active/step) was proposed and never approved.

---

## Sentinels and chaining

Each wave blocks until the sentinel files of every earlier wave exist, so they never
contend for the GPU. Sentinels present right now: **`SWEEP_DONE` only.**

| wave | script | writes | waits for |
|---|---|---|---|
| 1 | `run_sweep.sh` → superseded by `run_sweep_resume.sh` | `SWEEP_DONE` | — |
| 2 | `run_sweep_canonical.sh` | `SWEEP_DONE_CANONICAL` | `SWEEP_DONE` |
| 3 | `run_sweep_capacity.sh` (parameterised) | `$SENT` | `$WAIT` |

`run_sweep_capacity.sh` takes `NAP TPH POLICY TAGP LOGP SENT WAIT` from the environment
and **refuses** any config whose learnable table size is not 49,152.

---

## To restart — one line per wave, run from this directory

```bash
cd ~/projects/spiky/experiments/walker2d_lut/exp_c29_magnitude_anchors

# wave 2 (canonical). Starts immediately: SWEEP_DONE already exists.
nohup ./run_sweep_canonical.sh > run_sweep_canonical.log 2>&1 &

# wave 3 (nap5/tph128). Parks until wave 2 writes its sentinel.
NAP=5 TPH=128 TAGP=c29m LOGP=cellm SENT=SWEEP_DONE_MIDCAP \
  WAIT="SWEEP_DONE SWEEP_DONE_CANONICAL" \
  nohup ./run_sweep_capacity.sh > run_sweep_midcap.log 2>&1 &

# the live Slack bar (reuses the SAME message — do not drop --handle)
nohup ~/projects/walker2d_mjx/.venv/bin/python -u slack_bar.py \
  --task 26ca0490 --handle 439e8c66 --interval 150 > slack_bar.log 2>&1 &
```

Launch each as its own command. `cd X && nohup A & ... nohup B &` backgrounds the whole
`&&` chain, so the second launch runs in the wrong directory — that already happened once
here.

If the `random × 3 seeds` check is approved, it runs *before* wave 2:

```bash
# 3 runs, ~2 h. Delete SWEEP_DONE first ONLY if you want wave 2 to wait for it.
for s in 0 1 2; do
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  XLA_FLAGS=--xla_gpu_deterministic_ops=true CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  nohup ~/projects/walker2d_mjx/.venv/bin/python -u ./const_lut_sac.py \
    --seed $s --constants random --addressing anchors --anchor-policy balanced \
    --forward-mode hard --nap 6 --tph 64 --heads 1 --iters 10000 --envs 64 \
    --rollout 1 --updates 32 --batch 512 --warmup 500 --row-clip 1.0 \
    --eval-every 500 --eval-episodes 20 --tag _c29_random_s$s \
    > cell_random_s$s.log 2>&1 &
  sleep 25
done
```
(seed 0 already exists at 4092.0 and would simply reproduce; re-running it is a free
determinism check.)

---

## After any wave finishes

```bash
~/projects/walker2d_mjx/.venv/bin/python collect.py          # tables + results.json
MPLCONFIGDIR=/tmp/mplcfg ~/projects/spiky/.venv/bin/python plot_c29.py --wave c29c
```
`collect.py` runs in the **mjx** venv, `plot_c29.py` in the **spiky** venv (matplotlib
lives there, jax does not).

---

## Nothing is holding the GPU

`nvidia-smi` at stop: 0% utilisation, 636 MiB used, and the only compute process is the
desktop terminal (`ptyxis`, 140 MiB). No trainer, no driver, no bar process is alive.
