"""Shared loading + registry for the distillation analytics suite.

Everything here is POST-HOC and READ-ONLY: it opens saved history JSONs and checkpoint NPZs
and never constructs or runs a network, so it costs no GPU and is safe to run any time.

TWO DATA LIMITATIONS worth knowing before reading any figure produced by this suite:
  * the per-round history stores only `best` and `mean` of the pool EWMA, not the per-member
    vector, so true per-round variance is NOT recoverable post-hoc. (best - mean) is used as
    the spread proxy and is labelled as such. True pool spread IS available, but only at the
    final round, from the checkpoint's `ewma` array.
  * the checkpoint is REWRITTEN every round, so only the FINAL genome state exists. Structural
    drift over time is taken from the history's n_syn / n_exc / n_inh instead.
Both are fixable for future runs by logging the fitness vector per round; noted in the report.
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CHAPTER = os.path.dirname(os.path.dirname(HERE))     # experiments/neurodarwinism/
OUT = os.path.join(HERE, "out")
os.makedirs(OUT, exist_ok=True)

# Checkpoints are 12-49 MB each and are NOT in git. Point ND_CKPT_DIR at wherever the
# run left them (default: each experiment's own directory, which is where a fresh run
# launched with ND_OUT=$PWD will write).
CKPT_DIR = os.environ.get("ND_CKPT_DIR")

EXC, INH, INP, OUTP = 0, 1, 2, 3
FIELDS = ("src_pool", "src_idx", "tgt_pool", "tgt_idx", "delay", "weight")

# name -> (experiment directory, checkpoint npz basename or None, pool size K, d_max)
# Every experiment directory holds history.json; the checkpoint may or may not be there.
EXPERIMENTS = {
    "k32_baseline": ("exp001_k32-baseline", None, 32, 20),
    "k128":         ("exp002_k128-pool", "ck_k128.npz", 128, 20),
    "small8020":    ("exp003_small-sparse-80exc-20inh", "ck_small8020.npz", 32, 20),
    "delay148":     ("exp004_wide-exc-delays-1-48", "ck_delay148.npz", 32, 48),
    "tiepen01":     ("exp005_tie-penalty-0.1", "ck_tiepen01.npz", 32, 20),
    "drivegene":    ("exp006_drive-gene", "ck_drivegene.npz", 32, 20),
}


def available():
    """Experiments whose history exists right now."""
    out = {}
    for name, (expdir, ck, k, dmax) in EXPERIMENTS.items():
        d = os.path.join(CHAPTER, expdir)
        hp = os.path.join(d, "history.json")
        if os.path.exists(hp):
            cp = os.path.join(CKPT_DIR or d, ck) if ck else None
            out[name] = dict(history=hp, ckpt=cp if cp and os.path.exists(cp) else None,
                             k=k, d_max=dmax)
    return out


def load_history(path):
    return json.load(open(path))


def load_genomes(ckpt):
    """-> (list of genome dicts, ewma array, age, newborn, next_rnd)."""
    z = np.load(ckpt, allow_pickle=False)
    k = int(z["n_genomes"][0])
    genomes = [{f: z[f"g{i}_{f}"] for f in FIELDS} for i in range(k)]
    return (genomes, z["ewma"], z["age"], z["newborn"], int(z["next_rnd"][0]))


def convergence_round(hist, thresh=0.01):
    """First round where (best - mean) stays below `thresh` for 10 consecutive rounds.

    The pool has 'collapsed' when the best member is no longer meaningfully better than the
    average one -- at that point selection has almost nothing to choose between.
    """
    gaps = [r["best"] - r["mean"] for r in hist]
    run = 0
    for i, g in enumerate(gaps):
        run = run + 1 if g < thresh else 0
        if run >= 10:
            return hist[i - 9]["rnd"]
    return None


def pool_matrix(hist, key="fitness_vec"):
    """-> [n_rounds, K] of a per-member vector, or None if this run did not log one.

    Runs before 2026-08-09 recorded only scalar summaries per round, so per-round pool variance
    was recoverable only at the final round from the checkpoint's `ewma` array -- which is why
    every diversity claim about them leans on (best - mean). Newer runs log the full vectors
    (`fitness_vec`, `ewma_vec`, `tie_rate_vec`, `distinct_ticks_vec`, `age_vec`, `lineage_vec`).

    ALWAYS go through this rather than indexing the key: it returns None for old histories
    instead of raising, and it refuses ragged data rather than silently truncating -- K can
    legitimately differ between two histories, but never within one.
    """
    rows = [r.get(key) for r in hist]
    if not rows or any(v is None for v in rows):
        return None
    widths = {len(v) for v in rows}
    if len(widths) != 1:
        raise ValueError(f"{key}: ragged pool vectors across rounds, widths {sorted(widths)}")
    return np.asarray(rows, float)


def summarise(hist):
    best = max(hist, key=lambda r: r["best"])
    out = dict(rounds=len(hist), peak=best["best"], peak_round=best["rnd"],
               final_best=hist[-1]["best"], final_mean=hist[-1]["mean"],
               final_gap=hist[-1]["best"] - hist[-1]["mean"],
               convergence_round=convergence_round(hist),
               syn_first=hist[0]["n_syn"], syn_last=hist[-1]["n_syn"],
               wall_s=hist[-1]["wall"])
    # true per-round spread, when the run logged the vectors. Absent on older runs.
    fit = pool_matrix(hist, "fitness_vec")
    if fit is not None:
        out.update(pool_k=int(fit.shape[1]),
                   fitness_std_first=float(fit[0].std()),
                   fitness_std_last=float(fit[-1].std()),
                   fitness_std_min=float(fit.std(axis=1).min()),
                   fitness_spread_last=float(fit[-1].max() - fit[-1].min()))
    return out
