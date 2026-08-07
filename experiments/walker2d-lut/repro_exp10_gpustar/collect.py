"""Collect the gpustar reproduction of exp10_lut-anchor-pair-t32 and compare it to the
committed reference.

Writes the convention trio (config.json / metrics.csv / summary.json) into this folder,
using metric definitions taken VERBATIM from src/summarize_bench.py so the numbers are
comparable to the committed ones:
    final = last  ep_ret_mean in history
    best  = max   ep_ret_mean over history
    aggregate std = np.std  (population, ddof=0)   <- verified to reproduce 5488.4 +- 179.9

COLLAPSE CRITERION. No numeric definition is coded anywhere in the repo; the READMEs only
describe it ("seeds partially collapse in the last quarter"). The threshold below was
CALIBRATED against the committed labels rather than invented: final/best < 0.90 reproduces
exp02 2/3, exp03 1/3, exp04 1/3 and exp05 0/3 exactly, and there is a wide margin between
the collapsed (<=0.796) and healthy (>=0.940) seeds. Applied identically to both arms.

Usage:  python collect.py
"""
import csv
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REF = os.path.join(HERE, "..", "exp10_lut-anchor-pair-t32")
SEEDS = (0, 1, 2)
COLLAPSE_RATIO = 0.90


def series(run):
    h = run["history"]
    return (np.array([r["env_steps"] for r in h], float),
            np.array([r["ep_ret_mean"] for r in h], float))


def load(folder):
    out = {}
    for s in SEEDS:
        j = json.load(open(os.path.join(folder, f"ppo_s{s}.json")))
        xs, ys = series(j)
        ok = ys[np.isfinite(ys)]
        out[s] = dict(run=j, xs=xs, ys=ys, final=float(ys[-1]),
                      best=float(ok.max()) if len(ok) else float("nan"),
                      wall=j["wall_s"], thr=j["throughput_env_per_s"],
                      params=j["params"])
    return out


def agg(d):
    f = [d[s]["final"] for s in SEEDS]
    b = [d[s]["best"] for s in SEEDS]
    collapsed = [s for s in SEEDS if d[s]["final"] / d[s]["best"] < COLLAPSE_RATIO]
    return dict(final_mean=float(np.mean(f)), final_std=float(np.std(f)),
                best_mean=float(np.mean(b)), best_std=float(np.std(b)),
                collapsed=collapsed,
                wall_mean=float(np.mean([d[s]["wall"] for s in SEEDS])),
                thr_mean=float(np.mean([d[s]["thr"] for s in SEEDS])))


def welch(a, b):
    """Two-sample Welch t on the 3 per-seed finals (sample sd, ddof=1)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    se = np.sqrt(a.var(ddof=1) / a.size + b.var(ddof=1) / b.size)
    return float(a.mean() - b.mean()), float(se), float(abs(a.mean() - b.mean()) / se)


def main():
    rep = load(HERE)
    ref = load(REF)
    A, B = agg(rep), agg(ref)

    # ---- metrics.csv (same schema/column order as the committed one) -------
    with open(os.path.join(HERE, "metrics.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["seed", "update", "env_steps", "ep_ret_mean", "ep_ret_max",
                    "ep_len_mean", "lr", "logstd", "kl"])
        for s in SEEDS:
            for r in rep[s]["run"]["history"]:
                w.writerow([s, r["update"], r["env_steps"], r["ep_ret_mean"],
                            r["ep_ret_max"], r["ep_len_mean"], r["lr"],
                            r["logstd"], r["kl"]])

    # ---- config.json (the reference config + what this run is) ------------
    ref_cfg = json.load(open(os.path.join(REF, "config.json")))
    cfg = dict(ref_cfg)
    cfg.update(exp_name="repro_exp10_gpustar",
               description=("Reproduction of exp10_lut-anchor-pair-t32 on gpustar "
                            "(RTX 5090) to verify the framework runs cleanly on this "
                            "host. Flags verbatim from exp10/config.json; 3 seeds in "
                            "parallel. Not a new experiment — a host-verification rerun."),
               reproduces="exp10_lut-anchor-pair-t32",
               host="gpustar (RTX 5090, Blackwell sm_120)",
               torch=rep[0]["run"].get("torch", "2.9.1+cu130"))
    json.dump(cfg, open(os.path.join(HERE, "config.json"), "w"), indent=2)

    # ---- summary.json ------------------------------------------------------
    d_f, se_f, t_f = welch([rep[s]["final"] for s in SEEDS],
                           [ref[s]["final"] for s in SEEDS])
    summary = dict(
        exp_name="repro_exp10_gpustar", algo="ppo", n_seeds=3,
        reproduces="exp10_lut-anchor-pair-t32",
        ppo_best_mean=round(A["best_mean"], 1), ppo_best_std=round(A["best_std"], 1),
        ppo_final_mean=round(A["final_mean"], 1), ppo_final_std=round(A["final_std"], 1),
        params=rep[0]["params"],
        throughput_env_per_s_mean=round(A["thr_mean"]),
        training_time_hours_mean=round(A["wall_mean"] / 3600, 3),
        collapsed_seeds=A["collapsed"], collapse_criterion="final/best < 0.90",
        per_seed={str(s): dict(best=round(rep[s]["best"], 1),
                               final=round(rep[s]["final"], 1),
                               final_over_best=round(rep[s]["final"] / rep[s]["best"], 3))
                  for s in SEEDS},
        reference=dict(ppo_final_mean=round(B["final_mean"], 1),
                       ppo_final_std=round(B["final_std"], 1),
                       ppo_best_mean=round(B["best_mean"], 1),
                       collapsed_seeds=B["collapsed"],
                       throughput_env_per_s_mean=round(B["thr_mean"]),
                       training_time_hours_mean=round(B["wall_mean"] / 3600, 3)),
        comparison=dict(final_delta=round(d_f, 1), welch_se=round(se_f, 1),
                        welch_abs_t=round(t_f, 2),
                        pct_of_reference=round(100 * A["final_mean"] / B["final_mean"], 1),
                        speedup_vs_reference=round(A["thr_mean"] / B["thr_mean"], 2)))
    json.dump(summary, open(os.path.join(HERE, "summary.json"), "w"), indent=2)

    # ---- report ------------------------------------------------------------
    print(f"{'':6} {'best':>9} {'final':>9} {'f/b':>6}")
    for name, d in (("REPRO", rep), ("REF", ref)):
        for s in SEEDS:
            print(f"{name}{s:<2} {d[s]['best']:>9.1f} {d[s]['final']:>9.1f} "
                  f"{d[s]['final'] / d[s]['best']:>6.3f}")
    print(f"\nREPRO  final {A['final_mean']:.1f} +- {A['final_std']:.1f} | "
          f"best {A['best_mean']:.1f} +- {A['best_std']:.1f} | "
          f"collapse {len(A['collapsed'])}/3 | {A['thr_mean']:,.0f} steps/s | "
          f"{A['wall_mean'] / 60:.1f} min/seed")
    print(f"REF    final {B['final_mean']:.1f} +- {B['final_std']:.1f} | "
          f"best {B['best_mean']:.1f} +- {B['best_std']:.1f} | "
          f"collapse {len(B['collapsed'])}/3 | {B['thr_mean']:,.0f} steps/s | "
          f"{B['wall_mean'] / 60:.1f} min/seed")
    print(f"\ndelta {d_f:+.1f}  Welch se {se_f:.1f}  |t| {t_f:.2f}  "
          f"({summary['comparison']['pct_of_reference']:.1f}% of reference, "
          f"{summary['comparison']['speedup_vs_reference']:.2f}x throughput)")
    print("\nwrote config.json, metrics.csv, summary.json")


if __name__ == "__main__":
    main()
