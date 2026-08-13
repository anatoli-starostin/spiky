"""Collect exp16 (exp10 + trainable exponential output transform) and compare to exp10.

Writes the convention trio (config.json / metrics.csv / summary.json). Metric definitions
are taken verbatim from src/summarize_bench.py, exactly as in the exp10 reproduction, so
all three arms are directly comparable:
    final = last ep_ret_mean ; best = max ep_ret_mean ; aggregate std = np.std (ddof=0)

Collapse criterion `final/best < 0.90` — calibrated against the committed exp02-exp05
labels (reproduces 2/3, 1/3, 1/3, 0/3 exactly), not invented here.

exp16 also reports the learned transform parameters c and t (logged per history row via
models.FastLUTExpActorCritic.extra_log / ppo.py's extra_log hook).

Because gpustar's own exp10 reproduction exists, exp16 is compared against BOTH:
  - the committed exp10 reference (nebius, 5488.4 +- 179.9) -- the number the task names;
  - gpustar's own exp10 reproduction (6063.9 +- 879.3) -- the same-host control, which
    removes any host effect from the comparison.

Usage:  python collect.py
"""
import csv
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(HERE, "..")
REF = os.path.join(BASE, "exp10_lut-anchor-pair-t32")
REPRO = os.path.join(BASE, "repro_exp10_gpustar")
SEEDS = (0, 1, 2)
COLLAPSE_RATIO = 0.90


def load(folder):
    out = {}
    for s in SEEDS:
        j = json.load(open(os.path.join(folder, f"ppo_s{s}.json")))
        h = j["history"]
        ys = np.array([r["ep_ret_mean"] for r in h], float)
        ok = ys[np.isfinite(ys)]
        out[s] = dict(run=j, hist=h, ys=ys, final=float(ys[-1]),
                      best=float(ok.max()) if len(ok) else float("nan"),
                      wall=j["wall_s"], thr=j["throughput_env_per_s"],
                      params=j["params"], c=h[-1].get("c"), t=h[-1].get("t"))
    return out


def agg(d):
    f = [d[s]["final"] for s in SEEDS]
    b = [d[s]["best"] for s in SEEDS]
    return dict(final_mean=float(np.mean(f)), final_std=float(np.std(f)),
                best_mean=float(np.mean(b)), best_std=float(np.std(b)),
                collapsed=[s for s in SEEDS
                           if d[s]["final"] / d[s]["best"] < COLLAPSE_RATIO],
                wall_mean=float(np.mean([d[s]["wall"] for s in SEEDS])),
                thr_mean=float(np.mean([d[s]["thr"] for s in SEEDS])))


def welch(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    se = np.sqrt(a.var(ddof=1) / a.size + b.var(ddof=1) / b.size)
    return float(a.mean() - b.mean()), float(se), float(abs(a.mean() - b.mean()) / se)


def main():
    e16, e10, rp = load(HERE), load(REF), load(REPRO)
    A, B, C = agg(e16), agg(e10), agg(rp)

    with open(os.path.join(HERE, "metrics.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["seed", "update", "env_steps", "ep_ret_mean", "ep_ret_max",
                    "ep_len_mean", "lr", "logstd", "kl", "c", "t"])
        for s in SEEDS:
            for r in e16[s]["hist"]:
                w.writerow([s, r["update"], r["env_steps"], r["ep_ret_mean"],
                            r["ep_ret_max"], r["ep_len_mean"], r["lr"],
                            r["logstd"], r["kl"], r.get("c"), r.get("t")])

    cfg = dict(json.load(open(os.path.join(REF, "config.json"))))
    cfg.update(exp_name="exp16_lut-anchor-pair-t32-expout",
               description=("exp10 + a trainable exponential output transform on the "
                            "actor mean: mean -> c + exp(mean / t), with c a free "
                            "trainable scalar and t a trainable scalar constrained "
                            "positive via softplus. Everything else identical to exp10."),
               arch="fastlut_exp", forked_from="exp10_lut-anchor-pair-t32",
               output_transform="mean -> c + exp(mean / t)",
               c_init=-1.0, t_init=1.0,
               c_parameterization="free scalar (any sign)",
               t_parameterization="softplus(t_raw), constrained > 0",
               init_rationale=("c=-1, t=1 makes the transform a first-order match to the "
                               "identity at init (c+exp(0)=0, slope 1/t=1), so exp16 "
                               "starts behaviourally identical to exp10; verified to "
                               "7.6e-5 on the action mean."),
               exp_clamp=20.0, params=e16[0]["params"],
               host="gpustar (RTX 5090)")
    json.dump(cfg, open(os.path.join(HERE, "config.json"), "w"), indent=2)

    d_ref, se_ref, t_ref = welch([e16[s]["final"] for s in SEEDS],
                                 [e10[s]["final"] for s in SEEDS])
    d_rp, se_rp, t_rp = welch([e16[s]["final"] for s in SEEDS],
                              [rp[s]["final"] for s in SEEDS])
    summary = dict(
        exp_name="exp16_lut-anchor-pair-t32-expout", algo="ppo", n_seeds=3,
        forked_from="exp10_lut-anchor-pair-t32",
        output_transform="mean -> c + exp(mean / t)",
        ppo_best_mean=round(A["best_mean"], 1), ppo_best_std=round(A["best_std"], 1),
        ppo_final_mean=round(A["final_mean"], 1), ppo_final_std=round(A["final_std"], 1),
        params=e16[0]["params"],
        throughput_env_per_s_mean=round(A["thr_mean"]),
        training_time_hours_mean=round(A["wall_mean"] / 3600, 3),
        collapsed_seeds=A["collapsed"], collapse_criterion="final/best < 0.90",
        learned_transform={str(s): dict(c=round(e16[s]["c"], 4), t=round(e16[s]["t"], 4))
                           for s in SEEDS},
        learned_c_mean=round(float(np.mean([e16[s]["c"] for s in SEEDS])), 4),
        learned_t_mean=round(float(np.mean([e16[s]["t"] for s in SEEDS])), 4),
        per_seed={str(s): dict(best=round(e16[s]["best"], 1),
                               final=round(e16[s]["final"], 1),
                               final_over_best=round(e16[s]["final"] / e16[s]["best"], 3),
                               c=round(e16[s]["c"], 4), t=round(e16[s]["t"], 4))
                  for s in SEEDS},
        vs_exp10_committed=dict(
            reference_final=round(B["final_mean"], 1), reference_std=round(B["final_std"], 1),
            delta=round(d_ref, 1), welch_se=round(se_ref, 1), welch_abs_t=round(t_ref, 2),
            pct_of_reference=round(100 * A["final_mean"] / B["final_mean"], 1)),
        vs_exp10_gpustar_repro=dict(
            reference_final=round(C["final_mean"], 1), reference_std=round(C["final_std"], 1),
            delta=round(d_rp, 1), welch_se=round(se_rp, 1), welch_abs_t=round(t_rp, 2),
            pct_of_reference=round(100 * A["final_mean"] / C["final_mean"], 1)))
    json.dump(summary, open(os.path.join(HERE, "summary.json"), "w"), indent=2)

    print(f"{'':8} {'best':>9} {'final':>9} {'f/b':>6} {'c':>9} {'t':>8}")
    for s in SEEDS:
        e = e16[s]
        print(f"exp16 s{s} {e['best']:>9.1f} {e['final']:>9.1f} "
              f"{e['final'] / e['best']:>6.3f} {e['c']:>9.4f} {e['t']:>8.4f}")
    print(f"\nexp16                  final {A['final_mean']:8.1f} +- {A['final_std']:6.1f} "
          f"| best {A['best_mean']:8.1f} | collapse {len(A['collapsed'])}/3")
    print(f"exp10 (committed)      final {B['final_mean']:8.1f} +- {B['final_std']:6.1f} "
          f"| best {B['best_mean']:8.1f} | collapse {len(B['collapsed'])}/3")
    print(f"exp10 (gpustar repro)  final {C['final_mean']:8.1f} +- {C['final_std']:6.1f} "
          f"| best {C['best_mean']:8.1f} | collapse {len(C['collapsed'])}/3")
    print(f"\nvs committed exp10 : {d_ref:+9.1f}  se {se_ref:6.1f}  |t| {t_ref:.2f}  "
          f"({summary['vs_exp10_committed']['pct_of_reference']:.1f}%)")
    print(f"vs gpustar  exp10 : {d_rp:+9.1f}  se {se_rp:6.1f}  |t| {t_rp:.2f}  "
          f"({summary['vs_exp10_gpustar_repro']['pct_of_reference']:.1f}%)")
    print(f"\nlearned c mean {summary['learned_c_mean']:+.4f} (init -1.0)   "
          f"t mean {summary['learned_t_mean']:.4f} (init 1.0)")
    print(f"throughput {A['thr_mean']:,.0f} steps/s | {A['wall_mean'] / 60:.1f} min/seed")
    print("\nwrote config.json, metrics.csv, summary.json")


if __name__ == "__main__":
    main()
