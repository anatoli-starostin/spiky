"""Collect exp17 (log-sum-exp table readout) and compare to exp10 and exp16.

Writes the convention trio (config.json / metrics.csv with tau per row / summary.json).
Metric definitions verbatim from src/summarize_bench.py, as in exp16 and the exp10
reproduction, so every arm is directly comparable:
    final = last ep_ret_mean ; best = max ep_ret_mean ; aggregate std = np.std (ddof=0)

Collapse criterion `final/best < 0.90` — calibrated against the committed exp02-exp05
labels (reproduces 2/3, 1/3, 1/3, 0/3 exactly), not invented here.

Usage:  python collect.py
"""
import csv
import itertools
import json
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(HERE, "..")
ARMS = {
    "exp10_committed": os.path.join(BASE, "exp10_lut-anchor-pair-t32"),
    "exp10_gpustar_repro": os.path.join(BASE, "repro_exp10_gpustar"),
    "exp16_expout": os.path.join(BASE, "exp16_lut-anchor-pair-t32-expout"),
    # abandoned plain-log-sum-exp attempt (complete, 3/3 JSONs). The second abandoned
    # attempt was stopped early and has logs only, so it is not loadable here; its curve
    # is picked up from the logs by plot_diagnosis.py.
    "plain_lse_attempt1": os.path.join(HERE, "attempt1_additive_init"),
}
SEEDS = (0, 1, 2)
COLLAPSE_RATIO = 0.90


WARMUP_LEVELS = (1000, 3000, 5000)


def warmup(hist, level):
    """First update at which the return reaches `level`; None if never."""
    for r in hist:
        if r["ep_ret_mean"] >= level:
            return int(r["update"])
    return None


def load(folder):
    out = {}
    for s in SEEDS:
        j = json.load(open(os.path.join(folder, f"ppo_s{s}.json")))
        h = j["history"]
        ys = np.array([r["ep_ret_mean"] for r in h], float)
        ok = ys[np.isfinite(ys)]
        out[s] = dict(hist=h, final=float(ys[-1]),
                      best=float(ok.max()) if len(ok) else float("nan"),
                      wall=j["wall_s"], thr=j["throughput_env_per_s"],
                      params=j["params"], tau=h[-1].get("tau"),
                      warmup={lv: warmup(h, lv) for lv in WARMUP_LEVELS})
    return out


def warmup_summary(d):
    """Mean updates-to-threshold across seeds; None if any seed never got there."""
    out = {}
    for lv in WARMUP_LEVELS:
        vals = [d[s]["warmup"][lv] for s in SEEDS]
        out[str(lv)] = (None if any(v is None for v in vals)
                        else round(float(np.mean(vals)), 1))
        out[f"{lv}_reached"] = sum(v is not None for v in vals)
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


def separation(a, b):
    """Complete-rank-separation test: does one arm sit entirely below the other?
    Returns (separated, exact permutation p) for n_a of n_a+n_b lowest ranks."""
    sep = max(a) < min(b) or max(b) < min(a)
    p = 1.0 / math.comb(len(a) + len(b), len(a))
    return sep, p


def main():
    e17 = load(HERE)
    others = {k: load(v) for k, v in ARMS.items()}
    A = agg(e17)
    Ag = {k: agg(v) for k, v in others.items()}
    f17 = [e17[s]["final"] for s in SEEDS]

    with open(os.path.join(HERE, "metrics.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["seed", "update", "env_steps", "ep_ret_mean", "ep_ret_max",
                    "ep_len_mean", "lr", "logstd", "kl", "tau"])
        for s in SEEDS:
            for r in e17[s]["hist"]:
                w.writerow([s, r["update"], r["env_steps"], r["ep_ret_mean"],
                            r["ep_ret_max"], r["ep_len_mean"], r["lr"],
                            r["logstd"], r["kl"], r.get("tau")])

    cfg = dict(json.load(open(os.path.join(ARMS["exp10_committed"], "config.json"))))
    cfg.update(exp_name="exp17_lut-anchor-pair-t32-logsumexp",
               description=("exp10 with the anchor-pair actor's sum-over-tables readout "
                            "replaced by a SUM-SCALED (mean-normalised) temperature-tau "
                            "log-sum-exp: out = T * tau * log((1/T) * sum_t exp(w_t/tau)). "
                            "This generalises the SUM rather than the mean: tau->inf "
                            "recovers exp10's sum_t w_t exactly, tau->0 gives T*max(w). "
                            "tau is the only new trainable parameter (softplus, > 0). "
                            "Everything else identical to exp10."),
               arch="fastlut_lse_sum", forked_from="exp10_lut-anchor-pair-t32",
               readout="T * tau * log((1/T) * sum_tables exp(w / tau))",
               flag=('FastMultiHeadLut(exp_outputs=True, exp_outputs_scale="sum", '
                     'exp_outputs_init="additive", exp_outputs_tau_init=0.05)'),
               tau_init=0.05, tau_parameterization="softplus(tau_raw), floored at 1e-3",
               weight_init="plain additive U(+-initial_weights_noise) — no special init",
               exp_clamp=60.0,
               init_note=("at init the readout matches exp10's plain sum to 5.4% of the "
                          "output std (residual = the Jensen gap, predicted 1.07e-4, "
                          "measured 1.79e-4); gradient sums to 32 over tables, matching "
                          "exp10, against 1 for the plain log-sum-exp"),
               abandoned_attempts=("plain log-sum-exp (exp_outputs_scale='mean') with "
                                   "additive init (final 495) and with log-space init "
                                   "(plateaued ~350, stopped at update 430) — both kept "
                                   "under attempt*/"),
               params=e17[0]["params"], host="gpustar (RTX 5090)")
    json.dump(cfg, open(os.path.join(HERE, "config.json"), "w"), indent=2)

    comps = {}
    for k, d in others.items():
        fb = [d[s]["final"] for s in SEEDS]
        delta, se, t = welch(f17, fb)
        sep, p = separation(f17, fb)
        comps[k] = dict(reference_final=round(Ag[k]["final_mean"], 1),
                        reference_std=round(Ag[k]["final_std"], 1),
                        delta=round(delta, 1), welch_se=round(se, 1),
                        welch_abs_t=round(t, 2),
                        pct_of_reference=round(100 * A["final_mean"] / Ag[k]["final_mean"], 1),
                        complete_rank_separation=bool(sep),
                        separation_exact_p=round(p, 4))

    # pooled rank separation vs both exp10 arms together
    all_e10 = [others["exp10_committed"][s]["final"] for s in SEEDS] + \
              [others["exp10_gpustar_repro"][s]["final"] for s in SEEDS]
    sep_all, p_all = separation(f17, all_e10)

    summary = dict(
        exp_name="exp17_lut-anchor-pair-t32-logsumexp", algo="ppo", n_seeds=3,
        forked_from="exp10_lut-anchor-pair-t32",
        readout="T * tau * log((1/T) * sum_tables exp(w / tau))",
        ppo_best_mean=round(A["best_mean"], 1), ppo_best_std=round(A["best_std"], 1),
        ppo_final_mean=round(A["final_mean"], 1), ppo_final_std=round(A["final_std"], 1),
        params=e17[0]["params"],
        throughput_env_per_s_mean=round(A["thr_mean"]),
        training_time_hours_mean=round(A["wall_mean"] / 3600, 3),
        collapsed_seeds=A["collapsed"], collapse_criterion="final/best < 0.90",
        tau_init=0.05, init="additive (plain) — the sum-scaled readout needs no special init",
        warmup_updates_to=warmup_summary(e17),
        warmup_reference={k: warmup_summary(v) for k, v in others.items()},
        learned_tau={str(s): round(e17[s]["tau"], 5) for s in SEEDS},
        learned_tau_mean=round(float(np.mean([e17[s]["tau"] for s in SEEDS])), 5),
        per_seed={str(s): dict(best=round(e17[s]["best"], 1),
                               final=round(e17[s]["final"], 1),
                               final_over_best=round(e17[s]["final"] / e17[s]["best"], 3),
                               tau=round(e17[s]["tau"], 5)) for s in SEEDS},
        comparisons=comps,
        vs_all_exp10_seeds=dict(n=len(all_e10), complete_rank_separation=bool(sep_all),
                                exact_p=round(p_all, 4)))
    json.dump(summary, open(os.path.join(HERE, "summary.json"), "w"), indent=2)

    print(f"{'':10} {'best':>9} {'final':>9} {'f/b':>6} {'tau':>9}")
    for s in SEEDS:
        e = e17[s]
        print(f"exp17 s{s}  {e['best']:>9.1f} {e['final']:>9.1f} "
              f"{e['final'] / e['best']:>6.3f} {e['tau']:>9.5f}")
    print(f"\nexp17                  final {A['final_mean']:8.1f} +- {A['final_std']:6.1f} "
          f"| best {A['best_mean']:8.1f} | collapse {len(A['collapsed'])}/3")
    for k in ARMS:
        g = Ag[k]
        print(f"{k:22s} final {g['final_mean']:8.1f} +- {g['final_std']:6.1f} "
              f"| best {g['best_mean']:8.1f} | collapse {len(g['collapsed'])}/3")
    print()
    for k, c in comps.items():
        print(f"vs {k:22s} {c['delta']:+9.1f}  se {c['welch_se']:6.1f}  "
              f"|t| {c['welch_abs_t']:5.2f}  ({c['pct_of_reference']:.1f}%)"
              f"{'   RANK-SEPARATED p=' + str(c['separation_exact_p']) if c['complete_rank_separation'] else ''}")
    print(f"\nvs all {len(all_e10)} exp10 seeds pooled: complete separation "
          f"{sep_all}, exact p = {p_all:.4f}")
    print("\nWARMUP — mean updates to reach a return level (of 768; '-' = never)")
    print(f"{'arm':<32}" + "".join(f"{lv:>10}" for lv in WARMUP_LEVELS))
    rows = [("exp17 (attempt 2)", e17)] + [(k, v) for k, v in others.items()]
    for name, d in rows:
        cells = ""
        for lv in WARMUP_LEVELS:
            vals = [d[s]["warmup"][lv] for s in SEEDS]
            n_ok = sum(v is not None for v in vals)
            cells += (f"{'-':>10}" if n_ok == 0 else
                      f"{np.mean([v for v in vals if v is not None]):>8.0f}"
                      f"{'*' if n_ok < len(SEEDS) else ' '} ")
        print(f"{name:<32}{cells}")
    print("  (* = only some seeds reached that level; the mean covers those only)")

    print(f"\nlearned tau mean {summary['learned_tau_mean']:.5f} (init 0.05)")
    print(f"throughput {A['thr_mean']:,.0f} steps/s | {A['wall_mean'] / 60:.1f} min/seed")
    print("\nwrote config.json, metrics.csv, summary.json")


if __name__ == "__main__":
    main()
