"""Collect exp18 (exponential critic) and its control (plain-sum LUT critic).

Writes the convention trio into BOTH arm folders. Metric definitions verbatim from
src/summarize_bench.py, as everywhere else in this chapter:
    final = last ep_ret_mean ; best = max ep_ret_mean ; aggregate std = np.std (ddof=0)
Collapse criterion `final/best < 0.90`, calibrated against the committed exp02-05 labels.

THE DESIGN. exp13-15 already showed a plain LUT critic costs a lot against an MLP critic
(tph32: 2358.6 vs 5488.4), so treatment-vs-exp17 would only re-measure that. The control
holds the LUT critic fixed and toggles ONLY its readout, which isolates the exponential.
Together with exp13 and exp17 this fills a 2x2:

                        actor readout
                    plain sum        LSE-sum
    critic  MLP     exp10 5488.4     exp17 5403.8
            LUT     exp13 2358.6     exp18ctl (control)  /  exp18 (both exponential)

Usage:  python collect.py
"""
import csv
import json
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(HERE, "..")
TREAT = HERE
CTRL = os.path.join(BASE, "exp18ctl_lut-lse-plaincrit-t32")
REFS = {
    "exp10_mlpcrit_plainactor": os.path.join(BASE, "exp10_lut-anchor-pair-t32"),
    "exp13_lutcrit_plainactor": os.path.join(BASE, "exp13_lut-anchor-pair-lutcrit-t32"),
    "exp17_mlpcrit_lseactor": os.path.join(BASE, "exp17_lut-anchor-pair-t32-logsumexp"),
}
SEEDS = (0, 1, 2)
COLLAPSE_RATIO = 0.90
WARMUP_LEVELS = (1000, 3000, 5000)


def warmup(hist, level):
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
                      wall=j["wall_s"], thr=j["throughput_env_per_s"], params=j["params"],
                      tau_a=h[-1].get("tau_actor", h[-1].get("tau")),
                      tau_c=h[-1].get("tau_critic"),
                      warmup={lv: warmup(h, lv) for lv in WARMUP_LEVELS})
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


def sep(a, b):
    return (max(a) < min(b) or max(b) < min(a)), 1.0 / math.comb(len(a) + len(b), len(a))


def write_trio(folder, name, desc, arch, d, A, extra):
    with open(os.path.join(folder, "metrics.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["seed", "update", "env_steps", "ep_ret_mean", "ep_ret_max",
                    "ep_len_mean", "lr", "logstd", "kl", "tau_actor", "tau_critic"])
        for s in SEEDS:
            for r in d[s]["hist"]:
                w.writerow([s, r["update"], r["env_steps"], r["ep_ret_mean"],
                            r["ep_ret_max"], r["ep_len_mean"], r["lr"], r["logstd"],
                            r["kl"], r.get("tau_actor"), r.get("tau_critic")])
    cfg = dict(json.load(open(os.path.join(REFS["exp10_mlpcrit_plainactor"],
                                           "config.json"))))
    cfg.update(exp_name=name, description=desc, arch=arch,
               forked_from="exp17_lut-anchor-pair-t32-logsumexp",
               critic="anchor-pair LUT, tables_per_head=32 (matches the actor and exp13)",
               readout="T * tau * log((1/T) * sum_t exp(w_t / tau))",
               tau_init=0.05, weight_init="plain additive",
               params=d[0]["params"], host="gpustar (RTX 5090)", **extra)
    json.dump(cfg, open(os.path.join(folder, "config.json"), "w"), indent=2)


def main():
    t, c = load(TREAT), load(CTRL)
    refs = {k: load(v) for k, v in REFS.items()}
    At, Ac = agg(t), agg(c)
    Ar = {k: agg(v) for k, v in refs.items()}
    ft = [t[s]["final"] for s in SEEDS]
    fc = [c[s]["final"] for s in SEEDS]

    write_trio(TREAT, "exp18_lut-lse-lutcrit-t32",
               ("exp17's sum-scaled log-sum-exp actor paired with an anchor-pair LUT "
                "critic carrying the SAME exponential readout, each with its own "
                "trainable tau. Tests whether an exponential value head gives the actor's "
                "tau a reason to use the max-like regime."),
               "fastlut_lse_sum2", t, At, dict(critic_readout="log-sum-exp (sum-scaled)"))
    write_trio(CTRL, "exp18ctl_lut-lse-plaincrit-t32",
               ("CONTROL for exp18: identical in every respect except the critic's "
                "readout, which is the plain sum over tables. Isolates the effect of the "
                "exponential critic readout from 'LUT critic vs MLP critic', which "
                "exp13-15 already measured."),
               "fastlut_lse_sum2_plaincrit", c, Ac,
               dict(critic_readout="plain sum over tables"))

    d_ct, se_ct, t_ct = welch(ft, fc)
    sep_ct, p_ct = sep(ft, fc)
    comps = {"control_plain_lut_critic": dict(
        reference_final=round(Ac["final_mean"], 1), reference_std=round(Ac["final_std"], 1),
        delta=round(d_ct, 1), welch_se=round(se_ct, 1), welch_abs_t=round(t_ct, 2),
        pct_of_reference=round(100 * At["final_mean"] / Ac["final_mean"], 1),
        complete_rank_separation=bool(sep_ct), separation_exact_p=round(p_ct, 4))}
    for k, d in refs.items():
        fb = [d[s]["final"] for s in SEEDS]
        dd, ss, tt = welch(ft, fb)
        sp, pp = sep(ft, fb)
        comps[k] = dict(reference_final=round(Ar[k]["final_mean"], 1),
                        reference_std=round(Ar[k]["final_std"], 1),
                        delta=round(dd, 1), welch_se=round(ss, 1), welch_abs_t=round(tt, 2),
                        pct_of_reference=round(100 * At["final_mean"] / Ar[k]["final_mean"], 1),
                        complete_rank_separation=bool(sp), separation_exact_p=round(pp, 4))

    def tau_block(d, key):
        vals = [d[s][key] for s in SEEDS]
        vals = [v for v in vals if v is not None and v == v]
        return (dict(per_seed={str(s): round(d[s][key], 5) for s in SEEDS},
                     mean=round(float(np.mean(vals)), 5)) if vals else None)

    for folder, name, d, A, comp in ((TREAT, "exp18_lut-lse-lutcrit-t32", t, At, comps),
                                     (CTRL, "exp18ctl_lut-lse-plaincrit-t32", c, Ac, None)):
        summary = dict(
            exp_name=name, algo="ppo", n_seeds=3,
            ppo_best_mean=round(A["best_mean"], 1), ppo_best_std=round(A["best_std"], 1),
            ppo_final_mean=round(A["final_mean"], 1), ppo_final_std=round(A["final_std"], 1),
            params=d[0]["params"], throughput_env_per_s_mean=round(A["thr_mean"]),
            training_time_hours_mean=round(A["wall_mean"] / 3600, 3),
            collapsed_seeds=A["collapsed"], collapse_criterion="final/best < 0.90",
            tau_init=0.05,
            learned_tau_actor=tau_block(d, "tau_a"),
            learned_tau_critic=tau_block(d, "tau_c"),
            warmup_updates_to={str(lv): (None if any(d[s]["warmup"][lv] is None
                                                     for s in SEEDS)
                                         else round(float(np.mean(
                                             [d[s]["warmup"][lv] for s in SEEDS])), 1))
                               for lv in WARMUP_LEVELS},
            per_seed={str(s): dict(best=round(d[s]["best"], 1),
                                   final=round(d[s]["final"], 1),
                                   final_over_best=round(d[s]["final"] / d[s]["best"], 3),
                                   tau_actor=(round(d[s]["tau_a"], 5)
                                              if d[s]["tau_a"] is not None else None),
                                   tau_critic=(round(d[s]["tau_c"], 5)
                                               if d[s]["tau_c"] is not None
                                               and d[s]["tau_c"] == d[s]["tau_c"] else None))
                      for s in SEEDS})
        if comp:
            summary["comparisons"] = comp
        json.dump(summary, open(os.path.join(folder, "summary.json"), "w"), indent=2)

    # ---- report -----------------------------------------------------------
    print(f"{'':14} {'best':>9} {'final':>9} {'f/b':>6} {'tau_actor':>10} {'tau_crit':>9}")
    for tag, d in (("exp18   ", t), ("exp18ctl", c)):
        for s in SEEDS:
            tc = d[s]["tau_c"]
            print(f"{tag} s{s}   {d[s]['best']:>9.1f} {d[s]['final']:>9.1f} "
                  f"{d[s]['final'] / d[s]['best']:>6.3f} {d[s]['tau_a']:>10.5f} "
                  f"{(f'{tc:.5f}' if tc is not None and tc == tc else '—'):>9}")
    print()
    rows = [("exp18 exponential critic", At), ("exp18ctl plain LUT critic", Ac)]
    rows += [(k, Ar[k]) for k in REFS]
    for name, A in rows:
        print(f"{name:<34} final {A['final_mean']:8.1f} +- {A['final_std']:6.1f} "
              f"| collapse {len(A['collapsed'])}/3")
    print()
    for k, cc in comps.items():
        print(f"exp18 vs {k:<28} {cc['delta']:+9.1f}  se {cc['welch_se']:6.1f}  "
              f"|t| {cc['welch_abs_t']:5.2f}  ({cc['pct_of_reference']:.1f}%)"
              + ("  RANK-SEP" if cc["complete_rank_separation"] else ""))
    ta_t = np.mean([t[s]["tau_a"] for s in SEEDS])
    ta_c = np.mean([c[s]["tau_a"] for s in SEEDS])
    tc_t = np.mean([t[s]["tau_c"] for s in SEEDS])
    print(f"\nACTOR tau  — exp18 {ta_t:.5f} | control {ta_c:.5f} | exp17 (MLP critic) "
          f"{np.mean([refs['exp17_mlpcrit_lseactor'][s]['tau_a'] for s in SEEDS]):.5f} "
          f"| init 0.05")
    print(f"CRITIC tau — exp18 {tc_t:.5f} (init 0.05)")
    print("  (tau UP = more sum-like, tau->inf is the plain sum; DOWN = more max-like)")
    print(f"\nwall {At['wall_mean'] / 60:.1f} / {Ac['wall_mean'] / 60:.1f} min per seed")
    print("\nwrote config.json, metrics.csv, summary.json in both arm folders")


if __name__ == "__main__":
    main()
