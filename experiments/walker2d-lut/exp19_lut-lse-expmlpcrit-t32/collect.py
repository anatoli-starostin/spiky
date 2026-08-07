"""Collect exp19 (exponential MLP-critic readout) against its control, exp17.

exp17 IS the control: same LSE actor, same MLP critic backbone, plain LINEAR readout. The
only difference is the critic's readout, verified at init (actor bit-identical, critic
backbone bit-identical, value functions 98.6% shape-correlated).

Metric definitions verbatim from src/summarize_bench.py, as everywhere in this chapter.
Collapse criterion `final/best < 0.90`, calibrated against the committed exp02-05 labels.

Usage:  python collect.py
"""
import csv
import json
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(HERE, "..")
REFS = {
    "exp17_plain_linear_critic_CONTROL": os.path.join(
        BASE, "exp17_lut-anchor-pair-t32-logsumexp"),
    "exp10_plain_actor_mlp_critic": os.path.join(BASE, "exp10_lut-anchor-pair-t32"),
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


def main():
    e19 = load(HERE)
    refs = {k: load(v) for k, v in REFS.items()}
    A = agg(e19)
    Ar = {k: agg(v) for k, v in refs.items()}
    f19 = [e19[s]["final"] for s in SEEDS]

    with open(os.path.join(HERE, "metrics.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["seed", "update", "env_steps", "ep_ret_mean", "ep_ret_max",
                    "ep_len_mean", "lr", "logstd", "kl", "tau_actor", "tau_critic"])
        for s in SEEDS:
            for r in e19[s]["hist"]:
                w.writerow([s, r["update"], r["env_steps"], r["ep_ret_mean"],
                            r["ep_ret_max"], r["ep_len_mean"], r["lr"], r["logstd"],
                            r["kl"], r.get("tau_actor"), r.get("tau_critic")])

    cfg = dict(json.load(open(os.path.join(REFS["exp10_plain_actor_mlp_critic"],
                                           "config.json"))))
    cfg.update(exp_name="exp19_lut-lse-expmlpcrit-t32",
               description=("exp17's sum-scaled log-sum-exp actor + the exp10 MLP critic "
                            "with ONLY its final linear readout replaced by the matching "
                            "sum-scaled log-sum-exp over the 256 penultimate per-unit "
                            "contributions w_i*h_i. Critic backbone (obs -> [256,256] "
                            "Tanh, orthogonal gain 1.0) is bit-identical to exp10's."),
               arch="fastlut_lse_sum_expmlpcrit",
               control="exp17_lut-anchor-pair-t32-logsumexp (identical but for the "
                       "critic's readout)",
               critic_readout="T * tau_c * log((1/T) * sum_i exp(w_i*h_i / tau_c)) + b, "
                              "T = 256 penultimate units",
               actor_readout="T * tau_a * log((1/T) * sum_t exp(w_t / tau_a)), T = 32 tables",
               tau_actor_init=0.05, tau_critic_init=0.25,
               tau_critic_init_rationale=(
                   "measured on real normalised observations: at tau_c=0.25 the value "
                   "function stays 97.6% shape-correlated with exp17's plain-linear head "
                   "while the exponential remains measurably live; tau_c >= 1 is inert "
                   "(corr 0.999, all 256 units uniform) and would be a null result by "
                   "construction; tau_c <= 0.05 starts a substantially different value "
                   "function (corr 0.38)."),
               critic_numerics=("log(mean(exp(d))) computed as log1p(mean(expm1(d))) on "
                                "mean-centred d, not logsumexp(d)-log(T): the latter "
                                "subtracts two ~log(256)=5.55 values and loses the result "
                                "to fp32 cancellation when tau >> spread(u) (0.061 error "
                                "at tau=500). The tau->inf limit now lands on the plain "
                                "linear head to 1.2e-4."),
               exp_clamp=60.0, params=e19[0]["params"], host="gpustar (RTX 5090)")
    json.dump(cfg, open(os.path.join(HERE, "config.json"), "w"), indent=2)

    comps = {}
    for k, d in refs.items():
        fb = [d[s]["final"] for s in SEEDS]
        dd, ss, tt = welch(f19, fb)
        comps[k] = dict(reference_final=round(Ar[k]["final_mean"], 1),
                        reference_std=round(Ar[k]["final_std"], 1),
                        delta=round(dd, 1), welch_se=round(ss, 1), welch_abs_t=round(tt, 2),
                        pct_of_reference=round(100 * A["final_mean"] / Ar[k]["final_mean"], 1),
                        complete_rank_separation=bool(max(f19) < min(fb)
                                                      or max(fb) < min(f19)),
                        separation_exact_p=round(1.0 / math.comb(6, 3), 4))

    summary = dict(
        exp_name="exp19_lut-lse-expmlpcrit-t32", algo="ppo", n_seeds=3,
        control="exp17 (same actor, plain linear critic readout)",
        ppo_best_mean=round(A["best_mean"], 1), ppo_best_std=round(A["best_std"], 1),
        ppo_final_mean=round(A["final_mean"], 1), ppo_final_std=round(A["final_std"], 1),
        params=e19[0]["params"], throughput_env_per_s_mean=round(A["thr_mean"]),
        training_time_hours_mean=round(A["wall_mean"] / 3600, 3),
        collapsed_seeds=A["collapsed"], collapse_criterion="final/best < 0.90",
        tau_actor_init=0.05, tau_critic_init=0.25,
        learned_tau_actor={str(s): round(e19[s]["tau_a"], 5) for s in SEEDS},
        learned_tau_critic={str(s): round(e19[s]["tau_c"], 5) for s in SEEDS},
        learned_tau_actor_mean=round(float(np.mean([e19[s]["tau_a"] for s in SEEDS])), 5),
        learned_tau_critic_mean=round(float(np.mean([e19[s]["tau_c"] for s in SEEDS])), 5),
        exp17_tau_actor_mean=round(float(np.mean(
            [refs["exp17_plain_linear_critic_CONTROL"][s]["tau_a"] for s in SEEDS])), 5),
        warmup_updates_to={str(lv): (None if any(e19[s]["warmup"][lv] is None
                                                 for s in SEEDS)
                                     else round(float(np.mean(
                                         [e19[s]["warmup"][lv] for s in SEEDS])), 1))
                           for lv in WARMUP_LEVELS},
        warmup_reference={k: {str(lv): (None if any(d[s]["warmup"][lv] is None
                                                    for s in SEEDS)
                                        else round(float(np.mean(
                                            [d[s]["warmup"][lv] for s in SEEDS])), 1))
                              for lv in WARMUP_LEVELS} for k, d in refs.items()},
        per_seed={str(s): dict(best=round(e19[s]["best"], 1),
                               final=round(e19[s]["final"], 1),
                               final_over_best=round(e19[s]["final"] / e19[s]["best"], 3),
                               tau_actor=round(e19[s]["tau_a"], 5),
                               tau_critic=round(e19[s]["tau_c"], 5)) for s in SEEDS},
        comparisons=comps)
    json.dump(summary, open(os.path.join(HERE, "summary.json"), "w"), indent=2)

    print(f"{'':10} {'best':>9} {'final':>9} {'f/b':>6} {'tau_actor':>10} {'tau_crit':>10}")
    for s in SEEDS:
        e = e19[s]
        print(f"exp19 s{s}  {e['best']:>9.1f} {e['final']:>9.1f} "
              f"{e['final'] / e['best']:>6.3f} {e['tau_a']:>10.5f} {e['tau_c']:>10.5f}")
    print(f"\nexp19                          final {A['final_mean']:8.1f} +- "
          f"{A['final_std']:6.1f} | best {A['best_mean']:8.1f} | "
          f"collapse {len(A['collapsed'])}/3")
    for k in REFS:
        g = Ar[k]
        print(f"{k:<30} final {g['final_mean']:8.1f} +- {g['final_std']:6.1f} | "
              f"best {g['best_mean']:8.1f} | collapse {len(g['collapsed'])}/3")
    print()
    for k, c in comps.items():
        print(f"vs {k:<34} {c['delta']:+9.1f}  se {c['welch_se']:6.1f}  "
              f"|t| {c['welch_abs_t']:5.2f}  ({c['pct_of_reference']:.1f}%)"
              + ("  RANK-SEP" if c["complete_rank_separation"] else ""))
    print(f"\nTAU_ACTOR  exp19 {summary['learned_tau_actor_mean']:.5f}  vs  exp17 "
          f"{summary['exp17_tau_actor_mean']:.5f}   (init 0.05)")
    print(f"TAU_CRITIC exp19 {summary['learned_tau_critic_mean']:.5f}   (init 0.25)")
    print("  tau UP = more sum-like (tau->inf IS the plain sum) ; DOWN = more max-like")
    d_ta = summary["learned_tau_actor_mean"] - 0.05
    print(f"  -> actor tau moved {'UP' if d_ta > 0 else 'DOWN'} by {abs(d_ta):.5f} "
          f"from init; exp17 moved UP by {summary['exp17_tau_actor_mean'] - 0.05:.5f}")
    d_tc = summary["learned_tau_critic_mean"] - 0.25
    print(f"  -> critic tau moved {'UP' if d_tc > 0 else 'DOWN'} by {abs(d_tc):.5f} "
          f"from init")
    print(f"\nwall {A['wall_mean'] / 60:.1f} min/seed at {A['thr_mean']:,.0f} steps/s")
    print("\nwrote config.json, metrics.csv, summary.json")


if __name__ == "__main__":
    main()
