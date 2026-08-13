"""exp_c22 follow-up — two things the headline table raises but does not answer (#75).

1. IS THE LUT SCORE DISTRIBUTION BIMODAL, OR WAS THAT AN n=6 ARTEFACT?
   From exp_c18's six seeds I described this as bimodal -- "five at 4112 +/- 160 plus one
   at 5287" -- and the basin language in exp_c20/c21 leans on that reading. Twelve seeds
   look far more like a continuum. This tests it properly: the gap structure of the sorted
   scores, a dip test surrogate (largest gap vs mean gap), and -- the part that actually
   decides it -- the FORWARD VELOCITY of every seed, because exp_c18's behaviour analysis
   showed the score is essentially 1000 + 1000*velocity for a policy that never falls. If
   the gaits cluster, "basin" is right; if velocity is continuous, it is a continuum and I
   should stop calling it a basin.

2. IS THE RETENTION DIFFERENCE REAL? The table shows LUT 0.973 vs MLP 0.931 but no test.
   Claiming a stability edge without one would repeat the mistake this whole study exists
   to fix.
"""
import json, os, sys

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "..")
C09 = os.path.join(D, "exp_c09_lut_sac")
for p in ("exp_c02_mjx_scaffold", "exp_c06_jax_backprop", "exp_c07_robustness",
          "exp_c11_lut_sac_2x2", "exp_c09_lut_sac", "exp_c18_seed_variance"):
    sys.path.insert(0, os.path.join(D, p))

import perturb      # noqa: E402
import eval_cpu     # noqa: E402
import behavior     # noqa: E402

R = json.load(open(os.path.join(HERE, "matched_power_results.json")))
NEW = (6, 7, 8, 9, 10, 11)


def ckpt(seed):
    return os.path.join(C09, (f"lut_sac_c22_lut_s{seed}_actor.npz" if seed in NEW
                              else f"lut_sac_c18_seed{seed}_actor.npz"))


def main():
    m = perturb.make_model(None, 1.0)
    rows = []
    for r in sorted(R["lut"]["runs"], key=lambda r: r["seed"]):
        s = r["seed"]
        fn, _ = eval_cpu.load_actor(ckpt(s), forward_mode="hard")
        per_ep, _ = behavior.rollout_instrumented(m, fn)
        got = float(per_ep["ret"].mean())
        assert abs(got - r["score"]) < 0.5, f"seed {s}: {got:.1f} != {r['score']:.1f}"
        rows.append(dict(seed=s, score=r["score"],
                         vel=float(per_ep["vel_mean"].mean()),
                         full=int((per_ep["length"] >= 1000).sum()),
                         z=float(per_ep["z_mean"].mean())))
        print(f"  seed {s:>2}: {r['score']:7.1f}  vel {rows[-1]['vel']:.3f}  "
              f"full {rows[-1]['full']}/100", flush=True)

    print("\n=== 1. BIMODAL, OR A CONTINUUM? (12 LUT seeds) ===")
    by = sorted(rows, key=lambda r: r["score"])
    sc = np.array([r["score"] for r in by])
    vel = np.array([r["vel"] for r in by])
    gaps = np.diff(sc)
    print(f"{'rank':>5}{'seed':>6}{'score':>10}{'gap':>9}{'fwd vel':>10}{'full/100':>10}")
    for i, r in enumerate(by):
        g = f"{gaps[i-1]:.1f}" if i else "-"
        print(f"{i+1:>5}{r['seed']:>6}{r['score']:>10.1f}{g:>9}{r['vel']:>10.3f}"
              f"{r['full']:>7}/100")
    ratio = float(gaps.max() / gaps.mean())
    print(f"\n  largest gap {gaps.max():.1f} vs mean gap {gaps.mean():.1f} "
          f"-> ratio {ratio:.2f}")
    print(f"  velocity range {vel.min():.3f}-{vel.max():.3f} m/s, "
          f"sd {vel.std(ddof=1):.3f}")
    r_sv = float(np.corrcoef(sc, vel)[0, 1])
    print(f"  corr(score, velocity) = {r_sv:+.4f}")
    # With 11 gaps from a unimodal sample, the largest is typically ~3x the mean; a true
    # two-cluster split shows up as a gap many times the mean with nothing near it.
    if ratio > 4.0:
        v1 = (f"BIMODAL-ish: the largest gap is {ratio:.1f}x the mean gap, which is more "
              f"separation than a unimodal sample of 12 usually shows.")
    else:
        v1 = (f"A CONTINUUM, NOT TWO CLUSTERS. The largest gap is only {ratio:.1f}x the "
              f"mean gap, and velocity spans {vel.min():.2f}-{vel.max():.2f} m/s smoothly. "
              f"The 'bimodal, five at 4112 plus one at 5287' reading I gave from exp_c18's "
              f"six seeds was an n=6 artefact: seeds 8 and 11 land at 4814 and 5064, "
              f"filling the gap that made it look like two clusters. Seed 4 is the top of "
              f"a continuous right tail, not a separate mode.")
    print(f"\n  {v1}")

    print("\n=== 2. IS THE RETENTION EDGE REAL? ===")
    lr = np.array([r["final_mjx"] / r["max_mjx"] for r in R["lut"]["runs"]])
    mr = np.array([r["final_mjx"] / r["max_mjx"] for r in R["mlp"]["runs"]])
    t, p = stats.ttest_ind(lr, mr, equal_var=False)
    print(f"  within-evaluator retention (final MJX / best MJX):")
    print(f"    LUT {lr.mean():.3f} +/- {lr.std(ddof=1):.3f}   "
          f"MLP {mr.mean():.3f} +/- {mr.std(ddof=1):.3f}")
    print(f"    Welch t = {t:+.3f}, p = {p:.4g}  -> "
          f"{'SIGNIFICANT' if p < 0.05 else 'not significant'}")
    v2 = (f"The LUT retains {100*(lr.mean()-mr.mean()):+.1f} percentage points more of its "
          f"best (p = {p:.3g})."
          + ("" if p < 0.05 else " Not significant at n=12 -- report it as a consistent "
             "direction across three studies, not an established effect."))
    print(f"\n  {v2}")

    json.dump(dict(lut_gait=rows, gap_ratio=ratio, corr_score_velocity=r_sv,
                   distribution_verdict=v1,
                   retention_lut=lr.tolist(), retention_mlp=mr.tolist(),
                   retention_t=float(t), retention_p=float(p), retention_verdict=v2),
              open(os.path.join(HERE, "followup_results.json"), "w"), indent=1)
    print("\nwrote followup_results.json")


if __name__ == "__main__":
    main()
