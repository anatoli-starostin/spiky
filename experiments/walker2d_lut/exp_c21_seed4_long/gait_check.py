"""exp_c21 — did the extra 10k iters buy speed, or trade reliability for it? (#75). MJX venv.

The headline gain is +360.9, but the per-episode sd went from 51.3 at 10k to 592.2 at 20k --
an 11.5x jump. In this environment that signature almost always means the policy started
falling in some episodes, and a mean that improves while the failure rate rises is a
different result from a mean that improves cleanly. exp_c18's behaviour analysis showed the
score decomposes as steps_survived + forward distance - control cost, so the same
decomposition separates the two here.
"""
import json, os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "..")
C09 = os.path.join(D, "exp_c09_lut_sac")
for p in ("exp_c02_mjx_scaffold", "exp_c06_jax_backprop", "exp_c07_robustness",
          "exp_c11_lut_sac_2x2", "exp_c09_lut_sac", "exp_c18_seed_variance"):
    sys.path.insert(0, os.path.join(D, p))

import perturb      # noqa: E402
import eval_cpu     # noqa: E402
import behavior     # noqa: E402

TARGETS = [("lut_sac_c21_seed4_20k_at10000_actor.npz", "seed 4 @ 10k"),
           ("lut_sac_c21_seed4_20k_actor.npz", "seed 4 @ 20k")]


def main():
    m = perturb.make_model(None, 1.0)
    rows = []
    for ck, lab in TARGETS:
        fn, _ = eval_cpu.load_actor(os.path.join(C09, ck), forward_mode="hard")
        per_ep, _ = behavior.rollout_instrumented(m, fn)
        r = per_ep["ret"]
        full = int((per_ep["length"] >= 1000).sum())
        surv = per_ep["length"] >= 1000
        rows.append(dict(
            label=lab, checkpoint=ck, mean=float(r.mean()), sd=float(r.std(ddof=1)),
            median=float(np.median(r)), p10=float(np.percentile(r, 10)),
            worst=float(r.min()), best=float(r.max()),
            full=full, fell=int(per_ep["fell"].sum()),
            len_mean=float(per_ep["length"].mean()),
            vel=float(per_ep["vel_mean"].mean()),
            vel_surv=float(per_ep["vel_mean"][surv].mean()) if surv.any() else float("nan"),
            z=float(per_ep["z_mean"].mean()),
            energy=float(per_ep["act_energy"].mean())))
        print(f"  {lab}: {r.mean():.1f} +/- {r.std(ddof=1):.1f}, "
              f"full {full}/100, vel {rows[-1]['vel']:.3f}", flush=True)

    a, b = rows
    print("\n=== DID THE EXTRA BUDGET BUY SPEED, OR TRADE RELIABILITY FOR IT? ===")
    print(f"{'':<16}{'mean':>9}{'ep-sd':>8}{'median':>9}{'p10':>9}{'worst':>9}"
          f"{'full/100':>10}{'ep len':>8}{'vel(all)':>10}{'vel(survivors)':>16}")
    for r in rows:
        print(f"{r['label']:<16}{r['mean']:>9.1f}{r['sd']:>8.1f}{r['median']:>9.1f}"
              f"{r['p10']:>9.1f}{r['worst']:>9.1f}{r['full']:>7}/100"
              f"{r['len_mean']:>8.0f}{r['vel']:>10.3f}{r['vel_surv']:>16.3f}")

    d_full = b["full"] - a["full"]
    print(f"\n  falls: {100-a['full']}/100 at 10k -> {100-b['full']}/100 at 20k "
          f"({-d_full:+d} episodes)")
    print(f"  speed: {a['vel']:.3f} -> {b['vel']:.3f} m/s "
          f"({b['vel']-a['vel']:+.3f}); among survivors "
          f"{a['vel_surv']:.3f} -> {b['vel_surv']:.3f}")
    print(f"  median episode: {a['median']:.1f} -> {b['median']:.1f} "
          f"({b['median']-a['median']:+.1f}) — the median ignores the tail the mean "
          f"is exposed to")

    if b["full"] < a["full"] - 2:
        v = (f"SPEED BOUGHT AT THE COST OF RELIABILITY. The 20k policy is "
             f"{b['vel']-a['vel']:+.3f} m/s faster and its MEDIAN episode is "
             f"{b['median']-a['median']:+.0f} better, but it now falls in "
             f"{100-b['full']}/100 episodes against {100-a['full']}/100 at 10k. The "
             f"+{b['mean']-a['mean']:.0f} mean is real but it is a faster, more fragile "
             f"gait -- not a strictly better one. Which to prefer depends on whether the "
             f"metric that matters is mean return or worst-case survival.")
    elif b["mean"] > a["mean"] + 100:
        v = (f"CLEAN IMPROVEMENT. +{b['mean']-a['mean']:.0f} mean with survival "
             f"unchanged ({b['full']}/100 vs {a['full']}/100), driven by "
             f"{b['vel']-a['vel']:+.3f} m/s of extra speed. The extra budget bought "
             f"return without giving anything back.")
    else:
        v = (f"NO MATERIAL CHANGE: {b['mean']-a['mean']:+.0f} mean, survival "
             f"{b['full']}/100 vs {a['full']}/100.")
    print(f"\n  {v}")

    json.dump(dict(rows=rows, verdict=v), open(
        os.path.join(HERE, "gait_check.json"), "w"), indent=1)
    print("\nwrote gait_check.json")


if __name__ == "__main__":
    main()
