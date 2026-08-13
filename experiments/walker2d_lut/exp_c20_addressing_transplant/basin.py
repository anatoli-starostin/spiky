"""exp_c20 — did the transplanted runs land in seed 4's FAST-GAIT BASIN? (#75). MJX venv.

Why this exists, and why it supersedes the t-test in collect.py.

exp_c18 established that the outcome here is BIMODAL, not a continuum: five seeds at
4112 +/- 160 and one at 5287, and the behaviour deep-dive showed the whole difference is
forward velocity (4.29 vs 3.20 m/s) rather than a graded improvement. A difference-of-means
t-test on a bimodal outcome is the wrong statistic -- it spends all its power estimating a
mean that no run actually sits near, and with n=3 per arm it has essentially none left. Its
confidence interval came out [-1438, +2481], which does not support "the routing does not
transfer"; it supports "this test cannot tell".

The right question is binary: DID THE RUN FIND THE FAST GAIT? That is what this measures --
mean forward velocity from the same instrumented rollout used in exp_c18, plus a Fisher
exact test on basin membership pooling every comparable run in the chapter.
"""
import json, os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "..")
C09 = os.path.join(D, "exp_c09_lut_sac")
C18 = os.path.join(D, "exp_c18_seed_variance")
for p in ("exp_c02_mjx_scaffold", "exp_c06_jax_backprop", "exp_c07_robustness",
          "exp_c11_lut_sac_2x2", "exp_c09_lut_sac", "exp_c18_seed_variance"):
    sys.path.insert(0, os.path.join(D, p))

import perturb        # noqa: E402
import eval_cpu       # noqa: E402
import behavior       # noqa: E402  (reuse the SAME instrumented rollout as exp_c18)

SEEDS = (100, 101, 102)
ARMS = (("from4", "seed 4's routing"), ("from5", "seed 5's routing (control)"))
FAST = 5000.0        # basin boundary: exp_c18's pack topped out at 4370, seed 4 hit 5287


def log_comb(n, k):
    from math import lgamma
    return lgamma(n + 1) - lgamma(k + 1) - lgamma(n - k + 1)


def fisher_one_sided(a, b, c, d):
    """P(as extreme or more) for a 2x2 [[a,b],[c,d]], one-sided on a being large."""
    from math import exp
    n = a + b + c + d
    row1, col1 = a + b, a + c
    p = 0.0
    for k in range(a, min(row1, col1) + 1):
        p += exp(log_comb(col1, k) + log_comb(n - col1, row1 - k) - log_comb(n, row1))
    return p


def main():
    m = perturb.make_model(None, 1.0)
    rows = []
    for arm, lab in ARMS:
        for s in SEEDS:
            ck = os.path.join(C09, f"lut_sac_c20_{arm}_s{s}_actor.npz")
            ev = os.path.join(C09, f"lut_sac_c20_{arm}_s{s}_cpueval.json")
            score = json.load(open(ev))["cpu_reference_mean"]
            fn, _ = eval_cpu.load_actor(ck, forward_mode="hard")
            per_ep, _vis = behavior.rollout_instrumented(m, fn)
            got = float(per_ep["ret"].mean())
            assert abs(got - score) < 0.5, f"{arm}/s{s}: {got:.1f} != {score:.1f}"
            rows.append(dict(arm=arm, label=lab, seed=s, score=score,
                             vel=float(per_ep["vel_mean"].mean()),
                             z=float(per_ep["z_mean"].mean()),
                             full=int((per_ep["length"] >= 1000).sum()),
                             fell=int(per_ep["fell"].sum())))
            print(f"  {arm}/s{s}: {score:7.1f}  vel {rows[-1]['vel']:.3f}  "
                  f"full {rows[-1]['full']}/100", flush=True)

    print("\n=== DID THE TRANSPLANTED RUNS FIND THE FAST GAIT? ===")
    print(f"{'arm':<26}{'seed':>6}{'CPU-ref':>10}{'fwd vel':>10}{'z':>8}"
          f"{'full 1000':>11}{'basin':>10}")
    for r in rows:
        print(f"{r['label']:<26}{r['seed']:>6}{r['score']:>10.1f}{r['vel']:>10.3f}"
              f"{r['z']:>8.3f}{r['full']:>8}/100"
              f"{('FAST' if r['score'] >= FAST else 'slow'):>10}")
    print(f"\n  reference — exp_c18 seed 4 (joint training): 5286.6, vel 4.290, z 1.100")
    print(f"              exp_c18 pack (5 seeds):           3951-4370, vel 2.999-3.491")

    # ---- basin membership, pooled over every comparable run in the chapter ----
    a_hit = sum(1 for r in rows if r["arm"] == "from4" and r["score"] >= FAST)
    a_n = sum(1 for r in rows if r["arm"] == "from4")
    b_hit = sum(1 for r in rows if r["arm"] == "from5" and r["score"] >= FAST)
    b_n = sum(1 for r in rows if r["arm"] == "from5")
    # everything in the chapter NOT using seed 4's routing: arm B + exp_c18's 5 pack seeds
    other_hit, other_n = b_hit, b_n + 5
    print("\n=== BASIN MEMBERSHIP (>= %.0f) ===" % FAST)
    print(f"  seed 4's routing        : {a_hit}/{a_n}")
    print(f"  seed 5's routing        : {b_hit}/{b_n}")
    print(f"  everything else in the chapter (arm B + exp_c18's 5 pack seeds): "
          f"{other_hit}/{other_n}")
    p = fisher_one_sided(a_hit, a_n - a_hit, other_hit, other_n - other_hit)
    print(f"\n  Fisher exact, one-sided: p = {p:.4f}")
    if p < 0.05:
        v = (f"SEED 4'S ROUTING RAISES THE ODDS OF THE FAST BASIN. {a_hit} of {a_n} runs "
             f"with its frozen routing reached the fast gait against {other_hit} of "
             f"{other_n} without it (p = {p:.3f}). The routing is a genuine, transferable "
             f"carrier -- but NOT a guarantee: it still failed once in three.")
    else:
        v = (f"SUGGESTIVE, NOT ESTABLISHED. {a_hit} of {a_n} runs with seed 4's routing "
             f"reached the fast gait against {other_hit} of {other_n} without it, but "
             f"Fisher gives p = {p:.3f}. The pattern points at the routing carrying the "
             f"win; at these sample sizes it does not clear significance. More seeds on "
             f"arm A is the cheap way to settle it.")
    print(f"\n  {v}")

    json.dump(dict(runs=rows, fast_threshold=FAST, arm_a=[a_hit, a_n],
                   arm_b=[b_hit, b_n], other=[other_hit, other_n],
                   fisher_p=float(p), verdict=v),
              open(os.path.join(HERE, "basin_results.json"), "w"), indent=1)
    print("\nwrote basin_results.json")


if __name__ == "__main__":
    main()
