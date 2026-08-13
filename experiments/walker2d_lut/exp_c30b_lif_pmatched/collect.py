"""exp_c30b — collect the CPU-reference results for the PARAM-MATCHED LIF actor.

Two anchors, answering different questions:

  * exp_c18 hyperplane, nap6/tph32, 6 seeds, 4308.0 +/- 500.1, 49,152 params -- the same
    table geometry and SAC recipe with the ORIGINAL index front-end. This is the
    per-parameter comparison the experiment exists for, and it is only fair now that the
    LIF actor is 48,193 rather than exp_c30's 87,361.
  * exp_c30 dense-P LIF, 3 seeds, 3931.3 +/- 585.8, 87,361 params -- the same model with
    an unfactorised ordered-pair channel. This measures what the 44.8% cut actually cost.

Both comparisons are UNPAIRED: the three sets share no seeds, so the paired trick that
carried exp_c29 is unavailable here. Welch standard errors.

Usage:
  python collect.py
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SEEDS = (0, 1, 2)
TARGET_PARAMS = 49152
# (name, mean, sd, n_seeds, actor params)
ANCHORS = [("exp_c18 hyperplane nap6/tph32", 4308.0, 500.1, 6, 49152),
           ("exp_c30 dense-P LIF", 3931.3, 585.8, 3, 87361)]


def mean_sd(xs):
    n = len(xs)
    if n == 0:
        return float("nan"), float("nan")
    m = sum(xs) / n
    if n < 2:
        return m, float("nan")
    return m, (sum((x - m) ** 2 for x in xs) / (n - 1)) ** 0.5


def main():
    rows = {}
    for s in SEEDS:
        p = os.path.join(HERE, f"lif_sac_c30b_s{s}_cpueval.json")
        rows[s] = json.load(open(p)) if os.path.exists(p) else None

    print("=== exp_c30b — param-matched LIF-detector actor, nap6/tph32, 3 seeds ===")
    print(f"  {'seed':>4}{'CPU-ref 100ep':>16}{'ep-sd':>8}{'full':>8}{'vel':>7}"
          f"{'params':>10}")
    vals, n_par = [], None
    for s in SEEDS:
        d = rows[s]
        if not d:
            print(f"  {s:>4}{'—':>16}")
            continue
        vals.append(d["cpu_reference_mean"])
        n_par = d["params"]
        print(f"  {s:>4}{d['cpu_reference_mean']:>16.1f}{d['cpu_reference_std']:>8.1f}"
              f"{d['full_length']:>5}/100{d['velocity']:>7.3f}{d['params']:>10,}")

    m, sd = mean_sd(vals)
    print(f"\n  param-matched LIF mean over {len(vals)} seeds: {m:8.1f} +/- {sd:.1f}")
    if n_par:
        print(f"  actor params {n_par:,}  "
              f"({100 * n_par / TARGET_PARAMS - 100:+.2f}% vs the "
              f"{TARGET_PARAMS:,} of exp_c18)")

    out_anchors = []
    for name, amean, asd, an, apar in ANCHORS:
        delta = m - amean
        se = ((sd ** 2 / max(len(vals), 1)) + (asd ** 2 / an)) ** 0.5 \
            if len(vals) > 1 else float("nan")
        t = abs(delta) / se if se and se == se else float("nan")
        print(f"\n  vs {name}  ({amean:.1f} +/- {asd:.1f}, n={an}, {apar:,} params)")
        print(f"     delta {delta:+.1f}   unpaired Welch se {se:.1f}   |t| {t:.2f}")
        out_anchors.append(dict(name=name, mean=amean, sd=asd, n=an, params=apar,
                                delta=delta, welch_se=se))

    json.dump(dict(seeds={s: (rows[s]["cpu_reference_mean"] if rows[s] else None)
                          for s in SEEDS},
                   mean=m, sd=sd, n=len(vals), actor_params=n_par,
                   target_params=TARGET_PARAMS, param_matched=True,
                   anchors=out_anchors),
              open(os.path.join(HERE, "results.json"), "w"), indent=1)
    print("\nwrote results.json")


if __name__ == "__main__":
    main()
