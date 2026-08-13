"""exp_c30 — collect the CPU-reference results for the LIF-detector actor.

Reported against exp_c18's hyperplane cell at the SAME nap6/tph32 shape (6 seeds,
deterministic: 4308.0 +/- 500.1), which is the only fair anchor on disk -- same
architecture family, same table geometry, same SAC recipe, only the index front-end
differs.

The comparison is NOT param-matched and the table says so: the LIF actor carries 87,361
params against 49,152, because the ordered-pair channel P alone is 55,488. A win here is
therefore not evidence that LIF detectors are a better front-end per parameter; it is
evidence that they can drive the actor at all, at 1.8x the budget.

Usage:
  python collect.py
"""
import glob
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SEEDS = (0, 1, 2)
# exp_c18: hyperplane addressing, nap6/tph32, 6 seeds, determinism on.
ANCHOR_MEAN, ANCHOR_SD, ANCHOR_N = 4308.0, 500.1, 6


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
        p = os.path.join(HERE, f"lif_sac_c30_s{s}_cpueval.json")
        rows[s] = json.load(open(p)) if os.path.exists(p) else None

    print("=== exp_c30 — LIF-detector actor, nap6/tph32, 3 seeds ===")
    print(f"  {'seed':>4}{'CPU-ref 100ep':>16}{'ep-sd':>8}{'full':>8}{'vel':>7}"
          f"{'params':>10}")
    vals = []
    for s in SEEDS:
        d = rows[s]
        if not d:
            print(f"  {s:>4}{'—':>16}")
            continue
        vals.append(d["cpu_reference_mean"])
        print(f"  {s:>4}{d['cpu_reference_mean']:>16.1f}{d['cpu_reference_std']:>8.1f}"
              f"{d['full_length']:>5}/100{d['velocity']:>7.3f}{d['params']:>10,}")

    m, sd = mean_sd(vals)
    print(f"\n  LIF mean over {len(vals)} seeds: {m:8.1f} +/- {sd:.1f}")
    print(f"  exp_c18 hyperplane anchor, nap6/tph32, {ANCHOR_N} seeds: "
          f"{ANCHOR_MEAN:8.1f} +/- {ANCHOR_SD:.1f}")
    if vals:
        delta = m - ANCHOR_MEAN
        # Unpaired: the two sets share no seeds and were run at different times, so the
        # paired trick that carried exp_c29 is unavailable. Welch's standard error.
        se = ((sd ** 2 / max(len(vals), 1)) + (ANCHOR_SD ** 2 / ANCHOR_N)) ** 0.5 \
            if len(vals) > 1 else float("nan")
        print(f"  LIF - hyperplane: {delta:+.1f}  (unpaired, Welch se {se:.1f})")
        # CORRECTED 2026-08-03: the baseline is exp_c18 nap6/tph32 = 28,032 TOTAL
        # (24,576 table + 3,456 hyperplane w/b), not 49,152 -- that was exp_c29's
        # table-only figure for nap6/tph64. Both models share the same 24,576 table,
        # so the front-end ratio is the one that means anything.
        print(f"  NOT param-matched: 87,361 vs 28,032 total (3.12x)")
        print(f"  index front-end:   62,785 vs  3,456       (18.2x)  <- the real gap")

    out = dict(seeds={s: (rows[s]["cpu_reference_mean"] if rows[s] else None)
                      for s in SEEDS},
               mean=m, sd=sd, n=len(vals),
               anchor=dict(name="exp_c18 hyperplane nap6/tph32", mean=ANCHOR_MEAN,
                           sd=ANCHOR_SD, n=ANCHOR_N),
               param_matched=False)
    json.dump(out, open(os.path.join(HERE, "results.json"), "w"), indent=1)
    print("\nwrote results.json")


if __name__ == "__main__":
    main()
