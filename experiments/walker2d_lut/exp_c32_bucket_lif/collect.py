"""exp_c32 — collect the CPU-reference results for the Bucket-LIF actor.

FOUR anchors, because this sits at the end of a four-model line. All of them share the SAC
recipe, the critic, and the eval protocol; only the actor's index front-end differs.

  * exp_c18 hyperplane, 6 seeds, 4308.0 +/- 500.1 -- the standing anchor.
  * exp_c31 PureLIF (TTFS bits), 3 seeds, 2951.2 +/- 2109.2 -- the nearest relative: same
    membrane and first-spike machinery, but the spike time becomes `nap` independent bits
    instead of one bucket index. Its mean is bimodal (4262, 4073, 518) and is quoted here
    only because dropping the stuck seed would be cherry-picking; see the note printed
    below.
  * exp_c30b factorised-P, 3 seeds, 4086.8 +/- 991.2.
  * exp_c30 dense-P, 3 seeds, 3931.3 +/- 585.8.

All comparisons are UNPAIRED -- no shared seeds -- so Welch standard errors.

PARAMETERS. Every model above carries a 24,576-entry table (nap6/tph32 -> 64 rows). This
one does NOT: 16 buckets means 16 rows, so its table is 6,144. That makes it the only entry
in the chapter where the total and the front-end tell genuinely different stories, and both
are printed. The front-end ratio is the fair measure of the ADDRESSING idea; the total is
the fair measure of the deployed model.
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SEEDS = (0, 1, 2)
BASELINE_TOTAL, BASELINE_FRONT = 28032, 3456
# (name, mean, sd, n_seeds, total params, front-end params)
ANCHORS = [("exp_c18 hyperplane", 4308.0, 500.1, 6, 28032, 3456),
           ("exp_c31 PureLIF (TTFS bits)", 2951.2, 2109.2, 3, 31392, 6816),
           ("exp_c30b factorised-P LIF", 4086.8, 991.2, 3, 48193, 23617),
           ("exp_c30 dense-P LIF", 3931.3, 585.8, 3, 87361, 62785)]


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
        p = os.path.join(HERE, f"bucket_sac_c32_s{s}_cpueval.json")
        rows[s] = json.load(open(p)) if os.path.exists(p) else None

    print("=== exp_c32 — Bucket-LIF actor, 16 buckets x 32 tables, 3 seeds ===")
    print(f"  {'seed':>4}{'CPU-ref 100ep':>16}{'ep-sd':>8}{'full':>8}{'vel':>7}"
          f"{'params':>9}")
    vals, n_par, n_front = [], None, None
    for s in SEEDS:
        d = rows[s]
        if not d:
            print(f"  {s:>4}{'—':>16}")
            continue
        vals.append(d["cpu_reference_mean"])
        n_par, n_front = d["params"], d["frontend_params"]
        print(f"  {s:>4}{d['cpu_reference_mean']:>16.1f}{d['cpu_reference_std']:>8.1f}"
              f"{d['full_length']:>5}/100{d['velocity']:>7.3f}{d['params']:>9,}")

    m, sd = mean_sd(vals)
    print(f"\n  Bucket-LIF mean over {len(vals)} seeds: {m:8.1f} +/- {sd:.1f}")
    if n_par:
        print(f"  actor params {n_par:,} = {n_front:,} front-end + "
              f"{n_par - n_front:,} table")
        print(f"    total     {100 * n_par / BASELINE_TOTAL:5.1f}% of the "
              f"{BASELINE_TOTAL:,} hyperplane baseline")
        print(f"    front-end {100 * n_front / BASELINE_FRONT:5.1f}% of its "
              f"{BASELINE_FRONT:,}")
        print(f"    table     {100 * (n_par-n_front) / 24576:5.1f}% of the usual 24,576 "
              f"(16 rows, not 64)")

    out_anchors = []
    for name, amean, asd, an, apar, afront in ANCHORS:
        delta = m - amean
        se = ((sd ** 2 / max(len(vals), 1)) + (asd ** 2 / an)) ** 0.5 \
            if len(vals) > 1 else float("nan")
        t = abs(delta) / se if se and se == se else float("nan")
        print(f"\n  vs {name}  ({amean:.1f} +/- {asd:.1f}, n={an}, {apar:,} total / "
              f"{afront:,} front-end)")
        print(f"     delta {delta:+.1f}   unpaired Welch se {se:.1f}   |t| {t:.2f}")
        if n_par:
            print(f"     size: {n_par/apar:.2f}x total, {n_front/afront:.2f}x front-end")
        out_anchors.append(dict(name=name, mean=amean, sd=asd, n=an, params=apar,
                                front_end=afront, delta=delta, welch_se=se))

    print("\n  NOTE: exp_c31's 2951.2 +/- 2109.2 is bimodal (4262, 4073, 518). Comparing "
          "\n  a mean against it is weak in both directions; per-seed values are the "
          "honest read.")

    json.dump(dict(seeds={s: (rows[s]["cpu_reference_mean"] if rows[s] else None)
                          for s in SEEDS},
                   mean=m, sd=sd, n=len(vals), actor_params=n_par,
                   frontend_params=n_front,
                   table_params=(n_par - n_front) if n_par else None,
                   baseline_total=BASELINE_TOTAL, baseline_front=BASELINE_FRONT,
                   param_matched=False, anchors=out_anchors),
              open(os.path.join(HERE, "results.json"), "w"), indent=1)
    print("\nwrote results.json")


if __name__ == "__main__":
    main()
