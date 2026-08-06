"""exp_c43 — the pure-WIDTH end of the width-vs-count axis, and what it says about c36.

THE CONTROLLED TRIPLE. c38, c39 and c43 all have 32 tables, 64 cells per table and the same
24,576-entry table, under an identical recipe. They differ only in how the 64 cells are
addressed:

    c43   1 detector  x 64 buckets    digit COUNT 1, WIDTH 64   <- this run
    c39   3 detectors x  4 buckets    digit COUNT 3, WIDTH 4
    c38   6 detectors x  2 buckets    digit COUNT 6, WIDTH 2

so the axis is traversed end to end at fixed capacity.

A CAVEAT ON COMPARABILITY, stated because it cuts the useful way: c38 and c39 ran with the
STOCK table init (0.1), c43 with the fan-in-corrected one (0.1/sqrt(tph)), which the c42
line found is mildly better on average. So c43 gets the better initialisation and is still
being compared against configurations that did not -- any deficit it shows is if anything
understated.

THE SECOND, BROADER QUESTION. exp_c36's conclusion was that return tracks "the number of
independent indices SUMMED", i.e. tables. But a table with D detectors contains D
independent LIF cells, so the quantity that unifies both axes might be the TOTAL NUMBER OF
LIF DETECTORS, n_tables * n_det, regardless of how they are distributed. That is testable
against every configuration in this line at once, and it is tested below rather than
asserted -- including the Spearman correlation, so a weak relationship reads as weak.
"""
import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SEEDS = (0, 1, 2)
BASE_M, BASE_SD, BASE_N = 4308.0, 500.1, 6
TAKEOFF = 3000.0

# The identical-capacity triple: 32 tables, 64 cells/table, 24,576-entry table.
TRIPLE = [
    ("c43  1 det x 64 bkt", 1, 64, 27808, None, None),          # filled from results
    ("c39  3 det x  4 bkt", 3, 4, 28384, 2030.2, 1894.7),
    ("c38  6 det x  2 bkt", 6, 2, 31744, 3213.9, 1525.9),
]

# Every configuration in the LIF/bucket line, for the detector-count test.
# (name, n_tables, n_det, mean, sd, n_seeds)
ALL_RUNS = [
    ("c32b 16bkt x 32tab", 32, 1, 2041.2, 1230.1, 3),
    ("c33  64bkt x 32tab", 32, 1, 1536.2, 1416.8, 3),
    ("c37  32bkt x 64tab", 64, 1, 2531.1, 1266.1, 3),
    ("c36  16bkt x128tab", 128, 1, 4246.1, 298.4, 3),
    ("c39   4bkt x 32tab", 32, 3, 2030.2, 1894.7, 3),
    ("c42+b 4bkt x 32tab", 32, 3, 3043.7, 1480.5, 9),
    ("c38   2bkt x 32tab", 32, 6, 3213.9, 1525.9, 3),
]


def mean_sd(xs):
    n = len(xs)
    m = sum(xs) / n
    return (m, (sum((x - m) ** 2 for x in xs) / (n - 1)) ** 0.5) if n > 1 else (m, float("nan"))


def spearman(xs, ys):
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    rx, ry = rank(xs), rank(ys)
    n = len(xs)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den else float("nan")


def main():
    rows = {}
    for s in SEEDS:
        p = os.path.join(HERE, f"mhl_sac_c43_s{s}_cpueval.json")
        rows[s] = json.load(open(p)) if os.path.exists(p) else None

    print("=== exp_c43 — 1 detector x 64 buckets, 32 tables, 3 seeds ===")
    print(f"  {'seed':>4}{'CPU-ref 100ep':>16}{'ep-sd':>8}{'full':>9}{'vel':>8}"
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
              f"{d['full_length']:>6}/100{d['velocity']:>8.3f}{d['params']:>9,}")

    m, sd = mean_sd(vals)
    k = sum(1 for v in vals if v >= TAKEOFF)
    print(f"\n  mean over {len(vals)} seeds: {m:8.1f} +/- {sd:.1f}   takeoff {k}/{len(vals)}")
    if n_par:
        print(f"  params {n_par:,} = {n_front:,} front-end + {n_par-n_front:,} table")
        print(f"    {100*n_par/28032:5.1f}% of the 28,032 hyperplane baseline")
        print(f"    beta_raw alone is 2,016 of the {n_front:,} front-end "
              f"({100*2016/n_front:.0f}%) — 63 boundaries per detector")

    print("\n=== the width-vs-count axis at IDENTICAL capacity (32 tables, 64 cells) ===")
    print(f"  {'config':<22}{'det':>5}{'bkt':>5}{'params':>9}{'mean':>10}{'sd':>9}")
    TRIPLE[0] = (TRIPLE[0][0], 1, 64, 27808, m, sd)
    for name, nd, nb, par, mm, ss in TRIPLE:
        print(f"  {name:<22}{nd:>5}{nb:>5}{par:>9,}{mm:>10.1f}{ss:>9.1f}")
    print("  NOTE c38/c39 ran with the STOCK table init; c43 has the fan-in-corrected one,")
    print("  which the c42 line found is mildly BETTER — so c43's deficit is understated.")

    print("\n=== does return track the TOTAL number of LIF detectors? ===")
    runs = ALL_RUNS + [("c43  64bkt x 32tab", 32, 1, m, sd, len(vals))]
    runs.sort(key=lambda r: r[1] * r[2])
    print(f"  {'config':<22}{'tables':>8}{'det/tab':>9}{'DETECTORS':>11}{'mean':>10}")
    for name, nt, nd, mm, ss, nn in runs:
        print(f"  {name:<22}{nt:>8}{nd:>9}{nt*nd:>11}{mm:>10.1f}")
    xs = [r[1] * r[2] for r in runs]
    ys = [r[3] for r in runs]
    rho = spearman(xs, ys)
    print(f"\n  Spearman rho(total detectors, mean return) = {rho:+.3f} over "
          f"{len(runs)} configurations")
    xs2 = [r[1] for r in runs]
    print(f"  Spearman rho(tables only,     mean return) = {spearman(xs2, ys):+.3f}")

    se = math.sqrt(sd ** 2 / len(vals) + BASE_SD ** 2 / BASE_N)
    print(f"\n  vs exp_c18 hyperplane ({BASE_M:.1f} +/- {BASE_SD:.1f}, n={BASE_N}): "
          f"delta {m-BASE_M:+.1f}, Welch se {se:.1f}, |t| {abs(m-BASE_M)/se:.2f}")
    print(f"  vs exp_c33 (the same 27,808 params, old module, stock table init, "
          f"1536.2 +/- 1416.8): delta {m-1536.2:+.1f}")

    json.dump(dict(seeds={s: (rows[s]["cpu_reference_mean"] if rows[s] else None)
                          for s in SEEDS},
                   mean=m, sd=sd, n=len(vals), takeoff=k,
                   actor_params=n_par, frontend_params=n_front,
                   spearman_detectors=rho, spearman_tables=spearman(xs2, ys),
                   baseline_delta=m - BASE_M, welch_se=se),
              open(os.path.join(HERE, "results.json"), "w"), indent=1)
    print("\nwrote results.json")


if __name__ == "__main__":
    main()
