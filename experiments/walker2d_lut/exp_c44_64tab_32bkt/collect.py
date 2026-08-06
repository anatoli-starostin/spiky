"""exp_c44 — 64 tables x 1 detector x 32 buckets: a direct test of the detector-count reading.

TWO CONTROLLED COMPARISONS, and they are unusually clean for this chapter.

  vs exp_c37   IDENTICAL in every structural respect: 32 buckets x 64 tables, 1 detector,
               28,992 parameters, 4,416 front-end, 64 total LIF detectors. The only
               differences are the table init (c37 stock 0.1, c44 fan-in 0.1/sqrt(64) =
               0.0125) and delay_init_std (c37 predates it, so zeros; c44 half-normal at 4).
               c37 scored 2531.1 +/- 1266.1.

  vs exp_c43   DOUBLE the detectors (64 vs 32) at almost the same parameter count (28,992 vs
               27,808) and the same total cells (64 tables x 32 vs 32 tables x 64 = 2,048
               either way). c43 scored 1177.2 +/- 506.4.

So if the exp_c43 reading is right -- that return tracks the TOTAL number of LIF detectors
rather than tables, cells or bucket width -- c44 should sit near c37 and clearly above c43.
If it instead tracks something c44 shares with c43 (single detector, wide buckets), it
should sit near c43. The two predictions are far apart, which is what makes this worth
running rather than reasoning about.

The Spearman correlation is recomputed with c44 included, and reported with the caveat it
needs: eight-to-nine configurations at n=3 support an ORDINAL claim about direction, not a
regression, and several of the means overlap inside their own seed noise.
"""
import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SEEDS = (0, 1, 2)
BASE_M, BASE_SD, BASE_N = 4308.0, 500.1, 6
TAKEOFF = 3000.0

# (name, n_tables, n_det, mean, sd, n_seeds)
ALL_RUNS = [
    ("c43  64bkt x 32tab", 32, 1, 1177.2, 506.4, 3),
    ("c33  64bkt x 32tab", 32, 1, 1536.2, 1416.8, 3),
    ("c32b 16bkt x 32tab", 32, 1, 2041.2, 1230.1, 3),
    ("c37  32bkt x 64tab", 64, 1, 2531.1, 1266.1, 3),
    ("c39   4bkt x 32tab", 32, 3, 2030.2, 1894.7, 3),
    ("c42+b 4bkt x 32tab", 32, 3, 3043.7, 1480.5, 9),
    ("c36  16bkt x128tab", 128, 1, 4246.1, 298.4, 3),
    ("c38   2bkt x 32tab", 32, 6, 3213.9, 1525.9, 3),
]


def mean_sd(xs):
    n = len(xs)
    m = sum(xs) / n
    return (m, (sum((x - m) ** 2 for x in xs) / (n - 1)) ** 0.5) if n > 1 \
        else (m, float("nan"))


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


def welch(m1, s1, n1, m2, s2, n2):
    se = math.sqrt(s1 ** 2 / n1 + s2 ** 2 / n2)
    return m1 - m2, se, abs(m1 - m2) / se if se else float("nan")


def main():
    rows = {}
    for s in SEEDS:
        p = os.path.join(HERE, f"mhl_sac_c44_s{s}_cpueval.json")
        rows[s] = json.load(open(p)) if os.path.exists(p) else None

    print("=== exp_c44 — 64 tables x 1 detector x 32 buckets, 3 seeds ===")
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
    print(f"\n  mean over {len(vals)} seeds: {m:8.1f} +/- {sd:.1f}   "
          f"takeoff {k}/{len(vals)}")
    if n_par:
        print(f"  params {n_par:,} = {n_front:,} front-end + {n_par-n_front:,} table  "
              f"({100*n_par/28032:.1f}% of baseline)")
        print(f"    beta_raw 1,984 = {100*1984/n_front:.0f}% of the front-end "
              f"(31 boundaries x 64 tables) — vs c43's 62%")

    print("\n=== the two controlled comparisons ===")
    for nm, mm, ss, nn, note in (
            ("exp_c37 (IDENTICAL shape & params, stock table init)",
             2531.1, 1266.1, 3, "same 64 detectors, same 28,992 params"),
            ("exp_c43 (HALF the detectors, 1 det x 64 bkt)",
             1177.2, 506.4, 3, "32 detectors vs c44's 64"),
            ("exp_c18 hyperplane baseline", BASE_M, BASE_SD, BASE_N, "")):
        d, se, t = welch(m, sd, len(vals), mm, ss, nn)
        print(f"  vs {nm}")
        print(f"     {mm:.1f} +/- {ss:.1f} (n={nn}){('  — ' + note) if note else ''}")
        print(f"     delta {d:+.1f}   Welch se {se:.1f}   |t| {t:.2f}")

    print("\n=== return vs TOTAL LIF detectors, with c44 added ===")
    runs = ALL_RUNS + [("c44  32bkt x 64tab", 64, 1, m, sd, len(vals))]
    runs.sort(key=lambda r: (r[1] * r[2], r[3]))
    print(f"  {'config':<22}{'tables':>8}{'det/tab':>9}{'DETECTORS':>11}{'mean':>10}")
    for name, nt, nd, mm, ss, nn in runs:
        star = "  <-" if name.startswith("c44") else ""
        print(f"  {name:<22}{nt:>8}{nd:>9}{nt*nd:>11}{mm:>10.1f}{star}")
    xs = [r[1] * r[2] for r in runs]
    ys = [r[3] for r in runs]
    rho_det = spearman(xs, ys)
    rho_tab = spearman([r[1] for r in runs], ys)
    print(f"\n  Spearman rho(total detectors, mean) = {rho_det:+.3f}  "
          f"({len(runs)} configurations, was +0.822 over 8 before c44)")
    print(f"  Spearman rho(tables only,     mean) = {rho_tab:+.3f}")
    print("  ORDINAL claim only: n=3 per config, several means overlap in their own noise.")

    json.dump(dict(seeds={s: (rows[s]["cpu_reference_mean"] if rows[s] else None)
                          for s in SEEDS},
                   mean=m, sd=sd, n=len(vals), takeoff=k,
                   actor_params=n_par, frontend_params=n_front,
                   spearman_detectors=rho_det, spearman_tables=rho_tab,
                   n_configs=len(runs)),
              open(os.path.join(HERE, "results.json"), "w"), indent=1)
    print("\nwrote results.json")


if __name__ == "__main__":
    main()
