"""exp_c42b — pool the six confirmation seeds with exp_c42's three, n = 9.

exp_c42 gave 3/3 takeoffs at 4114.2 +/- 158.8. That is a striking result and it is three
seeds: a takeoff rate anywhere from about 0.4 upward produces 3/3 with probability >= 6%,
so the observation is consistent with a configuration that fails half the time. Six more
seeds (3..8, disjoint RNG streams) is the cheapest thing that narrows it.

TWO NUMBERS ARE REPORTED AND THEY ANSWER DIFFERENT QUESTIONS:

  the pooled MEAN   how good the configuration is, compared against the exp_c18 hyperplane
                    baseline (4308.0 +/- 500.1, n=6) with an unpaired Welch se.
  the TAKEOFF RATE  how reliable it is -- the fraction of seeds that reach the baseline band
                    rather than the ~1,000 stand-without-walking plateau. This is the
                    quantity exp_c40 and exp_c41 failed to move and the reason exp_c42 was
                    interesting, so it gets a Wilson interval rather than a bare fraction.

A seed is counted as "took off" at >= 3000, the threshold this chapter has used throughout;
the plateau band is quoted separately so the split is visible rather than asserted.
"""
import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
C42 = os.path.join(HERE, "..", "exp_c42_table_init_std")
NEW_SEEDS = (3, 4, 5, 6, 7, 8)
OLD_SEEDS = (0, 1, 2)
TAKEOFF = 3000.0
BASE_M, BASE_SD, BASE_N = 4308.0, 500.1, 6


def mean_sd(xs):
    n = len(xs)
    if n == 0:
        return float("nan"), float("nan")
    m = sum(xs) / n
    if n < 2:
        return m, float("nan")
    return m, (sum((x - m) ** 2 for x in xs) / (n - 1)) ** 0.5


def wilson(k, n, z=1.96):
    """Wilson score interval -- correct at the extremes, unlike k/n +/- z*sqrt(p q / n),
    which returns a zero-width interval when k == n and is therefore useless here."""
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def load(folder, tag, seeds):
    out = {}
    for s in seeds:
        p = os.path.join(folder, f"mhl_sac_{tag}_s{s}_cpueval.json")
        out[s] = json.load(open(p)) if os.path.exists(p) else None
    return out


def main():
    new = load(HERE, "c42b", NEW_SEEDS)
    old = load(C42, "c42", OLD_SEEDS)

    print("=== exp_c42b — six confirmation seeds of the exp_c42 config ===")
    print(f"  {'seed':>5}{'CPU-ref 100ep':>16}{'ep-sd':>8}{'full':>9}{'vel':>8}"
          f"{'takeoff':>9}")
    rows = []
    for tag, d, seeds in (("c42 ", old, OLD_SEEDS), ("c42b", new, NEW_SEEDS)):
        for s in seeds:
            r = d[s]
            if not r:
                print(f"  {tag}{s:>1}{'—':>16}")
                continue
            ok = r["cpu_reference_mean"] >= TAKEOFF
            rows.append((f"{tag}s{s}", r["cpu_reference_mean"], r["full_length"],
                         r["velocity"], ok))
            print(f"  {tag}{s:>1}{r['cpu_reference_mean']:>16.1f}"
                  f"{r['cpu_reference_std']:>8.1f}{r['full_length']:>6}/100"
                  f"{r['velocity']:>8.3f}{('YES' if ok else 'flat'):>9}")

    new_vals = [r[1] for r in rows if r[0].startswith("c42b")]
    all_vals = [r[1] for r in rows]
    nm, nsd = mean_sd(new_vals)
    pm, psd = mean_sd(all_vals)
    k_new = sum(1 for r in rows if r[0].startswith("c42b") and r[4])
    k_all = sum(1 for r in rows if r[4])

    print(f"\n  six NEW seeds:  {nm:8.1f} +/- {nsd:.1f}   takeoff {k_new}/{len(new_vals)}")
    print(f"  POOLED (n={len(all_vals)}): {pm:8.1f} +/- {psd:.1f}   "
          f"takeoff {k_all}/{len(all_vals)}")
    lo, hi = wilson(k_all, len(all_vals))
    print(f"    takeoff rate {k_all/len(all_vals):.2f}, Wilson 95% CI "
          f"[{lo:.2f}, {hi:.2f}]")

    band = sum(1 for r in rows if abs(r[1] - BASE_M) <= BASE_SD)
    # "Never learned to walk": returns under ~1,500 are a walker that survives some or all
    # of the episode without travelling. The stock-init failures sat near 1,000; these sit
    # near 635, which is worse -- they fall early AND do not travel.
    plateau = sum(1 for r in rows if r[1] < 1500)
    mid = sum(1 for r in rows if 1500 <= r[1] < TAKEOFF)
    print(f"    inside the baseline band ({BASE_M:.0f} +/- {BASE_SD:.0f}): "
          f"{band}/{len(all_vals)}")
    print(f"    never learned to walk (< 1,500): {plateau}/{len(all_vals)}")
    print(f"    intermediate (1,500 - {TAKEOFF:.0f}): {mid}/{len(all_vals)}")

    se = math.sqrt(psd ** 2 / len(all_vals) + BASE_SD ** 2 / BASE_N)
    print(f"\n  vs exp_c18 hyperplane ({BASE_M:.1f} +/- {BASE_SD:.1f}, n={BASE_N})")
    print(f"     delta {pm - BASE_M:+.1f}   unpaired Welch se {se:.1f}   "
          f"|t| {abs(pm - BASE_M)/se:.2f}")
    print(f"  seed sd: c42b pooled {psd:.1f} vs the baseline's {BASE_SD:.1f} "
          f"({BASE_SD/psd:.1f}x tighter)" if psd > 0 else "")

    json.dump(dict(new_seeds={r[0]: r[1] for r in rows if r[0].startswith("c42b")},
                   pooled={r[0]: r[1] for r in rows},
                   new_mean=nm, new_sd=nsd, new_takeoff=k_new,
                   pooled_mean=pm, pooled_sd=psd, pooled_n=len(all_vals),
                   pooled_takeoff=k_all, wilson95=[lo, hi],
                   in_band=band, on_plateau=plateau, intermediate=mid,
                   baseline_delta=pm - BASE_M, welch_se=se,
                   actor_params=28384),
              open(os.path.join(HERE, "results.json"), "w"), indent=1)
    print("\nwrote results.json")


if __name__ == "__main__":
    main()
