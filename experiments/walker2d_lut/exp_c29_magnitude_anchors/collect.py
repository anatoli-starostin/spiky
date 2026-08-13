"""exp_c29 — collect the CPU-reference results and the constant-usage instrumentation.

One table per wave, plus the none-vs-grid contrast that the experiment exists to
measure. The contrast is reported as a per-seed PAIRED difference as well as a
difference of means: the two arms share a seed, and at n=3 against a seed spread of
several hundred the paired view is the only one with any power. It is also reported
against the run-to-run noise floor this chapter established (exp_c16: ~663 without
determinism; exp_c17 killed it, so seed spread is now the only noise), and against
exp_c13's 3-seed mean for this cell.

Usage:
  python collect.py [--waves c29 c29c c29m]
"""
import argparse
import glob
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
WAVES = {"c29": "wave 1 · balanced · nap6/tph64",
         "c29c": "wave 2 · canonical_full_coverage · nap6/tph64",
         "c29m": "wave 3 · balanced · nap5/tph128",
         "c29mc": "wave 4 · canonical_full_coverage · nap5/tph128"}
ARMS = ("none", "grid")
EXTRA = ("random", "clumped")
SEEDS = (0, 1, 2)


def load(tag, arm, seed):
    p = os.path.join(HERE, f"lut_sac_{tag}_{arm}_s{seed}_cpueval.json")
    if not os.path.exists(p):
        return None
    d = json.load(open(p))
    b = os.path.join(HERE, f"lut_sac_{tag}_{arm}_s{seed}_bitusage.json")
    if os.path.exists(b):
        d["bits"] = json.load(open(b))
    return d


def mean_sd(xs):
    n = len(xs)
    if n == 0:
        return float("nan"), float("nan")
    m = sum(xs) / n
    if n < 2:
        return m, float("nan")
    return m, (sum((x - m) ** 2 for x in xs) / (n - 1)) ** 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--waves", nargs="+", default=list(WAVES))
    a = ap.parse_args()

    out = {}
    for tag in a.waves:
        rows = {arm: {s: load(tag, arm, s) for s in SEEDS} for arm in ARMS}
        if not any(v for d in rows.values() for v in d.values()):
            continue
        print(f"\n=== {WAVES.get(tag, tag)} ===")
        print(f"  {'arm':<8}{'seed':>5}{'CPU-ref 100ep':>16}{'ep-sd':>8}{'full':>7}"
              f"{'vel':>7}{'rows/tbl':>10}{'deadTh':>8}{'H(oc)':>7}{'H(oo)':>7}")
        for arm in ARMS:
            for s in SEEDS:
                d = rows[arm][s]
                if not d:
                    print(f"  {arm:<8}{s:>5}{'—':>16}")
                    continue
                b = d.get("bits", {})
                k = b.get("by_kind", {})
                oc, oo = k.get("obs-const"), k.get("obs-obs")
                c_dead = "{}/{}".format(oc["dead"], oc["n"]) if oc else "—"
                c_hoc = "{:.3f}".format(oc["mean_H"]) if oc else "—"
                c_hoo = "{:.3f}".format(oo["mean_H"]) if oo else "—"
                print(f"  {arm:<8}{s:>5}{d['cpu_reference_mean']:>16.1f}"
                      f"{d['cpu_reference_std']:>8.1f}{d['full_length']:>5}/100"
                      f"{d['velocity']:>7.3f}"
                      f"{b.get('rows_reached_mean', float('nan')):>10.1f}"
                      f"{c_dead:>8}{c_hoc:>7}{c_hoo:>7}")
        # arm means and the paired contrast
        vals = {arm: [rows[arm][s]["cpu_reference_mean"] for s in SEEDS
                      if rows[arm][s]] for arm in ARMS}
        stat = {}
        for arm in ARMS:
            m, sd = mean_sd(vals[arm])
            stat[arm] = dict(n=len(vals[arm]), mean=m, sd=sd)
            print(f"  {arm:<8} mean over {len(vals[arm])} seeds: {m:8.1f} "
                  f"+/- {sd:.1f}")
        paired = [(s, rows['grid'][s]['cpu_reference_mean']
                   - rows['none'][s]['cpu_reference_mean'])
                  for s in SEEDS if rows['none'][s] and rows['grid'][s]]
        if paired:
            dm, dsd = mean_sd([d for _, d in paired])
            print(f"  grid - none, PAIRED per seed: "
                  + ", ".join(f"s{s} {d:+.1f}" for s, d in paired))
            print(f"  paired mean {dm:+.1f} +/- {dsd:.1f} "
                  f"(se {dsd / max(len(paired), 1) ** 0.5:.1f}); "
                  f"{sum(1 for _, d in paired if d > 0)}/{len(paired)} seeds favour grid")
            stat["contrast"] = dict(paired=[d for _, d in paired], mean=dm, sd=dsd)
        out[tag] = dict(label=WAVES.get(tag, tag), arms=stat,
                        per_seed={arm: {s: (rows[arm][s]["cpu_reference_mean"]
                                            if rows[arm][s] else None)
                                        for s in SEEDS} for arm in ARMS})

    sal = {}
    for arm in EXTRA:
        d = load("c29", arm, 0)
        if d:
            sal[arm] = d["cpu_reference_mean"]
    if sal:
        print("\n=== dropped arms, wave-1 seed 0 only (compute already spent) ===")
        for arm, v in sal.items():
            print(f"  {arm:<8}{v:>10.1f}")
        out["salvaged_wave1_seed0"] = sal

    json.dump(out, open(os.path.join(HERE, "results.json"), "w"), indent=1)
    print("\nwrote results.json")


if __name__ == "__main__":
    main()
