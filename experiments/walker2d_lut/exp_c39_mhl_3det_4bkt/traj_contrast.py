"""exp_c39 diagnosis, phase 1 — does ANY logged diagnostic separate the seeds that took
off from the ones that did not?

Six seeds are available, not three: exp_c38 (6 det x 2 bkt) and exp_c39 (3 det x 4 bkt)
share the trainer, the recipe and the diagnostic definitions, and between them they give
THREE takeoffs and THREE flats. Analysing c39's three alone would be one winner against
two losers, which cannot distinguish a predictor from a coincidence. Pooling the two
configurations is legitimate here precisely because the question is about seeds within a
configuration, not about the configurations.

The four logged mechanical diagnostics:
    eff_cells     2**entropy of the per-table cell occupancy -- how many of the 64 rows a
                  table actually uses
    row_coverage  fraction of (table, cell) pairs that have EVER received an update
    nospike       fraction of detectors whose membrane never crossed theta_mem
    digit         mean hard bucket digit (0..M-1); balance of the detectors

For each, this reports the value at several early checkpoints and asks whether takeoff and
flat seeds are separated -- and in WHICH DIRECTION, which is the part that matters. The
standing hypothesis in this chapter has been that failure is addressing collapse: dead
detectors, digits going constant, a table stuck on a couple of rows. If that were right the
flat seeds would show LOWER eff_cells and LOWER coverage than the takeoff seeds.

Usage:
  python traj_contrast.py
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "..")
C38 = os.path.join(D, "exp_c38_mhl_6det_2bkt")

# (label, dir, file-stem, seed, final CPU reference, took off?)
SEEDS = [
    ("c38 s0", C38, "mhl_sac_c38_s0", 0, 1452.1, False),
    ("c38 s1", C38, "mhl_sac_c38_s1", 1, 4117.4, True),
    ("c38 s2", C38, "mhl_sac_c38_s2", 2, 4072.3, True),
    ("c39 s0", HERE, "mhl_sac_c39_s0", 0, 890.8, False),
    ("c39 s1", HERE, "mhl_sac_c39_s1", 1, 982.3, False),
    ("c39 s2", HERE, "mhl_sac_c39_s2", 2, 4217.3, True),
]
METRICS = ["eff_cells", "row_coverage", "nospike", "digit", "mjx_return"]
PROBES = [1000, 2000, 3000, 4000, 5000, 10000]


def at(hist, it, key):
    for e in hist:
        if e["iter"] == it:
            return e.get(key)
    return None


def main():
    hs = {}
    for lab, d, stem, s, ret, ok in SEEDS:
        hs[lab] = json.load(open(os.path.join(d, stem + ".json")))["history"]

    out = {}
    print("=== exp_c38 + exp_c39: 3 takeoff seeds vs 3 flat seeds ===")
    print("  Both configurations share the trainer, recipe and diagnostic definitions.")
    print("  `digit` is NOT comparable across configs (0..1 at 2 buckets, 0..3 at 4), so")
    print("  it is reported but never pooled.\n")

    for m in METRICS:
        print(f"--- {m} ---")
        hdr = "  " + f"{'seed':<9}{'takeoff':>9}" + "".join(f"{p:>9}" for p in PROBES)
        print(hdr)
        rows = {}
        for lab, d, stem, s, ret, ok in SEEDS:
            vals = [at(hs[lab], p, m) for p in PROBES]
            rows[lab] = vals
            cells = "".join(f"{v:>9.2f}" if v is not None else f"{'—':>9}"
                            for v in vals)
            print(f"  {lab:<9}{('YES' if ok else 'flat'):>9}{cells}")

        # separation: at each probe, does the takeoff group sit entirely above or entirely
        # below the flat group? A predictor must separate with NO overlap.
        took = [s[0] for s in SEEDS if s[5]]
        flat = [s[0] for s in SEEDS if not s[5]]
        print(f"  {'':<9}{'separation':>9}", end="")
        sep_line = {}
        for i, p in enumerate(PROBES):
            tk = [rows[l][i] for l in took if rows[l][i] is not None]
            fl = [rows[l][i] for l in flat if rows[l][i] is not None]
            if len(tk) < 2 or len(fl) < 2:
                tag = "?"
            elif min(tk) > max(fl):
                tag = "UP"          # takeoff strictly higher
            elif max(tk) < min(fl):
                tag = "DOWN"        # takeoff strictly lower
            else:
                tag = "overlap"
            sep_line[p] = tag
            print(f"{tag:>9}", end="")
        print("\n")
        out[m] = dict(rows=rows, separation=sep_line)

    json.dump(dict(probes=PROBES, metrics=out,
                   seeds=[dict(label=l, final=r, takeoff=o)
                          for l, _, _, _, r, o in SEEDS]),
              open(os.path.join(HERE, "traj_contrast.json"), "w"), indent=1)
    print("wrote traj_contrast.json")


if __name__ == "__main__":
    main()
