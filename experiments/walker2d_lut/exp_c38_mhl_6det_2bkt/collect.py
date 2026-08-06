"""exp_c38 — collect the CPU-reference results for the MHL-LIF actor.

THE COMPARISON THAT MATTERS IS exp_c31, and it is the cleanest controlled comparison in
this whole chapter. Both models have:

    32 tables · 64 rows per table · a 24,576-entry table · 12 outputs per row

and their totals differ by 1.1% (31,744 vs 31,392). The SAC recipe, critic, replay,
trust region, learning rates and eval protocol are identical. The one thing that differs
is HOW THE SIX BITS ARE PRODUCED:

    c31  ONE LIF neuron per table. Its single first-spike time t* is compared against six
         learned deadlines L_k, giving bits 1[t* < L_k]. Six views of ONE scalar -- the
         bits cannot be independent, and the 64 rows they address are really a
         thermometer code on one number.
    c38  SIX LIF neurons per table, each with its own 17 delays, its own 17 synapses and
         its own tau, each compared against its OWN learned boundary. Six genuinely
         independent scalars, packed mixed-radix into the same 64 rows.

So this experiment isolates one variable -- the independence of the address bits -- at
matched capacity. c31 scored 2951.2 +/- 2109.2, and that mean is bimodal (4262, 4073,
518): two seeds reached the baseline band and one never took off. If bit independence is
what c31 lacked, c38 should reach the band on all three.

THE SECOND COMPARISON IS exp_c36, which is within 1.2% on parameters (31,360) and scored
4246.1 +/- 298.4 -- the only bucket configuration to match the hyperplane baseline. c36
bought its capacity by adding TABLES (128 of them, each a separate summand). c38 buys it
by adding DETECTORS INSIDE a table (which multiplies the row count and adds NO summand).
The c36 finding was that per-table addressing entropy predicts nothing and the number of
independent indices SUMMED predicts everything; c38 is the test of whether "summed" was
load-bearing in that sentence.

All comparisons are UNPAIRED -- no shared seeds -- so Welch standard errors.
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SEEDS = (0, 1, 2)
BASELINE_TOTAL, BASELINE_FRONT = 28032, 3456
# (name, mean, sd, n_seeds, total params, front-end params)
ANCHORS = [("exp_c18 hyperplane", 4308.0, 500.1, 6, 28032, 3456),
           ("exp_c31 PureLIF 6 deadlines x 32 tab", 2951.2, 2109.2, 3, 31392, 6816),
           ("exp_c36 bucket 16bkt x 128tab", 4246.1, 298.4, 3, 31360, 6784),
           ("exp_c37 bucket 32bkt x 64tab", 2531.1, 1266.1, 3, 28992, 4416),
           ("exp_c32b bucket 16bkt x 32tab", 2041.2, 1230.1, 3, 7840, 1696),
           ("exp_c33 bucket 64bkt x 32tab", 1536.2, 1416.8, 3, 27808, 3232),
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
        p = os.path.join(HERE, f"mhl_sac_c38_s{s}_cpueval.json")
        rows[s] = json.load(open(p)) if os.path.exists(p) else None

    print("=== exp_c38 — LIFMultiHeadLUT actor, 32 tables x 6 detectors x 2 buckets, "
          "3 seeds ===")
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
    print(f"\n  MHL-LIF mean over {len(vals)} seeds: {m:8.1f} +/- {sd:.1f}")
    if n_par:
        print(f"  actor params {n_par:,} = {n_front:,} front-end + "
              f"{n_par - n_front:,} table  ({2 * 32} of them frozen)")
        print(f"    total     {100 * n_par / BASELINE_TOTAL:5.1f}% of the "
              f"{BASELINE_TOTAL:,} hyperplane baseline")
        print(f"    front-end {100 * n_front / BASELINE_FRONT:5.1f}% of its "
              f"{BASELINE_FRONT:,}")
        print(f"    table     {100 * (n_par-n_front) / 24576:5.1f}% of the usual 24,576 "
              f"(2**6 = 64 rows, the chapter's standard height)")

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

    print("\n  NOTE: exp_c31's 2951.2 +/- 2109.2 is bimodal (4262, 4073, 518) and exp_c38"
          "\n  is its matched control -- same 32 tables, same 64 rows, same 24,576 table,"
          "\n  totals within 1.1%, differing only in whether the six address bits come"
          "\n  from six deadlines on ONE neuron or from SIX independent neurons."
          "\n  Comparing means against a bimodal anchor is weak in both directions; the"
          "\n  per-seed values and the count of seeds clearing 3,000 are the honest read.")

    json.dump(dict(seeds={s: (rows[s]["cpu_reference_mean"] if rows[s] else None)
                          for s in SEEDS},
                   mean=m, sd=sd, n=len(vals), actor_params=n_par,
                   frontend_params=n_front,
                   table_params=(n_par - n_front) if n_par else None,
                   frozen_params=64,
                   baseline_total=BASELINE_TOTAL, baseline_front=BASELINE_FRONT,
                   param_matched=False, anchors=out_anchors),
              open(os.path.join(HERE, "results.json"), "w"), indent=1)
    print("\nwrote results.json")


if __name__ == "__main__":
    main()
