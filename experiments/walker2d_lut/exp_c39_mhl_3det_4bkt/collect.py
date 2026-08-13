"""exp_c39 — collect the CPU-reference results for the MHL-LIF actor.

THE COMPARISON THAT MATTERS IS exp_c38, and together with exp_c31 the three of them now
form the tightest controlled triple in this chapter. All three have:

    32 tables · 64 rows per table · a 24,576-entry table · 12 outputs per row

with identical SAC recipe, critic, replay, trust region, learning rates and eval protocol.
They differ ONLY in how those 64 rows are addressed:

    c31  ONE LIF per table. Its single first-spike time t* against six learned deadlines
         L_k, bits 1[t* < L_k]. Six views of ONE scalar -- the bits cannot be independent,
         and the 64 rows are really a thermometer code on one number.  2951 +/- 2109
    c38  SIX LIFs per table, each with its own 17 delays, 17 synapses and tau, each
         against its OWN boundary. Digit COUNT 6, digit WIDTH 2.       3214 +/- 1526
    c39  THREE LIFs per table, each quantising its own spike time into 4 ORDERED buckets.
         Digit COUNT 3, digit WIDTH 4.                                 <- this run

So c38-vs-c39 is a clean width-against-count trade at fixed capacity, and c39 does it on
HALF the front-end (3,808 vs 7,168) because each detector carries its own 17 delays and 17
synapses. Three readings make different predictions: if raw addressing capacity is what
matters the two tie; if the number of INDEPENDENT scalars matters, c39 falls back toward
c31; if the ORDERED structure within a digit matters -- bucket indices are monotone in
spike time, bits are not -- c39 beats c38 with fewer parameters.

PARAMETERS. At 28,384 total this is the closest parameter match to the 28,032 hyperplane
baseline of ANY model in the chapter (101.3%), and it arrives there without being tuned
for it.

THE OTHER ANCHOR IS exp_c36, the only bucket configuration to match the baseline
(4246.1 +/- 298.4). c36 bought capacity by adding TABLES -- 128 of them, each a separate
summand. c38 and c39 buy it by adding DETECTORS INSIDE a table, which multiplies the row
count and adds NO summand. c38's result already favoured the "summed" reading: it broke
the addressing-entropy plateau (1.7-2.5 -> 7.6-10.8, the first configuration ever to do
so) and still did not separate from c31 in return.

All comparisons are UNPAIRED -- no shared seeds -- so Welch standard errors.
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SEEDS = (0, 1, 2)
BASELINE_TOTAL, BASELINE_FRONT = 28032, 3456
# (name, mean, sd, n_seeds, total params, front-end params)
ANCHORS = [("exp_c18 hyperplane", 4308.0, 500.1, 6, 28032, 3456),
           ("exp_c38 mhl 6det x 2bkt x 32tab", 3213.9, 1525.9, 3, 31744, 7168),
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
        p = os.path.join(HERE, f"mhl_sac_c39_s{s}_cpueval.json")
        rows[s] = json.load(open(p)) if os.path.exists(p) else None

    print("=== exp_c39 — LIFMultiHeadLUT actor, 32 tables x 3 detectors x 4 buckets, "
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
              f"(4**3 = 64 rows, the chapter's standard height)")

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

    print("\n  NOTE: both anchors in the controlled triple are BIMODAL -- c31 (4262, 4073,"
          "\n  518) and c38 (4117, 4072, 1452) each put two seeds in or near the baseline"
          "\n  band and one far below. Comparing means against a bimodal anchor is weak in"
          "\n  both directions; the per-seed values and the count of seeds clearing 3,000"
          "\n  are the honest read, and whether the 1-in-3 collapse recurs is a separate"
          "\n  question from where the mean lands.")

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
