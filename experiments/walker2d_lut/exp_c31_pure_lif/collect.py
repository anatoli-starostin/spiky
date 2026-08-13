"""exp_c31 — collect the CPU-reference results for the PureLIF (TTFS) actor.

THREE anchors, because this experiment sits at the end of a line rather than beside a
single control:

  * exp_c18 hyperplane, nap6/tph32, 6 seeds, 4308.0 +/- 500.1, 28,032 params -- the same
    table geometry and SAC recipe with the ORIGINAL index front-end. The standing anchor
    for the whole chapter.
  * exp_c30 dense-P LIF, 3 seeds, 3931.3 +/- 585.8, 87,361 params -- the first LIF
    front-end, with the full ordered-pair channel.
  * exp_c30b factorised-P LIF, 3 seeds, 4086.8 +/- 991.2, 48,193 params -- the same model
    with `P` cut to rank 2 plus a source bias.

All three comparisons are UNPAIRED: the sets share no seeds, so the paired trick that
carried exp_c29 is unavailable. Welch standard errors.

THE BASELINE IS 28,032, NOT 49,152 -- corrected 2026-08-03, and exp_c30/c30b were written
against the wrong figure. 49,152 is exp_c29's TABLE-only count for its nap6/tph64 cells
(tph * 2**nap * 12 = 64 * 64 * 12); exp_c29's own totals were 56,064-70,912. exp_c18 is
nap6/tph32, whose table is 24,576 and whose TOTAL learnable count is 28,032 (24,576 table +
3,456 hyperplane w/b). Every model compared here shares that same 24,576-entry table, so
the only thing that actually differs is the INDEX FRONT-END, and that is what `front_end`
below reports:

    exp_c18 hyperplane   3,456      exp_c30b factorised-P  23,617
    exp_c31 PureLIF      6,816      exp_c30  dense-P       62,785

Read the front-end column, not the total: it is the entire subject of exp_c30/c30b/c31 and
the totals are dominated by a table none of them changed.

Return-per-1k-params is printed as a blunt ratio, NOT as a test -- it inherits both
distributions' noise and there is no sense in which three seeds resolve it.
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SEEDS = (0, 1, 2)
BASELINE_PARAMS = 28032
TABLE_PARAMS = 24576          # shared by every model below; nap6/tph32 -> 32*64*12
# (name, mean, sd, n_seeds, total params, index front-end params)
ANCHORS = [("exp_c18 hyperplane nap6/tph32", 4308.0, 500.1, 6, 28032, 3456),
           ("exp_c30 dense-P LIF", 3931.3, 585.8, 3, 87361, 62785),
           ("exp_c30b factorised-P LIF", 4086.8, 991.2, 3, 48193, 23617)]


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
        p = os.path.join(HERE, f"pure_lif_sac_c31_s{s}_cpueval.json")
        rows[s] = json.load(open(p)) if os.path.exists(p) else None

    print("=== exp_c31 — PureLIF (TTFS) detector actor, nap6/tph32, 3 seeds ===")
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
    front = (n_par - TABLE_PARAMS) if n_par else None
    print(f"\n  PureLIF mean over {len(vals)} seeds: {m:8.1f} +/- {sd:.1f}")
    if n_par:
        print(f"  actor params {n_par:,} total = {front:,} front-end + "
              f"{TABLE_PARAMS:,} table")
        print(f"    total    {100 * n_par / BASELINE_PARAMS:6.1f}% of the "
              f"{BASELINE_PARAMS:,} hyperplane baseline")
        print(f"    FRONT-END{100 * front / 3456:6.1f}% of the 3,456 hyperplane "
              f"front-end  <- the quantity actually under test")

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
            # A blunt ratio, deliberately not dressed up as a test.
            print(f"     front-end size: {front:,} vs {afront:,}  "
                  f"({front/afront:.2f}x)")
        out_anchors.append(dict(name=name, mean=amean, sd=asd, n=an, params=apar,
                                front_end=afront, delta=delta, welch_se=se,
                                front_end_ratio=(front/afront) if n_par else None))

    json.dump(dict(seeds={s: (rows[s]["cpu_reference_mean"] if rows[s] else None)
                          for s in SEEDS},
                   mean=m, sd=sd, n=len(vals), actor_params=n_par,
                   front_end_params=front, table_params=TABLE_PARAMS,
                   baseline_params=BASELINE_PARAMS, param_matched=False,
                   anchors=out_anchors),
              open(os.path.join(HERE, "results.json"), "w"), indent=1)
    print("\nwrote results.json")


if __name__ == "__main__":
    main()
