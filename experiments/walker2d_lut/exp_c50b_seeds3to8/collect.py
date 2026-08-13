"""exp_c50 — collect the 3 seed results AND the end-of-training delay distributions.

The second half is the point of the experiment. c49's autopsy found 94.6-94.9% of its 2,176
delays sitting at or below `clamp(delay, 0, t_window)`'s floor, where the forward returns 0
and the gradient is exactly 0 -- dead, permanently. This run removed that floor. Whether the
delays actually SPREAD is a separate question from whether the return recovers, and both are
reported: a run that recovered the return without moving the delays would mean the clamp was
never the mechanism, and a run that spread the delays without recovering the return would
mean the delay parameterisation is not where the missing 2,000 points live.

WHAT IS COUNTED, and why each number rather than a summary statistic:

  range          min..max of the learned delay. c36 (unclamped, old module) spans
                 -10.08..+12.67. c49 (clamped) spans -0.006..+6.7/11.3/10.1 -- the -0.006 is
                 float noise around a floor, not a negative delay.
  % negative     the number c49 could not produce at all (0%) and c36 sat at ~40%. Under
                 this run's forward a negative delay is meaningful: that synapse arrives
                 EARLIER than its latency code, and since the timeline origin is arbitrary
                 a global shift renormalises the minimum to 0 while preserving every
                 relative arrival. So this is capacity regained, not a pathology.
  % dead-at-floor  |delay| <= 1e-6, i.e. entries that never left the initialisation. With
                 delay_init_std=0 EVERY delay starts exactly at 0, so this is the direct
                 read of how many never moved. Under c49's clamp this was 94.6-94.9%.
  % at the cap   delay >= t_window - 1e-6. The upper bound is RETAINED (float32 safety in
                 the reference's cumsum membrane), so it can still trap. If this is
                 non-trivial the experiment has replaced one trap with another and the
                 result must be read with that in mind.

Run in the MJX venv (numpy only, no jax needed).

Usage:
  python collect.py
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
C50 = os.path.join(HERE, "..", "exp_c50_no_delay_clamp")
SEEDS = (3, 4, 5, 6, 7, 8)
PRIOR = (0, 1, 2)           # exp_c50's original three, pooled in below
TAG = "c50"
T_WINDOW = 32.0
TAKEOFF = 3000.0            # the chapter's threshold, unchanged since c18
EPS = 1e-6


def delay_stats(seed, base=None):
    z = np.load(os.path.join(base or HERE, f"mhl_sac_{TAG}_s{seed}_actor.npz"))
    d = np.asarray(z["delay"]).ravel().astype(np.float64)
    # `delay_sd`, not `sd`: the per-seed CPU-eval std is merged into this same dict below
    # and would otherwise silently overwrite it -- two different quantities, one key.
    return dict(n=int(d.size),
                min=float(d.min()), max=float(d.max()),
                mean=float(d.mean()), delay_sd=float(d.std()),
                pct_negative=float(100.0 * (d < 0).mean()),
                pct_dead_at_zero=float(100.0 * (np.abs(d) <= EPS).mean()),
                pct_at_cap=float(100.0 * (d >= T_WINDOW - EPS).mean()),
                pct_below_minus_one=float(100.0 * (d < -1.0).mean()))


def main():
    seeds, dly = {}, {}
    for s in SEEDS:
        p = os.path.join(HERE, f"mhl_sac_{TAG}_s{s}_cpueval.json")
        if not os.path.exists(p):
            print(f"  seed {s}: no CPU eval yet — skipping")
            continue
        j = json.load(open(p))
        seeds[str(s)] = j["cpu_reference_mean"]
        dly[str(s)] = delay_stats(s)
        dly[str(s)].update(velocity=j["velocity"], length_mean=j["length_mean"],
                           full_length=j["full_length"],
                           return_sd=j["cpu_reference_std"])

    vals = list(seeds.values())
    m = sum(vals) / len(vals)
    sd = (sum((v - m) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5 \
        if len(vals) > 1 else float("nan")
    out = dict(seeds=seeds, mean=m, sd=sd, n=len(vals),
               takeoff=sum(1 for v in vals if v >= TAKEOFF),
               actor_params=31360, frontend_params=6784, delays=dly)
    json.dump(out, open(os.path.join(HERE, "results.json"), "w"), indent=1)

    print(f"\nexp_{TAG}: {m:.1f} +/- {sd:.1f} over {len(vals)} seeds, "
          f"takeoff {out['takeoff']}/{len(vals)}")
    print(f"  {'seed':>4} {'return':>9} {'+/-':>7} {'vel':>6} {'len':>7} {'full':>5}")
    for s in sorted(seeds):
        d = dly[s]
        print(f"  {s:>4} {seeds[s]:9.1f} {d['return_sd']:7.1f} {d['velocity']:6.3f} "
              f"{d['length_mean']:7.1f} {d['full_length']:5d}")
    print(f"\n  DELAY DISTRIBUTION at end of training ({dly[sorted(dly)[0]]['n']} delays "
          f"per seed, all starting at exactly 0.0)")
    print(f"  {'seed':>4} {'min':>8} {'max':>8} {'mean':>8} {'sd':>7} "
          f"{'%neg':>7} {'%<-1':>7} {'%dead@0':>8} {'%at cap':>8}")
    for s in sorted(dly):
        d = dly[s]
        print(f"  {s:>4} {d['min']:8.3f} {d['max']:8.3f} {d['mean']:8.3f} "
              f"{d['delay_sd']:7.3f} {d['pct_negative']:7.1f} {d['pct_below_minus_one']:7.1f} "
              f"{d['pct_dead_at_zero']:8.1f} {d['pct_at_cap']:8.2f}")
    print("\n  for comparison: c49 (clamped) 0.0% negative, 94.6-94.9% dead at the floor; "
          "\n                  c36 (unclamped old module) ~40% negative, span -10.08..+12.67")

    # ---- POOLED n=9: these six plus exp_c50's original three -------------------------
    # Same configuration, same code, different seeds -- so they pool. Reported separately
    # from the six-seed mean because the six on their own are what says whether the new
    # draws behave like the old ones, and the pool is what answers the question.
    prior = {}
    for s in PRIOR:
        f = os.path.join(C50, f"mhl_sac_{TAG}_s{s}_cpueval.json")
        if os.path.exists(f):
            prior[str(s)] = json.load(open(f))["cpu_reference_mean"]
    if prior:
        pool = list(seeds.values()) + list(prior.values())
        pm = sum(pool) / len(pool)
        psd = (sum((v - pm) ** 2 for v in pool) / (len(pool) - 1)) ** 0.5
        ptk = sum(1 for v in pool if v >= TAKEOFF)
        out.update(pooled_mean=pm, pooled_sd=psd, pooled_n=len(pool),
                   pooled_takeoff=ptk, prior_seeds=prior)
        json.dump(out, open(os.path.join(HERE, "results.json"), "w"), indent=1)
        print(f"\n  exp_c50 seeds 0/1/2 (already banked): "
              f"{', '.join(f'{k}={v:.1f}' for k, v in sorted(prior.items()))}")
        print(f"  POOLED n={len(pool)}: {pm:.1f} +/- {psd:.1f}, "
              f"takeoff {ptk}/{len(pool)}")
        # Welch against the two reference points this chapter reads everything against.
        for nm, m2, sd2, n2 in (("c36", 4246.1, 298.4, 3), ("c49", 2232.9, 1259.1, 3)):
            se = (psd ** 2 / len(pool) + sd2 ** 2 / n2) ** 0.5
            print(f"    vs {nm} ({m2:.1f} +/- {sd2:.1f}, n={n2}): "
                  f"{pm - m2:+.1f}, Welch se {se:.1f}, |t| {abs(pm - m2)/se:.2f}")
    print(f"\nwrote {os.path.join(HERE, 'results.json')}")


if __name__ == "__main__":
    main()
