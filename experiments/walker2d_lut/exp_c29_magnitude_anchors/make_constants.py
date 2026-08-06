"""exp_c29 — derive the two constant sets the A/B arms append to the LUT input (#75).

THE HYPOTHESIS THIS SERVES. Anchor addressing computes bit_i = 1[x[a] - x[b] > 0]: a
PAIRWISE comparison of two observation coordinates. Such a bit is magnitude-blind in a
precise sense -- it is invariant to adding any constant to the whole (normalised)
observation vector, and it can never ask "is the knee velocity above 3 rad/s". A
hyperplane bit, 1[<w, x> + b > 0], can. If that missing faculty is what separates the
two addressing modes, then handing the anchors extra input coordinates that hold FIXED
NUMBERS restores it for free: a pair (obs j, const k) becomes 1[x_j > c_k], a threshold
test, with no new parameters and no new forward.

WHAT SPACE THE CONSTANTS LIVE IN. The addressing sees the STANDARDISED observation
x = (o - mean) / (std + 1e-6), with the SAC teacher's frozen statistics. So a constant
is a threshold in standardised units, and a single shared constant c corresponds to a
DIFFERENT raw threshold in every channel, mean_j + std_j * c. That is a feature, not a
compromise: it is the same per-channel-scaled grid structure exp_c28 validated, with the
scale supplied by the normaliser instead of by a per-channel percentile.

THE TWO SETS.

  grid    16 uniformly spaced levels tiling the MEDIAN PER-CHANNEL 0.5/99.5 percentile
          of the standardised observation stream. This is exp_c28's rule -- 16 levels
          between the 0.5 and the 99.5 percentile -- carried over to a set that has to
          be SHARED by all 17 channels. exp_c28 showed 16 levels on that rule preserve
          the gait to 100.0% of return, so these are literally "quantisation values
          already known to be sufficient".

          MEDIAN OF THE PER-CHANNEL RANGES, NOT THE POOLED RANGE. The first version of
          this file pooled all channels and took the 0.5/99.5 percentile of the union.
          That range is [-9.04, +3.38], and it is that wide because of two heavy-tailed
          channels: r_hip reaches -9.55 and r_knee -8.63 while eight other channels
          never go below -2.6. Sixteen levels tiling the union therefore put seven
          constants in a region only one or two channels ever visit. Measured on a
          trained-briefly policy with bit_usage.py, 66% of the resulting threshold bits
          were DEAD -- permanently stuck, costing an address bit each and halving their
          tables' reachable rows. The augmented arms would have been handicapped by a
          quarter of their capacity, and the experiment would have measured that instead
          of magnitude. exp_c28's rule was PER CHANNEL; the honest shared-set analogue is
          the typical channel's range, which is the median, not the union.

          BIN CENTRES, not endpoint-inclusive linspace. exp_c28 used the levels as
          RECONSTRUCTION values, where landing on the range endpoint is harmless. Here
          they are THRESHOLDS, and a threshold at the top of the range is dead: the
          liveness check below fired on the first version of this file because the
          pooled 99.5 percentile sits exactly on the joint-velocity saturation atom
          (the env clips qvel to +/-10, so >0.5% of samples share that one value), and
          `x > hi` was true for exactly zero samples. Bin centres are the standard
          reconstruction set for a K-level uniform quantiser on a bounded interval and
          keep all 16 strictly interior, which is what a comparator needs.

  random  16 values drawn once from a fixed RNG, uniform over the SAME pooled range,
          then sorted. Same count, same range, same liveness -- so every threshold is
          just as usable a comparator as a grid level. What differs is only that the
          positions are irregular: they clump and leave gaps instead of tiling.

  clumped 16 values drawn over the CENTRAL FIFTH of the range only. Still 16, still all
          live (they sit where the data is densest), but they resolve one narrow band of
          magnitude and leave both tails unresolved.

WHY THERE ARE THREE AND NOT TWO. `random` was specified as the control for `grid`, and
it is the natural one to reach for -- but this script measures the two sets against each
other and they come out only HALF A BIN WIDTH apart on average. That is not a fluke of
the seed: the order statistics of 16 uniform draws are themselves near-evenly spaced, so
"sorted uniform over the range" is a slightly jittered grid, not an alternative to one.
Run as-is, `grid` vs `random` would be a near-null contrast dressed up as a control, and
a null result from it would be uninterpretable. `clumped` is the set that actually
differs while holding the count and the liveness fixed, so it is what can carry a
negative result. `random` is kept and run anyway, because "does jitter matter" is a
cheap, real question once the machinery exists.

The ladder that results:
  none vs {grid, random, clumped}  -- does magnitude access help AT ALL? This is the
                                     hypothesis. A null here kills it outright.
  grid vs clumped                  -- does the SPREAD of the thresholds matter, at
                                     matched count?
  grid vs random                   -- does the regular tiling matter, given the spread?
                                     Expected null; reported for completeness.

POOLED, NOT PER-CHANNEL. The constants are extra input DIMENSIONS, so each one is
visible to every comparator; there is no way to give channel j its own set. Pooling the
percentile over all channels is therefore not a simplification, it is the only
well-posed version of the question. The per-channel spread that pooling hides is
reported in the output so the cost of that is visible rather than assumed.

Writes constants.json. Reads the exp_c09 checkpoint read-only.

Usage:
  python make_constants.py [--levels 16] [--episodes 20] [--seed 12345]
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "..")
for p in ("exp_c02_mjx_scaffold", "exp_c06_jax_backprop", "exp_c07_robustness",
          "exp_c11_lut_sac_2x2", "exp_c09_lut_sac", "exp_c26_action_quant",
          "exp_c28_input_quant"):
    sys.path.insert(0, os.path.join(D, p))

import eval_cpu                                            # noqa: E402
import perturb                                             # noqa: E402
import input_quant as IQ                                   # noqa: E402

TEACHER = os.path.join(D, "exp_c09_lut_sac",
                       "lut_sac_c21_seed4_20k_at10000_actor.npz")
NAMES = IQ.NAMES


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", type=int, default=16)
    ap.add_argument("--episodes", type=int, default=20,
                    help="rollout episodes used to MEASURE the percentile range; the "
                         "range is a distributional summary, so it converges long "
                         "before a return estimate would")
    ap.add_argument("--seed", type=int, default=12345,
                    help="draws the `random` arm's constants, and nothing else")
    a = ap.parse_args()
    K = a.levels

    stats = json.load(open(os.path.join(D, "exp_c03_distillation",
                                        "dataset_stats.json")))
    mean = np.asarray(stats["obs_mean"], np.float64)
    std = np.asarray(stats["obs_std"], np.float64)

    fn, n = eval_cpu.load_actor(TEACHER, forward_mode="hard")
    m = perturb.make_model(None, 1.0)
    O = IQ.rollout_record_obs(m, fn, a.episodes)             # [N, 17] RAW observations
    X = (O - mean) / (std + 1e-6)                            # standardised: what the
                                                             # comparators actually see

    # per-channel percentiles first: they define the range AND the liveness check
    plo = np.percentile(X, 0.5, axis=0)
    phi = np.percentile(X, 99.5, axis=0)
    lo, hi = float(np.median(plo)), float(np.median(phi))
    pooled_lo = float(np.percentile(X, 0.5))
    pooled_hi = float(np.percentile(X, 99.5))
    span = hi - lo
    step = span / K
    rng = np.random.default_rng(a.seed)
    sets = dict(
        grid=lo + (np.arange(K) + 0.5) * step,       # bin centres -- see the header
        random=np.sort(rng.uniform(lo, hi, K)),
        clumped=np.sort(rng.uniform(lo + 0.4 * span, lo + 0.6 * span, K)),
    )

    # --- verification, before any of this reaches a training run ------------
    live = {}
    for nm, v in sets.items():
        # 1. Strictly increasing and inside the range. A duplicate constant would give
        #    two input dimensions that are bit-identical, silently halving the distinct
        #    comparators available at those two slots.
        assert np.all(np.diff(v) > 0), f"{nm} is not strictly increasing"
        assert v[0] >= lo - 1e-9 and v[-1] <= hi + 1e-9, f"{nm} escapes [lo, hi]"
        # 2. Every constant must be LIVE: samples on BOTH sides of it, or that threshold
        #    is a constant bit against every channel and the arm silently loses capacity.
        #    Checked against the real standardised stream, per constant. This is the
        #    check that caught the saturation atom at the top of the range.
        frac = np.array([float((X > c).mean()) for c in v])
        live[nm] = frac
        assert np.all((frac > 1e-6) & (frac < 1 - 1e-6)), \
            f"{nm}: a constant is never straddled by the data: {frac}"
    # 3. At least one pair must be genuinely different, or nothing here is a contrast.
    #    Deliberately NOT asserted per-pair: grid-vs-random being small is the measured
    #    fact that put `clumped` in this file, so asserting it away would hide it.
    sep = {f"{x}|{y}": float(np.abs(sets[x] - sets[y]).mean())
           for x, y in (("grid", "random"), ("grid", "clumped"),
                        ("random", "clumped"))}
    # 0.15 * span is ~2.4 bin widths of mean displacement -- comfortably more than the
    # grid's own resolution, so a set that clears it is asking a different question of
    # the data rather than jittering the same one.
    assert max(sep.values()) > 0.15 * span, \
        f"no pair of constant sets differs by more than 15% of the span ({span:.3f}): {sep}"
    # 4. PER-CHANNEL liveness. Check 2 asks whether a constant is straddled by the
    #    pooled stream, which one heavy-tailed channel can satisfy on its own. What a
    #    comparator actually needs is that the SPECIFIC channel it is wired to crosses
    #    the threshold, and the sampler wires channels to constants at random. So the
    #    quantity that predicts dead bits is the fraction of (channel, constant) pairs
    #    where the constant lies inside that channel's own range. This is the check that
    #    would have caught the pooled-range mistake before a GPU-hour was spent on it.
    inside = {nm: float(((v[None, :] > plo[:, None]) &
                         (v[None, :] < phi[:, None])).mean())
              for nm, v in sets.items()}
    for nm, f in inside.items():
        assert f > 0.40, \
            f"{nm}: only {100*f:.1f}% of (channel, constant) pairs are live; most " \
            f"threshold bits would be permanently stuck"

    print(f"teacher {os.path.basename(TEACHER)} ({n:,} params), {a.episodes} episodes, "
          f"{len(X):,} observation vectors", flush=True)
    print(f"median per-channel 0.5/99.5 percentile: [{lo:.4f}, {hi:.4f}]  "
          f"span {span:.4f}, bin width {step:.4f}", flush=True)
    print(f"  (the POOLED range would have been [{pooled_lo:.4f}, {pooled_hi:.4f}] -- "
          f"{(pooled_hi - pooled_lo) / span:.1f}x wider, and mostly empty; see header)",
          flush=True)
    for nm, v in sets.items():
        print(f"  {nm:<8}" + " ".join(f"{c:+.2f}" for c in v), flush=True)
    print("  pairwise mean |difference|:  "
          + "   ".join(f"{k} {v:.3f} ({100 * v / span:.0f}% of span)"
                       for k, v in sep.items()), flush=True)
    print("  (channel, constant) pairs that are LIVE: "
          + "  ".join(f"{k} {100*v:.1f}%" for k, v in inside.items()), flush=True)
    for nm, frac in live.items():
        print(f"  frac of samples above each constant, {nm:<8}"
              + " ".join(f"{f:.2f}" for f in frac), flush=True)
    print(f"\n  per-channel standardised range (pooling hides this spread):", flush=True)
    print(f"  {'ch':<14}{'p0.5':>9}{'p99.5':>9}", flush=True)
    for j, nm in enumerate(NAMES):
        print(f"  {nm:<14}{plo[j]:9.3f}{phi[j]:9.3f}", flush=True)

    json.dump(dict(levels=K, teacher=os.path.basename(TEACHER),
                   episodes=a.episodes, n_samples=int(len(X)),
                   rule="MEDIAN PER-CHANNEL 0.5/99.5 percentile of the STANDARDISED "
                        "observation stream; grid = K bin centres, random = K uniform "
                        "draws over the whole range, clumped = K uniform draws over the "
                        "central fifth. All sorted.",
                   lo=lo, hi=hi, pooled_lo=pooled_lo, pooled_hi=pooled_hi,
                   bin_width=step, random_seed=a.seed,
                   sets={k: v.tolist() for k, v in sets.items()},
                   frac_above={k: v.tolist() for k, v in live.items()},
                   frac_channel_constant_live=inside,
                   pairwise_mean_abs_diff=sep,
                   channels=NAMES,
                   per_channel_lo=plo.tolist(), per_channel_hi=phi.tolist()),
              open(os.path.join(HERE, "constants.json"), "w"), indent=1)
    print("\nwrote constants.json", flush=True)


if __name__ == "__main__":
    main()
