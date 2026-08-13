"""exp_c27 — did the soft penalty actually put the raw outputs ON the grid?

For a soft-penalty arm the interesting question is not just "what does it score" but
"did the property emerge". So each actor is scored TWICE on the same deterministic
100-episode CPU reference —

  raw      the continuous tanh output, exactly as trained
  snapped  the same actor with its output rounded to the K-level grid at every step

— and the gap between them is the real measurement. If the penalty worked, the raw
outputs already sit on the grid, snapping is close to a no-op, and the two numbers
agree. If it did not, snapping costs what post-hoc rounding always costs.

Alongside that, the RESIDUAL |a - nearest_level| is reported from the raw rollout:
mean, 95th percentile, per joint, plus how much of the action mass lands within 1% of
a level. A residual near zero is the property; a residual near the quantizer's own
worst case (half a step = 0.25 at K=5) means the penalty did nothing.

Writes <actor>_soft_eval.json.

Usage:
  python eval_soft.py <actor.npz> [--episodes 100] [--levels K]
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "..")
for p in ("exp_c02_mjx_scaffold", "exp_c06_jax_backprop", "exp_c07_robustness",
          "exp_c11_lut_sac_2x2", "exp_c09_lut_sac", "exp_c26_action_quant"):
    sys.path.insert(0, os.path.join(D, p))

import eval_cpu                                            # noqa: E402
import perturb                                             # noqa: E402
import action_quant as AQ                                  # noqa: E402

ACT = 6


def quantize(a, K):
    return np.rint((a + 1.0) * 0.5 * (K - 1)) / (K - 1) * 2.0 - 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("actor")
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--levels", type=int, default=None)
    a = ap.parse_args()
    path = a.actor if os.path.isabs(a.actor) else os.path.join(HERE, a.actor)
    z = np.load(path)
    K = a.levels or (int(z["penalty_levels"]) if "penalty_levels" in z.files else 5)
    lam = float(z["quant_penalty"]) if "quant_penalty" in z.files else float("nan")

    fn, n = eval_cpu.load_actor(path, forward_mode="hard")
    m = perturb.make_model(None, 1.0)

    raw = AQ.rollout(m, fn, episodes=a.episodes, record=True)
    A = raw.pop("actions")
    snap = AQ.rollout(m, lambda o: quantize(np.asarray(fn(o), np.float64), K),
                      episodes=a.episodes)

    resid = np.abs(A - quantize(A, K))
    step = 2.0 / (K - 1)
    per_joint = dict(mean=resid.mean(0).tolist(), p95=np.percentile(resid, 95, axis=0).tolist())
    on_grid = float((resid < 0.01).mean())

    print(f"{os.path.basename(path)} ({n:,} params, lambda={lam:g}, K={K})", flush=True)
    print(f"  raw     {raw['mean']:8.1f} +/- {raw['sd']:7.1f}  full {raw['full']:>3}/"
          f"{a.episodes}  vel {raw['vel']:.3f}", flush=True)
    print(f"  snapped {snap['mean']:8.1f} +/- {snap['sd']:7.1f}  full {snap['full']:>3}/"
          f"{a.episodes}  vel {snap['vel']:.3f}   (snap cost "
          f"{snap['mean'] - raw['mean']:+.1f})", flush=True)
    print(f"  residual |a - nearest level|: mean {resid.mean():.4f}  p95 "
          f"{np.percentile(resid, 95):.4f}   (grid step {step:.3f}, worst case "
          f"{step/2:.3f}); {on_grid*100:.1f}% of action values within 0.01 of a level",
          flush=True)
    print("  per joint mean/p95: " + "  ".join(
        f"j{j} {per_joint['mean'][j]:.3f}/{per_joint['p95'][j]:.3f}"
        for j in range(ACT)), flush=True)

    hist = np.histogram(A.ravel(), bins=np.linspace(-1, 1, 81))[0]
    levels = np.linspace(-1, 1, K)
    near = [float(((np.abs(A - lv) < step * 0.1).mean())) for lv in levels]
    print("  mass within 10% of a step of each level: " + "  ".join(
        f"{lv:+.2f}:{p*100:5.1f}%" for lv, p in zip(levels, near)), flush=True)

    json.dump(dict(actor=os.path.basename(path), params=n, quant_penalty=lam,
                   levels=K, episodes=a.episodes, raw=raw, snapped=snap,
                   residual_mean=float(resid.mean()),
                   residual_p95=float(np.percentile(resid, 95)),
                   residual_per_joint=per_joint, frac_within_0p01=on_grid,
                   mass_near_levels=near, level_values=levels.tolist(),
                   hist_counts=hist.tolist(), grid_step=step),
              open(path.replace("_actor.npz", "_soft_eval.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
