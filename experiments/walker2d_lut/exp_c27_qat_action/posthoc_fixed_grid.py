"""exp_c27 — post-hoc rounding of the c21@10k teacher on the SAME fixed [-1, 1] grid the
QAT arms use.

exp_c26 quantized on each joint's observed min/max range. Those ranges came out within
0.008 of [-1, +1], so the two grids are nearly identical -- but "nearly" is not a basis
for a paired comparison. The QAT arms snap to a fixed [-1, 1] grid, so the post-hoc half
of every pair is recomputed here on exactly that grid. Any difference from the exp_c26
row is then the range rule and nothing else.

Usage:
  python posthoc_fixed_grid.py [--levels 16 15 ...] [--episodes 100]
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

TEACHER = os.path.join(D, "exp_c09_lut_sac",
                       "lut_sac_c21_seed4_20k_at10000_actor.npz")
ON_RECORD = 5286.557120404921


def quantize(a, K):
    return np.rint((a + 1.0) * 0.5 * (K - 1)) / (K - 1) * 2.0 - 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", type=int, nargs="+", default=[16, 15])
    ap.add_argument("--episodes", type=int, default=100)
    a = ap.parse_args()

    fn, n = eval_cpu.load_actor(TEACHER, forward_mode="hard")
    m = perturb.make_model(None, 1.0)
    base = AQ.rollout(m, fn, episodes=a.episodes)
    if abs(base["mean"] - ON_RECORD) > 1.0:
        raise SystemExit(f"teacher reproduced {base['mean']:.1f}, not {ON_RECORD:.1f}")
    print(f"teacher unquantized: {base['mean']:.1f} +/- {base['sd']:.1f}  "
          f"full {base['full']}/{a.episodes}  vel {base['vel']:.3f}", flush=True)

    rows = [dict(K=None, **base)]
    for K in a.levels:
        r = AQ.rollout(m, lambda o, K=K: quantize(np.asarray(fn(o), np.float64), K),
                       episodes=a.episodes)
        r["K"] = K
        r["retention"] = 100.0 * r["mean"] / base["mean"]
        rows.append(r)
        print(f"  K={K:>3} (fixed [-1,1]): {r['mean']:8.1f} +/- {r['sd']:7.1f}  "
              f"full {r['full']:>3}/{a.episodes}  vel {r['vel']:.3f}  "
              f"ret {r['retention']:6.1f}%", flush=True)

    json.dump(dict(teacher=os.path.basename(TEACHER), params=n,
                   episodes=a.episodes, grid="fixed [-1, 1], midtread, K levels",
                   rows=rows),
              open(os.path.join(HERE, "posthoc_fixed_grid.json"), "w"), indent=1)
    print("wrote posthoc_fixed_grid.json", flush=True)


if __name__ == "__main__":
    main()
