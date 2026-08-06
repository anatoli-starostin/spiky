"""exp_c27 — deterministic 100-episode CPU eval of a QAT actor, ON ITS GRID.

Same protocol as exp_c09's `eval_cpu.py`: gymnasium's Walker2d-v5 on CPU MuJoCo at the
stock solver, 100 episodes seeded 0..99, deterministic tanh(mu). The one addition is that
the action is snapped to the actor's own K-level grid before it reaches the environment —
that is the deployment condition, and scoring a QAT actor continuously would flatter it.

K is read from the checkpoint (`action_levels`, written by the trainer) and a checkpoint
without it is refused rather than guessed at.

Also checks the quantizer itself before using it, because a silent off-by-one in the grid
would move every number in the study.

Writes <actor>_qat_cpueval.json.

Usage:
  python eval_qat_cpu.py <actor.npz> [--episodes 100] [--levels K]
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
import action_quant as AQ                                  # noqa: E402  (rollout harness)


def quantize(a, K):
    """numpy twin of qat_lut_sac.quantize_action: midtread uniform on [-1, 1]."""
    return np.rint((a + 1.0) * 0.5 * (K - 1)) / (K - 1) * 2.0 - 1.0


def _self_check():
    """The grid must contain the endpoints and exactly K distinct values."""
    for K in (3, 5, 7, 9, 11, 15):
        g = np.unique(quantize(np.linspace(-1, 1, 20001), K))
        assert len(g) == K, f"K={K} produced {len(g)} levels"
        assert abs(g[0] + 1) < 1e-12 and abs(g[-1] - 1) < 1e-12, f"K={K} endpoints {g}"
    assert np.allclose(np.unique(quantize(np.linspace(-1, 1, 999), 3)), [-1, 0, 1])


def main():
    _self_check()
    ap = argparse.ArgumentParser()
    ap.add_argument("actor")
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--levels", type=int, default=None,
                    help="override K (default: read from the checkpoint)")
    a = ap.parse_args()

    path = a.actor if os.path.isabs(a.actor) else os.path.join(HERE, a.actor)
    z = np.load(path)
    if a.levels is None:
        if "action_levels" not in z.files:
            raise SystemExit(
                f"{os.path.basename(path)} has no 'action_levels'. Refusing to guess: a "
                f"QAT actor scored on the wrong grid (or none) is a different policy and "
                f"the number would be meaningless. Pass --levels to override knowingly.")
        K = int(z["action_levels"])
    else:
        K = a.levels

    fn, n = eval_cpu.load_actor(path, forward_mode="hard")
    pol = fn if not K else (lambda o: quantize(np.asarray(fn(o), np.float64), K))
    r = AQ.rollout(perturb.make_model(None, 1.0), pol, episodes=a.episodes)
    print(f"{os.path.basename(path)} ({n:,} params, K={K or 'continuous'}) | "
          f"CPU-reference {a.episodes}-ep deterministic: {r['mean']:.1f} +/- {r['sd']:.1f}"
          f"  full {r['full']}/{a.episodes}  vel {r['vel']:.3f} m/s", flush=True)
    json.dump(dict(actor=os.path.basename(path), params=n, action_levels=K,
                   episodes=a.episodes, forward_mode="hard", **r),
              open(path.replace("_actor.npz", "_qat_cpueval.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
