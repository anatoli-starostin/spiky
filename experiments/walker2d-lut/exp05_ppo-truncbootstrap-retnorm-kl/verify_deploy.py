"""End-to-end check of the DEPLOYABLE artifacts, in the server's own environment.

The MLP analogue of exp19's verify_deploy.py. This does not test our training code — it
tests exactly what nucstar will ship:

  1. builds a throwaway package that mirrors the server layout
     (actors/base.py + actors/mlp_ppo.py + models/<npz>),
  2. imports the actor the same way `server/actors/__init__.py` auto-discovers it,
  3. checks the Actor contract: act(obs) -> (6,) float32 within [-1, 1],
  4. runs N full episodes of **gymnasium Walker2d-v5** — the exact env the server steps —
     and reports the return.

If this prints a healthy return, the artifact works on the server. If it fails here, it
would have failed there.

Usage:  python verify_deploy.py [--episodes 30]
"""
import argparse
import os
import shutil
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DEPLOY = os.path.join(HERE, "deploy")
VIZ = os.path.join(HERE, "..", "..", "..", "landing", "walker2d-viz", "server")
NPZ = "walker2d_mlp_ppo_exp05.npz"


def build_fake_server(tmp):
    """Mirror the server's actors/ + models/ layout so relative imports and paths work."""
    pkg = os.path.join(tmp, "actors")
    os.makedirs(pkg)
    os.makedirs(os.path.join(tmp, "models"))
    shutil.copy(os.path.join(VIZ, "actors", "base.py"), os.path.join(pkg, "base.py"))
    shutil.copy(os.path.join(DEPLOY, "mlp_ppo.py"), os.path.join(pkg, "mlp_ppo.py"))
    shutil.copy(os.path.join(DEPLOY, NPZ), os.path.join(tmp, "models", NPZ))
    open(os.path.join(pkg, "__init__.py"), "w").close()
    return pkg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=30)
    a = ap.parse_args()

    tmp = tempfile.mkdtemp(prefix="viz_verify_")
    build_fake_server(tmp)
    sys.path.insert(0, tmp)

    import gymnasium as gym
    from actors.mlp_ppo import MLPPPOActor                      # noqa: E402

    env = gym.make("Walker2d-v5")
    print(f"env {env.spec.id}  obs {env.observation_space.shape}  "
          f"act {env.action_space.shape} in [{env.action_space.low[0]}, "
          f"{env.action_space.high[0]}]")

    actor = MLPPPOActor(env.action_space)
    print(f"actor name: {actor.name!r}   obs {actor.n_obs}   act {actor.n_act}   "
          f"hidden {actor.W[0].shape[1]}/{actor.W[1].shape[1]}")

    # ---- contract checks ---------------------------------------------------
    obs, _ = env.reset(seed=0)
    act = actor.act(obs)
    ok = True

    def chk(name, cond, detail=""):
        nonlocal ok
        print(f"  [{'OK ' if cond else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""))
        ok = ok and cond

    chk("returns a numpy array", isinstance(act, np.ndarray), type(act).__name__)
    chk("shape == action_space.shape", act.shape == env.action_space.shape, str(act.shape))
    chk("dtype float32", act.dtype == np.float32, str(act.dtype))
    chk("inside [-1, 1]", bool(np.all(np.abs(act) <= 1.0)),
        f"max|a| {np.abs(act).max():.4f}")
    chk("finite", bool(np.all(np.isfinite(act))))
    chk("deterministic (same obs -> same action)",
        bool(np.array_equal(act, actor.act(obs))))

    # ---- real rollouts -----------------------------------------------------
    rets, lens = [], []
    for ep in range(a.episodes):
        obs, _ = env.reset(seed=1000 + ep)
        total, steps, done = 0.0, 0, False
        while not done:
            obs, r, term, trunc, _ = env.step(actor.act(obs))
            total += r
            steps += 1
            done = term or trunc
        rets.append(total)
        lens.append(steps)
    rets, lens = np.array(rets), np.array(lens)
    print(f"\n{a.episodes} episodes of gymnasium Walker2d-v5 (the server's own env):")
    print(f"  return  mean {rets.mean():8.1f}  std {rets.std():7.1f}  "
          f"min {rets.min():8.1f}  max {rets.max():8.1f}")
    print(f"  length  mean {lens.mean():8.1f}  full-1000 episodes {int((lens >= 1000).sum())}"
          f"/{a.episodes}")
    chk("walks (mean return > 1000)", rets.mean() > 1000, f"{rets.mean():.1f}")

    shutil.rmtree(tmp, ignore_errors=True)
    print(f"\n{'ARTIFACT VERIFIED — ready to deploy' if ok else 'VERIFICATION FAILED'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
