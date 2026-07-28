"""exp_c05 — re-measure an ES winner in the CPU reference env (#75).

The ES loop optimises an MJX horizon-400 fitness, which is NOT comparable to the
project's headline numbers. This converts any saved mean vector into the one number
the project quotes: deterministic, 100 episodes, gymnasium Walker2d-v5 on CPU MuJoCo.

Usage:
  XLA_PYTHON_CLIENT_PREALLOCATE=false python eval_es_cpu.py es_mlp_openai_mu.npy --policy mlp
"""
import argparse, json, os, sys

import jax.numpy as jnp
import numpy as np
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "exp_c02_mjx_scaffold"))
import mjx_walker2d as W          # noqa: E402
import es_mjx                     # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mu")
    ap.add_argument("--policy", default="mlp", choices=["mlp", "lut"])
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--nap", type=int, default=6)
    ap.add_argument("--tph", type=int, default=16)
    ap.add_argument("--episodes", type=int, default=100)
    a = ap.parse_args()

    st = json.load(open(os.path.join(HERE, "..", "exp_c03_distillation",
                                     "dataset_stats.json")))
    norm = (jnp.asarray(st["obs_mean"], jnp.float32),
            jnp.asarray(st["obs_std"], jnp.float32))
    if a.policy == "mlp":
        _, apply, _ = es_mjx.mlp_spec(a.hidden)
    else:
        _, apply, _ = es_mjx.lut_spec(a.nap, a.tph)
    flat = jnp.asarray(np.load(os.path.join(HERE, a.mu)))

    m = mujoco.MjModel.from_xml_path(W.XML)      # stock 100/50 — the reference
    dt = m.opt.timestep * W.FRAME_SKIP
    rets = []
    for ep in range(a.episodes):
        d = mujoco.MjData(m)
        rng = np.random.default_rng(ep)
        d.qpos[:] += rng.uniform(-5e-3, 5e-3, m.nq)
        d.qvel[:] += rng.uniform(-5e-3, 5e-3, m.nv)
        mujoco.mj_forward(m, d)
        R = 0.0
        for _ in range(1000):
            obs = np.concatenate([d.qpos[1:], np.clip(d.qvel, -10, 10)])
            act = np.asarray(apply(flat, jnp.asarray(obs, jnp.float32), norm))
            act = np.clip(act, -1, 1)
            x0 = d.qpos[0]
            d.ctrl[:] = act
            for _ in range(W.FRAME_SKIP):
                mujoco.mj_step(m, d)
            R += 1.0 + (d.qpos[0] - x0) / dt - 1e-3 * float(np.sum(act ** 2))
            z, ang = d.qpos[1], d.qpos[2]
            if not (0.8 < z < 2.0 and -1.0 < ang < 1.0):
                break
        rets.append(R)
    mean, sd = float(np.mean(rets)), float(np.std(rets))
    print(f"{a.mu} [{a.policy}] CPU-reference {a.episodes}-ep deterministic: "
          f"{mean:.1f} +/- {sd:.1f}  [bar 3000 -> "
          f"{'SOLVED' if mean >= 3000 else 'below'}]", flush=True)
    json.dump(dict(mu=a.mu, policy=a.policy, episodes=a.episodes,
                   cpu_reference_mean=mean, cpu_reference_std=sd,
                   solved=bool(mean >= 3000)),
              open(os.path.join(HERE, a.mu.replace("_mu.npy", "_cpueval.json")), "w"),
              indent=1)


if __name__ == "__main__":
    main()
