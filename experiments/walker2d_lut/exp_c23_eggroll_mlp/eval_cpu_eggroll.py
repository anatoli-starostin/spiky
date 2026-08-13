"""exp_c23 — re-measure an evolved policy in the CPU reference env (#75).

The EGGROLL loop optimises an MJX fitness at the approved 10/8 solver setting, which is
NOT the number this project quotes. This converts a saved mean into the one that is:
deterministic, 100 episodes, gymnasium Walker2d-v5 on stock CPU MuJoCo (100/50 solver).

Deliberately a line-for-line mirror of `exp_c05_es/eval_es_cpu.py` -- same episode seeds,
same reward reconstruction, same termination test -- so an EGGROLL policy and exp_c05's
full-rank ES policies are measured by identical code and the numbers are comparable.

The training proxy has mis-ranked policies before in this track (RESULTS.md, exp_c12), so
the proxy is never quoted as a result.

Usage:
  XLA_PYTHON_CLIENT_PREALLOCATE=false python eval_cpu_eggroll.py <name>_theta.npz
"""
import argparse, json, os, sys

import jax.numpy as jnp
import numpy as np
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "exp_c02_mjx_scaffold"))

import mjx_walker2d as W          # noqa: E402
import eggroll as E               # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("theta", help="<name>_theta.npz written by eggroll.py")
    ap.add_argument("--episodes", type=int, default=100)
    a = ap.parse_args()

    st = json.load(open(os.path.join(HERE, "..", "exp_c03_distillation",
                                     "dataset_stats.json")))
    norm = (jnp.asarray(st["obs_mean"], jnp.float32),
            jnp.asarray(st["obs_std"], jnp.float32))
    theta = E.load_theta(os.path.join(HERE, a.theta))

    m = mujoco.MjModel.from_xml_path(W.XML)      # stock 100/50 -- the reference
    dt = m.opt.timestep * W.FRAME_SKIP
    rets, lengths = [], []
    for ep in range(a.episodes):
        d = mujoco.MjData(m)
        rng = np.random.default_rng(ep)
        d.qpos[:] += rng.uniform(-5e-3, 5e-3, m.nq)
        d.qvel[:] += rng.uniform(-5e-3, 5e-3, m.nv)
        mujoco.mj_forward(m, d)
        R, t = 0.0, 0
        for t in range(1, 1001):
            obs = np.concatenate([d.qpos[1:], np.clip(d.qvel, -10, 10)])
            act = np.asarray(E.apply_policy(theta, None, jnp.asarray(obs, jnp.float32),
                                            norm, 0.0))
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
        lengths.append(t)
    mean, sd = float(np.mean(rets)), float(np.std(rets))
    full = int(sum(1 for L in lengths if L >= 1000))
    print(f"{a.theta} CPU-reference {a.episodes}-ep deterministic: {mean:.1f} +/- {sd:.1f}"
          f"  | {full}/{a.episodes} episodes ran the full 1000 steps"
          f"  | bar 3000 -> {'SOLVED' if mean >= 3000 else 'below'}", flush=True)
    json.dump(dict(theta=a.theta, episodes=a.episodes, cpu_reference_mean=mean,
                   cpu_reference_std=sd, full_length_episodes=full,
                   mean_length=float(np.mean(lengths)), returns=rets, lengths=lengths,
                   solved=bool(mean >= 3000)),
              open(os.path.join(HERE, a.theta.replace("_theta.npz", "_cpueval.json")), "w"),
              indent=1)


if __name__ == "__main__":
    main()
