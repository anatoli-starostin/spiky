"""exp_c06 — evaluate the from-scratch LUT policy in the CPU reference env (#75).

Deterministic (mean action), 100 episodes, gymnasium Walker2d-v5 on CPU MuJoCo —
the same protocol as every other number in this project.
"""
import argparse, json, os, sys

import jax.numpy as jnp
import numpy as np
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "exp_c02_mjx_scaffold"))
import mjx_walker2d as W          # noqa: E402
import jax_lut_grad as L         # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", default=os.path.join(HERE, "lut_scratch_params.npz"))
    ap.add_argument("--episodes", type=int, default=100)
    a = ap.parse_args()

    z = np.load(a.params)
    p = dict(w=jnp.asarray(z["w"]), b=jnp.asarray(z["b"]),
             weights=jnp.asarray(z["weights"]),
             log_T_soft=jnp.asarray(z["log_T_soft"]),
             log_T_sel=jnp.asarray(z["log_T_sel"]),
             n_heads=int(z["n_heads"]), tph=int(z["tph"]),
             obs_mean=jnp.asarray(z["obs_mean"]), obs_std=jnp.asarray(z["obs_std"]))
    n = int(np.prod(z["weights"].shape) + np.prod(z["w"].shape) + np.prod(z["b"].shape))

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
            act = np.asarray(L.policy(p, jnp.asarray(obs, jnp.float32)[None]))[0]
            act = np.clip(act, -1, 1)
            x0 = d.qpos[0]
            d.ctrl[:] = act
            for _ in range(W.FRAME_SKIP):
                mujoco.mj_step(m, d)
            R += 1.0 + (d.qpos[0] - x0) / dt - 1e-3 * float(np.sum(act ** 2))
            z_, ang = d.qpos[1], d.qpos[2]
            if not (0.8 < z_ < 2.0 and -1.0 < ang < 1.0):
                break
        rets.append(R)
    mean, sd = float(np.mean(rets)), float(np.std(rets))
    print(f"LUT trained FROM SCRATCH ({n:,} params) | CPU-reference "
          f"{a.episodes}-ep deterministic: {mean:.1f} +/- {sd:.1f}  "
          f"[bar 3000 -> {'SOLVED' if mean >= 3000 else 'below'}]", flush=True)
    json.dump(dict(params=n, episodes=a.episodes, cpu_reference_mean=mean,
                   cpu_reference_std=sd, solved=bool(mean >= 3000)),
              open(os.path.join(HERE, "eval_scratch_cpu.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
