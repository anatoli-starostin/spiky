"""exp_c09 — render a LUT-SAC actor walking in the CPU reference env (#75).

Same conventions as every other video in this project so they sit side by side:
gymnasium's own `walker2d_v5.xml`, the model's `track` camera (NOT a camera
reconstructed from DEFAULT_CAMERA_CONFIG — see RESULTS.md), 480x480, 50 fps.

Frames are dumped to .npz and encoded by `encode_frames.py` under the torch venv,
because this (JAX/MJX) venv deliberately has no imageio.
"""
import argparse, json, os, sys

import jax.numpy as jnp
import numpy as np
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "exp_c02_mjx_scaffold"))
import mjx_walker2d as W          # noqa: E402
from eval_cpu import load_actor   # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("actor")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="lut_sac_walk_frames.npz")
    a = ap.parse_args()

    fn, n = load_actor(os.path.join(HERE, a.actor))
    m = mujoco.MjModel.from_xml_path(W.XML)
    d = mujoco.MjData(m)
    rng = np.random.default_rng(a.seed)
    d.qpos[:] += rng.uniform(-5e-3, 5e-3, m.nq)
    d.qvel[:] += rng.uniform(-5e-3, 5e-3, m.nv)
    mujoco.mj_forward(m, d)

    r = mujoco.Renderer(m, height=480, width=480)
    dt = m.opt.timestep * W.FRAME_SKIP
    frames, R, steps = [], 0.0, 0
    for _ in range(1000):
        r.update_scene(d, camera="track")
        frames.append(r.render())
        obs = np.concatenate([d.qpos[1:], np.clip(d.qvel, -10, 10)])[None]
        act = np.clip(fn(obs.astype(np.float32))[0], -1, 1)
        x0 = d.qpos[0]
        d.ctrl[:] = act
        for _ in range(W.FRAME_SKIP):
            mujoco.mj_step(m, d)
        R += 1.0 + (d.qpos[0] - x0) / dt - 1e-3 * float(act @ act)
        steps += 1
        z, ang = d.qpos[1], d.qpos[2]
        if not (0.8 < z < 2.0 and -1.0 < ang < 1.0):
            break
    np.savez_compressed(os.path.join(HERE, a.out),
                        frames=np.stack(frames), fps=50)
    print(f"{a.actor} ({n:,} params) | episode return {R:.1f} over {steps} steps "
          f"-> {a.out}", flush=True)


if __name__ == "__main__":
    main()
