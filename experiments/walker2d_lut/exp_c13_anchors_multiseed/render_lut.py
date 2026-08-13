"""exp_c13 — render a LUT-SAC actor walking in the CPU reference env (#75). MJX venv.

Same policy path as eval_cpu.py (obs standardiser, the requested forward mode, tanh of
the means only — the 6 log-sigmas are never read), and the same gymnasium Walker2d-v5
physics and camera as exp_c02's renderer, so these clips are directly comparable to the
SAC and PPO videos already produced.

Frames are dumped to .npz here and encoded by exp_c02_mjx_scaffold/encode_frames.py under
the spiky venv, which is the one with imageio — the MJX venv deliberately stays lean.
"""
import argparse, json, os, sys

import jax, jax.numpy as jnp
import numpy as np
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
for p in ("exp_c02_mjx_scaffold", "exp_c06_jax_backprop", "exp_c11_lut_sac_2x2"):
    sys.path.insert(0, os.path.join(HERE, "..", p))
import mjx_walker2d as W      # noqa: E402
import jax_lut_ext as X       # noqa: E402

C09 = os.path.join(HERE, "..", "exp_c09_lut_sac")
ACT = 6
CAMERA = "track"              # the model camera gymnasium actually uses (see exp_c02)


def load_actor(path, forward_mode):
    z = np.load(path)
    p = {k: jnp.asarray(z[k]) for k in
         ("w", "b", "weights", "log_T_soft", "log_T_sel")}
    heads, tph = int(z["n_heads"]), int(z["tph"])
    stats = json.load(open(os.path.join(HERE, "..", "exp_c03_distillation",
                                        "dataset_stats.json")))
    om = jnp.asarray(stats["obs_mean"], jnp.float32)
    osd = jnp.asarray(stats["obs_std"], jnp.float32)

    @jax.jit
    def act(obs):
        x = (obs - om) / (osd + 1e-6)
        y = X.apply(forward_mode)(x, p["w"], p["b"], p["weights"],
                                  p["log_T_soft"], p["log_T_sel"],
                                  heads, tph).sum(1)
        return jnp.tanh(y[:, :ACT])
    return lambda o: np.asarray(act(jnp.asarray(o[None])))[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("actor")
    ap.add_argument("--forward-mode", default="hard",
                    choices=["hard", "hybrid_smooth"])
    ap.add_argument("--seed", type=int, default=0, help="episode reset seed")
    ap.add_argument("--out", required=True, help="output basename (no extension)")
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--width", type=int, default=640)
    a = ap.parse_args()

    fn = load_actor(os.path.join(C09, a.actor), a.forward_mode)
    m = mujoco.MjModel.from_xml_path(W.XML)          # stock 100/50 reference solver
    d = mujoco.MjData(m)
    rng = np.random.default_rng(a.seed)
    d.qpos[:] += rng.uniform(-5e-3, 5e-3, m.nq)
    d.qvel[:] += rng.uniform(-5e-3, 5e-3, m.nv)
    mujoco.mj_forward(m, d)

    renderer = mujoco.Renderer(m, height=a.height, width=a.width)
    dt = m.opt.timestep * W.FRAME_SKIP
    frames, R, steps = [], 0.0, 0
    for _ in range(1000):
        renderer.update_scene(d, camera=CAMERA)
        frames.append(renderer.render())
        obs = np.concatenate([d.qpos[1:], np.clip(d.qvel, -10, 10)]).astype(np.float32)
        act = np.clip(fn(obs).astype(np.float64), -1.0, 1.0)
        x0 = d.qpos[0]
        d.ctrl[:] = act
        for _ in range(W.FRAME_SKIP):
            mujoco.mj_step(m, d)
        R += 1.0 + (d.qpos[0] - x0) / dt - 1e-3 * float(act @ act)
        steps += 1
        z, ang = d.qpos[1], d.qpos[2]
        if not (0.8 < z < 2.0 and -1.0 < ang < 1.0):
            break

    fps = int(round(1 / dt))
    npz = os.path.join(HERE, f"{a.out}_frames.npz")
    np.savez_compressed(npz, frames=np.stack(frames), fps=fps)
    print(f"{a.actor} [{a.forward_mode}] seed {a.seed}: return {R:.1f} over {steps} "
          f"steps ({steps / fps:.1f}s at {fps} fps) -> {npz}", flush=True)
    json.dump(dict(actor=a.actor, forward_mode=a.forward_mode, seed=a.seed,
                   episode_return=R, steps=steps, fps=fps,
                   duration_s=steps / fps, frames_npz=npz),
              open(os.path.join(HERE, f"{a.out}_render.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
