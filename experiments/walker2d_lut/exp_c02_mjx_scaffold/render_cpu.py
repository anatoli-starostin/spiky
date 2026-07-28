"""exp_c02e — evaluate + render an MJX-trained policy in the CPU REFERENCE env (#75).

The policy is trained under MJX @10/8, but the number and the video that count are
produced in gymnasium's CPU `Walker2d-v5` — same engine the SAC baseline is measured
in, so the two videos are apples-to-apples.

Runs in the JAX venv (it needs flax to rebuild the policy) but drives plain CPU
`mujoco` with Walker2d-v5's documented obs/reward/termination.

Usage:
    MUJOCO_GL=glfw XLA_PYTHON_CLIENT_PREALLOCATE=false \
        python render_cpu.py --params ppo_policy_full.msgpack --episodes 100
"""
import argparse, json, os

import jax, jax.numpy as jnp
import numpy as np
import mujoco

import mjx_walker2d as W
from cross_check import load_policy

HERE = os.path.dirname(os.path.abspath(__file__))


# The camera gymnasium actually renders Walker2d-v5 with. Inspecting a live
# gymnasium viewer shows cam.type == mjCAMERA_FIXED and cam.fixedcamid == 0 — i.e. it
# uses the camera DEFINED IN THE MODEL, not a constructed one:
#
#   walker2d_v5.xml:18
#   <camera name="track" mode="trackcom" pos="0 -3 -0.25" xyaxes="1 0 0 0 0 1"/>
#
# `mode="trackcom"` follows the centre of mass, and the xyaxes give the side-on view.
# DEFAULT_CAMERA_CONFIG (trackbodyid/distance/lookat/elevation) is inert here — those
# fields only apply to a free/tracking camera, which is why reconstructing an
# MjvCamera from them does NOT reproduce gymnasium's framing.
CAMERA = "track"


def episode(net, params, m, seed, frames=None, renderer=None, camera=None):
    d = mujoco.MjData(m)
    rng = np.random.default_rng(seed)
    d.qpos[:] += rng.uniform(-5e-3, 5e-3, m.nq)
    d.qvel[:] += rng.uniform(-5e-3, 5e-3, m.nv)
    mujoco.mj_forward(m, d)
    dt = m.opt.timestep * W.FRAME_SKIP
    R, steps = 0.0, 0
    for _ in range(1000):
        if frames is not None:
            renderer.update_scene(d, camera=camera)
            frames.append(renderer.render())
        obs = np.concatenate([d.qpos[1:], np.clip(d.qvel, -10, 10)])
        mean, _, _ = net.apply(params, jnp.array(obs, jnp.float32)[None])
        a = np.clip(np.asarray(mean)[0], -1, 1)
        x0 = d.qpos[0]
        d.ctrl[:] = a
        for _ in range(W.FRAME_SKIP):
            mujoco.mj_step(m, d)
        R += 1.0 + (d.qpos[0] - x0) / dt - 1e-3 * float(np.sum(a ** 2))
        steps += 1
        z, ang = d.qpos[1], d.qpos[2]
        if not (0.8 < z < 2.0 and -1.0 < ang < 1.0):
            break
    return R, steps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", default="ppo_policy_full.msgpack")
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--video", default="walker2d_mjx_ppo_cpu_eval.mp4")
    ap.add_argument("--steps-label", default="")
    a = ap.parse_args()

    net, params = load_policy(os.path.join(HERE, a.params))
    m = mujoco.MjModel.from_xml_path(W.XML)      # stock 100/50 — the reference
    print(f"policy {a.params} | CPU Walker2d-v5 reference "
          f"(solver {m.opt.iterations}/{m.opt.ls_iterations})", flush=True)

    rets = [episode(net, params, m, seed=s)[0] for s in range(a.episodes)]
    mean, std = float(np.mean(rets)), float(np.std(rets))
    print(f"deterministic {a.episodes}-episode eval in the CPU reference env: "
          f"{mean:.1f} +/- {std:.1f}   [solved bar = 3000 -> "
          f"{'SOLVED' if mean >= 3000 else 'below'}]", flush=True)

    # This venv has no imageio (it is the lean JAX/MJX one). Dump raw frames and let
    # encode_frames.py — run under the torch venv, which already has imageio-ffmpeg —
    # do the MP4. Avoids an install into the JAX env purely for video encoding.
    renderer = mujoco.Renderer(m, height=480, width=480)   # same 480x480 as the SAC video
    frames = []
    R, steps = episode(net, params, m, seed=0, frames=frames, renderer=renderer,
                       camera=CAMERA)
    npz = os.path.join(HERE, a.video.replace(".mp4", "_frames.npz"))
    np.savez_compressed(npz, frames=np.stack(frames),
                        fps=int(round(1 / (m.opt.timestep * W.FRAME_SKIP))))
    path = os.path.join(HERE, a.video)
    print(f"rendered episode: return {R:.1f} over {steps} steps -> frames {npz}",
          flush=True)

    json.dump(dict(params=a.params, episodes=a.episodes, mean_return=mean,
                   std_return=std, solved=bool(mean >= 3000), video=path,
                   video_return=R, video_steps=steps),
              open(os.path.join(HERE, "render_cpu_results.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
