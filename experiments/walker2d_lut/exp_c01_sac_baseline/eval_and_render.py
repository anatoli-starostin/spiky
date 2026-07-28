"""exp_c01 — evaluate a SAC checkpoint and render an episode to MP4 (issue #75).

Eval protocol per the issue: deterministic policy, N episodes (100 for the final
number), report mean +/- std return.

Usage:
    MUJOCO_GL=glfw python eval_and_render.py --ckpt run_seed0/ckpt/sac_walker2d_200000_steps.zip
    MUJOCO_GL=glfw python eval_and_render.py --ckpt <...> --episodes 100 --no-video
"""
import argparse, glob, json, os, time

import numpy as np
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy

HERE = os.path.dirname(os.path.abspath(__file__))
ENV_ID = "Walker2d-v5"


def latest_ckpt(run_dir):
    """Newest checkpoint in a run dir: the final save if present, else the highest step."""
    final = os.path.join(run_dir, "sac_walker2d_final.zip")
    if os.path.exists(final):
        return final
    cks = glob.glob(os.path.join(run_dir, "ckpt", "*_steps.zip"))
    if not cks:
        return None
    return max(cks, key=lambda p: int(os.path.basename(p).split("_")[-2]))


def render_episode(model, path, seed=0, fps=50, max_steps=1000):
    import imageio
    env = gym.make(ENV_ID, render_mode="rgb_array")
    obs, _ = env.reset(seed=seed)
    frames, R, steps = [], 0.0, 0
    for _ in range(max_steps):
        frames.append(env.render())
        a, _ = model.predict(obs, deterministic=True)
        obs, r, term, trunc, _ = env.step(a)
        R += r
        steps += 1
        if term or trunc:
            break
    env.close()
    imageio.mimwrite(path, frames, fps=fps, codec="libx264",
                     output_params=["-pix_fmt", "yuv420p"])
    return R, steps, len(frames)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None, help="checkpoint .zip (default: latest in --run)")
    ap.add_argument("--run", default=os.path.join(HERE, "run_seed0"))
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--video", default=None)
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    ckpt = a.ckpt or latest_ckpt(a.run)
    if ckpt is None:
        raise SystemExit(f"no checkpoint found under {a.run}")
    step = os.path.basename(ckpt).split("_")[-2] if "_steps" in ckpt else "final"
    print(f"checkpoint: {ckpt}  (step {step})")

    model = SAC.load(ckpt, device="cuda")
    env = Monitor(gym.make(ENV_ID))
    t0 = time.time()
    mean, std = evaluate_policy(model, env, n_eval_episodes=a.episodes,
                                deterministic=True, warn=False)
    env.close()
    print(f"deterministic eval over {a.episodes} episodes: {mean:.1f} +/- {std:.1f}"
          f"   ({time.time()-t0:.1f}s)   [solved bar = 3000]")

    out = dict(ckpt=ckpt, step=step, episodes=a.episodes,
               mean_return=float(mean), std_return=float(std),
               solved=bool(mean >= 3000))

    if not a.no_video:
        vid = a.video or os.path.join(HERE, f"walker2d_sac_step{step}.mp4")
        R, steps, nf = render_episode(model, vid, seed=a.seed)
        print(f"rendered episode: return {R:.1f} over {steps} steps "
              f"({nf} frames) -> {vid}")
        out.update(video=vid, video_return=float(R), video_steps=int(steps))

    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
