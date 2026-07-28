"""exp_c03 1c — evaluate a policy in the CPU reference env (#75).

The ONLY number this project quotes: deterministic, 100 episodes, gymnasium
Walker2d-v5 on CPU MuJoCo (stock solver), same protocol as both baselines and the
same 3000 bar. Never a training proxy.
"""
import numpy as np
import torch
import gymnasium as gym


@torch.no_grad()
def eval_policy(model, episodes=100, seed0=0, device="cuda", max_steps=1000,
                render=False, camera="track"):
    """model: obs[B,17] -> action[B,6] in [-1,1]. Returns (mean, std, returns, frames)."""
    env = gym.make("Walker2d-v5", render_mode="rgb_array" if render else None)
    rets, frames = [], []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed0 + ep)
        R = 0.0
        for _ in range(max_steps):
            if render and ep == 0:
                frames.append(env.render())
            t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            a = model(t).squeeze(0).cpu().numpy()
            obs, r, term, trunc, _ = env.step(np.clip(a, -1, 1))
            R += r
            if term or trunc:
                break
        rets.append(R)
    env.close()
    rets = np.asarray(rets)
    return float(rets.mean()), float(rets.std()), rets, frames


if __name__ == "__main__":
    import argparse, json
    from lut_policy import load
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--render", default=None, help="write an MP4 here")
    a = ap.parse_args()
    m = load(a.ckpt)
    print(m.describe())
    mean, std, rets, frames = eval_policy(m, episodes=a.episodes,
                                          render=bool(a.render))
    print(f"deterministic {a.episodes}-episode eval (CPU reference): "
          f"{mean:.1f} +/- {std:.1f}  [bar 3000 -> "
          f"{'SOLVED' if mean >= 3000 else 'below'}]")
    if a.render and frames:
        import imageio
        imageio.mimwrite(a.render, frames, fps=50, codec="libx264",
                         output_params=["-pix_fmt", "yuv420p"])
        print("wrote", a.render)
