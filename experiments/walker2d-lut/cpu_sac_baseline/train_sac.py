"""Pure-PyTorch SAC baseline on Walker2d-v5 — faithful reproduction of
walker2d-lut branch exp_c01_sac_baseline (issue #75, roadmap step 1).

Hyperparameters are the issue's spec verbatim:
  MLP [256, 256] ReLU actor + critic, lr 3e-4, buffer 1e6, batch 256, gamma 0.99,
  tau 0.005, train_freq 1, gradient_steps 1, learning_starts 10k, auto entropy
  (target -6 = -|A|), 1M timesteps, seed 0.

Physics: gymnasium/MuJoCo on CPU (single env). Nets: torch on CUDA. NO JAX.

Usage:
    python train_sac.py --seed 0 [--steps 1000000] [--device cuda]
"""
import argparse, json, os, time, sys

import numpy as np
import gymnasium as gym
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy

HERE = os.path.dirname(os.path.abspath(__file__))
ENV_ID = "Walker2d-v5"

MILESTONES = (100_000, 300_000, 1_000_000)


class ProgressCallback(BaseCallback):
    """Periodic eval + a single-line progress log with a live ETA."""

    def __init__(self, eval_every=25_000, n_eval_episodes=10, seed=0, log_path=None):
        super().__init__()
        self.eval_every = eval_every
        self.n_eval_episodes = n_eval_episodes
        self.seed = seed
        self.log_path = log_path
        self.rows = []
        self.t0 = None
        self._next = eval_every

    def _on_training_start(self):
        self.t0 = time.time()

    def _on_step(self):
        if self.num_timesteps < self._next:
            return True
        self._next += self.eval_every
        eval_env = Monitor(gym.make(ENV_ID))
        mean, std = evaluate_policy(
            self.model, eval_env, n_eval_episodes=self.n_eval_episodes,
            deterministic=True, warn=False)
        eval_env.close()
        el = time.time() - self.t0
        fps = self.num_timesteps / max(el, 1e-9)
        remaining = (self.total_steps - self.num_timesteps) / max(fps, 1e-9)
        row = dict(step=int(self.num_timesteps), mean_return=float(mean),
                   std_return=float(std), elapsed_s=round(el, 1),
                   fps=round(fps, 1), eta_s=round(remaining, 1))
        self.rows.append(row)
        print(f"[{self.num_timesteps:>8,}/{self.total_steps:,}] "
              f"eval({self.n_eval_episodes}ep) = {mean:8.1f} +/- {std:6.1f} | "
              f"{fps:5.0f} fps | elapsed {el/60:5.1f}m | ETA {remaining/60:5.1f}m",
              flush=True)
        if self.log_path:
            with open(self.log_path, "w") as f:
                json.dump(self.rows, f, indent=1)
        return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=1_000_000)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--eval-every", type=int, default=25_000)
    ap.add_argument("--learning-starts", type=int, default=10_000)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    run = os.path.join(HERE, f"run_seed{a.seed}{a.tag}")
    os.makedirs(run, exist_ok=True)

    env = Monitor(gym.make(ENV_ID), filename=os.path.join(run, "monitor.csv"))
    env.reset(seed=a.seed)
    env.action_space.seed(a.seed)

    model = SAC(
        "MlpPolicy", env,
        learning_rate=3e-4,
        buffer_size=1_000_000,
        learning_starts=a.learning_starts,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        ent_coef="auto",                       # auto entropy, target -|A| = -6
        target_entropy="auto",
        policy_kwargs=dict(net_arch=[256, 256], activation_fn=torch.nn.ReLU),
        verbose=0,
        seed=a.seed,
        device=a.device,
        tensorboard_log=None,   # pure logging; omitted (tensorboard not installed) — no effect on training
    )
    print(f"env={ENV_ID} obs={env.observation_space.shape} act={env.action_space.shape}")
    print(f"device={model.device}  target_entropy={model.target_entropy}  "
          f"actor_params={sum(p.numel() for p in model.actor.parameters()):,}",
          flush=True)

    prog = ProgressCallback(eval_every=a.eval_every, seed=a.seed,
                            log_path=os.path.join(run, "progress.json"))
    prog.total_steps = a.steps
    ckpt = CheckpointCallback(save_freq=50_000, save_path=os.path.join(run, "ckpt"),
                              name_prefix="sac_walker2d")

    t0 = time.time()
    model.learn(total_timesteps=a.steps, callback=[prog, ckpt], progress_bar=False)
    wall = time.time() - t0
    model.save(os.path.join(run, "sac_walker2d_final"))

    # Final protocol eval: deterministic, 100 episodes (the issue's spec).
    eval_env = Monitor(gym.make(ENV_ID))
    mean, std = evaluate_policy(model, eval_env, n_eval_episodes=100,
                                deterministic=True, warn=False)
    eval_env.close()
    print(f"[FINAL] deterministic 100-episode eval: {mean:.1f} +/- {std:.1f}  "
          f"(solved bar = 3000)", flush=True)

    import importlib.metadata as md
    summary = dict(
        env=ENV_ID, algo="SAC", seed=a.seed, total_steps=a.steps,
        final_eval_mean=float(mean), final_eval_std=float(std),
        solved=bool(mean >= 3000), wall_clock_h=round(wall / 3600, 3),
        fps=round(a.steps / wall, 1), device=str(model.device),
        milestones={str(m): next((r["mean_return"] for r in prog.rows
                                  if r["step"] >= m), None) for m in MILESTONES},
        versions={p: md.version(p) for p in
                  ("gymnasium", "mujoco", "stable-baselines3", "torch", "numpy")},
        progress=prog.rows,
    )
    with open(os.path.join(run, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print("wrote", os.path.join(run, "summary.json"), flush=True)


if __name__ == "__main__":
    main()
