"""exp_c02c — PPO smoke-train over batched GPU envs (issue #75).

Proves the pipeline end-to-end: batched MJX physics -> on-policy rollouts -> PPO
update, all resident on the GPU. This is a SMOKE TEST (a few iterations), not the
real training run.

Uses brax's PPO driving brax's `walker2d` on the **mjx backend** — i.e. MuJoCo's own
dynamics via MJX, with obs 17 / act 6, matching Walker2d-v5's spaces exactly.

Note recorded during scoping: brax now emits a UserWarning that the library "is not
actively being maintained" and points at MJX / mujoco_playground. Its PPO still works
and is a fine scaffold, but see the issue comment for the recommendation.

Usage:
    XLA_PYTHON_CLIENT_PREALLOCATE=false python ppo_smoke.py [--timesteps 300000]
"""
import argparse, functools, json, os, time

import jax

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timesteps", type=int, default=300_000)
    ap.add_argument("--num-envs", type=int, default=2048)
    ap.add_argument("--unroll-length", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--episode-length", type=int, default=1000)
    ap.add_argument("--backend", default="mjx")
    a = ap.parse_args()

    from brax import envs
    from brax.training.agents.ppo import train as ppo

    env = envs.get_environment("walker2d", backend=a.backend)
    print(f"env walker2d backend={a.backend}  obs={env.observation_size} "
          f"act={env.action_size}  device={jax.devices()[0].device_kind}", flush=True)

    rows = []
    t_start = time.time()

    def progress(step, metrics):
        r = metrics.get("eval/episode_reward")
        el = time.time() - t_start
        sps = step / max(el, 1e-9)
        row = dict(step=int(step), reward=(float(r) if r is not None else None),
                   elapsed_s=round(el, 1), env_steps_per_sec=round(sps, 1))
        rows.append(row)
        print(f"[{step:>9,}] reward={r if r is None else round(float(r), 1):>9} "
              f"| {sps:>10,.0f} env-steps/s | {el/60:5.1f}m", flush=True)

    train_fn = functools.partial(
        ppo.train,
        num_timesteps=a.timesteps,
        num_evals=4,
        episode_length=a.episode_length,
        num_envs=a.num_envs,
        batch_size=a.batch_size,
        unroll_length=a.unroll_length,
        num_minibatches=8,
        num_updates_per_batch=4,
        learning_rate=3e-4,
        entropy_cost=1e-2,
        discounting=0.97,
        normalize_observations=True,
        reward_scaling=1.0,
        seed=0,
    )

    t0 = time.time()
    make_inference_fn, params, metrics = train_fn(environment=env, progress_fn=progress)
    wall = time.time() - t0

    print(f"\nPPO smoke-train finished: {a.timesteps:,} env-steps in {wall/60:.1f} min "
          f"({a.timesteps/wall:,.0f} env-steps/s end-to-end incl. compile + evals)")
    print(f"final eval reward: {metrics.get('eval/episode_reward')}")

    out = dict(backend=a.backend, num_envs=a.num_envs, timesteps=a.timesteps,
               wall_clock_s=round(wall, 1),
               end_to_end_env_steps_per_sec=round(a.timesteps / wall, 1),
               final_eval_reward=float(metrics.get("eval/episode_reward", float("nan"))),
               progress=rows)
    with open(os.path.join(HERE, "ppo_smoke_results.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("wrote ppo_smoke_results.json")


if __name__ == "__main__":
    main()
