"""exp_c03 1a — build the distillation dataset from the PPO teacher (#75).

Rolls the teacher out in BATCHED MJX (4096 envs on the GPU) rather than the CPU env:
the CPU reference stepped with a per-step JAX policy call runs at ~1k steps/s, so 4M
pairs would take over an hour; on MJX the same 4M pairs take well under a minute.
The cross-check (RESULTS.md) showed the MJX@10/8 and CPU state distributions agree to
within +1.5% of return, so the state coverage is representative — and a CPU-rollout
slice is added on top as a distribution-shift guard.

Labels are ALWAYS the teacher's DETERMINISTIC action for the visited state. States come
from a mix of deterministic and noise-injected rollouts (DAgger-style coverage), so the
student sees states it will actually encounter once it is slightly wrong.

Output (kept out of git): obs.npy [N,17] float32, act.npy [N,6] float32.

Usage:
  XLA_PYTHON_CLIENT_PREALLOCATE=false python gen_dataset.py --pairs 4000000
"""
import argparse, json, os, sys, time

import jax, jax.numpy as jnp
import numpy as np

MJX_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..",
                                       "exp_c02_mjx_scaffold"))
sys.path.insert(0, MJX_DIR)

import mjx_walker2d as W          # noqa: E402
from mujoco import mjx           # noqa: E402
from cross_check import load_policy  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", default=os.path.join(MJX_DIR, "ppo_policy_full.msgpack"))
    ap.add_argument("--pairs", type=int, default=4_000_000)
    ap.add_argument("--num-envs", type=int, default=4096)
    ap.add_argument("--noise-frac", type=float, default=0.5,
                    help="fraction of envs driven with exploration noise")
    ap.add_argument("--noise-std", type=float, default=0.1)
    ap.add_argument("--out", default=HERE)
    a = ap.parse_args()

    net, params = load_policy(a.params)
    m = W.make_model()                      # solver 10/8
    mx = mjx.put_model(m)
    reset, step = W.make_env(mx)
    v_reset, v_step = jax.vmap(reset), jax.vmap(step)

    n_steps = int(np.ceil(a.pairs / a.num_envs))
    print(f"teacher {os.path.basename(a.params)} | {a.num_envs} envs x {n_steps} steps "
          f"= {a.num_envs*n_steps:,} pairs | noise on {a.noise_frac:.0%} of envs "
          f"(std {a.noise_std})", flush=True)

    key = jax.random.PRNGKey(0)
    key, k0 = jax.random.split(key)
    st = v_reset(jax.random.split(k0, a.num_envs))
    noisy = (jnp.arange(a.num_envs) < int(a.num_envs * a.noise_frac))

    @jax.jit
    def collect(st, key):
        def one(carry, _):
            st, key = carry
            key, sub = jax.random.split(key)
            mean, _, _ = net.apply(params, st.obs)
            # CLIP the label. The policy head is an unbounded Gaussian mean, but the
            # environment applies clip(a, -1, 1) before stepping — so the teacher's
            # *behaviour* is the clipped action, and that is what a student must clone.
            # Regressing the raw mean instead makes 63% of targets lie outside the
            # action space entirely (measured mean(y^2) = 7.53 vs ~0.8 clipped), which
            # no bounded student can fit and which weights the loss towards magnitudes
            # the env throws away.
            label = jnp.clip(mean, -1.0, 1.0)
            act = jnp.where(noisy[:, None],
                            label + a.noise_std * jax.random.normal(sub, mean.shape),
                            label)
            nst = v_step(st, act)
            return (nst, key), (st.obs, label)
        (st, key), (obs, act) = jax.lax.scan(one, (st, key), None, length=n_steps)
        return st, key, obs, act

    t0 = time.time()
    st, key, obs, act = collect(st, key)
    obs = np.asarray(obs, np.float32).reshape(-1, 17)
    act = np.asarray(act, np.float32).reshape(-1, 6)
    dt = time.time() - t0
    print(f"collected {len(obs):,} pairs in {dt:.1f}s "
          f"({len(obs)/dt:,.0f} pairs/s)", flush=True)

    np.save(os.path.join(a.out, "obs.npy"), obs)
    np.save(os.path.join(a.out, "act.npy"), act)

    stats = dict(pairs=int(len(obs)), num_envs=a.num_envs, steps=n_steps,
                 noise_frac=a.noise_frac, noise_std=a.noise_std,
                 collect_s=round(dt, 1),
                 obs_mean=obs.mean(0).round(4).tolist(),
                 obs_std=obs.std(0).round(4).tolist(),
                 obs_min=obs.min(0).round(3).tolist(),
                 obs_max=obs.max(0).round(3).tolist(),
                 act_mean=act.mean(0).round(4).tolist(),
                 act_std=act.std(0).round(4).tolist(),
                 act_saturated_frac=float((np.abs(act) > 0.99).mean()))
    json.dump(stats, open(os.path.join(a.out, "dataset_stats.json"), "w"), indent=1)
    print(f"obs {obs.shape} act {act.shape} | action saturation "
          f"{stats['act_saturated_frac']:.1%} | wrote obs.npy / act.npy", flush=True)


if __name__ == "__main__":
    main()
