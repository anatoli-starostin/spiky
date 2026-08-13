"""exp_c02c — compact PPO over the batched MJX Walker2d (issue #75).

purejaxrl-style: everything (physics, policy, rollout, update) is jitted and lives on
the GPU; no host round-trip inside a training iteration. Self-contained — flax/optax
only, deliberately NOT brax (its PPO calls the removed `jax.device_put_replicated`
and the library now warns it is unmaintained).

This is the scaffold + smoke test, not the full training run.

Usage:
    XLA_PYTHON_CLIENT_PREALLOCATE=false python ppo_mjx.py --iters 10 --num-envs 2048
"""
import argparse, json, os, time
from typing import NamedTuple

import jax, jax.numpy as jnp
import numpy as np
import optax
from flax import linen as nn
from flax.training.train_state import TrainState

import mjx_walker2d as W
from mujoco import mjx

HERE = os.path.dirname(os.path.abspath(__file__))
OBS, ACT = 17, 6


class ActorCritic(nn.Module):
    hidden: int = 256

    @nn.compact
    def __call__(self, x):
        a = nn.tanh(nn.Dense(self.hidden)(x))
        a = nn.tanh(nn.Dense(self.hidden)(a))
        mean = nn.Dense(ACT)(a)
        logstd = self.param("logstd", nn.initializers.zeros, (ACT,))
        c = nn.tanh(nn.Dense(self.hidden)(x))
        c = nn.tanh(nn.Dense(self.hidden)(c))
        v = nn.Dense(1)(c)
        return mean, logstd, jnp.squeeze(v, -1)


def log_prob(mean, logstd, a):
    std = jnp.exp(logstd)
    return (-0.5 * jnp.square((a - mean) / std) - logstd
            - 0.5 * jnp.log(2 * jnp.pi)).sum(-1)


class Transition(NamedTuple):
    obs: jnp.ndarray
    action: jnp.ndarray
    logp: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    done: jnp.ndarray


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--num-envs", type=int, default=2048)
    ap.add_argument("--rollout", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--minibatches", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--ent", type=float, default=1e-3)
    ap.add_argument("--solver-iterations", type=int, default=W.SOLVER_ITERATIONS)
    ap.add_argument("--ls-iterations", type=int, default=W.SOLVER_LS_ITERATIONS)
    ap.add_argument("--save-params", default=None,
                    help="write the trained policy params here (msgpack)")
    ap.add_argument("--out", default="ppo_mjx_results.json")
    a = ap.parse_args()

    m = W.make_model(solver_iterations=a.solver_iterations,
                     ls_iterations=a.ls_iterations)
    mx = mjx.put_model(m)
    reset, step = W.make_env(mx)
    v_reset, v_step = jax.vmap(reset), jax.vmap(step)

    key = jax.random.PRNGKey(0)
    key, k_init, k_reset = jax.random.split(key, 3)
    net = ActorCritic()
    params = net.init(k_init, jnp.zeros((1, OBS)))
    tx = optax.chain(optax.clip_by_global_norm(0.5), optax.adam(a.lr))
    ts = TrainState.create(apply_fn=net.apply, params=params, tx=tx)
    env_state = v_reset(jax.random.split(k_reset, a.num_envs))

    n_batch = a.num_envs * a.rollout
    mb = n_batch // a.minibatches
    print(f"PPO on MJX Walker2d | envs={a.num_envs} rollout={a.rollout} "
          f"batch={n_batch:,} minibatch={mb:,} | solver_iters={a.solver_iterations} "
          f"| {jax.devices()[0].device_kind}", flush=True)

    def rollout(ts, env_state, key):
        def one(carry, _):
            ts, st, key = carry
            key, sub = jax.random.split(key)
            mean, logstd, val = net.apply(ts.params, st.obs)
            act = mean + jnp.exp(logstd) * jax.random.normal(sub, mean.shape)
            lp = log_prob(mean, logstd, act)
            nst = v_step(st, act)
            return (ts, nst, key), Transition(st.obs, act, lp, val, nst.reward, nst.done)
        (ts, st, key), traj = jax.lax.scan(one, (ts, env_state, key), None,
                                           length=a.rollout)
        _, _, last_v = net.apply(ts.params, st.obs)
        return st, traj, last_v, key

    def gae(traj, last_v):
        def one(carry, x):
            adv, nv = carry
            rew, val, done = x
            delta = rew + a.gamma * nv * (1 - done) - val
            adv = delta + a.gamma * a.lam * (1 - done) * adv
            return (adv, val), adv
        _, advs = jax.lax.scan(
            one, (jnp.zeros_like(last_v), last_v),
            (traj.reward, traj.value, traj.done), reverse=True)
        return advs, advs + traj.value

    def loss_fn(params, obs, act, old_lp, adv, ret):
        mean, logstd, val = net.apply(params, obs)
        lp = log_prob(mean, logstd, act)
        ratio = jnp.exp(lp - old_lp)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        pl = -jnp.minimum(ratio * adv,
                          jnp.clip(ratio, 1 - a.clip, 1 + a.clip) * adv).mean()
        vl = jnp.square(val - ret).mean()
        ent = (logstd + 0.5 * jnp.log(2 * jnp.pi * jnp.e)).sum()
        return pl + 0.5 * vl - a.ent * ent, (pl, vl, ent)

    @jax.jit
    def train_iter(ts, env_state, key):
        env_state, traj, last_v, key = rollout(ts, env_state, key)
        adv, ret = gae(traj, last_v)
        flat = jax.tree.map(lambda x: x.reshape((n_batch,) + x.shape[2:]),
                            (traj.obs, traj.action, traj.logp, adv, ret))

        def epoch(carry, _):
            ts, key = carry
            key, sub = jax.random.split(key)
            perm = jax.random.permutation(sub, n_batch)
            shuf = jax.tree.map(lambda x: x[perm].reshape((a.minibatches, mb) +
                                                          x.shape[1:]), flat)

            def upd(ts, batch):
                (l, aux), g = jax.value_and_grad(loss_fn, has_aux=True)(ts.params, *batch)
                return ts.apply_gradients(grads=g), (l,) + aux
            ts, out = jax.lax.scan(upd, ts, shuf)
            return (ts, key), out
        (ts, key), out = jax.lax.scan(epoch, (ts, key), None, length=a.epochs)
        return ts, env_state, key, traj, out

    rows, t_start = [], time.time()
    total_steps = 0
    for it in range(a.iters):
        t0 = time.time()
        key, sub = jax.random.split(key)
        ts, env_state, key, traj, out = train_iter(ts, env_state, sub)
        jax.block_until_ready(ts.params)
        dt = time.time() - t0
        total_steps += n_batch
        # mean reward per env-step * 1000 ~ the return of a 1000-step episode
        rew = float(traj.reward.mean())
        loss, pl, vl, ent = (float(x.mean()) for x in out)
        sps = n_batch / dt
        rows.append(dict(iter=it, env_steps=total_steps, mean_step_reward=rew,
                         est_return_1000=round(rew * 1000, 1), loss=loss,
                         policy_loss=pl, value_loss=vl, entropy=ent,
                         env_steps_per_sec=round(sps, 1), iter_s=round(dt, 2)))
        print(f"[iter {it:>3}] steps={total_steps:>10,} | rew/step {rew:+.4f} "
              f"(~{rew*1000:7.1f}/1000-step ep) | loss {loss:+.3f} "
              f"| {sps:>9,.0f} env-steps/s | {dt:5.1f}s", flush=True)
        # incremental progress file, so an external poller (the Slack bar) can read
        # live state without parsing the log
        if it % 5 == 0 or it == a.iters - 1:
            el = time.time() - t_start
            eta = (a.iters - it - 1) * (el / (it + 1))
            json.dump(dict(iter=it, iters=a.iters, env_steps=total_steps,
                           target_env_steps=n_batch * a.iters,
                           est_return_1000=round(rew * 1000, 1),
                           env_steps_per_sec=round(sps, 1),
                           elapsed_s=round(el, 1), eta_s=round(eta, 1),
                           done=False),
                      open(os.path.join(HERE, a.out + ".partial"), "w"), indent=1)

    wall = time.time() - t_start
    print(f"\nsmoke-train OK: {a.iters} iterations, {total_steps:,} env-steps "
          f"in {wall:.1f}s ({total_steps/wall:,.0f} env-steps/s incl. compile)")

    if a.save_params:
        import flax.serialization as fs
        with open(os.path.join(HERE, a.save_params), "wb") as f:
            f.write(fs.to_bytes(ts.params))
        print("saved policy params ->", a.save_params)
    json.dump(dict(num_envs=a.num_envs, rollout=a.rollout, iters=a.iters,
                   total_env_steps=total_steps, wall_s=round(wall, 1),
                   end_to_end_sps=round(total_steps / wall, 1),
                   solver_iterations=a.solver_iterations, progress=rows),
              open(os.path.join(HERE, a.out), "w"), indent=1)
    json.dump(dict(iter=a.iters - 1, iters=a.iters, env_steps=total_steps,
                   target_env_steps=n_batch * a.iters,
                   est_return_1000=round(rows[-1]["mean_step_reward"] * 1000, 1),
                   env_steps_per_sec=rows[-1]["env_steps_per_sec"],
                   elapsed_s=round(wall, 1), eta_s=0.0, done=True),
              open(os.path.join(HERE, a.out + ".partial"), "w"), indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
