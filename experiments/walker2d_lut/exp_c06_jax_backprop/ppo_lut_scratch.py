"""exp_c06 — train a LUT policy FROM SCRATCH with gradients (#75, Phase 4).

No teacher, no distillation: random-init LUT, PPO on the batched MJX rollout loop,
gradients through the verified full-K softmax surrogate (`jax_lut_grad`).

The question: can a LUT be trained end-to-end by backprop — not just filled by
distillation (Phase 1: yes, 99.2% at 5.4k params) and not just evolved (Phase 3: no,
904 in the CPU reference)?

Design notes:
  * the LUT is the POLICY (the thing under test). The critic is a small MLP — a value
    function is scaffolding for the update, not the representation being studied.
  * a learnable per-dimension log-std makes the policy stochastic, as PPO requires; the
    LUT emits the mean.
  * the table is initialised small (0.05) so the initial policy is near-zero-action
    rather than saturated bang-bang.

Usage:
  XLA_PYTHON_CLIENT_PREALLOCATE=false python ppo_lut_scratch.py --iters 600 --nap 8 --tph 16
"""
import argparse, json, os, sys, time
from typing import NamedTuple

import jax, jax.numpy as jnp
import numpy as np
import optax

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "exp_c02_mjx_scaffold"))

import mjx_walker2d as W          # noqa: E402
from mujoco import mjx           # noqa: E402
import jax_lut_grad as L         # noqa: E402

OBS, ACT = 17, 6


class Tr(NamedTuple):
    obs: jnp.ndarray
    act: jnp.ndarray
    logp: jnp.ndarray
    val: jnp.ndarray
    rew: jnp.ndarray
    done: jnp.ndarray


def critic_init(key, hidden=64):
    k1, k2, k3 = jax.random.split(key, 3)
    return dict(w1=jax.random.normal(k1, (OBS, hidden)) * 0.1,
                b1=jnp.zeros(hidden),
                w2=jax.random.normal(k2, (hidden, hidden)) * 0.1,
                b2=jnp.zeros(hidden),
                w3=jax.random.normal(k3, (hidden, 1)) * 0.01,
                b3=jnp.zeros(1))


def critic(p, x):
    h = jnp.tanh(x @ p["w1"] + p["b1"])
    h = jnp.tanh(h @ p["w2"] + p["b2"])
    return jnp.squeeze(h @ p["w3"] + p["b3"], -1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=600)
    ap.add_argument("--num-envs", type=int, default=1024)
    ap.add_argument("--rollout", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--minibatches", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--nap", type=int, default=8)
    ap.add_argument("--tph", type=int, default=16)
    ap.add_argument("--heads", type=int, default=1)
    ap.add_argument("--out", default="ppo_lut_scratch.json")
    a = ap.parse_args()

    stats = json.load(open(os.path.join(HERE, "..", "exp_c03_distillation",
                                        "dataset_stats.json")))
    obs_mean = jnp.asarray(stats["obs_mean"], jnp.float32)
    obs_std = jnp.asarray(stats["obs_std"], jnp.float32)

    key = jax.random.PRNGKey(0)
    key, kl, kc, kr = jax.random.split(key, 4)
    lut = L.init(kl, a.nap, a.tph, a.heads, OBS, ACT, obs_mean, obs_std)
    params = dict(lut={k: v for k, v in lut.items()
                       if k not in ("n_heads", "tph", "obs_mean", "obs_std")},
                  logstd=jnp.zeros(ACT) - 0.5,
                  critic=critic_init(kc))
    static = dict(n_heads=a.heads, tph=a.tph, obs_mean=obs_mean, obs_std=obs_std)

    n_table = int(np.prod(lut["weights"].shape))
    n_idx = int(np.prod(lut["w"].shape) + np.prod(lut["b"].shape))
    print(f"LUT policy nap={a.nap} tph={a.tph} heads={a.heads} | table {n_table:,} "
          f"+ addressing {n_idx:,} = {n_table+n_idx:,} params | "
          f"{jax.devices()[0].device_kind}", flush=True)

    def act_mean(p, obs):
        full = dict(p["lut"], **static)
        return L.policy(full, obs)

    def logp_of(p, obs, act):
        mu = act_mean(p, obs)
        std = jnp.exp(p["logstd"])
        return (-0.5 * jnp.square((act - mu) / std) - p["logstd"]
                - 0.5 * jnp.log(2 * jnp.pi)).sum(-1)

    m = W.make_model()
    mx = mjx.put_model(m)
    reset, step = W.make_env(mx)
    v_reset, v_step = jax.vmap(reset), jax.vmap(step)
    st0 = v_reset(jax.random.split(kr, a.num_envs))

    n_batch = a.num_envs * a.rollout
    mb = n_batch // a.minibatches
    tx = optax.chain(optax.clip_by_global_norm(0.5), optax.adam(a.lr))
    opt_state = tx.init(params)

    def rollout(params, st, key):
        def one(carry, _):
            params, st, key = carry
            key, sub = jax.random.split(key)
            mu = act_mean(params, st.obs)
            std = jnp.exp(params["logstd"])
            act = mu + std * jax.random.normal(sub, mu.shape)
            lp = (-0.5 * jnp.square((act - mu) / std) - params["logstd"]
                  - 0.5 * jnp.log(2 * jnp.pi)).sum(-1)
            val = critic(params["critic"],
                         (st.obs - obs_mean) / (obs_std + 1e-6))
            nst = v_step(st, act)
            return (params, nst, key), Tr(st.obs, act, lp, val, nst.reward, nst.done)
        (params, st, key), tr = jax.lax.scan(one, (params, st, key), None,
                                             length=a.rollout)
        last_v = critic(params["critic"], (st.obs - obs_mean) / (obs_std + 1e-6))
        return st, tr, last_v, key

    def gae(tr, last_v):
        def one(carry, xs):
            adv, nv = carry
            rew, val, done = xs
            delta = rew + a.gamma * nv * (1 - done) - val
            adv = delta + a.gamma * a.lam * (1 - done) * adv
            return (adv, val), adv
        _, advs = jax.lax.scan(one, (jnp.zeros_like(last_v), last_v),
                               (tr.rew, tr.val, tr.done), reverse=True)
        return advs, advs + tr.val

    def loss_fn(p, obs, act, old_lp, adv, ret):
        lp = logp_of(p, obs, act)
        ratio = jnp.exp(lp - old_lp)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        pl = -jnp.minimum(ratio * adv,
                          jnp.clip(ratio, 1 - a.clip, 1 + a.clip) * adv).mean()
        v = critic(p["critic"], (obs - obs_mean) / (obs_std + 1e-6))
        vl = jnp.square(v - ret).mean()
        ent = (p["logstd"] + 0.5 * jnp.log(2 * jnp.pi * jnp.e)).sum()
        return pl + 0.5 * vl - 1e-3 * ent

    @jax.jit
    def train_iter(params, opt_state, st, key):
        st, tr, last_v, key = rollout(params, st, key)
        adv, ret = gae(tr, last_v)
        flat = jax.tree.map(lambda z: z.reshape((n_batch,) + z.shape[2:]),
                            (tr.obs, tr.act, tr.logp, adv, ret))

        def epoch(carry, _):
            params, opt_state, key = carry
            key, sub = jax.random.split(key)
            perm = jax.random.permutation(sub, n_batch)
            shuf = jax.tree.map(
                lambda z: z[perm].reshape((a.minibatches, mb) + z.shape[1:]), flat)

            def upd(c, batch):
                params, opt_state = c
                l, g = jax.value_and_grad(loss_fn)(params, *batch)
                updates, opt_state = tx.update(g, opt_state, params)
                return (optax.apply_updates(params, updates), opt_state), l
            (params, opt_state), ls = jax.lax.scan(upd, (params, opt_state), shuf)
            return (params, opt_state, key), ls
        (params, opt_state, key), ls = jax.lax.scan(
            epoch, (params, opt_state, key), None, length=a.epochs)
        return params, opt_state, st, key, tr, ls

    rows, t0, total = [], time.time(), 0
    st = st0
    for it in range(a.iters):
        key, sub = jax.random.split(key)
        params, opt_state, st, key, tr, ls = train_iter(params, opt_state, st, sub)
        jax.block_until_ready(params["logstd"])
        total += n_batch
        rew = float(tr.rew.mean())
        el = time.time() - t0
        rows.append(dict(iter=it, env_steps=total, mean_step_reward=rew,
                         proxy_return=round(rew * 1000, 1), loss=float(ls.mean()),
                         elapsed_s=round(el, 1)))
        if it % 10 == 0 or it == a.iters - 1:
            eta = (a.iters - it - 1) * el / (it + 1)
            print(f"[iter {it:>4}/{a.iters}] steps {total:>11,} | rew/step {rew:+.4f} "
                  f"(proxy ~{rew*1000:7.1f}) | loss {float(ls.mean()):+.3f} | "
                  f"{total/el:>8,.0f} sps | ETA {eta/60:5.1f}m", flush=True)
            json.dump(dict(iter=it, iters=a.iters, env_steps=total,
                           target_env_steps=n_batch * a.iters,
                           proxy_return=round(rew * 1000, 1),
                           env_steps_per_sec=round(total / el, 1),
                           eta_s=round(eta, 1), done=False),
                      open(os.path.join(HERE, a.out + ".partial"), "w"), indent=1)

    np.savez(os.path.join(HERE, "lut_scratch_params.npz"),
             **{k: np.asarray(v) for k, v in params["lut"].items()},
             logstd=np.asarray(params["logstd"]),
             obs_mean=np.asarray(obs_mean), obs_std=np.asarray(obs_std),
             n_heads=np.int32(a.heads), tph=np.int32(a.tph))
    json.dump(dict(nap=a.nap, tph=a.tph, heads=a.heads,
                   table_params=n_table, index_params=n_idx,
                   total_env_steps=total, wall_s=round(time.time() - t0, 1),
                   history=rows),
              open(os.path.join(HERE, a.out), "w"), indent=1)
    json.dump(dict(iter=a.iters - 1, iters=a.iters, env_steps=total,
                   target_env_steps=n_batch * a.iters,
                   proxy_return=rows[-1]["proxy_return"],
                   env_steps_per_sec=round(total / (time.time() - t0), 1),
                   eta_s=0.0, done=True),
              open(os.path.join(HERE, a.out + ".partial"), "w"), indent=1)
    print(f"done: {total:,} env-steps in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
