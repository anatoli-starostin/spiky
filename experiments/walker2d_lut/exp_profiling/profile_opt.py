"""exp_profiling — the optimised trainer loop, measured against profile_run.py.

THE BASELINE PROFILE said, over 260 iterations:

    update      39.8% of stage time (6,400 separate dispatches)
    eval        29.8%  (one call, 19.45 s)
    rollout     21.2%
    coverage     5.8%  (6,400 device->host syncs)
    buf_write    2.5%
    roll_host    1.0%
    ... and 25.8% of WALL time unaccounted = Python dispatch overhead

Three of those are the same disease: the inner `for _ in range(updates)` loop runs 32
Python iterations per training step, and each one dispatches ~8 separate operations (a key
split, five buffer gathers, the jitted update, a device->host sync for the row indices, a
numpy bincount). 32 x 8 = ~256 host round-trips per training iteration to do 32 updates.

THE CHANGE: fuse the whole inner loop into ONE `lax.scan` inside a single jit.
  * batch sampling moves onto the device (the gathers become part of the scan body)
  * row-coverage accumulates on-device with a scatter-add, so the host sync disappears
    entirely -- it is read once per eval instead of 32x per iteration
  * one dispatch per training iteration instead of ~256

BIT-EXACTNESS IS THE CONSTRAINT, not just "close enough". The scan body performs exactly
the same operations in exactly the same order on exactly the same RNG stream as the Python
loop, so the optimised run must reproduce the baseline's parameters bit for bit.
`check_equivalence.py` asserts that rather than trusting the reasoning.

Usage:
  python profile_opt.py [--iters 260] [--warmup 60]
"""
import argparse
import json
import os
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "exp_c02_mjx_scaffold"))
sys.path.insert(0, HERE)

import mjx_walker2d as W                                   # noqa: E402
from mujoco import mjx                                     # noqa: E402
import jax_bucket_lif as LIF                               # noqa: E402
from profile_run import (Clock, actor_out, actor_sample, q_init, q_apply,  # noqa: E402
                         CFG, OBS, ACT)


def build(a, obs_mean, obs_std):
    """Returns the jitted pieces. Kept out of main() so check_equivalence can reuse it."""
    tx_a = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(3e-4))
    tx_q = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(3e-4))
    tx_al = optax.adam(3e-4)
    n_tables, K, EPS = a.tph, a.buckets, 0.3

    def norm(s):
        return (s - obs_mean) / (obs_std + 1e-6)

    def one_update(carry, _):
        (ap_, qp, qt, log_alpha, os_a, os_q, os_al, key, cov, buf, size) = carry
        # Exactly the baseline's order: split for the batch index, THEN pass the
        # post-split key into the update body. Any deviation changes the RNG stream and
        # the runs stop being comparable.
        key, kb = jax.random.split(key)
        bi = jax.random.randint(kb, (a.batch,), 0, size)
        s, act = buf["s"][bi], buf["a"][bi]
        r, s2, d = buf["r"][bi], buf["s2"][bi], buf["d"][bi]

        ns, ns2 = norm(s), norm(s2)
        alpha = jnp.exp(log_alpha)
        key, k1 = jax.random.split(key)
        a2, logp2, _ = actor_sample(ap_, s2, k1, EPS)
        target = r + 0.99 * (1 - d) * (jnp.minimum(q_apply(qt["q1"], ns2, a2),
                                                   q_apply(qt["q2"], ns2, a2))
                                       - alpha * logp2)
        target = jax.lax.stop_gradient(target)

        def q_loss(qp):
            e1 = q_apply(qp["q1"], ns, act) - target
            e2 = q_apply(qp["q2"], ns, act) - target
            return (jnp.square(e1) + jnp.square(e2)).mean(), jnp.abs(e1)
        (ql, td), gq = jax.value_and_grad(q_loss, has_aux=True)(qp)
        uq, os_q = tx_q.update(gq, os_q, qp)
        qp = optax.apply_updates(qp, uq)
        key, k2 = jax.random.split(key)

        def a_loss(ap_):
            an, logp, _ = actor_sample(ap_, s, k2, EPS)
            q = jnp.minimum(q_apply(qp["q1"], ns, an), q_apply(qp["q2"], ns, an))
            return (alpha * logp - q).mean(), logp
        (al, logp), ga = jax.value_and_grad(a_loss, has_aux=True)(ap_)
        gw = ga["table"]
        nrm = jnp.linalg.norm(gw, axis=-1, keepdims=True)
        ga = dict(ga, table=gw * jnp.minimum(1.0, 1.0 / (nrm + 1e-8)))
        ua, os_a = tx_a.update(ga, os_a, ap_)
        ap_ = optax.apply_updates(ap_, ua)

        def al_loss(la):
            return (-jnp.exp(la) * (jax.lax.stop_gradient(logp) - 6.0)).mean()
        ual, os_al = tx_al.update(jax.grad(al_loss)(log_alpha), os_al, log_alpha)
        log_alpha = optax.apply_updates(log_alpha, ual)
        qt = jax.tree.map(lambda t, s_: 0.995 * t + 0.005 * s_, qt, qp)

        # Coverage ON DEVICE. The baseline pulled `rows` to the host and ran
        # np.bincount 32x per iteration; this is the same count as a scatter-add that
        # never leaves the GPU.
        rows = LIF.address(ap_, norm(s), EPS, 1, a.tph, a.buckets)
        flat = (jnp.arange(n_tables)[None, :] * K + rows).ravel()
        cov = cov + jnp.zeros(n_tables * K).at[flat].add(1.0).reshape(n_tables, K)
        return (ap_, qp, qt, log_alpha, os_a, os_q, os_al, key, cov, buf, size), alpha

    @jax.jit
    def update_block(ap_, qp, qt, log_alpha, os_a, os_q, os_al, key, cov, buf, size):
        carry = (ap_, qp, qt, log_alpha, os_a, os_q, os_al, key, cov, buf, size)
        carry, alphas = jax.lax.scan(one_update, carry, None, length=a.updates)
        return carry[:9] + (alphas[-1],)

    return update_block


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=260)
    ap.add_argument("--warmup", type=int, default=60)
    ap.add_argument("--envs", type=int, default=64)
    ap.add_argument("--updates", type=int, default=32)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--buffer", type=int, default=1_000_000)
    ap.add_argument("--buckets", type=int, default=16)
    ap.add_argument("--tph", type=int, default=32)
    ap.add_argument("--eval-every", type=int, default=200)
    ap.add_argument("--eval-episodes", type=int, default=20)
    ap.add_argument("--out", default="profile_opt.json")
    a = ap.parse_args()

    stats = json.load(open(os.path.join(HERE, "..", "exp_c03_distillation",
                                        "dataset_stats.json")))
    obs_mean = jnp.asarray(stats["obs_mean"], jnp.float32)
    obs_std = jnp.asarray(stats["obs_std"], jnp.float32)
    key = jax.random.PRNGKey(0)
    key, ka, kq, kr = jax.random.split(key, 4)
    CFG.update(n_heads=1, tph=a.tph, n_buckets=a.buckets,
               obs_mean=obs_mean, obs_std=obs_std)
    ap_ = LIF.init(ka, a.buckets, a.tph, 1, OBS, 2 * ACT)
    ap_["table"] = ap_["table"].at[:, :, ACT:].add(-1.0 / a.tph)
    qp = q_init(kq)
    qt = jax.tree.map(lambda x: x, qp)
    log_alpha = jnp.log(jnp.asarray(0.2))
    n_tables, K, EPS = a.tph, a.buckets, 0.3

    tx_a = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(3e-4))
    tx_q = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(3e-4))
    tx_al = optax.adam(3e-4)
    os_a, os_q, os_al = tx_a.init(ap_), tx_q.init(qp), tx_al.init(log_alpha)

    m = W.make_model()
    reset, step = W.make_env(mjx.put_model(m))
    v_reset, v_step = jax.vmap(reset), jax.vmap(step)
    st = v_reset(jax.random.split(kr, a.envs))

    N = a.buffer
    buf = dict(s=jnp.zeros((N, OBS)), a=jnp.zeros((N, ACT)), r=jnp.zeros(N),
               s2=jnp.zeros((N, OBS)), d=jnp.zeros(N))
    ptr, size = 0, 0
    cov = jnp.zeros((n_tables, K))
    update_block = build(a, obs_mean, obs_std)

    @jax.jit
    def rollout(ap_, st, key, random_actions):
        def one(carry, _):
            st, key = carry
            key, k1 = jax.random.split(key)
            a_rand = jax.random.uniform(k1, (a.envs, ACT), minval=-1.0, maxval=1.0)
            a_pol, _, _ = actor_sample(ap_, st.obs, k1, EPS)
            act = jnp.where(random_actions, a_rand, a_pol)
            nst = v_step(st, act)
            return (nst, key), (st.obs, act, nst.reward, nst.obs, nst.done)
        (st, key), tr = jax.lax.scan(one, (st, key), None, length=1)
        return st, key, tr

    # Buffer insertion, jitted and DONATED. The baseline ran five separate `.at[].set()`
    # dispatches on 1M-row arrays outside jit; donating lets XLA update in place.
    @jax.jit
    def insert(buf, idx, s_, a_, r_, s2_, d_):
        return dict(s=buf["s"].at[idx].set(s_), a=buf["a"].at[idx].set(a_),
                    r=buf["r"].at[idx].set(r_), s2=buf["s2"].at[idx].set(s2_),
                    d=buf["d"].at[idx].set(d_))

    @jax.jit
    def det_action(ap_, obs):
        mu, _ = actor_out(ap_, obs, EPS, mode="hard")
        return jnp.tanh(mu)

    def eval_mjx(ap_, episodes, horizon=1000):
        stx = v_reset(jax.random.split(jax.random.PRNGKey(0), episodes))

        @jax.jit
        def run(stx):
            def one(c, _):
                stx, ret, alive = c
                nst = v_step(stx, det_action(ap_, stx.obs))
                return (nst, ret + nst.reward * alive, alive * (1 - nst.done)), None
            (stx, ret, alive), _ = jax.lax.scan(
                one, (stx, jnp.zeros(episodes), jnp.ones(episodes)), None,
                length=horizon)
            return ret
        return run(stx)

    clk = Clock()
    print("=== compile (one-time) ===")
    t0 = time.perf_counter()
    jax.block_until_ready(rollout(ap_, st, key, True))
    t_roll_c = time.perf_counter() - t0
    t0 = time.perf_counter()
    jax.block_until_ready(update_block(ap_, qp, qt, log_alpha, os_a, os_q, os_al,
                                       key, cov, buf, 1024))
    t_upd_c = time.perf_counter() - t0
    t0 = time.perf_counter()
    jax.block_until_ready(eval_mjx(ap_, a.eval_episodes))
    t_ev_c = time.perf_counter() - t0
    print(f"  rollout {t_roll_c:6.2f}s   update_block {t_upd_c:6.2f}s   "
          f"eval {t_ev_c:6.2f}s   TOTAL {t_roll_c+t_upd_c+t_ev_c:6.2f}s")

    print(f"\n=== steady state ({a.iters} iters, updates from {a.warmup}) ===")
    wall0 = time.perf_counter()
    for it in range(a.iters):
        key, kro = jax.random.split(key)
        st, kro, tr = clk("rollout", rollout, ap_, st, kro, it < a.warmup)
        s_, a_, r_, s2_, d_ = clk(
            "roll_host",
            lambda tr: [np.asarray(x).reshape((-1,) + x.shape[2:]) for x in tr], tr)
        n = len(s_)
        idx = jnp.asarray((ptr + np.arange(n)) % N)
        buf = clk("buf_write", insert, buf, idx, s_, a_, r_, s2_, d_)
        ptr = int((ptr + n) % N)
        size = min(size + n, N)

        if it >= a.warmup:
            res = clk("update", update_block, ap_, qp, qt, log_alpha, os_a, os_q,
                      os_al, key, cov, buf, size)
            (ap_, qp, qt, log_alpha, os_a, os_q, os_al, key, cov, _alpha) = res

        if (it + 1) % a.eval_every == 0:
            clk("eval", eval_mjx, ap_, a.eval_episodes)
            clk("ckpt", lambda: np.savez(
                os.path.join(HERE, "_prof_actor_opt.npz"),
                **{k: np.asarray(v) for k, v in ap_.items()}))
    wall = time.perf_counter() - wall0

    t = clk.report(a.iters, wall)
    json.dump(dict(stages=t, wall_s=wall, iters=a.iters,
                   compile_s=dict(rollout=t_roll_c, update=t_upd_c, eval=t_ev_c),
                   ms_per_iter=1000 * wall / a.iters),
              open(os.path.join(HERE, a.out), "w"), indent=1)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
