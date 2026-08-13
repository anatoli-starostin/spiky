"""exp_profiling — where does a Walker2d bucket-LIF SAC iteration actually spend its time?

Instruments the exp_c32b trainer stage by stage. Every stage is closed with
`block_until_ready`, so the attribution is honest rather than an artefact of JAX's async
dispatch — without it, work queued in one stage is billed to whichever later stage happens
to force it.

Two costs are separated because they scale completely differently:
  * COMPILE — paid once, and only matters for short runs.
  * STEADY STATE — per-iteration, and is what a 10,000-iteration run is made of.

Stages, matching the structure of the training loop:
  rollout        jitted env step + policy (MJX)
  roll_host      pulling the rollout to the host as numpy
  buf_write      the replay-buffer functional update
  update         the `--updates` SAC steps (critic + actor + alpha + target)
  coverage       the row-coverage bookkeeping after each update
  eval           the 20-episode MJX proxy
  ckpt           np.savez of the actor

Usage:
  python profile_run.py [--iters 260] [--warmup 60] [--variant base|opt]
"""
import argparse
import collections
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

OBS, ACT = 17, 6
LOGSTD_MIN, LOGSTD_MAX = -5.0, 2.0
CFG = {}


class Clock:
    """Accumulate wall time per stage, forcing completion at each boundary.

    CAVEAT, learned the hard way: `block_until_ready` at a stage boundary bills that stage
    for draining everything JAX had already queued. In the first baseline profile the
    rollout stage read 51 ms/iter, but the same jitted rollout measured standalone costs
    8.9 ms — the extra 42 ms was the previous iteration's update block finishing. Stage
    attribution is therefore an UPPER bound per stage and the shares are only indicative.

    Set PROFILE_NOSYNC=1 to skip the per-stage sync entirely. That destroys the breakdown
    but leaves the loop running exactly as it does in production, so the wall time is the
    honest number to quote for a speedup.
    """

    NOSYNC = os.environ.get("PROFILE_NOSYNC") == "1"

    def __init__(self):
        self.t = collections.defaultdict(float)
        self.n = collections.defaultdict(int)

    def __call__(self, name, fn, *a, **k):
        if self.NOSYNC:
            return fn(*a, **k)
        t0 = time.perf_counter()
        out = fn(*a, **k)
        jax.block_until_ready(out)
        self.t[name] += time.perf_counter() - t0
        self.n[name] += 1
        return out

    def report(self, iters, total):
        if self.NOSYNC:
            print(f"\n  PROFILE_NOSYNC=1 — no per-stage breakdown.")
            print(f"  {'wall':<12}{total:>10.2f}{1000*total/iters:>10.3f} ms/iter")
            return {}
        rows = sorted(self.t.items(), key=lambda kv: -kv[1])
        acc = sum(self.t.values())
        print(f"\n  {'stage':<12}{'total s':>10}{'ms/iter':>10}{'% of stages':>13}"
              f"{'calls':>9}")
        for k, v in rows:
            print(f"  {k:<12}{v:>10.2f}{1000*v/iters:>10.3f}{100*v/acc:>12.1f}%"
                  f"{self.n[k]:>9,}")
        print(f"  {'-'*54}")
        print(f"  {'stages':<12}{acc:>10.2f}{1000*acc/iters:>10.3f}{100.0:>12.1f}%")
        print(f"  {'wall':<12}{total:>10.2f}{1000*total/iters:>10.3f}"
              f"   (unaccounted {100*(total-acc)/total:.1f}% = python overhead)")
        return dict(self.t)


def actor_out(p, obs, eps, mode="st"):
    x = (obs - CFG["obs_mean"]) / (CFG["obs_std"] + 1e-6)
    y = LIF.apply(p, x, eps, CFG["n_heads"], CFG["tph"], CFG["n_buckets"],
                  mode=mode).sum(1)
    return y[:, :ACT], jnp.clip(y[:, ACT:], LOGSTD_MIN, LOGSTD_MAX)


def actor_sample(p, obs, key, eps):
    mu, log_std = actor_out(p, obs, eps, mode="st")
    e = jax.random.normal(key, mu.shape)
    a = jnp.tanh(mu + jnp.exp(log_std) * e)
    logp = (-0.5 * jnp.square(e) - log_std - 0.5 * jnp.log(2 * jnp.pi)).sum(-1)
    logp -= jnp.log(1.0 - jnp.square(a) + 1e-6).sum(-1)
    return a, logp, jnp.tanh(mu)


def q_init(key, hidden=256):
    def one(k):
        k1, k2, k3 = jax.random.split(k, 3)
        return dict(w1=jax.random.normal(k1, (OBS + ACT, hidden)) * 0.1,
                    b1=jnp.zeros(hidden),
                    w2=jax.random.normal(k2, (hidden, hidden)) * 0.1,
                    b2=jnp.zeros(hidden),
                    w3=jax.random.normal(k3, (hidden, 1)) * 0.01, b3=jnp.zeros(1))
    k1, k2 = jax.random.split(key)
    return dict(q1=one(k1), q2=one(k2))


def q_apply(qp, s, a):
    h = jnp.concatenate([s, a], -1)
    h = jax.nn.relu(h @ qp["w1"] + qp["b1"])
    h = jax.nn.relu(h @ qp["w2"] + qp["b2"])
    return jnp.squeeze(h @ qp["w3"] + qp["b3"], -1)


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
    ap.add_argument("--dev-coverage", action="store_true",
                    help="accumulate row coverage on-device with a scatter-add instead "
                         "of pulling `rows` to the host and running np.bincount 32x per "
                         "iteration. Ablation for the fused loop's second component.")
    ap.add_argument("--jit-insert", action="store_true",
                    help="jit the replay-buffer write instead of five separate eager "
                         "`.at[].set()` dispatches on 1M-row arrays.")
    ap.add_argument("--out", default="profile_base.json")
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
    n_tables, K = a.tph, a.buckets
    EPS = 0.3

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
    row_updates = jnp.zeros((n_tables, K))

    def norm(s):
        return (s - obs_mean) / (obs_std + 1e-6)

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

    @jax.jit
    def update(ap_, qp, qt, log_alpha, os_a, os_q, os_al, batch, key):
        s, act, r, s2, d = batch
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
        x = (s - obs_mean) / (obs_std + 1e-6)
        rows = LIF.address(ap_, x, EPS, 1, a.tph, a.buckets)
        return (ap_, qp, qt, log_alpha, os_a, os_q, os_al, key,
                dict(alpha=alpha, rows=rows))

    @jax.jit
    def insert(buf, idx, s_, a_, r_, s2_, d_):
        return dict(s=buf["s"].at[idx].set(s_), a=buf["a"].at[idx].set(a_),
                    r=buf["r"].at[idx].set(r_), s2=buf["s2"].at[idx].set(s2_),
                    d=buf["d"].at[idx].set(d_))

    @jax.jit
    def dev_cov(rows):
        flat = (jnp.arange(n_tables)[None, :] * K + rows).ravel()
        return jnp.zeros(n_tables * K).at[flat].add(1.0).reshape(n_tables, K)

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
    # ---- one-time compile cost, measured separately -------------------------
    print("=== compile (one-time) ===")
    t0 = time.perf_counter()
    st2, k2, tr = rollout(ap_, st, key, True)
    jax.block_until_ready(tr)
    t_roll_c = time.perf_counter() - t0
    bi = jnp.arange(a.batch)
    batch0 = (buf["s"][bi], buf["a"][bi], buf["r"][bi], buf["s2"][bi], buf["d"][bi])
    t0 = time.perf_counter()
    out = update(ap_, qp, qt, log_alpha, os_a, os_q, os_al, batch0, key)
    jax.block_until_ready(out)
    t_upd_c = time.perf_counter() - t0
    t0 = time.perf_counter()
    jax.block_until_ready(eval_mjx(ap_, a.eval_episodes))
    t_ev_c = time.perf_counter() - t0
    print(f"  rollout {t_roll_c:6.2f}s   update {t_upd_c:6.2f}s   eval {t_ev_c:6.2f}s"
          f"   TOTAL {t_roll_c+t_upd_c+t_ev_c:6.2f}s")

    # ---- steady state --------------------------------------------------------
    print(f"\n=== steady state ({a.iters} iters, updates from {a.warmup}) ===")
    wall0 = time.perf_counter()
    for it in range(a.iters):
        key, kro = jax.random.split(key)
        st, kro, tr = clk("rollout", rollout, ap_, st, kro, it < a.warmup)
        s_, a_, r_, s2_, d_ = clk(
            "roll_host",
            lambda tr: [np.asarray(x).reshape((-1,) + x.shape[2:]) for x in tr], tr)
        n = len(s_)
        idx = (ptr + np.arange(n)) % N
        if a.jit_insert:
            buf = clk("buf_write", insert, buf, jnp.asarray(idx), s_, a_, r_, s2_, d_)
        else:
            buf = clk("buf_write", lambda: dict(
                s=buf["s"].at[idx].set(s_), a=buf["a"].at[idx].set(a_),
                r=buf["r"].at[idx].set(r_), s2=buf["s2"].at[idx].set(s2_),
                d=buf["d"].at[idx].set(d_)))
        ptr = int((ptr + n) % N)
        size = min(size + n, N)

        if it >= a.warmup:
            for _ in range(a.updates):
                key, kb = jax.random.split(key)
                bi = jax.random.randint(kb, (a.batch,), 0, size)
                batch = (buf["s"][bi], buf["a"][bi], buf["r"][bi],
                         buf["s2"][bi], buf["d"][bi])
                res = clk("update", update, ap_, qp, qt, log_alpha, os_a, os_q,
                          os_al, batch, key)
                (ap_, qp, qt, log_alpha, os_a, os_q, os_al, key, info) = res

                if a.dev_coverage:
                    row_updates = row_updates + clk("coverage", dev_cov, info["rows"])
                else:
                    def cov(info=info):
                        rr = np.asarray(info["rows"])
                        flat = (np.arange(n_tables)[None, :] * K + rr).ravel()
                        return jnp.asarray(np.bincount(
                            flat, minlength=n_tables * K).reshape(n_tables, K))
                    row_updates = row_updates + clk("coverage", cov)

        if (it + 1) % a.eval_every == 0:
            clk("eval", eval_mjx, ap_, a.eval_episodes)
            clk("ckpt", lambda: np.savez(
                os.path.join(HERE, "_prof_actor.npz"),
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
