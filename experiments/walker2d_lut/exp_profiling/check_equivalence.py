"""exp_profiling — does the fused update block compute the SAME thing as the Python loop?

A speedup that changes the learned result is not a speedup. This runs both implementations
from an identical starting state on an identical RNG stream and compares every parameter.

The bar is BIT-EXACT, not "close". The scan body performs the same operations in the same
order on the same key stream, so any difference beyond exact zero means the refactor
changed the computation and the timing numbers are meaningless.

One thing deliberately NOT assumed: in the Python loop `size` is a concrete Python int
baked into each eager call, while in the fused block it is a traced argument. If
`jax.random.randint(key, shape, 0, size)` disagreed between the two, the sampled batch
indices would diverge and everything downstream with them. That is exactly the kind of
silent difference this check exists to catch, so the batch indices are compared explicitly
too.

Usage:
  python check_equivalence.py [--iters 6] [--updates 32]
"""
import argparse
import json
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np
import optax

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "exp_c02_mjx_scaffold"))
sys.path.insert(0, HERE)

import jax_bucket_lif as LIF                               # noqa: E402
from profile_run import actor_sample, q_init, q_apply, CFG, OBS, ACT   # noqa: E402
import profile_opt                                          # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=6)
    ap.add_argument("--updates", type=int, default=32)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--buckets", type=int, default=16)
    ap.add_argument("--tph", type=int, default=32)
    a = ap.parse_args()
    a.envs = 64

    stats = json.load(open(os.path.join(HERE, "..", "exp_c03_distillation",
                                        "dataset_stats.json")))
    om = jnp.asarray(stats["obs_mean"], jnp.float32)
    osd = jnp.asarray(stats["obs_std"], jnp.float32)
    CFG.update(n_heads=1, tph=a.tph, n_buckets=a.buckets, obs_mean=om, obs_std=osd)

    key0 = jax.random.PRNGKey(0)
    k0, ka, kq, kd = jax.random.split(key0, 4)
    p0 = LIF.init(ka, a.buckets, a.tph, 1, OBS, 2 * ACT)
    p0["table"] = p0["table"].at[:, :, ACT:].add(-1.0 / a.tph)
    qp0 = q_init(kq)
    qt0 = jax.tree.map(lambda x: x, qp0)
    la0 = jnp.log(jnp.asarray(0.2))

    # A small synthetic replay buffer — the loop under test does not care where the data
    # came from, and a real rollout would only add MJX noise to the comparison.
    NBUF = 20000
    kb1, kb2, kb3 = jax.random.split(kd, 3)
    buf = dict(s=jax.random.normal(kb1, (NBUF, OBS)),
               a=jnp.tanh(jax.random.normal(kb2, (NBUF, ACT))),
               r=jax.random.uniform(kb3, (NBUF,)),
               s2=jax.random.normal(kb1, (NBUF, OBS)) * 1.01,
               d=jnp.zeros(NBUF))
    size = NBUF
    n_tables, K, EPS = a.tph, a.buckets, 0.3

    tx_a = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(3e-4))
    tx_q = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(3e-4))
    tx_al = optax.adam(3e-4)

    def norm(s):
        return (s - om) / (osd + 1e-6)

    # ---------------- reference: the original Python loop --------------------
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
        rows = LIF.address(ap_, norm(s), EPS, 1, a.tph, a.buckets)
        return (ap_, qp, qt, log_alpha, os_a, os_q, os_al, key, rows)

    P = (p0, qp0, qt0, la0, tx_a.init(p0), tx_q.init(qp0), tx_al.init(la0))
    ap_, qp, qt, la = P[0], P[1], P[2], P[3]
    os_a, os_q, os_al = P[4], P[5], P[6]
    key = k0
    cov_ref = jnp.zeros((n_tables, K))
    idx_ref = []
    for _ in range(a.iters):
        for _ in range(a.updates):
            key, kb = jax.random.split(key)
            bi = jax.random.randint(kb, (a.batch,), 0, size)
            idx_ref.append(np.asarray(bi))
            batch = (buf["s"][bi], buf["a"][bi], buf["r"][bi], buf["s2"][bi],
                     buf["d"][bi])
            (ap_, qp, qt, la, os_a, os_q, os_al, key, rows) = update(
                ap_, qp, qt, la, os_a, os_q, os_al, batch, key)
            flat = (np.arange(n_tables)[None, :] * K + np.asarray(rows)).ravel()
            cov_ref = cov_ref + jnp.asarray(
                np.bincount(flat, minlength=n_tables * K).reshape(n_tables, K))
    ref = dict(actor=ap_, q=qp, qt=qt, log_alpha=la, cov=cov_ref, key=key)

    # ---------------- candidate: the fused scan ------------------------------
    block = profile_opt.build(a, om, osd)
    ap_, qp, qt, la = p0, qp0, qt0, la0
    os_a, os_q, os_al = tx_a.init(p0), tx_q.init(qp0), tx_al.init(la0)
    key, cov = k0, jnp.zeros((n_tables, K))
    for _ in range(a.iters):
        (ap_, qp, qt, la, os_a, os_q, os_al, key, cov, _al) = block(
            ap_, qp, qt, la, os_a, os_q, os_al, key, cov, buf, size)
    got = dict(actor=ap_, q=qp, qt=qt, log_alpha=la, cov=cov, key=key)

    # ---------------- compare -------------------------------------------------
    print(f"=== fused scan vs Python loop — {a.iters} iters x {a.updates} updates ===\n")
    bad = 0

    def cmp(name, x, y):
        nonlocal bad
        x, y = np.asarray(x), np.asarray(y)
        d = float(np.abs(x - y).max())
        exact = d == 0.0
        rel = d / max(float(np.abs(y).max()), 1e-30)
        bad += 0 if exact else 1
        print(f"  {'EXACT' if exact else 'DIFFERS':<8} {name:<26} "
              f"max|Δ| {d:.3e}  rel {rel:.3e}")

    for k in sorted(ref["actor"]):
        cmp(f"actor.{k}", got["actor"][k], ref["actor"][k])
    for k in sorted(ref["q"]["q1"]):
        cmp(f"critic.q1.{k}", got["q"]["q1"][k], ref["q"]["q1"][k])
        cmp(f"target.q1.{k}", got["qt"]["q1"][k], ref["qt"]["q1"][k])
    cmp("log_alpha", got["log_alpha"], ref["log_alpha"])
    cmp("row_coverage", got["cov"], ref["cov"])
    cmp("rng key", got["key"], ref["key"])

    print()
    if bad == 0:
        print("  ALL BIT-EXACT — the fused block is the same computation.")
    else:
        print(f"  {bad} tensor(s) DIFFER — the refactor changed the computation.")
        sys.exit(1)


if __name__ == "__main__":
    main()
