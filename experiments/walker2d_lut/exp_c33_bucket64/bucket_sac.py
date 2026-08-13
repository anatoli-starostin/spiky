"""exp_c32 — Bucket-LIF SAC: the exp_c09 LUT-SAC trainer with a bucket-addressed actor (#75).

A fork of exp_c31/pure_lif_sac.py with the front-end module swapped. Everything else is
deliberately untouched -- twin-Q MLP critic, replay, learned per-cell log-sigma, the trust
region on row deltas, learning rates, target entropy, tau, gamma, batch, warmup -- so any
difference is attributable to the front-end.

PARAMETER COUNT: 7,840 total = 1,696 front-end + 6,144 table. This is BY FAR the smallest
actor in the chapter:

    model                     front-end   table    total   total vs c18
    exp_c32 bucket (16 bkt)       1,696   6,144    7,840          0.28x
    exp_c18 hyperplane            3,456  24,576   28,032          1.00x
    exp_c31 PureLIF               6,816  24,576   31,392          1.12x
    exp_c30b factorised-P        23,617  24,576   48,193          1.72x
    exp_c30  dense-P             62,785  24,576   87,361          3.12x

NOT param-matched, and not forced to be -- Anatoli asked for a small configuration and this
is what 16 buckets x 32 tables costs. Note the table shrank too: 16 rows per table instead
of 64, because rows = n_buckets rather than 2**nap. So this experiment changes TWO things
at once relative to exp_c31 (addressing scheme AND capacity), and a poor result cannot be
attributed to either alone. That is a deliberate consequence of the requested config, not
an oversight; the clean follow-up if it underperforms is 64 buckets, which restores the row
count at ~10.5k params.

WHAT IS ACTUALLY NEW HERE, beyond size: the address is a MONOTONE QUANTISATION of one
scalar. Every other front-end in this chapter addressed with a set of independent sign
tests, so rows were an unordered set and row 5 had nothing to do with row 4. Here row m
means "the neuron fired in the m-th time interval", the boundaries are trainable and kept
sorted by construction, and adjacent rows are adjacent in time. The trust region on row
deltas is doing something different in that geometry -- neighbouring rows now hold
genuinely related actions -- and `bucket-spread` is logged every eval to show whether the
policy uses the ordering or collapses onto a couple of intervals.

`eps` IS INERT, as in exp_c31, and run_parity.sh asserts it (0.0 sensitivity between
eps=0.7 and eps=0.05 on both forward modes). Sharpness lives in two trainable per-LUT
parameters, `T_cross` (crossing sharpness) and `T_bkt` (bucket-partition softness), both
starting at 1.0 and neither frozen. There is no schedule and nothing to anneal, so -- as in
exp_c31 -- the terminal sharpening dip that hit 6 of 6 runs in exp_c30/c30b cannot occur by
construction.

COLD START. At init the boundaries are evenly spaced every 2.0 across (0, 32) but the
membrane rarely crosses the fixed theta_mem = 1.0, so non-firing neurons fold into the LAST
bucket: the torch reference reaches only 7 of 16 buckets at init. `bucket-cov` and
`bkt-spread` are logged so a run that never leaves that corner is visible immediately.
"""
import argparse, json, os, sys, time
from functools import partial

import jax, jax.numpy as jnp
import numpy as np
import optax

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "exp_c02_mjx_scaffold"))
sys.path.insert(0, HERE)

import mjx_walker2d as W          # noqa: E402
from mujoco import mjx           # noqa: E402
import jax_bucket_lif as LIF     # noqa: E402

OBS, ACT = 17, 6
LOGSTD_MIN, LOGSTD_MAX = -5.0, 2.0

CFG = {}


def actor_init(key, n_buckets, tph, heads):
    p = LIF.init(key, n_buckets, tph, heads, OBS, 2 * ACT)
    # Same modest, uniform starting spread as exp_c09: bias the log-sigma half of every
    # row so the tph tables sum to -1.0 rather than to 0.
    p["table"] = p["table"].at[:, :, ACT:].add(-1.0 / (heads * tph))
    return p


def actor_out(p, obs, eps, mode="st"):
    x = (obs - CFG["obs_mean"]) / (CFG["obs_std"] + 1e-6)
    y = LIF.apply(p, x, eps, CFG["n_heads"], CFG["tph"], CFG["n_buckets"],
                  mode=mode).sum(1)                                  # [B, 12]
    mu, log_std = y[:, :ACT], jnp.clip(y[:, ACT:], LOGSTD_MIN, LOGSTD_MAX)
    return mu, log_std


def actor_sample(p, obs, key, eps):
    """Reparameterised tanh-Gaussian: returns (action, logp, mu_tanh)."""
    mu, log_std = actor_out(p, obs, eps, mode="st")
    std = jnp.exp(log_std)
    e = jax.random.normal(key, mu.shape)
    pre = mu + std * e
    a = jnp.tanh(pre)
    logp = (-0.5 * jnp.square(e) - log_std - 0.5 * jnp.log(2 * jnp.pi)).sum(-1)
    logp -= jnp.log(1.0 - jnp.square(a) + 1e-6).sum(-1)
    return a, logp, jnp.tanh(mu)


def actor_rows(p, obs, eps):
    """The bucket each table addresses — the coverage diagnostic. [B, n_tables]"""
    x = (obs - CFG["obs_mean"]) / (CFG["obs_std"] + 1e-6)
    return LIF.address(p, x, eps, CFG["n_heads"], CFG["tph"], CFG["n_buckets"])


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
    ap.add_argument("--iters", type=int, default=10000)
    ap.add_argument("--envs", type=int, default=64)
    ap.add_argument("--rollout", type=int, default=1)
    ap.add_argument("--updates", type=int, default=32)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--buffer", type=int, default=1_000_000)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--actor-lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--tau", type=float, default=0.005)
    ap.add_argument("--buckets", type=int, default=64)
    ap.add_argument("--tph", type=int, default=32)
    ap.add_argument("--heads", type=int, default=1)
    ap.add_argument("--target-entropy", type=float, default=-6.0)
    ap.add_argument("--row-clip", type=float, default=1.0)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--eval-episodes", type=int, default=20)
    ap.add_argument("--eval-eps", type=float, default=0.3,
                    help="INERT — the module ignores eps (run_parity.sh verifies 0.0 "
                         "sensitivity). Recorded in the checkpoint for schema parity.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out_name = a.out or f"bucket_sac{a.tag}"

    stats = json.load(open(os.path.join(HERE, "..", "exp_c03_distillation",
                                        "dataset_stats.json")))
    obs_mean = jnp.asarray(stats["obs_mean"], jnp.float32)
    obs_std = jnp.asarray(stats["obs_std"], jnp.float32)

    key = jax.random.PRNGKey(a.seed)
    key, ka, kq, kr = jax.random.split(key, 4)
    CFG.update(n_heads=a.heads, tph=a.tph, n_buckets=a.buckets,
               obs_mean=obs_mean, obs_std=obs_std)
    ap_ = actor_init(ka, a.buckets, a.tph, a.heads)

    qp = q_init(kq)
    qt = jax.tree.map(lambda x: x, qp)
    log_alpha = jnp.log(jnp.asarray(0.2))

    n_tables = a.heads * a.tph
    K = a.buckets
    n_det, n_tab = LIF.n_params(ap_)
    print(f"Bucket-LIF SAC actor buckets={a.buckets} tph={a.tph} heads={a.heads} "
          f"seed={a.seed} | 12 outputs/cell (6 mu + 6 log-sigma) | front-end {n_det:,} + "
          f"table {n_tab:,} = {n_det + n_tab:,} params "
          f"({100*(n_det+n_tab)/28032:.1f}% of the 28,032 hyperplane baseline; front-end "
          f"{100*n_det/3456:.1f}% of its 3,456) | rows {n_tables}x{K} = {n_tables*K:,} | "
          f"eps INERT; sharpness is trainable T_cross/T_bkt", flush=True)

    tx_a = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(a.actor_lr))
    tx_q = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(a.lr))
    tx_al = optax.adam(a.lr)
    os_a, os_q, os_al = tx_a.init(ap_), tx_q.init(qp), tx_al.init(log_alpha)

    m = W.make_model()
    mx = mjx.put_model(m)
    reset, step = W.make_env(mx)
    v_reset, v_step = jax.vmap(reset), jax.vmap(step)
    st = v_reset(jax.random.split(kr, a.envs))

    N = a.buffer
    buf = dict(s=jnp.zeros((N, OBS)), a=jnp.zeros((N, ACT)), r=jnp.zeros(N),
               s2=jnp.zeros((N, OBS)), d=jnp.zeros(N))
    ptr, size = 0, 0
    row_updates = jnp.zeros((n_tables, K))
    EPS = a.eval_eps                      # inert; passed only for signature parity

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
        (st, key), tr = jax.lax.scan(one, (st, key), None, length=a.rollout)
        return st, key, tr

    @partial(jax.jit, static_argnums=())
    def update(ap_, qp, qt, log_alpha, os_a, os_q, os_al, batch, key):
        s, act, r, s2, d = batch
        ns, ns2 = norm(s), norm(s2)
        alpha = jnp.exp(log_alpha)

        key, k1 = jax.random.split(key)
        a2, logp2, _ = actor_sample(ap_, s2, k1, EPS)
        q1t = q_apply(qt["q1"], ns2, a2)
        q2t = q_apply(qt["q2"], ns2, a2)
        target = r + a.gamma * (1 - d) * (jnp.minimum(q1t, q2t) - alpha * logp2)
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

        # Per-row trust region, as in exp_c09. Applies to the table only; the LIF
        # front-end is continuous and has no row semantics.
        if a.row_clip > 0:
            gw = ga["table"]
            nrm = jnp.linalg.norm(gw, axis=-1, keepdims=True)
            ga = dict(ga, table=gw * jnp.minimum(1.0, a.row_clip / (nrm + 1e-8)))
        ua, os_a = tx_a.update(ga, os_a, ap_)
        ap_ = optax.apply_updates(ap_, ua)

        def al_loss(log_alpha):
            return (-jnp.exp(log_alpha)
                    * (jax.lax.stop_gradient(logp) + a.target_entropy)).mean()
        gal = jax.grad(al_loss)(log_alpha)
        ual, os_al = tx_al.update(gal, os_al, log_alpha)
        log_alpha = optax.apply_updates(log_alpha, ual)

        qt = jax.tree.map(lambda t, s_: (1 - a.tau) * t + a.tau * s_, qt, qp)
        rows = actor_rows(ap_, s, EPS)
        return (ap_, qp, qt, log_alpha, os_a, os_q, os_al, key,
                dict(q_loss=ql, a_loss=al, alpha=alpha, logp=logp.mean(),
                     td=td.mean(), rows=rows))

    @jax.jit
    def det_action(ap_, obs):
        mu, _ = actor_out(ap_, obs, EPS, mode="hard")
        return jnp.tanh(mu)

    @jax.jit
    def bucket_stats(ap_, obs):
        """(mean bucket index, sd across tables of the per-table mean).

        The cold-start diagnostic. Non-firing neurons fold into the LAST bucket, so a run
        stuck near index 15 has a membrane that never crosses threshold; one stuck at a
        single index anywhere is using a 1-row table."""
        r = actor_rows(ap_, obs, EPS).astype(jnp.float32)
        return r.mean(), r.std()

    def eval_mjx(ap_, episodes=20, horizon=1000, seed=0):
        """mode="hard" — byte-for-byte the training forward value, as in exp_c31."""
        stx = v_reset(jax.random.split(jax.random.PRNGKey(seed), episodes))

        @jax.jit
        def run(stx):
            def one(c, _):
                stx, ret, alive = c
                act = det_action(ap_, stx.obs)
                nst = v_step(stx, act)
                return (nst, ret + nst.reward * alive, alive * (1 - nst.done)), None
            (stx, ret, alive), _ = jax.lax.scan(
                one, (stx, jnp.zeros(episodes), jnp.ones(episodes)), None,
                length=horizon)
            return ret
        return float(np.asarray(run(stx)).mean())

    rows_log, t0, best = [], time.time(), -1e9
    total_steps = 0

    for it in range(a.iters):
        key, kro = jax.random.split(key)
        st, kro, tr = rollout(ap_, st, kro, it < a.warmup)
        s_, a_, r_, s2_, d_ = [np.asarray(x).reshape((-1,) + x.shape[2:]) for x in tr]
        n = len(s_)
        idx = (ptr + np.arange(n)) % N
        buf = dict(s=buf["s"].at[idx].set(s_), a=buf["a"].at[idx].set(a_),
                   r=buf["r"].at[idx].set(r_), s2=buf["s2"].at[idx].set(s2_),
                   d=buf["d"].at[idx].set(d_))
        ptr = int((ptr + n) % N)
        size = min(size + n, N)
        total_steps += n

        if it >= a.warmup:
            for _ in range(a.updates):
                key, kb = jax.random.split(key)
                bi = jax.random.randint(kb, (a.batch,), 0, size)
                batch = (buf["s"][bi], buf["a"][bi], buf["r"][bi],
                         buf["s2"][bi], buf["d"][bi])
                (ap_, qp, qt, log_alpha, os_a, os_q, os_al, key,
                 info) = update(ap_, qp, qt, log_alpha, os_a, os_q, os_al, batch, key)
                rr = np.asarray(info["rows"])
                flat = (np.arange(n_tables)[None, :] * K + rr).ravel()
                cnt = np.bincount(flat, minlength=n_tables * K).reshape(n_tables, K)
                row_updates = row_updates + jnp.asarray(cnt)

        if (it + 1) % a.eval_every == 0 or it == a.iters - 1:
            ret = eval_mjx(ap_, episodes=a.eval_episodes)
            cov = float((row_updates > 0).mean())
            bmean, bsd = [float(v) for v in bucket_stats(ap_, st.obs)]
            tbkt = float(np.exp(np.asarray(ap_["log_T_bkt"])).mean())
            tcr = float(np.exp(np.asarray(ap_["log_T_cross"])).mean())
            best = max(best, ret)
            el = time.time() - t0
            rows_log.append(dict(iter=it + 1, env_steps=total_steps, mjx_return=ret,
                                 row_coverage=cov, bucket_mean=bmean, bucket_sd=bsd,
                                 t_bkt=tbkt, t_cross=tcr,
                                 alpha=float(info["alpha"]) if it >= a.warmup else None,
                                 elapsed_s=round(el, 1)))
            print(f"[{it+1:>5}/{a.iters}] steps {total_steps:>9,} | MJX ret {ret:8.1f} "
                  f"| bkt-cov {cov*100:5.1f}% | bkt {bmean:5.2f}±{bsd:4.2f} "
                  f"| Tbkt {tbkt:5.3f} | Tcr {tcr:5.3f} | best {best:8.1f} "
                  f"| {el/60:5.1f}m", flush=True)
            json.dump(dict(iter=it + 1, iters=a.iters, env_steps=total_steps,
                           mjx_return=ret, best=best, row_coverage=cov,
                           eta_s=(a.iters - it - 1) * el / (it + 1), done=False),
                      open(os.path.join(HERE, out_name + ".partial"), "w"), indent=1)
            ck = dict({k: np.asarray(v) for k, v in ap_.items()},
                      n_heads=np.int32(a.heads), tph=np.int32(a.tph),
                      n_buckets=np.int32(a.buckets), eval_eps=np.float32(a.eval_eps))
            np.savez(os.path.join(HERE, f"{out_name}_actor.npz"), **ck)
            # Also keep the BEST actor. `_actor.npz` is rewritten at every eval, which is
            # what made the exp_c32b seed-0 dip un-autopsiable: by the time anyone looked,
            # the state that produced it had been overwritten twice. The CPU reference
            # still scores `_actor.npz` (the final policy) so no published number changes;
            # this is purely a forensic artefact.
            if ret >= best:
                np.savez(os.path.join(HERE, f"{out_name}_best_actor.npz"),
                         **ck, best_iter=np.int32(it + 1),
                         best_mjx=np.float32(ret))

    json.dump(dict(config=vars(a), frontend_params=n_det, table_params=n_tab,
                   total_params=n_det + n_tab, total_env_steps=total_steps,
                   wall_s=round(time.time() - t0, 1), best_mjx=best, history=rows_log),
              open(os.path.join(HERE, out_name + ".json"), "w"), indent=1)
    json.dump(dict(iter=a.iters, iters=a.iters, env_steps=total_steps,
                   mjx_return=rows_log[-1]["mjx_return"] if rows_log else 0.0,
                   best=best, row_coverage=rows_log[-1]["row_coverage"] if rows_log else 0.0,
                   eta_s=0.0, done=True),
              open(os.path.join(HERE, out_name + ".partial"), "w"), indent=1)
    print(f"done: best MJX {best:.1f} over {total_steps:,} env-steps in "
          f"{(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
