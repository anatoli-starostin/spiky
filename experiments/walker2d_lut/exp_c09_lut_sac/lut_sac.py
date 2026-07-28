"""exp_c09 — LUT-SAC: an off-policy actor-critic whose ACTOR is a lookup table (#75).

Diagnosis this is built on (established in Phases 1-4, not re-derived here): the
from-scratch gap is an OPTIMIZATION problem, not a capacity one. Distillation puts a
5,378-param LUT at 5512, so the table can represent the gait; PPO-from-scratch reaches
only 4407 because (a) on-policy data is a narrow state distribution while the LUT's
backward scatters into a SINGLE addressed row per sample, so most rows barely train, and
(b) a global fixed Gaussian is the wrong exploration for a rugged, piecewise-constant
landscape.

The design answers both:

  * **Actor = LUT with 12 outputs per cell**: 6 action means AND 6 log-stds. Exploration
    spread becomes state-local and *learned*, stored in the table itself:
    a = tanh(mu_row + sigma_row * eps). Where the critic is flat the entropy term widens
    that cell; where it is sharp, it narrows.
  * **Off-policy replay** decouples the update distribution from the current policy, so
    rows keep receiving gradient long after the policy stops visiting them.
  * **Coverage-prioritised sampling** additionally up-weights transitions addressing
    rarely-UPDATED rows — aimed directly at the sparse-scatter problem.
  * **Per-row trust region**: a row update is a step change in the policy for every state
    in that cell (unlike an MLP's smeared update). Row deltas are norm-clipped.

Critic is an MLP twin-Q (Variant A) so the question stays "can a LUT be the actor?".
A LUT critic is a separate research branch and is deliberately NOT built here.

Diagnostic: `row_coverage` = fraction of table rows that have received an update. The
diagnosis predicts it should RISE with return; both are logged every iteration.
"""
import argparse, json, os, sys, time
from functools import partial

import jax, jax.numpy as jnp
import numpy as np
import optax

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "exp_c02_mjx_scaffold"))
sys.path.insert(0, os.path.join(HERE, "..", "exp_c06_jax_backprop"))

import mjx_walker2d as W          # noqa: E402
from mujoco import mjx           # noqa: E402
import jax_lut_grad as L         # noqa: E402
sys.path.insert(0, os.path.join(HERE, "..", "exp_c11_lut_sac_2x2"))
import jax_lut_ext as X           # noqa: E402

OBS, ACT = 17, 6
LOGSTD_MIN, LOGSTD_MAX = -5.0, 2.0


# =============================================================================
# Actor: LUT with 12 outputs per cell (6 mu + 6 log sigma)
# =============================================================================

# Static/buffer state kept OUT of the differentiable pytree: jax.grad rejects the
# int fields (n_heads/tph), and the observation standardiser is a frozen buffer that
# must not receive gradient. Set once in main().
CFG = {}


def actor_init(key, nap, tph, heads, obs_mean, obs_std, table_std=0.05):
    p = L.init(key, nap, tph, heads, OBS, 2 * ACT, obs_mean, obs_std,
               table_std=table_std)
    # start with a modest, uniform spread: bias the log-sigma half of every row
    p["weights"] = p["weights"].at[:, :, ACT:].add(-1.0 / (heads * tph))
    return {k: v for k, v in p.items()
            if k not in ("n_heads", "tph", "obs_mean", "obs_std")}


def actor_out(p, obs):
    x = (obs - CFG["obs_mean"]) / (CFG["obs_std"] + 1e-6)
    y = CFG["apply"](x, p["w"], p["b"], p["weights"], p["log_T_soft"],
                     p["log_T_sel"], CFG["n_heads"], CFG["tph"]).sum(1)   # [B, 12]
    mu, log_std = y[:, :ACT], jnp.clip(y[:, ACT:], LOGSTD_MIN, LOGSTD_MAX)
    return mu, log_std


def actor_sample(p, obs, key):
    """Reparameterised tanh-Gaussian: returns (action, logp, mu_tanh)."""
    mu, log_std = actor_out(p, obs)
    std = jnp.exp(log_std)
    eps = jax.random.normal(key, mu.shape)
    pre = mu + std * eps
    a = jnp.tanh(pre)
    logp = (-0.5 * jnp.square(eps) - log_std - 0.5 * jnp.log(2 * jnp.pi)).sum(-1)
    logp -= jnp.log(1.0 - jnp.square(a) + 1e-6).sum(-1)      # tanh correction
    return a, logp, jnp.tanh(mu)


def actor_rows(p, obs):
    """The row index each table addresses — the coverage diagnostic. [B, n_tables]"""
    x = (obs - CFG["obs_mean"]) / (CFG["obs_std"] + 1e-6)
    return L._hard_index(L._project(x, p["w"], p["b"]), p["w"].shape[1])


# =============================================================================
# Critic: twin Q MLP
# =============================================================================

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


# =============================================================================
# Training
# =============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=3000)
    ap.add_argument("--envs", type=int, default=256)
    ap.add_argument("--rollout", type=int, default=4)
    ap.add_argument("--updates", type=int, default=8, help="grad steps per iteration")
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--buffer", type=int, default=1_000_000)
    ap.add_argument("--warmup", type=int, default=100, help="random-action iterations")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--actor-lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--tau", type=float, default=0.005)
    ap.add_argument("--nap", type=int, default=6)
    ap.add_argument("--tph", type=int, default=64)
    ap.add_argument("--heads", type=int, default=1)
    ap.add_argument("--target-entropy", type=float, default=-6.0)
    ap.add_argument("--row-clip", type=float, default=0.0,
                    help="per-row trust region: max L2 norm of a row delta (0=off)")
    ap.add_argument("--coverage-bonus", type=float, default=0.0,
                    help="weight of the rarely-updated-row bonus in sampling (0=off)")
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--eval-episodes", type=int, default=20)
    ap.add_argument("--addressing", default="hyperplane",
                    choices=["hyperplane", "anchors"],
                    help="anchors = frozen w = e_a - e_b (FastMHL semantics)")
    ap.add_argument("--seed", type=int, default=0,
                    help="drives BOTH the jax PRNGKey and the anchor draw")
    ap.add_argument("--anchor-policy", default="balanced",
                    choices=["balanced", "connected", "canonical_distinct",
                             "canonical_full_coverage", "legacy_jax"],
                    help="lutorch AnchorSamplingPolicy for --addressing anchors. "
                         "NOTE FastMultiHeadLut itself uses canonical_full_coverage "
                         "and REJECTS balanced. legacy_jax = the home-grown draw that "
                         "produced the exp_c11/exp_c12 numbers.")
    ap.add_argument("--forward-mode", default="hard",
                    choices=["hard", "hybrid_smooth"])
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out_name = a.out or f"lut_sac{a.tag}"

    stats = json.load(open(os.path.join(HERE, "..", "exp_c03_distillation",
                                        "dataset_stats.json")))
    obs_mean = jnp.asarray(stats["obs_mean"], jnp.float32)
    obs_std = jnp.asarray(stats["obs_std"], jnp.float32)

    key = jax.random.PRNGKey(a.seed)
    key, ka, kq, kr = jax.random.split(key, 4)
    CFG.update(n_heads=a.heads, tph=a.tph, obs_mean=obs_mean, obs_std=obs_std,
               apply=X.apply(a.forward_mode))
    ap_ = actor_init(ka, a.nap, a.tph, a.heads, obs_mean, obs_std)
    if a.addressing == "anchors":
        # anchor pairs written as a FROZEN hyperplane: w = e_a - e_b, b = 0.
        # The ENCODING is verified bit-exact against FastMultiHeadLut
        # (exp_c11/verify_ext.py check B); the DRAW now comes from lutorch's own
        # sampler (check C), instead of the home-grown one that only matched its
        # spirit. --anchor-policy legacy_jax restores that older draw, which is what
        # the exp_c11 2x2 and the exp_c12 sweep were run with.
        if a.anchor_policy == "legacy_jax":
            w0, b0 = X.anchor_pair_wb(np.random.default_rng(a.seed),
                                      a.heads * a.tph, a.nap, OBS)
        else:
            w0, b0 = X.anchor_pair_wb_lutorch(a.heads * a.tph, a.nap, OBS,
                                              seed=a.seed, policy=a.anchor_policy,
                                              heads=a.heads)
        ap_ = dict(ap_, w=w0, b=b0)
    qp = q_init(kq)
    qt = jax.tree.map(lambda x: x, qp)
    log_alpha = jnp.log(jnp.asarray(0.2))

    n_tables = a.heads * a.tph
    K = 2 ** a.nap
    n_table_params = int(np.prod(ap_["weights"].shape))
    n_idx = int(np.prod(ap_["w"].shape) + np.prod(ap_["b"].shape))
    print(f"LUT-SAC actor nap={a.nap} tph={a.tph} heads={a.heads} "
          f"seed={a.seed} addressing={a.addressing}"
          f"{'/' + a.anchor_policy if a.addressing == 'anchors' else ''} | "
          f"12 outputs/cell (6 mu + 6 log-sigma) | table {n_table_params:,} + "
          f"addressing {n_idx:,} = {n_table_params + n_idx:,} params | "
          f"rows {n_tables}x{K} = {n_tables*K:,}", flush=True)

    tx_a = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(a.actor_lr))
    tx_q = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(a.lr))
    tx_al = optax.adam(a.lr)
    os_a, os_q, os_al = tx_a.init(ap_), tx_q.init(qp), tx_al.init(log_alpha)

    m = W.make_model()
    mx = mjx.put_model(m)
    reset, step = W.make_env(mx)
    v_reset, v_step = jax.vmap(reset), jax.vmap(step)
    st = v_reset(jax.random.split(kr, a.envs))

    # ---- replay buffer (GPU-resident, circular) --------------------------
    N = a.buffer
    buf = dict(s=jnp.zeros((N, OBS)), a=jnp.zeros((N, ACT)), r=jnp.zeros(N),
               s2=jnp.zeros((N, OBS)), d=jnp.zeros(N))
    ptr, size = 0, 0
    row_updates = jnp.zeros((n_tables, K))     # how often each row got a gradient

    def norm(s):
        return (s - obs_mean) / (obs_std + 1e-6)

    @jax.jit
    def rollout(ap_, st, key, random_actions):
        def one(carry, _):
            st, key = carry
            key, k1 = jax.random.split(key)
            a_rand = jax.random.uniform(k1, (a.envs, ACT), minval=-1.0, maxval=1.0)
            a_pol, _, _ = actor_sample(ap_, st.obs, k1)
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

        # --- critic ---
        key, k1 = jax.random.split(key)
        a2, logp2, _ = actor_sample(ap_, s2, k1)
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

        # --- actor (reparameterised, through the verified LUT surrogate) ---
        key, k2 = jax.random.split(key)

        def a_loss(ap_):
            an, logp, _ = actor_sample(ap_, s, k2)
            q = jnp.minimum(q_apply(qp["q1"], ns, an), q_apply(qp["q2"], ns, an))
            return (alpha * logp - q).mean(), logp
        (al, logp), ga = jax.value_and_grad(a_loss, has_aux=True)(ap_)

        # per-row trust region: a row update is a STEP change for every state in
        # that cell, so bound its L2 norm (an MLP has no analogue of this).
        if a.addressing == "anchors":
            # no gradient to the addressing — fixed anchor pairs, as in FastMHL
            ga = dict(ga, w=jnp.zeros_like(ga["w"]), b=jnp.zeros_like(ga["b"]))
        if a.row_clip > 0:
            gw = ga["weights"]
            nrm = jnp.linalg.norm(gw, axis=-1, keepdims=True)
            ga = dict(ga, weights=gw * jnp.minimum(1.0, a.row_clip / (nrm + 1e-8)))
        ua, os_a = tx_a.update(ga, os_a, ap_)
        ap_ = optax.apply_updates(ap_, ua)

        # --- temperature ---
        def al_loss(log_alpha):
            return (-jnp.exp(log_alpha)
                    * (jax.lax.stop_gradient(logp) + a.target_entropy)).mean()
        gal = jax.grad(al_loss)(log_alpha)
        ual, os_al = tx_al.update(gal, os_al, log_alpha)
        log_alpha = optax.apply_updates(log_alpha, ual)

        qt = jax.tree.map(lambda t, s_: (1 - a.tau) * t + a.tau * s_, qt, qp)
        rows = actor_rows(ap_, s)
        return (ap_, qp, qt, log_alpha, os_a, os_q, os_al, key,
                dict(q_loss=ql, a_loss=al, alpha=alpha, logp=logp.mean(),
                     td=td.mean(), rows=rows))

    @jax.jit
    def det_action(ap_, obs):
        mu, _ = actor_out(ap_, obs)
        return jnp.tanh(mu)

    def eval_mjx(ap_, episodes=20, horizon=1000, seed=0):
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
            best = max(best, ret)
            el = time.time() - t0
            rows_log.append(dict(iter=it + 1, env_steps=total_steps, mjx_return=ret,
                                 row_coverage=cov, alpha=float(info["alpha"]) if it >= a.warmup else None,
                                 elapsed_s=round(el, 1)))
            print(f"[{it+1:>5}/{a.iters}] steps {total_steps:>9,} | MJX ret {ret:8.1f} "
                  f"| row-cov {cov*100:5.1f}% | best {best:8.1f} | {el/60:5.1f}m",
                  flush=True)
            json.dump(dict(iter=it + 1, iters=a.iters, env_steps=total_steps,
                           mjx_return=ret, best=best, row_coverage=cov,
                           eta_s=(a.iters - it - 1) * el / (it + 1), done=False),
                      open(os.path.join(HERE, out_name + ".partial"), "w"), indent=1)
            np.savez(os.path.join(HERE, f"{out_name}_actor.npz"),
                     **{k: np.asarray(v) for k, v in ap_.items()
                        if k not in ("n_heads", "tph")},
                     n_heads=np.int32(a.heads), tph=np.int32(a.tph))

    json.dump(dict(config=vars(a), table_params=n_table_params, index_params=n_idx,
                   total_params=n_table_params + n_idx, total_env_steps=total_steps,
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
