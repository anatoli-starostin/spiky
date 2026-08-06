"""exp_c30 — LIF-detector SAC: the exp_c09 LUT-SAC trainer with a LIF-detector actor (#75).

A fork of exp_c09/lut_sac.py with ONE thing changed: the actor's index front-end. The
hyperplane/anchor sign-tests are replaced by the combined LIF detectors of
`spiky.lutorch.lif_detectors_mhl.LIFDetectorsMHL`, ported to JAX in `jax_lif_mhl.py` and
held to the torch reference by `run_parity.sh` (13/13, gradients to fp32 noise).

Everything else is deliberately untouched -- twin-Q MLP critic, replay, learned per-cell
log-sigma, the trust region on row deltas, learning rates, target entropy, tau, gamma,
batch, warmup. The point is to attribute any difference to the front-end, so the SAC
recipe stays the known-good one.

THIS RUN IS PARAM-MATCHED, which exp_c30 was not. exp_c30's actor was 87,361 params
against 49,152 for the LUT actors, so its near-parity result said nothing per parameter.
Here the ordered-pair channel is factorised (`jax_lif_lowrank.py`: rank 2 plus a
per-source-channel bias) and the actor is **48,193** params -- 1.95% UNDER the 49,152 of
exp_c13/c18/c29, and 44.8% smaller than exp_c30. Under rather than over on purpose: a
result that holds at 98% of the baseline's budget is a conservative claim.

Anchors: exp_c18 hyperplane at this exact nap6/tph32 shape, 6 seeds, deterministic,
4308.0 +/- 500.1; and exp_c30, the same LIF model with a dense P, 3931.3 +/- 585.8.

TWO KNOBS THE LUT ACTOR DOES NOT HAVE:

  * `eps` -- the detector gate sharpness. Annealed 2.0 -> 0.3 across the whole run,
    proportionally to the distillation recipe (which annealed over its own 6,000 Adam
    steps). Here the horizon is the gradient steps actually taken,
    (iters - warmup) * updates, so the schedule stretches to the run rather than
    finishing in its first 2%. It is passed as a TRACED argument, not a static one, so
    the jitted update is compiled once and not re-traced per step.
  * `temp_bit` -- the bit temperature, trainable (as `log_temp_bit`, init log 1.0) and
    left to self-sharpen. Not frozen, no schedule, per the module's design.

EVAL runs in mode="hard" at the CURRENT annealed eps. `eps` is not a display setting: it
enters the membrane, so it changes the bits, so it changes which row every table
addresses. A policy trained at eps=2.0 and scored at eps=0.3 is a different function, not
a blurred one, and pinning the eval to 0.3 would make the first ~70% of every run read as
failure no matter what the actor had learned. The anneal ENDS at --eval-eps, so the last
training proxy and the 100-episode CPU reference are taken in the same regime -- which is
also the deployment regime, and the eps the checkpoint records.
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
import jax_lif_lowrank as LIF   # noqa: E402

OBS, ACT = 17, 6
LOGSTD_MIN, LOGSTD_MAX = -5.0, 2.0

CFG = {}


# =============================================================================
# Actor: LIF-detector MHL with 12 outputs per cell (6 mu + 6 log sigma)
# =============================================================================

def actor_init(key, nap, tph, heads):
    p = LIF.init(key, nap, tph, heads, OBS, 2 * ACT)
    # Same modest, uniform starting spread as exp_c09: bias the log-sigma half of every
    # row so the tph tables sum to -1.0 rather than to 0.
    p["table"] = p["table"].at[:, :, ACT:].add(-1.0 / (heads * tph))
    return p


def actor_out(p, obs, eps, mode="st"):
    x = (obs - CFG["obs_mean"]) / (CFG["obs_std"] + 1e-6)
    y = LIF.apply(p, x, eps, CFG["n_heads"], CFG["tph"], CFG["nap"],
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
    """The row index each table addresses — the coverage diagnostic. [B, n_tables]"""
    x = (obs - CFG["obs_mean"]) / (CFG["obs_std"] + 1e-6)
    return LIF.address(p, x, eps, CFG["n_heads"], CFG["tph"], CFG["nap"])


# =============================================================================
# Critic: twin Q MLP — verbatim from exp_c09
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
    ap.add_argument("--nap", type=int, default=6)
    ap.add_argument("--tph", type=int, default=32)
    ap.add_argument("--heads", type=int, default=1)
    ap.add_argument("--target-entropy", type=float, default=-6.0)
    ap.add_argument("--row-clip", type=float, default=1.0)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--eval-episodes", type=int, default=20)
    ap.add_argument("--eps-start", type=float, default=2.0)
    ap.add_argument("--eps-end", type=float, default=0.3)
    ap.add_argument("--eval-eps", type=float, default=0.3,
                    help="the DEPLOYMENT gate sharpness, recorded in the checkpoint and "
                         "used by eval_lif_cpu.py. Training evals use the current "
                         "annealed eps; since the anneal ends here, the two coincide by "
                         "the end of the run. Set it to --eps-end unless you mean to "
                         "score the policy outside the regime it was trained into.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out_name = a.out or f"lif_sac{a.tag}"

    stats = json.load(open(os.path.join(HERE, "..", "exp_c03_distillation",
                                        "dataset_stats.json")))
    obs_mean = jnp.asarray(stats["obs_mean"], jnp.float32)
    obs_std = jnp.asarray(stats["obs_std"], jnp.float32)

    key = jax.random.PRNGKey(a.seed)
    key, ka, kq, kr = jax.random.split(key, 4)
    CFG.update(n_heads=a.heads, tph=a.tph, nap=a.nap,
               obs_mean=obs_mean, obs_std=obs_std)
    ap_ = actor_init(ka, a.nap, a.tph, a.heads)

    qp = q_init(kq)
    qt = jax.tree.map(lambda x: x, qp)
    log_alpha = jnp.log(jnp.asarray(0.2))

    n_tables = a.heads * a.tph
    K = 2 ** a.nap
    n_det, n_tab = LIF.n_params(ap_)
    print(f"LIF-SAC actor nap={a.nap} tph={a.tph} heads={a.heads} seed={a.seed} | "
          f"12 outputs/cell (6 mu + 6 log-sigma) | detectors {n_det:,} + table "
          f"{n_tab:,} = {n_det + n_tab:,} params | rows {n_tables}x{K} = "
          f"{n_tables*K:,} | eps {a.eps_start}->{a.eps_end}, eval at {a.eval_eps}",
          flush=True)

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

    # The anneal runs over the gradient steps actually taken, not over iterations: the
    # first `warmup` iterations collect random-action data and perform no update.
    total_gsteps = max(1, (a.iters - a.warmup) * a.updates)
    gstep = 0

    def norm(s):
        return (s - obs_mean) / (obs_std + 1e-6)

    @jax.jit
    def rollout(ap_, st, key, random_actions, eps):
        def one(carry, _):
            st, key = carry
            key, k1 = jax.random.split(key)
            a_rand = jax.random.uniform(k1, (a.envs, ACT), minval=-1.0, maxval=1.0)
            a_pol, _, _ = actor_sample(ap_, st.obs, k1, eps)
            act = jnp.where(random_actions, a_rand, a_pol)
            nst = v_step(st, act)
            return (nst, key), (st.obs, act, nst.reward, nst.obs, nst.done)
        (st, key), tr = jax.lax.scan(one, (st, key), None, length=a.rollout)
        return st, key, tr

    @partial(jax.jit, static_argnums=())
    def update(ap_, qp, qt, log_alpha, os_a, os_q, os_al, batch, key, eps):
        s, act, r, s2, d = batch
        ns, ns2 = norm(s), norm(s2)
        alpha = jnp.exp(log_alpha)

        key, k1 = jax.random.split(key)
        a2, logp2, _ = actor_sample(ap_, s2, k1, eps)
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
            an, logp, _ = actor_sample(ap_, s, k2, eps)
            q = jnp.minimum(q_apply(qp["q1"], ns, an), q_apply(qp["q2"], ns, an))
            return (alpha * logp - q).mean(), logp
        (al, logp), ga = jax.value_and_grad(a_loss, has_aux=True)(ap_)

        # Per-row trust region, exactly as in exp_c09 — a row update is a step change
        # for every state in that cell. Applies to the table only; the detector bank is
        # a continuous front-end with no row semantics.
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
        rows = actor_rows(ap_, s, eps)
        return (ap_, qp, qt, log_alpha, os_a, os_q, os_al, key,
                dict(q_loss=ql, a_loss=al, alpha=alpha, logp=logp.mean(),
                     td=td.mean(), rows=rows))

    @jax.jit
    def det_action(ap_, obs, eps):
        mu, _ = actor_out(ap_, obs, eps, mode="hard")
        return jnp.tanh(mu)

    def eval_mjx(ap_, eps, episodes=20, horizon=1000, seed=0):
        """Scored at the CURRENT eps, not at a pinned one.

        `eps` changes the membrane, hence the bits, hence which row every table
        addresses -- so a policy trained at eps=2.0 and scored at eps=0.3 is a different
        function, not a slightly blurred one. Pinning the eval to 0.3 would make the
        first ~70% of the curve read as failure regardless of what the actor had
        learned. Because the anneal ENDS at --eval-eps, the last training proxy and the
        CPU reference are taken in the same regime anyway."""
        stx = v_reset(jax.random.split(jax.random.PRNGKey(seed), episodes))

        @jax.jit
        def run(stx):
            def one(c, _):
                stx, ret, alive = c
                act = det_action(ap_, stx.obs, eps)
                nst = v_step(stx, act)
                return (nst, ret + nst.reward * alive, alive * (1 - nst.done)), None
            (stx, ret, alive), _ = jax.lax.scan(
                one, (stx, jnp.zeros(episodes), jnp.ones(episodes)), None,
                length=horizon)
            return ret
        return float(np.asarray(run(stx)).mean())

    rows_log, t0, best = [], time.time(), -1e9
    total_steps = 0
    eps_now = a.eps_start

    for it in range(a.iters):
        key, kro = jax.random.split(key)
        st, kro, tr = rollout(ap_, st, kro, it < a.warmup, eps_now)
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
                frac = gstep / max(1, total_gsteps - 1)
                eps_now = a.eps_start + (a.eps_end - a.eps_start) * frac
                key, kb = jax.random.split(key)
                bi = jax.random.randint(kb, (a.batch,), 0, size)
                batch = (buf["s"][bi], buf["a"][bi], buf["r"][bi],
                         buf["s2"][bi], buf["d"][bi])
                (ap_, qp, qt, log_alpha, os_a, os_q, os_al, key,
                 info) = update(ap_, qp, qt, log_alpha, os_a, os_q, os_al, batch,
                                key, eps_now)
                rr = np.asarray(info["rows"])
                flat = (np.arange(n_tables)[None, :] * K + rr).ravel()
                cnt = np.bincount(flat, minlength=n_tables * K).reshape(n_tables, K)
                row_updates = row_updates + jnp.asarray(cnt)
                gstep += 1

        if (it + 1) % a.eval_every == 0 or it == a.iters - 1:
            ret = eval_mjx(ap_, eps_now, episodes=a.eval_episodes)
            cov = float((row_updates > 0).mean())
            best = max(best, ret)
            el = time.time() - t0
            rows_log.append(dict(iter=it + 1, env_steps=total_steps, mjx_return=ret,
                                 row_coverage=cov, eps=eps_now,
                                 temp_bit=float(np.exp(np.asarray(ap_["log_temp_bit"]))),
                                 alpha=float(info["alpha"]) if it >= a.warmup else None,
                                 elapsed_s=round(el, 1)))
            print(f"[{it+1:>5}/{a.iters}] steps {total_steps:>9,} | MJX ret {ret:8.1f} "
                  f"| row-cov {cov*100:5.1f}% | eps {eps_now:5.2f} | tbit "
                  f"{float(np.exp(np.asarray(ap_['log_temp_bit']))):5.3f} "
                  f"| best {best:8.1f} | {el/60:5.1f}m", flush=True)
            json.dump(dict(iter=it + 1, iters=a.iters, env_steps=total_steps,
                           mjx_return=ret, best=best, row_coverage=cov,
                           eta_s=(a.iters - it - 1) * el / (it + 1), done=False),
                      open(os.path.join(HERE, out_name + ".partial"), "w"), indent=1)
            np.savez(os.path.join(HERE, f"{out_name}_actor.npz"),
                     **{k: np.asarray(v) for k, v in ap_.items()},
                     n_heads=np.int32(a.heads), tph=np.int32(a.tph),
                     nap=np.int32(a.nap), eval_eps=np.float32(a.eval_eps))

    json.dump(dict(config=vars(a), detector_params=n_det, table_params=n_tab,
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
