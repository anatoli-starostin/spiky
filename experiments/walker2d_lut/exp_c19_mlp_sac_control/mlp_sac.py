"""exp_c19 — MLP-actor SAC: the like-for-like control for the LUT actor (#75).

WHY THIS EXISTS. exp_c18 measures how much a LUT-actor SAC run moves when only the seed
changes. That number is uninterpretable on its own: SAC on Walker2d is a famously
seed-sensitive combination, so a large LUT spread could be the LUT's doing or could be
what any SAC actor does here. This run answers that by changing ONE thing -- the actor's
representation -- and holding the rest fixed.

WHAT IS HELD IDENTICAL TO exp_c09/lut_sac.py (line for line, deliberately):
  * the twin-Q MLP critic and its init, the alpha/target-entropy machinery, tau, gamma
  * envs / rollout / updates / batch / warmup / buffer / iters / lrs / grad clipping
  * the MJX env, the replay buffer's circular indexing, the eval-every cadence
  * the RNG structure: PRNGKey(seed) -> split 4 -> (actor, critic, env-reset), so a given
    seed resets the environments to the SAME states as the LUT run of that seed
  * the deterministic 100-episode CPU-reference eval (eval_cpu_mlp.py mirrors eval_cpu.py)

WHAT NECESSARILY DIFFERS, and why each is unavoidable rather than a choice:
  * The actor is a 2x256 MLP emitting 6 mu + 6 log-sigma, instead of a table of rows.
  * NO per-row trust region (--row-clip). It has no MLP analogue: clipping bounds the step
    a single addressed row takes, and an MLP has no addressed row. Global-norm clipping at
    1.0 is kept, as in the LUT runs.
  * PARAMETER COUNT IS NOT MATCHED: 2x256 is 73,740 actor params against the LUT's 28,032.
    Matching them was rejected: shrinking the MLP to 28k would make it a small-MLP study
    and confound representation with capacity, when the question asked is "is this spread
    just SAC-on-Walker2d?" -- for which the right control is a competent standard MLP.
    A param-matched MLP is a separate, also-interesting run.
  * Actor init is He-normal (std = sqrt(2/fan_in)) with a small output head. The critic's
    fixed 0.1 would be off-scale for the 17-input layer, and a deliberately weak baseline
    would make the control useless.
"""
import argparse, json, os, sys, time

import jax, jax.numpy as jnp
import numpy as np
import optax

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "exp_c02_mjx_scaffold"))

import mjx_walker2d as W          # noqa: E402
from mujoco import mjx           # noqa: E402

OBS, ACT = 17, 6
LOGSTD_MIN, LOGSTD_MAX = -5.0, 2.0

CFG = {}


# =============================================================================
# Actor: plain MLP, 6 mu + 6 log-sigma  (the ONE thing that differs from exp_c09)
# =============================================================================

def actor_init(key, hidden=256):
    k1, k2, k3 = jax.random.split(key, 3)
    def he(k, shape):
        return jax.random.normal(k, shape) * np.sqrt(2.0 / shape[0])
    return dict(w1=he(k1, (OBS, hidden)), b1=jnp.zeros(hidden),
                w2=he(k2, (hidden, hidden)), b2=jnp.zeros(hidden),
                w3=jax.random.normal(k3, (hidden, 2 * ACT)) * 0.01,
                b3=jnp.zeros(2 * ACT))


def actor_out(p, obs):
    x = (obs - CFG["obs_mean"]) / (CFG["obs_std"] + 1e-6)
    h = jax.nn.relu(x @ p["w1"] + p["b1"])
    h = jax.nn.relu(h @ p["w2"] + p["b2"])
    y = h @ p["w3"] + p["b3"]
    return y[:, :ACT], jnp.clip(y[:, ACT:], LOGSTD_MIN, LOGSTD_MAX)


def actor_sample(p, obs, key):
    """Reparameterised tanh-Gaussian — identical algebra to exp_c09/lut_sac.py."""
    mu, log_std = actor_out(p, obs)
    std = jnp.exp(log_std)
    eps = jax.random.normal(key, mu.shape)
    pre = mu + std * eps
    a = jnp.tanh(pre)
    logp = (-0.5 * jnp.square(eps) - log_std - 0.5 * jnp.log(2 * jnp.pi)).sum(-1)
    logp -= jnp.log(1.0 - jnp.square(a) + 1e-6).sum(-1)
    return a, logp, jnp.tanh(mu)


# =============================================================================
# Critic: twin Q MLP — copied verbatim from exp_c09 so it cannot drift
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
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--target-entropy", type=float, default=-6.0)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--eval-episodes", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out_name = a.out or f"mlp_sac{a.tag}"

    stats = json.load(open(os.path.join(HERE, "..", "exp_c03_distillation",
                                        "dataset_stats.json")))
    obs_mean = jnp.asarray(stats["obs_mean"], jnp.float32)
    obs_std = jnp.asarray(stats["obs_std"], jnp.float32)
    CFG.update(obs_mean=obs_mean, obs_std=obs_std)

    # Same split structure as lut_sac.py, so seed s resets the envs identically.
    key = jax.random.PRNGKey(a.seed)
    key, ka, kq, kr = jax.random.split(key, 4)
    ap_ = actor_init(ka, a.hidden)
    qp = q_init(kq)
    qt = jax.tree.map(lambda x: x, qp)
    log_alpha = jnp.log(jnp.asarray(0.2))

    n_actor = int(sum(np.prod(v.shape) for v in ap_.values()))
    # The critic width is q_init's own default and is deliberately NOT tied to --hidden:
    # --hidden resizes the ACTOR only, so a param-matched actor study leaves the critic
    # identical to exp_c19's. Read it back from the built params rather than assuming, so
    # the banner cannot drift from what was actually constructed (it did: this line used
    # to print a.hidden for the critic, which was silently wrong for any --hidden != 256).
    q_hidden = int(qp["q1"]["w1"].shape[1])
    print(f"MLP-SAC actor {OBS}x{a.hidden}x{a.hidden}x{2*ACT} seed={a.seed} | "
          f"6 mu + 6 log-sigma | actor {n_actor:,} params "
          f"(control for the LUT's 28,032) | critic twin-Q {q_hidden}x{q_hidden}",
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

    @jax.jit
    def update(ap_, qp, qt, log_alpha, os_a, os_q, os_al, batch, key):
        s, act, r, s2, d = batch
        ns, ns2 = norm(s), norm(s2)
        alpha = jnp.exp(log_alpha)

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

        key, k2 = jax.random.split(key)

        def a_loss(ap_):
            an, logp, _ = actor_sample(ap_, s, k2)
            q = jnp.minimum(q_apply(qp["q1"], ns, an), q_apply(qp["q2"], ns, an))
            return (alpha * logp - q).mean(), logp
        (al, logp), ga = jax.value_and_grad(a_loss, has_aux=True)(ap_)
        # No per-row trust region: there is no addressed row in an MLP. Global-norm
        # clipping (inside tx_a) is the part that does carry over.
        ua, os_a = tx_a.update(ga, os_a, ap_)
        ap_ = optax.apply_updates(ap_, ua)

        def al_loss(log_alpha):
            return (-jnp.exp(log_alpha)
                    * (jax.lax.stop_gradient(logp) + a.target_entropy)).mean()
        gal = jax.grad(al_loss)(log_alpha)
        ual, os_al = tx_al.update(gal, os_al, log_alpha)
        log_alpha = optax.apply_updates(log_alpha, ual)

        qt = jax.tree.map(lambda t, s_: (1 - a.tau) * t + a.tau * s_, qt, qp)
        return (ap_, qp, qt, log_alpha, os_a, os_q, os_al, key,
                dict(q_loss=ql, a_loss=al, alpha=alpha, logp=logp.mean(),
                     td=td.mean()))

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

        if (it + 1) % a.eval_every == 0 or it == a.iters - 1:
            ret = eval_mjx(ap_, episodes=a.eval_episodes)
            best = max(best, ret)
            el = time.time() - t0
            rows_log.append(dict(iter=it + 1, env_steps=total_steps, mjx_return=ret,
                                 alpha=float(info["alpha"]) if it >= a.warmup else None,
                                 elapsed_s=round(el, 1)))
            # Same line shape as lut_sac.py so one bar/parser reads both; row-cov is
            # printed as n/a rather than omitted, since an MLP has no rows.
            print(f"[{it+1:>5}/{a.iters}] steps {total_steps:>9,} | MJX ret {ret:8.1f} "
                  f"| row-cov   n/a | best {best:8.1f} | {el/60:5.1f}m", flush=True)
            json.dump(dict(iter=it + 1, iters=a.iters, env_steps=total_steps,
                           mjx_return=ret, best=best,
                           eta_s=(a.iters - it - 1) * el / (it + 1), done=False),
                      open(os.path.join(HERE, out_name + ".partial"), "w"), indent=1)
            np.savez(os.path.join(HERE, f"{out_name}_actor.npz"),
                     **{k: np.asarray(v) for k, v in ap_.items()},
                     hidden=np.int32(a.hidden))

    json.dump(dict(config=vars(a), actor_params=n_actor, total_env_steps=total_steps,
                   wall_s=round(time.time() - t0, 1), best_mjx=best, history=rows_log),
              open(os.path.join(HERE, out_name + ".json"), "w"), indent=1)
    json.dump(dict(iter=a.iters, iters=a.iters, env_steps=total_steps,
                   mjx_return=rows_log[-1]["mjx_return"] if rows_log else 0.0,
                   best=best, eta_s=0.0, done=True),
              open(os.path.join(HERE, out_name + ".partial"), "w"), indent=1)
    print(f"done: best MJX {best:.1f} over {total_steps:,} env-steps in "
          f"{(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
