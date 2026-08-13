"""exp_c05 — gradient-free evolution strategies on MJX (#75, Phase 3 + CMA-ES-on-LUT).

Establishes the gradient-free ceiling with a small MLP, then points the SAME loop at a
LUT policy (via the bit-exact JAX forward from exp_c04) — the representation-under-
evolution comparison.

**Why OpenAI-ES / sep-CMA-ES and not vanilla CMA-ES:** vanilla CMA-ES maintains a full
d x d covariance, which is O(d^2) memory and O(d^3) per update. Even the smallest LUT
here has d ~ 5k parameters (25M covariance entries) and the MLP has ~1.5k; full CMA-ES
is simply not applicable at this scale. This uses:
  * `openai`  — antithetic mirrored sampling + rank normalisation, isotropic sigma;
  * `sepcma`  — a diagonal (separable) covariance, so O(d) per-coordinate step sizes.
Both scale linearly in d and both consume exactly what MJX is good at: thousands of
independent rollouts per generation.

Fitness is the mean undiscounted return over `episodes_per_candidate` MJX episodes.
The FINAL number for any winner is always re-measured in the CPU reference env.

Usage:
  XLA_PYTHON_CLIENT_PREALLOCATE=false python es_mjx.py --policy mlp --gens 200
  XLA_PYTHON_CLIENT_PREALLOCATE=false python es_mjx.py --policy lut --nap 6 --tph 16
"""
import argparse, json, os, sys, time

import jax, jax.numpy as jnp
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(BASE, "exp_c02_mjx_scaffold"))
sys.path.insert(0, os.path.join(BASE, "exp_c04_jax_lut"))

import mjx_walker2d as W          # noqa: E402
from mujoco import mjx           # noqa: E402
import jax_lut                   # noqa: E402

OBS, ACT = 17, 6


# =============================================================================
# Policies as flat parameter vectors
# =============================================================================

def mlp_spec(hidden=32):
    shapes = [(OBS, hidden), (hidden,), (hidden, hidden), (hidden,), (hidden, ACT), (ACT,)]
    sizes = [int(np.prod(s)) for s in shapes]
    d = int(sum(sizes))

    def apply(flat, obs, norm):
        i, ps = 0, []
        for s, n in zip(shapes, sizes):
            ps.append(flat[i:i + n].reshape(s)); i += n
        x = (obs - norm[0]) / (norm[1] + 1e-6)
        x = jnp.tanh(x @ ps[0] + ps[1])
        x = jnp.tanh(x @ ps[2] + ps[3])
        return jnp.clip(x @ ps[4] + ps[5], -1.0, 1.0)

    def init(key):
        return 0.1 * jax.random.normal(key, (d,))
    return d, apply, init


def lut_spec(nap=6, tph=16, heads=1):
    T = heads * tph
    K = 2 ** nap
    sw, sb, sv = (T, nap, OBS), (T, nap), (T, K, ACT)
    d = jax_lut.n_flat_params(sw, sb, sv)
    nw, nb = int(np.prod(sw)), int(np.prod(sb))

    def apply(flat, obs, norm):
        p = dict(w=flat[:nw].reshape(sw), b=flat[nw:nw + nb].reshape(sb),
                 weights=flat[nw + nb:].reshape(sv), n_heads=heads, tph=tph)
        x = (obs - norm[0]) / (norm[1] + 1e-6)
        # `apply` is called under vmap, so obs arrives rank-1 [D]; lut_forward is
        # written for a batch [B, D] (its dot_general contracts axis 1). Add and drop
        # a singleton batch axis rather than special-casing the ported forward, which
        # is verified bit-exact against torch and should not be touched.
        y = jax_lut.lut_forward(p, x[None])[0]          # [n_heads, ACT]
        return jnp.clip(y.sum(0), -1.0, 1.0)

    def init(key):
        k1, k2, k3 = jax.random.split(key, 3)
        # anchor-pair-like init for the addressing, small noise in the table body
        w = jax.random.normal(k1, sw) * 0.5
        b = jax.random.normal(k2, sb) * 0.1
        v = jax.random.normal(k3, sv) * 0.3
        return jnp.concatenate([w.ravel(), b.ravel(), v.ravel()])
    return d, apply, init


# =============================================================================
# Batched fitness: evaluate a whole population in one vmapped MJX rollout
# =============================================================================

def make_fitness(mx, apply, norm, episodes, horizon):
    reset, step = W.make_env(mx)
    v_reset, v_step = jax.vmap(reset), jax.vmap(step)

    def fitness(pop, key):
        P = pop.shape[0]
        n = P * episodes
        st = v_reset(jax.random.split(key, n))
        who = jnp.repeat(jnp.arange(P), episodes)          # env -> candidate
        # Materialise the per-env parameter vector ONCE, outside the rollout scan.
        # Gathering pop[i] inside the scan re-ran the gather every physics step and
        # dominated the generation time (12.5 s/gen -> see the log).
        pop_per_env = pop[who]

        def one(carry, _):
            st, ret, alive = carry
            act = jax.vmap(lambda p, o: apply(p, o, norm))(pop_per_env, st.obs)
            nst = v_step(st, act)
            ret = ret + nst.reward * alive
            alive = alive * (1.0 - nst.done)
            return (nst, ret, alive), None
        (st, ret, alive), _ = jax.lax.scan(
            one, (st, jnp.zeros(n), jnp.ones(n)), None, length=horizon)
        return ret.reshape(P, episodes).mean(1)
    return jax.jit(fitness)


# =============================================================================
# The two scalable ES variants
# =============================================================================

def rank_normalise(f):
    order = jnp.argsort(jnp.argsort(f))
    return order / (f.shape[0] - 1) - 0.5


def run_es(algo, d, fitness, key, gens, pop, sigma, lr, log, init_mu):
    mu = init_mu
    s = jnp.ones(d) * sigma if algo == "sepcma" else sigma
    m_adam = jnp.zeros(d); v_adam = jnp.zeros(d)
    half = pop // 2
    best = -1e9
    rows = []
    t0 = time.time()
    for g in range(gens):
        key, k1, k2 = jax.random.split(key, 3)
        eps = jax.random.normal(k1, (half, d))
        eps = jnp.concatenate([eps, -eps])              # antithetic pairs
        cand = mu[None] + (s * eps if algo == "sepcma" else s * eps)
        f = fitness(cand, k2)
        u = rank_normalise(f)
        grad = (u[:, None] * eps).mean(0) / (s if algo != "sepcma" else 1.0)
        if algo == "sepcma":
            grad = grad / (s + 1e-8)
        # Adam on the ES gradient estimate (standard for OpenAI-ES)
        m_adam = 0.9 * m_adam + 0.1 * grad
        v_adam = 0.999 * v_adam + 0.001 * grad ** 2
        mhat = m_adam / (1 - 0.9 ** (g + 1)); vhat = v_adam / (1 - 0.999 ** (g + 1))
        mu = mu + lr * mhat / (jnp.sqrt(vhat) + 1e-8)
        if algo == "sepcma":
            # per-coordinate step-size adaptation from the weighted spread
            w_ = jnp.abs(u)[:, None] * (eps ** 2)
            s = jnp.clip(s * jnp.exp(0.05 * (w_.mean(0) / (w_.mean() + 1e-8) - 1.0)),
                         1e-3, 1.0)
        fb, fm = float(f.max()), float(f.mean())
        best = max(best, fb)
        rows.append(dict(gen=g, best=fb, mean=fm, elapsed_s=round(time.time() - t0, 1)))
        if g % 10 == 0 or g == gens - 1:
            log(f"  gen {g:>4}/{gens}  best {fb:8.1f}  mean {fm:8.1f}  "
                f"(running best {best:8.1f})  {time.time()-t0:6.0f}s")
    return mu, best, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="mlp", choices=["mlp", "lut"])
    ap.add_argument("--algo", default="openai", choices=["openai", "sepcma"])
    ap.add_argument("--gens", type=int, default=200)
    ap.add_argument("--pop", type=int, default=256)
    ap.add_argument("--episodes", type=int, default=4)
    ap.add_argument("--horizon", type=int, default=500)
    ap.add_argument("--sigma", type=float, default=0.05)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--nap", type=int, default=6)
    ap.add_argument("--tph", type=int, default=16)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    stats_path = os.path.join(BASE, "exp_c03_distillation", "dataset_stats.json")
    st = json.load(open(stats_path))
    norm = (jnp.asarray(st["obs_mean"], jnp.float32),
            jnp.asarray(st["obs_std"], jnp.float32))

    if a.policy == "mlp":
        d, apply, init = mlp_spec(a.hidden)
        desc = f"MLP[{a.hidden},{a.hidden}]"
    else:
        d, apply, init = lut_spec(a.nap, a.tph)
        desc = f"LUT nap={a.nap} tph={a.tph}"

    m = W.make_model()
    mx = mjx.put_model(m)
    fitness = make_fitness(mx, apply, norm, a.episodes, a.horizon)

    name = f"es_{a.policy}_{a.algo}{a.tag}"
    print(f"[{name}] {desc} | d={d:,} params | pop={a.pop} x {a.episodes} eps "
          f"x {a.horizon} steps = {a.pop*a.episodes*a.horizon:,} env-steps/gen "
          f"| {a.gens} gens", flush=True)

    key = jax.random.PRNGKey(0)
    key, ki = jax.random.split(key)
    mu, best, rows = run_es(a.algo, d, fitness, key, a.gens, a.pop, a.sigma, a.lr,
                            lambda s: print(s, flush=True), init(ki))

    np.save(os.path.join(HERE, f"{name}_mu.npy"), np.asarray(mu))
    total_steps = a.gens * a.pop * a.episodes * a.horizon
    out = dict(name=name, policy=a.policy, algo=a.algo, desc=desc, d=d,
               gens=a.gens, pop=a.pop, episodes=a.episodes, horizon=a.horizon,
               sigma=a.sigma, lr=a.lr, best_mjx_fitness=best,
               total_env_steps=total_steps,
               final_gen_best=rows[-1]["best"], final_gen_mean=rows[-1]["mean"],
               wall_s=rows[-1]["elapsed_s"], history=rows)
    json.dump(out, open(os.path.join(HERE, f"{name}.json"), "w"), indent=1)
    print(f"[{name}] best MJX fitness {best:.1f} over {total_steps:,} env-steps "
          f"in {rows[-1]['elapsed_s']/60:.1f} min "
          f"(NOTE: MJX horizon-{a.horizon} fitness, NOT the CPU reference number)",
          flush=True)


if __name__ == "__main__":
    main()
