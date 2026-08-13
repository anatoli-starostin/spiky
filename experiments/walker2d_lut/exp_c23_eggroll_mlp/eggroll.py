"""exp_c23 — EGGROLL (low-rank evolution strategies) on the MJX Walker2d env (#75).

Pure evolution. No PPO, no policy gradient, no distillation: a population is sampled
around a mean parameter vector, each member is rolled out in the GPU env, and the mean
moves along a fitness-weighted average of the perturbations. Phase 1 (this file) evolves
an **MLP** controller; Phase 2 will point the same loop at a LUT controller.

Implements **Algorithm 1 of "Evolution Strategies at the Hyperscale"** (Sarkar et al.,
arXiv:2511.16652) — EGGROLL = Evolution Guided GeneRal Optimisation via Low-rank
Learning. The paper's own construction, verbatim from Algorithm 1:

    A_i ~ p(A_i),  B_i ~ p(B_i),   E_i <- (1/sqrt(r)) A_i B_i^T
    f_i <- f(W = M + sigma E_i)
    M   <- M + (alpha / N) sum_j E_j f_j

with Assumption 1 requiring only that the entries of A and B be i.i.d., zero-mean,
symmetric, unit-variance (Gaussian here). Note there is no explicit 1/sigma in their
update — "we absorb the constant 1/sigma into the tunable learning rate" — so this file
does the same, which also makes the paper's published learning rates directly usable.

WHAT EGGROLL CHANGES, AND WHY IT MATTERS HERE
---------------------------------------------
Classical ES (what `exp_c05_es/es_mjx.py` already does) perturbs a weight matrix
W in R^{m x n} with a full-rank Gaussian E ~ N(0,1)^{m x n}. For a population of P that
is P*m*n numbers to sample, hold and multiply by. EGGROLL instead draws two thin
factors and uses their outer product:

    A ~ N(0,1)^{m x r},  B ~ N(0,1)^{n x r},  E = A B^T / sqrt(r),   r << min(m, n)

Two consequences, and only the second one is free:

  1. MEMORY / SAMPLING drops from P*m*n to P*r*(m+n).
  2. The PERTURBED FORWARD never has to materialise W + sigma*E:

         x (W + sigma E) = x W + (sigma/sqrt(r)) (x A) B^T

     which costs r(m+n) per row instead of m*n. This is the property that makes the
     method scale, and it is the reason to bother at all.

The 1/sqrt(r) is not cosmetic: it makes Var(E_ij) = 1, i.e. exactly the per-entry
variance of the full-rank noise it replaces, so `--sigma` means the same thing in both
methods and the two are directly comparable. `test_eggroll.py` asserts this numerically
rather than trusting the algebra.

The aggregate ES update also stays factored -- the fitness-weighted sum of P outer
products is one einsum, never P matrices:

    g_W = (1 / (P sigma sqrt(r))) * sum_i u_i A_i B_i^T

Bias vectors are 1-D and have no low-rank structure to exploit, so they are perturbed
full-rank. That is what the method prescribes and it costs nothing (they are tiny).

HONEST SCOPE NOTE. At the size of exp_c05's control (MLP[32,32] = 1,830 parameters)
EGGROLL's efficiency claim is untestable: the run is dominated by MJX physics, not by
the controller's matmul or by noise memory. That size is kept only as the like-for-like
comparison against the full-rank ES number already in RESULTS.md. The DEFAULT here is
the paper's own RL architecture -- 3 layers of 256 units, where the 256x256 matrix at
rank 1-4 is a 64-256x reduction in noise and the claim becomes measurable. This is also
the setting the paper tuned on Brax ant/humanoid, the nearest published relatives of
Walker2d.

The paper's own RL conclusion is the bar to clear: over 16 environments EGGROLL is
"competitive with OpenES on 7/16, underperforms on 2/16, and outperforms on 7/16" -- so
the expected Phase-1 result is PARITY WITH FULL-RANK ES AT LOWER COST, not a better
policy. Anything else would be the surprise worth chasing.

Fitness is the mean undiscounted return over `--episodes` MJX episodes of `--horizon`
steps. As everywhere in this track, the FINAL number for any winner is re-measured in
the deterministic 100-episode CPU reference env (`eval_cpu_eggroll.py`) -- the MJX
fitness is a training proxy and has mis-ranked policies before (see RESULTS.md).

Usage:
  XLA_PYTHON_CLIENT_PREALLOCATE=false python eggroll.py --rank 4 --gens 200
  XLA_PYTHON_CLIENT_PREALLOCATE=false python eggroll.py --rank 0 --gens 200   # full-rank control
"""
import argparse, json, os, sys, time

import jax, jax.numpy as jnp
import numpy as np

# JAX defaults to TF32 tensor cores for float32 matmuls on this card, which costs ~1e-3
# relative accuracy. That was enough to make the update einsum disagree with an explicit
# sum of the same outer products by 1.8e-4 (test_eggroll.py test 3 catches it). The
# matmuls here are tiny next to the MJX physics, so buying exactness costs nothing
# measurable -- and an ES update that is quietly approximate is not worth debugging later.
jax.config.update("jax_default_matmul_precision", "highest")

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(BASE, "exp_c02_mjx_scaffold"))

import mjx_walker2d as W          # noqa: E402
from mujoco import mjx            # noqa: E402

OBS, ACT = 17, 6


# =============================================================================
# The controller, as a list of (matrix, bias) layers
# =============================================================================

def mlp_spec(hidden=256, layers=3):
    """`layers` weight matrices of width `hidden`, clipped linear head.

    Default (3, 256) is the paper's RL architecture. At (3, 32) the shapes match
    `exp_c05_es/es_mjx.py:mlp_spec` exactly, so the full-rank ES numbers already in
    RESULTS.md are a like-for-like control.
    """
    assert layers >= 2, "need at least an input and an output matrix"
    return [(OBS, hidden)] + [(hidden, hidden)] * (layers - 2) + [(hidden, ACT)]


def init_theta(shapes, key, scale=0.1):
    ks = jax.random.split(key, len(shapes))
    return dict(W=[scale * jax.random.normal(k, s) for k, s in zip(ks, shapes)],
                b=[jnp.zeros((s[1],)) for s in shapes])


def n_params(shapes):
    return int(sum(m * n + n for m, n in shapes))


def apply_policy(theta, pert, obs, norm, sigma):
    """One forward pass. `pert` is None for the unperturbed mean policy.

    When `pert` is given it holds this member's factors; the perturbed matrix is NEVER
    formed -- the low-rank term is applied as (x A) B^T, which is the whole point of
    EGGROLL. Called under vmap, so `obs` arrives rank-1 [OBS].
    """
    x = (obs - norm[0]) / (norm[1] + 1e-6)
    L = len(theta["W"])
    for i in range(L):
        h = x @ theta["W"][i] + theta["b"][i]
        if pert is not None:
            if pert["A"][i] is not None:                 # low-rank branch
                h = h + sigma * (x @ pert["A"][i]) @ pert["B"][i].T
            else:                                        # full-rank control branch
                h = h + sigma * (x @ pert["E"][i])
            h = h + sigma * pert["e"][i]
        x = jnp.tanh(h) if i < L - 1 else jnp.clip(h, -1.0, 1.0)
    return x


# =============================================================================
# Perturbation sampling
# =============================================================================

def sample_pert(key, shapes, half, rank):
    """Antithetic population of P = 2*half perturbations.

    rank > 0  -> EGGROLL: E_i = A_i B_i^T / sqrt(rank), per-entry variance 1.
    rank == 0 -> the full-rank Gaussian control, same interface.

    The mirrored half negates A (equivalently E), which flips the sign of the outer
    product without a second draw. The negated copy is materialised rather than carried
    as a sign flag: at this size it costs nothing, and the alternative complicates the
    vmapped gather for no measurable gain. For Phase 2's larger tables, revisit.
    """
    A, B, E, e = [], [], [], []
    keys = jax.random.split(key, 3 * len(shapes))
    for i, (m, n) in enumerate(shapes):
        kb = keys[3 * i + 2]
        eh = jax.random.normal(kb, (half, n))
        e.append(jnp.concatenate([eh, -eh]))
        if rank > 0:
            ah = jax.random.normal(keys[3 * i], (half, m, rank)) / jnp.sqrt(rank)
            bh = jax.random.normal(keys[3 * i + 1], (half, n, rank))
            A.append(jnp.concatenate([ah, -ah]))
            B.append(jnp.concatenate([bh, bh]))
            E.append(None)
        else:
            Eh = jax.random.normal(keys[3 * i], (half, m, n))
            E.append(jnp.concatenate([Eh, -Eh]))
            A.append(None)
            B.append(None)
    return dict(A=A, B=B, E=E, e=e)


def es_gradient(pert, u, shapes):
    """(1/P) sum_i u_i E_i -- the EGGROLL update direction of Algorithm 1.

    Kept in factored form: sum_i u_i A_i B_i^T is a single einsum over the population,
    so the P individual m x n matrices are never built. Individual perturbations are
    rank r, but this SUM is full-rank (up to rank P*r), which is the paper's point --
    the parameter update is not restricted to a low-rank subspace.

    No 1/sigma: the paper absorbs it into the learning rate, and Adam is invariant to a
    constant rescale of the whole estimate anyway.
    """
    P = u.shape[0]
    gW, gb = [], []
    for i, _ in enumerate(shapes):
        if pert["A"][i] is not None:
            g = jnp.einsum("p,pmr,pnr->mn", u, pert["A"][i], pert["B"][i])
        else:
            g = jnp.einsum("p,pmn->mn", u, pert["E"][i])
        gW.append(g / P)
        gb.append(jnp.einsum("p,pn->n", u, pert["e"][i]) / P)
    return dict(W=gW, b=gb)


def shape_fitness(f, rank_transform):
    """The paper's `rank_transform` switch, tuned per environment in its HPO.

    True  -> centred ranks in [-0.5, +0.5] (standard OpenAI-ES shaping): insensitive to
             the reward scale and to the members that fall instantly and score ~0.
    False -> z-scored raw fitness, i.e. Algorithm 1's bare f_i, standardised so the step
             size does not track the reward magnitude. Walker2d returns span 0..5000+
             across a population, so feeding raw f_i unscaled would make the effective
             learning rate drift by orders of magnitude over a run.
    """
    if rank_transform:
        order = jnp.argsort(jnp.argsort(f))
        return order / (f.shape[0] - 1) - 0.5
    return (f - f.mean()) / (f.std() + 1e-8)


# =============================================================================
# Batched fitness: one vmapped MJX rollout for the whole population
# =============================================================================

def make_fitness(mx, norm, shapes, episodes, horizon, eval_episodes):
    """One rollout per generation, returning (population fitness, mean-policy fitness).

    The mean policy is evaluated IN THE SAME SCAN as a member whose perturbation is
    exactly zero, rather than in a second rollout. This is not a micro-optimisation: an
    MJX rollout is latency-bound on the sequential physics scan, so a separate 500-step
    eval of 16 envs cost almost as much as the 512-env population rollout -- 40% of
    generation time for a diagnostic. Appending `eval_episodes` envs to an existing
    batch is nearly free.

    Why the mean policy is worth measuring at all: population best is a max over P noisy
    draws and drifts upward with P even when the mean is standing still, so it is not a
    progress signal. The mean is the thing that has to improve.
    """
    reset, step = W.make_env(mx)
    v_reset, v_step = jax.vmap(reset), jax.vmap(step)

    def fitness(theta, pert, key, sigma):
        P = pert["e"][0].shape[0]
        n = P * episodes + eval_episodes
        st = v_reset(jax.random.split(key, n))
        # member index P is the zero perturbation == the unperturbed mean policy
        who = jnp.concatenate([jnp.repeat(jnp.arange(P), episodes),
                               jnp.full((eval_episodes,), P)])
        padded = jax.tree.map(
            lambda x: (jnp.concatenate([x, jnp.zeros_like(x[:1])])
                       if x is not None else None),
            pert, is_leaf=lambda x: x is None)
        # Materialise the per-env perturbation ONCE, outside the scan. Gathering inside
        # re-ran the gather every physics step and dominated generation time in exp_c05.
        pe = jax.tree.map(lambda x: x[who] if x is not None else None, padded,
                          is_leaf=lambda x: x is None)

        def one(carry):
            st, ret, alive, t = carry
            act = jax.vmap(lambda p, o: apply_policy(theta, p, o, norm, sigma))(pe, st.obs)
            nst = v_step(st, act)
            ret = ret + nst.reward * alive
            alive = alive * (1.0 - nst.done)
            return (nst, ret, alive, t + 1)

        # while_loop, not scan: `alive` latches to 0 on an env's first termination and
        # the env auto-resets, so every step after the LAST env falls adds exactly
        # nothing to any return. Early in evolution nearly every member falls inside
        # ~60 steps, so a fixed 1000-step scan spends >90% of its physics on dead
        # batches. Stopping when the whole batch is dead is numerically identical --
        # only steps whose contribution is multiplied by alive=0 are skipped.
        def cond(carry):
            _, _, alive, t = carry
            return (t < horizon) & (alive.sum() > 0)

        st, ret, alive, steps = jax.lax.while_loop(
            cond, one, (st, jnp.zeros(n), jnp.ones(n), 0))
        pop_ret, mean_ret = ret[:P * episodes], ret[P * episodes:]
        return pop_ret.reshape(P, episodes).mean(1), mean_ret.mean(), steps

    return jax.jit(fitness)


# =============================================================================
# The loop
# =============================================================================

def run(a, theta, fitness, shapes, key, log):
    m_ad = jax.tree.map(jnp.zeros_like, theta)
    v_ad = jax.tree.map(jnp.zeros_like, theta)
    half = a.pop // 2
    best_mean, rows = -1e9, []
    steps_done = 0
    t0 = time.time()
    for g in range(a.gens):
        key, k1, k2 = jax.random.split(key, 3)
        # Both decays are the paper's (lr_decay / sigma_decay, tuned per environment).
        sigma_g = a.sigma * (a.sigma_decay ** g)
        lr_g = a.lr * (a.lr_decay ** g)
        pert = sample_pert(k1, shapes, half, a.rank)
        f, fm, rollout_steps = fitness(theta, pert, k2, sigma_g)
        u = shape_fitness(f, not a.no_rank_transform)
        grad = es_gradient(pert, u, shapes)

        b1, b2 = 0.9, 0.999
        m_ad = jax.tree.map(lambda m, gr: b1 * m + (1 - b1) * gr, m_ad, grad)
        v_ad = jax.tree.map(lambda v, gr: b2 * v + (1 - b2) * gr ** 2, v_ad, grad)
        mc = 1 - b1 ** (g + 1)
        vc = 1 - b2 ** (g + 1)
        theta = jax.tree.map(
            lambda t, m, v: t + lr_g * (m / mc) / (jnp.sqrt(v / vc) + 1e-8),
            theta, m_ad, v_ad)

        # `fm` is the mean policy measured at the START of this generation -- the same
        # theta the population was sampled around, from the same rollout. It is not the
        # post-update mean; the post-update number is generation g+1's.
        fm = float(fm)
        best_mean = max(best_mean, fm)
        steps_done += int(rollout_steps) * (a.pop * a.episodes + a.eval_episodes)
        rows.append(dict(gen=g, mean_policy=fm, pop_best=float(f.max()),
                         pop_mean=float(f.mean()), sigma=round(float(sigma_g), 5),
                         lr=round(float(lr_g), 6), rollout_steps=int(rollout_steps),
                         env_steps=steps_done,
                         elapsed_s=round(time.time() - t0, 1)))
        if g % 10 == 0 or g == a.gens - 1:
            log(f"  gen {g:>4}/{a.gens}  mean-policy {fm:8.1f}  pop best {float(f.max()):8.1f}"
                f"  pop mean {float(f.mean()):8.1f}  (best mean {best_mean:8.1f})"
                f"  rollout {int(rollout_steps):>4}/{a.horizon}  {time.time()-t0:6.0f}s")
    return theta, best_mean, rows, steps_done


def save_theta(path, theta):
    """Save as named arrays, not a flat vector.

    exp_c05 saved a flat `mu.npy` that only means anything alongside the exact
    `--hidden` it was trained with; reconstructing it wrongly is silent, not an error.
    """
    np.savez(path, **{f"W{i}": np.asarray(w) for i, w in enumerate(theta["W"])},
             **{f"b{i}": np.asarray(b) for i, b in enumerate(theta["b"])})


def load_theta(path):
    z = np.load(path)
    n = sum(1 for k in z.files if k.startswith("W"))
    return dict(W=[jnp.asarray(z[f"W{i}"]) for i in range(n)],
                b=[jnp.asarray(z[f"b{i}"]) for i in range(n)])


def main():
    ap = argparse.ArgumentParser()
    # Defaults follow the paper's tuned brax/ant EGGROLL column (Table 19), the nearest
    # published relative of Walker2d: 3x256 MLP, adam, lr 0.01, sigma 0.05, and BOTH
    # decays 0.9995. Read those decays carefully -- 0.995 vs 0.9995 is not a rounding
    # difference: over 300 generations they leave 22% vs 86% of the starting sigma, i.e.
    # a completely different exploration schedule. An earlier version of this file had
    # 0.995, taken from the HPO *range* table rather than the tuned ant row.
    #
    # Two deliberate departures, both documented at their flags: --pop (budget) and
    # --no-rank-transform (ant tuned to false, humanoid to true; it is per-environment
    # and untuned for Walker2d, so this keeps centred ranks to match exp_c05's ES arm).
    ap.add_argument("--rank", type=int, default=4,
                    help="EGGROLL perturbation rank; 0 = full-rank Gaussian control")
    ap.add_argument("--gens", type=int, default=300)
    ap.add_argument("--pop", type=int, default=1024,
                    help="must be even (antithetic). The paper used 2048 on brax/ant; "
                         "1024 keeps a generation at ~2M env-steps so a 300-gen arm "
                         "fits in ~2.5h on one 5090")
    ap.add_argument("--episodes", type=int, default=2,
                    help="the paper's n_parallel_evaluations; it used 1 on brax/ant, "
                         "2 here to halve the fitness noise")
    ap.add_argument("--horizon", type=int, default=1000)
    ap.add_argument("--sigma", type=float, default=0.05)
    ap.add_argument("--sigma-decay", type=float, default=0.9995)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--lr-decay", type=float, default=0.9995)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--layers", type=int, default=3, help="number of weight matrices")
    ap.add_argument("--no-rank-transform", action="store_true",
                    help="use z-scored raw fitness instead of centred ranks")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-episodes", type=int, default=16,
                    help="MJX episodes for the mean-policy progress number")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    assert a.pop % 2 == 0, "--pop must be even: sampling is antithetic"

    st = json.load(open(os.path.join(BASE, "exp_c03_distillation", "dataset_stats.json")))
    norm = (jnp.asarray(st["obs_mean"], jnp.float32),
            jnp.asarray(st["obs_std"], jnp.float32))

    shapes = mlp_spec(a.hidden, a.layers)
    d = n_params(shapes)
    key = jax.random.PRNGKey(a.seed)
    key, ki = jax.random.split(key)
    theta = init_theta(shapes, ki)

    mx = mjx.put_model(W.make_model())
    fitness = make_fitness(mx, norm, shapes, a.episodes, a.horizon, a.eval_episodes)

    kind = f"eggroll-r{a.rank}" if a.rank > 0 else "fullrank-ES"
    name = f"eggroll_mlp{a.hidden}x{a.layers}_r{a.rank}_s{a.seed}{a.tag}"
    noise_floats = (a.pop * sum(a.rank * (m + n) for m, n in shapes) if a.rank > 0
                    else a.pop * sum(m * n for m, n in shapes))
    full_floats = a.pop * sum(m * n for m, n in shapes)
    print(f"[{name}] {kind} | MLP {a.layers}x{a.hidden} d={d:,} | pop={a.pop} x "
          f"{a.episodes} eps x {a.horizon} steps = {a.pop*a.episodes*a.horizon:,} "
          f"env-steps/gen | {a.gens} gens | noise {noise_floats:,} floats/gen "
          f"({full_floats/max(noise_floats,1):.1f}x less than full-rank)", flush=True)

    theta, best_mean, rows, steps_done = run(a, theta, fitness, shapes, key,
                                             lambda s: print(s, flush=True))

    save_theta(os.path.join(HERE, f"{name}_theta.npz"), theta)
    out = dict(name=name, kind=kind, rank=a.rank, hidden=a.hidden, layers=a.layers,
               d=d, seed=a.seed, rank_transform=not a.no_rank_transform,
               gens=a.gens, pop=a.pop, episodes=a.episodes, horizon=a.horizon,
               sigma=a.sigma, sigma_decay=a.sigma_decay, lr=a.lr, lr_decay=a.lr_decay,
               noise_floats_per_gen=noise_floats, full_rank_floats_per_gen=full_floats,
               total_env_steps=steps_done,
               budget_env_steps=a.gens * (a.pop * a.episodes + a.eval_episodes) * a.horizon,
               best_mean_policy_mjx=best_mean,
               final_mean_policy_mjx=rows[-1]["mean_policy"],
               wall_s=rows[-1]["elapsed_s"], history=rows)
    json.dump(out, open(os.path.join(HERE, f"{name}.json"), "w"), indent=1)
    print(f"[{name}] best mean-policy MJX fitness {best_mean:.1f} in "
          f"{rows[-1]['elapsed_s']/60:.1f} min (MJX horizon-{a.horizon} PROXY, not the "
          f"CPU reference number -- run eval_cpu_eggroll.py)", flush=True)


if __name__ == "__main__":
    main()
