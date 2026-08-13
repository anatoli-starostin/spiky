"""exp_c24 — sep-CMA-ES, the real one, in JAX (#75).

Implements **Ros & Hansen (2008), "A Simple Modification in CMA-ES Achieving Linear Time
and Space Complexity"**: standard CMA-ES with the covariance matrix restricted to its
diagonal, and the covariance learning rate multiplied by (n+2)/3 to exploit the fact that
only n parameters are being learned instead of n(n+1)/2.

WHY THIS FILE EXISTS. `exp_c05_es/es_mjx.py` has an `--algo sepcma` branch that RESULTS.md
reports as sep-CMA-ES. It is not. Read line by line, that branch does isotropic antithetic
sampling, centred-rank fitness shaping, an **Adam** step on the ES gradient estimate, and
an ad-hoc per-coordinate step-size heuristic. None of CMA-ES's actual machinery is
present: no weighted recombination of the best mu of lambda, no evolution paths, no
cumulative step-size adaptation, no rank-one or rank-mu covariance update. It is
OpenAI-ES with a diagonal step size. This file is the algorithm that name refers to.

WHAT IS ACTUALLY DIFFERENT, and why it might matter here:
  * **Weighted recombination.** The mean jumps to a log-weighted average of the best mu
    candidates. ES instead nudges the mean along a rank-weighted average of ALL of them.
  * **Evolution paths.** p_sigma and p_c accumulate correlated progress across
    generations, so consistent directions are amplified and random walk is not.
  * **CSA.** The step size is driven by whether the evolution path is longer or shorter
    than a random walk would be -- a principled signal, unlike a hand-tuned decay.
  * **A real covariance.** Per-coordinate variances are learned from the actual spread of
    successful steps. This is the part most likely to matter for the eventual LUT target,
    where addressing parameters and table entries have genuinely different natural scales
    and a single isotropic sigma has to compromise between them.

CONVENTIONS. The optimiser MINIMISES internally, which is what every CMA-ES reference and
every test function assumes. `tell()` takes a `maximise` flag and negates once, at the
sorting step, so nothing downstream has to remember the sign.

Constants follow the standard CMA-ES formulation (Hansen's tutorial, arXiv:1604.00772)
with the Ros & Hansen (n+2)/3 scaling applied to c_1 and c_mu. Ros & Hansen's own paper
writes c_cov in the older mu_cov form; the two agree to within the usual reparametrisation
and the modern form is what the reference implementations use.

A NOTE ON NOISE. Walker2d fitness is stochastic (2 episodes per candidate), and CMA-ES is
more sensitive to fitness noise than rank-shaped ES is, because the covariance is learned
from which candidates actually won. The mitigation used here is a large population, which
is nearly free on this harness (an MJX rollout is latency-bound on the sequential physics
scan, so 4x the envs costs ~1.5x the wall-clock). It is not a complete answer; if the
covariance turns out to be noise-driven, the next step is more episodes per candidate.
"""
import math

import jax
import jax.numpy as jnp

# Same reasoning as exp_c23: TF32 tensor cores cost ~1e-3 relative accuracy, and the
# matmuls here are trivial next to the physics. An optimiser that is quietly approximate
# is not worth debugging later.
jax.config.update("jax_default_matmul_precision", "highest")


def default_lambda(n):
    """CMA-ES's textbook population size.

    NOTE this is 26 at n=1830 and is the WRONG choice on the MJX harness, where a
    26-member generation costs nearly the same wall-clock as a 1024-member one. It is
    provided because the test functions (cheap, serial) want it, not because Walker2d
    should use it.
    """
    return 4 + int(3 * math.log(n))


def init(n, lam=None, sigma0=0.3, x0=None):
    """Build the sep-CMA-ES state. All constants are computed once, here."""
    lam = default_lambda(n) if lam is None else int(lam)
    assert lam >= 4, "population must be at least 4"
    mu = lam // 2

    # log-decreasing recombination weights over the best mu, normalised to sum 1
    w = jnp.log(mu + 0.5) - jnp.log(jnp.arange(1, mu + 1))
    w = w / w.sum()
    mu_eff = float(1.0 / jnp.sum(w ** 2))

    # step-size control (CSA)
    c_sigma = (mu_eff + 2.0) / (n + mu_eff + 5.0)
    d_sigma = (1.0 + 2.0 * max(0.0, math.sqrt((mu_eff - 1.0) / (n + 1.0)) - 1.0)
               + c_sigma)
    # covariance path
    c_c = (4.0 + mu_eff / n) / (n + 4.0 + 2.0 * mu_eff / n)
    # covariance learning rates, then the Ros & Hansen separable speed-up
    c_1 = 2.0 / ((n + 1.3) ** 2 + mu_eff)
    c_mu = min(1.0 - c_1,
               2.0 * (mu_eff - 2.0 + 1.0 / mu_eff) / ((n + 2.0) ** 2 + mu_eff))
    sep = (n + 2.0) / 3.0
    c_1, c_mu = c_1 * sep, c_mu * sep
    if c_1 + c_mu > 1.0:                      # the scaling can overshoot at small n
        s = 1.0 / (c_1 + c_mu)
        c_1, c_mu = c_1 * s, c_mu * s

    chi_n = math.sqrt(n) * (1.0 - 1.0 / (4.0 * n) + 1.0 / (21.0 * n * n))

    return dict(
        n=n, lam=lam, mu=mu, w=w, mu_eff=mu_eff,
        c_sigma=c_sigma, d_sigma=d_sigma, c_c=c_c, c_1=c_1, c_mu=c_mu, chi_n=chi_n,
        m=jnp.zeros(n) if x0 is None else jnp.asarray(x0, jnp.float32),
        sigma=float(sigma0),
        C=jnp.ones(n),          # DIAGONAL of the covariance -- the whole point
        p_sigma=jnp.zeros(n), p_c=jnp.zeros(n),
        gen=0,
    )


def ask(state, key):
    """Sample lambda candidates. Returns (x [lam, n], z [lam, n]).

    x = m + sigma * D * z with D = sqrt(diag(C)); z is kept because the covariance and
    path updates are written in terms of it.
    """
    z = jax.random.normal(key, (state["lam"], state["n"]))
    D = jnp.sqrt(state["C"])
    x = state["m"][None, :] + state["sigma"] * z * D[None, :]
    return x, z


def tell(state, x, fitness, maximise=False):
    """One full sep-CMA-ES update from the evaluated population."""
    n, w, mu = state["n"], state["w"], state["mu"]
    mu_eff, chi_n = state["mu_eff"], state["chi_n"]
    c_sigma, d_sigma = state["c_sigma"], state["d_sigma"]
    c_c, c_1, c_mu = state["c_c"], state["c_1"], state["c_mu"]

    # CMA-ES minimises; negate once, here, if the caller is maximising.
    key_f = -fitness if maximise else fitness
    order = jnp.argsort(key_f)[:mu]                     # best mu, best first
    x_sel = x[order]                                    # [mu, n]

    m_old, sigma, C = state["m"], state["sigma"], state["C"]
    D = jnp.sqrt(C)

    m_new = jnp.sum(w[:, None] * x_sel, axis=0)
    y_w = (m_new - m_old) / sigma                       # weighted mean step, in z-space

    # --- step-size path. C^{-1/2} is just 1/D for a diagonal covariance, which is the
    # entire reason sep-CMA-ES is O(n): no eigendecomposition anywhere.
    p_sigma = ((1.0 - c_sigma) * state["p_sigma"]
               + math.sqrt(c_sigma * (2.0 - c_sigma) * mu_eff) * (y_w / D))

    g = state["gen"] + 1
    # h_sigma stalls the rank-one update when the path is suspiciously long, which is what
    # happens right after a large step-size increase.
    denom = math.sqrt(1.0 - (1.0 - c_sigma) ** (2 * g))
    h_sigma = (jnp.linalg.norm(p_sigma) / denom
               < (1.4 + 2.0 / (n + 1.0)) * chi_n).astype(jnp.float32)

    p_c = ((1.0 - c_c) * state["p_c"]
           + h_sigma * math.sqrt(c_c * (2.0 - c_c) * mu_eff) * y_w)

    # --- covariance: rank-one from the path, rank-mu from the selected steps. Diagonal
    # throughout, so these are elementwise squares rather than outer products.
    y_sel = (x_sel - m_old[None, :]) / sigma            # [mu, n]
    rank_mu = jnp.sum(w[:, None] * y_sel ** 2, axis=0)
    delta_h = (1.0 - h_sigma) * c_c * (2.0 - c_c)       # compensates the stalled path
    C_new = ((1.0 - c_1 - c_mu) * C
             + c_1 * (p_c ** 2 + delta_h * C)
             + c_mu * rank_mu)
    C_new = jnp.maximum(C_new, 1e-20)                   # variances stay positive

    sigma_new = sigma * jnp.exp((c_sigma / d_sigma)
                                * (jnp.linalg.norm(p_sigma) / chi_n - 1.0))

    new = dict(state)
    new.update(m=m_new, sigma=float(sigma_new), C=C_new, p_sigma=p_sigma, p_c=p_c, gen=g)
    return new
