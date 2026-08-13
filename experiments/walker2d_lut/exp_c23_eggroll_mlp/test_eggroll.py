"""exp_c23 — correctness tests for the EGGROLL implementation (#75).

Four things can be silently wrong in a low-rank ES and none of them show up as a crash;
they show up as a run that merely learns badly, which is indistinguishable from "ES is
hard". So each is asserted directly against a naive reference:

  1. The factored forward x(W + sigma A B^T/sqrt(r)) must equal the forward through the
     explicitly materialised perturbed matrix. This is the optimisation that makes
     EGGROLL worth using; if it is wrong, every fitness is wrong.
  2. Var(E_ij) must be 1, matching the full-rank N(0,1) noise it replaces. This is what
     the paper's 1/sqrt(r) buys and it is why --sigma means the same thing at any rank.
     Get it wrong and rank silently rescales the search radius.
  3. The einsum aggregate must equal the explicit sum_i u_i E_i. This is the update.
  4. Antithetic pairing must give E_{i+half} = -E_i exactly.

Test 3 earned its keep immediately: it failed at 1.8e-4 on first run, which was not a
bug in the einsum but JAX's default TF32 tensor-core matmul. `eggroll.py` now pins
`jax_default_matmul_precision=highest` because of it.

Run (no GPU needed, but it will use one if present):
  XLA_PYTHON_CLIENT_PREALLOCATE=false python test_eggroll.py
"""
import os
import sys

import jax, jax.numpy as jnp
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import eggroll as E  # noqa: E402


def materialise(pert, i, member, sigma):
    """The naive perturbed matrix, built the expensive way, for reference only."""
    if pert["A"][i] is not None:
        return sigma * (pert["A"][i][member] @ pert["B"][i][member].T)
    return sigma * pert["E"][i][member]


def test_factored_forward_matches_materialised():
    key = jax.random.PRNGKey(0)
    shapes = E.mlp_spec(hidden=16, layers=3)
    half, rank, sigma = 8, 3, 0.07
    theta = E.init_theta(shapes, key)
    pert = E.sample_pert(jax.random.PRNGKey(1), shapes, half, rank)
    norm = (jnp.zeros(E.OBS), jnp.ones(E.OBS))
    obs = jax.random.normal(jax.random.PRNGKey(2), (E.OBS,))

    for member in (0, 5, half, 2 * half - 1):
        one = jax.tree.map(lambda x: x[member] if x is not None else None, pert,
                           is_leaf=lambda x: x is None)
        got = E.apply_policy(theta, one, obs, norm, sigma)

        # reference: build W + sigma*E explicitly and run a plain MLP
        x = obs
        L = len(shapes)
        for i in range(L):
            Wp = theta["W"][i] + materialise(pert, i, member, sigma)
            h = x @ Wp + theta["b"][i] + sigma * pert["e"][i][member]
            x = jnp.tanh(h) if i < L - 1 else jnp.clip(h, -1.0, 1.0)
        err = float(jnp.abs(got - x).max())
        assert err < 2e-5, f"member {member}: factored forward differs by {err:.2e}"
    print("  1. factored forward == materialised forward           OK")


def test_perturbation_variance_is_unit():
    shapes = [(64, 96)]
    half = 4000
    for rank in (1, 2, 4, 16):
        pert = E.sample_pert(jax.random.PRNGKey(rank), shapes, half, rank)
        Emat = jnp.einsum("pmr,pnr->pmn", pert["A"][0], pert["B"][0])
        v = float(Emat.var())
        assert abs(v - 1.0) < 0.05, f"rank {rank}: Var(E_ij) = {v:.4f}, expected 1.0"
        print(f"  2. rank {rank:>2}: Var(E_ij) = {v:.4f}                          OK")

    full = E.sample_pert(jax.random.PRNGKey(9), shapes, half, 0)
    vf = float(full["E"][0].var())
    assert abs(vf - 1.0) < 0.05, f"full-rank control: Var = {vf:.4f}"
    print(f"  2. full-rank control: Var(E_ij) = {vf:.4f}                 OK")


def test_gradient_matches_explicit_sum():
    shapes = E.mlp_spec(hidden=12, layers=3)
    half, rank = 16, 2
    u = jax.random.normal(jax.random.PRNGKey(3), (2 * half,))
    for r in (rank, 0):
        pert = E.sample_pert(jax.random.PRNGKey(4), shapes, half, r)
        g = E.es_gradient(pert, u, shapes)
        for i, _ in enumerate(shapes):
            ref = sum(u[p] * materialise(pert, i, p, 1.0) for p in range(2 * half))
            ref = ref / (2 * half)
            err = float(jnp.abs(g["W"][i] - ref).max())
            scale = float(jnp.abs(ref).max())
            assert err < 1e-5 * max(scale, 1.0), \
                f"rank {r} layer {i}: einsum aggregate differs by {err:.2e}"
        print(f"  3. rank {r}: einsum aggregate == explicit sum_i u_i E_i    OK")


def test_antithetic_pairs_are_exact_mirrors():
    shapes = E.mlp_spec(hidden=8, layers=2)
    half, rank = 5, 3
    for r in (rank, 0):
        pert = E.sample_pert(jax.random.PRNGKey(5), shapes, half, r)
        for i, _ in enumerate(shapes):
            for p in range(half):
                a = materialise(pert, i, p, 1.0)
                b = materialise(pert, i, p + half, 1.0)
                err = float(jnp.abs(a + b).max())
                assert err < 1e-6, f"rank {r} layer {i} pair {p}: not mirrored ({err:.2e})"
            eb = pert["e"][i]
            assert float(jnp.abs(eb[:half] + eb[half:]).max()) < 1e-6, "bias not mirrored"
        print(f"  4. rank {r}: antithetic pairs are exact mirrors           OK")


def test_low_rank_update_approaches_full_rank():
    """The paper's central claim, checked empirically at small scale.

    Individual perturbations are rank r, but the POPULATION-WEIGHTED SUM is not: with P
    members it can reach rank P*r. So the low-rank update should look increasingly like
    a full-rank one as the population grows -- measured here as the rank of the
    aggregate, which is the property that lets EGGROLL move in any direction at all.
    """
    shapes = [(40, 50)]
    for half in (2, 8, 64):
        pert = E.sample_pert(jax.random.PRNGKey(7), shapes, half, 1)
        u = jax.random.normal(jax.random.PRNGKey(8), (2 * half,))
        g = E.es_gradient(pert, u, shapes)["W"][0]
        rk = int(np.linalg.matrix_rank(np.asarray(g), tol=1e-6))
        # antithetic halves share B, so P=2*half members contribute `half` distinct
        # outer products; the aggregate rank is bounded by min(half, 40).
        print(f"  5. P={2*half:>3} rank-1 perturbations -> aggregate rank {rk:>2} "
              f"(cap {min(half, 40)})                OK")
        assert rk == min(half, 40), f"expected rank {min(half, 40)}, got {rk}"


if __name__ == "__main__":
    print("EGGROLL implementation tests")
    test_factored_forward_matches_materialised()
    test_perturbation_variance_is_unit()
    test_gradient_matches_explicit_sum()
    test_antithetic_pairs_are_exact_mirrors()
    test_low_rank_update_approaches_full_rank()
    print("ALL TESTS PASSED")
