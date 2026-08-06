"""exp_c38 — decompose the membrane's cost and test spellings that survive the
determinism flag.

The membrane is ~95% of this actor's cost and the flag `--xla_gpu_deterministic_ops=true`
changes which spellings are usable, so both axes have to be measured together. ONE VARIANT
PER PROCESS, selected by argv, because `lax.sort` under the flag does not merely run slowly
-- it produces no output for 16+ minutes -- and a single process benchmarking everything
would never reach the later variants.

Variants:
  argsort_only     jnp.argsort alone (no permutation applied)
  sort_only        jnp.sort alone (values, no indices)
  gather_only      one take_along_axis with a precomputed permutation
  argsort_gather   argsort + 2 x take_along_axis            <- the c30-c37 spelling
  lax_sort         lax.sort, 2 operands, num_keys=1         <- 39x faster, flag-hostile
  rank_matmul      pairwise rank + permutation applied as a ONE-HOT CONTRACTION
  membrane_*       the same four, through the full membrane

`rank_matmul` is the interesting one. The permutation is recovered without a sort at all:
rank_k = #{j : a_j < a_k, ties broken by index} is a pairwise comparison, and applying it
is P @ a with P[r,k] = 1[rank_k == r] -- a MATMUL, which is deterministic by construction
and needs no serialisation. N=17, so the pairwise tensor is (B,T,D,17,17). Gradients are
correct because P is piecewise-constant in a, exactly as torch.sort's permutation is.

Usage:
  python bench_sort_variants.py <variant>          # add XLA_FLAGS to test under the flag
"""
import os
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import jax_mhl_lut as LIF                                   # noqa: E402

B, OBS = 512, 17
HEADS, TPH, NDET, NB = 1, 32, 6, 2
REPS = 30
TW = LIF.T_WINDOW


def arrivals(p, x):
    lat = LIF.latency(x)
    d = jnp.clip(p["delay"], 0.0, TW)
    return lat[:, None, None, :] + d[None]


# --- the permutation spellings -----------------------------------------------
def s_argsort_gather(a, w):
    idx = jnp.argsort(a, axis=-1, stable=True)
    return (jnp.take_along_axis(a, idx, axis=-1),
            jnp.take_along_axis(jnp.broadcast_to(w[None], a.shape), idx, axis=-1))


def s_lax_sort(a, w):
    return jax.lax.sort((a, jnp.broadcast_to(w[None], a.shape)),
                        dimension=-1, num_keys=1, is_stable=True)


def s_rank_matmul(a, w):
    """Stable rank by pairwise comparison, then permute by a one-hot contraction."""
    n = a.shape[-1]
    ai, aj = a[..., :, None], a[..., None, :]              # (...,N,1) vs (...,1,N)
    ii = jnp.arange(n)
    earlier = (aj < ai) | ((aj == ai) & (ii[None, :] < ii[:, None]))
    rank = earlier.sum(-1)                                  # (...,N) stable rank of each k
    P = (rank[..., None, :] == jnp.arange(n)[:, None]).astype(a.dtype)   # (...,N_r,N_k)
    wb = jnp.broadcast_to(w[None], a.shape)
    return ((P * a[..., None, :]).sum(-1), (P * wb[..., None, :]).sum(-1))


SORTS = dict(argsort_gather=s_argsort_gather, lax_sort=s_lax_sort,
             rank_matmul=s_rank_matmul)


def main():
    variant = sys.argv[1]
    key = jax.random.PRNGKey(0)
    kp, kx = jax.random.split(key)
    p = LIF.init(kp, NB, NDET, TPH, HEADS, OBS, 12, delay_init_std=4.0)
    x = jax.random.normal(kx, (B, OBS))
    a0 = arrivals(p, x)
    perm = jnp.argsort(a0, axis=-1, stable=True)
    tau = LIF._tau(p["tau_raw"])[None, :, :, None]
    flag = os.environ.get("XLA_FLAGS", "(none)")

    if variant == "argsort_only":
        fn = jax.jit(lambda p, x: jnp.argsort(arrivals(p, x), axis=-1, stable=True))
        args = (p, x)
    elif variant == "sort_only":
        fn = jax.jit(lambda p, x: jnp.sort(arrivals(p, x), axis=-1))
        args = (p, x)
    elif variant == "gather_only":
        fn = jax.jit(lambda a, i: jnp.take_along_axis(a, i, axis=-1))
        args = (a0, perm)
    elif variant in SORTS:
        s = SORTS[variant]
        fn = jax.jit(lambda p, x: s(arrivals(p, x), LIF.synapses(p)))
        args = (p, x)
    elif variant.startswith("membrane_"):
        s = SORTS[variant[len("membrane_"):]]

        def m(p, x):
            a_s, w_s = s(arrivals(p, x), LIF.synapses(p))
            return LIF.membrane_linear(a_s, w_s, tau)
        fn = jax.jit(m)
        args = (p, x)
    elif variant.startswith("grad_"):
        s = SORTS[variant[len("grad_"):]]

        def loss(p, x):
            a_s, w_s = s(arrivals(p, x), LIF.synapses(p))
            return LIF.membrane_linear(a_s, w_s, tau).sum()
        fn = jax.jit(jax.grad(loss))
        args = (p, x)
    else:
        raise SystemExit(f"unknown variant {variant!r}")

    t0 = time.time()
    out = jax.block_until_ready(fn(*args))
    compile_s = time.time() - t0
    t0 = time.time()
    for _ in range(REPS):
        out = fn(*args)
    jax.block_until_ready(out)
    ms = (time.time() - t0) / REPS * 1e3
    print(f"{variant:<24} {ms:9.3f} ms   (compile {compile_s:6.2f} s)   "
          f"XLA_FLAGS={flag}", flush=True)

    # Correctness: every permutation spelling must agree with the reference argsort form.
    if variant in SORTS or variant.startswith("membrane_"):
        ref_a, ref_w = s_argsort_gather(a0, LIF.synapses(p))
        got = SORTS[variant.replace("membrane_", "")](a0, LIF.synapses(p)) \
            if variant.startswith("membrane_") else out
        da = float(np.abs(np.asarray(got[0]) - np.asarray(ref_a)).max())
        dw = float(np.abs(np.asarray(got[1]) - np.asarray(ref_w)).max())
        print(f"{'':<24} vs argsort_gather: max|Δa| {da:.3e}  max|Δw| {dw:.3e}",
              flush=True)


if __name__ == "__main__":
    main()
