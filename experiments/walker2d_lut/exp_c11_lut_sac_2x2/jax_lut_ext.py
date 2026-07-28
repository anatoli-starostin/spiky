"""exp_c11 — two additions to the JAX LUT needed for a real-training 2x2 (#75).

1. **Anchor-pair addressing.** No new forward is required: an anchor pair (p1, p2) is
   exactly the hyperplane w_i = e_p1 - e_p2, b_i = 0. So this writes those sparse +/-1
   rows (sampled the same balanced way `get_balanced_anchor_pairs` does) and the caller
   freezes w/b, giving `FastMultiHeadLut` semantics through the existing code path.

2. **hybrid_smooth forward**, ported from
   `hyperplane_multi_head_lut._hyperplane_smooth_fwd_embedding`:

       main = affine sign-pack of (a_i > 0)
       alt  = main with the bit at argmin|a| flipped
       u    = sigmoid(-Delta / T_sel),  Delta = 2*a_min / (T_soft + a_min)
       out  = sum_t [(1-u) * W[main] + u * W[alt]]

   Backward reuses the same full-K softmax surrogate as the hard forward (that is what
   torch does too — the surrogate is shared; only the forward differs), so the verified
   gradient path is untouched.

Both are checked against torch by `verify_ext.py`.
"""
import functools

import jax
import jax.numpy as jnp
import numpy as np

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "exp_c06_jax_backprop"))
import jax_lut_grad as L  # noqa: E402


# =============================================================================
# 1. Anchor-pair addressing as a frozen hyperplane
# =============================================================================

def anchor_pair_wb(rng, n_tables, nap, input_dim):
    """w[t,i] = e_a - e_b, b = 0 — i.e. bit_i = 1[x[a] - x[b] > 0].

    Balanced sampling: every table draws its 2*NAP endpoints without replacement so no
    coordinate is used twice inside a table, matching the spirit of
    `get_balanced_anchor_pairs`.
    """
    w = np.zeros((n_tables, nap, input_dim), np.float32)
    for t in range(n_tables):
        need = min(2 * nap, input_dim)
        pick = rng.choice(input_dim, size=need, replace=False)
        if need < 2 * nap:                       # tiny input_dim: allow reuse
            pick = np.concatenate([pick, rng.choice(input_dim, 2 * nap - need)])
        for i in range(nap):
            w[t, i, pick[2 * i]] = 1.0
            w[t, i, pick[2 * i + 1]] = -1.0
    return jnp.asarray(w), jnp.zeros((n_tables, nap), jnp.float32)


# =============================================================================
# 2. hybrid_smooth forward (top-2 blend), sharing the verified soft backward
# =============================================================================

def _smooth_out(x, w, b, weights, log_T_soft, log_T_sel, n_heads, tph):
    nap = w.shape[1]
    T_soft, T_sel = jnp.exp(log_T_soft), jnp.exp(log_T_sel)
    a = L._project(x, w, b)                                   # [B, T, NAP]
    powers = 2 ** jnp.arange(nap - 1, -1, -1)
    main = ((a > 0).astype(jnp.int32) * powers[None, None, :]).sum(-1)
    abs_a = jnp.abs(a)
    p_star = jnp.argmin(abs_a, axis=-1)                       # least-confident bit
    alt = jnp.bitwise_xor(main, powers[p_star])
    a_min = jnp.take_along_axis(abs_a, p_star[..., None], axis=-1).squeeze(-1)
    u = jax.nn.sigmoid(-(2.0 * a_min / (T_soft + a_min)) / T_sel)   # in (0, 0.5]
    mr = jnp.take_along_axis(weights[None], main[:, :, None, None], axis=2).squeeze(2)
    ar = jnp.take_along_axis(weights[None], alt[:, :, None, None], axis=2).squeeze(2)
    blended = mr * (1.0 - u)[..., None] + ar * u[..., None]
    return blended.reshape(x.shape[0], n_heads, tph, -1).sum(2), main, alt, u


@functools.partial(jax.custom_vjp, nondiff_argnums=(6, 7))
def lut_apply_smooth(x, w, b, weights, log_T_soft, log_T_sel, n_heads, tph):
    return _smooth_out(x, w, b, weights, log_T_soft, log_T_sel, n_heads, tph)[0]


def _fwd(x, w, b, weights, log_T_soft, log_T_sel, n_heads, tph):
    out, main, alt, u = _smooth_out(x, w, b, weights, log_T_soft, log_T_sel,
                                    n_heads, tph)
    return out, (x, w, b, weights, log_T_soft, log_T_sel, main, alt, u)


def _bwd(n_heads, tph, res, g):
    """Hybrid-smooth backward — a HYBRID of a different kind from the hard one.

    Matching `HyperplaneMultiHeadLUT._HyperplaneHybridSmooth.backward`:

      * x / w / b / temperatures -> the SAME full-K softmax surrogate as the hard
        forward, pinned to `main` (torch calls the shared body with
        `compute_weight_grad=False` for exactly this reason);
      * table weights -> a **2-ROW** scatter, (1-u) at `main` and u at `alt`.

    The 2-row weight gradient is the part that differs. An earlier version of this
    file reused the hard backward wholesale, which scatters a SINGLE row at `main` —
    correct for the hard forward, wrong here: it drops the `alt` row's contribution
    entirely and mis-weights `main` by 1 instead of (1-u). Every gradient except the
    table values was already right, which is why it trained at all and why the error
    was easy to miss.
    """
    x, w, b, weights, log_T_soft, log_T_sel, main, alt, u = res
    T, nap, _ = w.shape
    K, n_out = weights.shape[1], weights.shape[2]
    B = x.shape[0]
    grad_pt = jnp.broadcast_to(g[:, :, None, :], (B, n_heads, tph, n_out)
                               ).reshape(B, T, n_out)

    # --- table weights: 2-row scatter, (1-u) at main and u at alt -----------
    offs = (jnp.arange(T) * K)[None, :]
    gw = jnp.zeros((T * K, n_out), grad_pt.dtype)
    gw = gw.at[(main + offs).reshape(-1)].add(
        ((1.0 - u)[..., None] * grad_pt).reshape(-1, n_out))
    gw = gw.at[(alt + offs).reshape(-1)].add(
        (u[..., None] * grad_pt).reshape(-1, n_out))
    grad_weights = gw.reshape(weights.shape)

    # --- x, w, b, temperatures: the shared full-K softmax surrogate ---------
    bm = L.bit_matrix_msb(nap)
    wt = jax.lax.stop_gradient(weights)
    f = lambda x_, w_, b_, ts_, tl_: L._soft_surrogate(x_, w_, b_, wt, ts_, tl_,
                                                       main, bm)
    _, vjp = jax.vjp(f, x, w, b, log_T_soft, log_T_sel)
    gx, gw_hp, gb, gts, gtl = vjp(grad_pt)
    return gx, gw_hp, gb, grad_weights, gts, gtl


lut_apply_smooth.defvjp(_fwd, _bwd)


def apply(mode):
    return lut_apply_smooth if mode == "hybrid_smooth" else L.lut_apply
