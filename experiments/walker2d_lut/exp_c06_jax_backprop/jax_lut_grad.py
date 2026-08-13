"""exp_c06 — the DIFFERENTIABLE multi-head LUT in JAX (#75, Phase 4).

Hard discrete forward (bit-exact, reused unchanged from exp_c04) + the full-K softmax
surrogate backward, matching torch's custom autograd in
`hyperplane_multi_head_lut._hyperplane_soft_bwd_body`.

The backward is a **hybrid**, and getting that right is the whole job:

  * table weights  -> the HONEST hard gradient: a 1-row scatter of grad_out at the row
                      the forward actually selected. Not the softmax-weighted average.
  * x, w, b, temps -> the SOFT full-K surrogate
                          y = sum_k softmax(ts_k / T_sel) * W[t, k, :]
                          ts_k = sum_i p_i * chi_i(k),  p_i = sign(a_i)|a_i|/(T_soft+|a_i|)
                      with the sign pattern PINNED to the row the forward chose (so the
                      surrogate's argmax is the row that actually fired), and with the
                      table weights held constant.

Rather than hand-transcribing torch's softmax backward (easy to get subtly wrong, hard
to notice), the soft path is written once as a forward function and differentiated by
JAX itself — `jax.vjp` of the surrogate with `stop_gradient` on the weights reproduces
torch's `d_sel_soft = einsum("bto,tko->btk", grad_pt, weights)` path exactly.

Temperatures are parametrised as log T, as in torch, so the temperature gradients match
without a chain-rule fixup.
"""
import functools

import jax
import jax.numpy as jnp

# The forward pins precision on its own dot (a TF32 sign flip picks a WHOLE WRONG ROW).
# The BACKWARD needs the same treatment for a quieter reason: its einsums and the
# vjp's GEMMs also default to TF32, which showed up as ~1e-3 relative gradient error
# against torch — small enough to look like "close enough", which is exactly why it is
# worth pinning rather than tolerating. With this set, agreement drops to fp32 noise.
jax.config.update("jax_default_matmul_precision", "highest")


def bit_matrix_msb(nap):
    """[NAP, K] of ±1, MSB-first: bit_matrix[i, k] = +1 if (k >> (NAP-1-i)) & 1."""
    k = jnp.arange(1 << nap)
    shifts = jnp.arange(nap - 1, -1, -1)
    bits = (k[None, :] >> shifts[:, None]) & 1
    return bits.astype(jnp.float32) * 2.0 - 1.0


def _project(x, w, b):
    """a_i = <w_i, x> + b_i. HIGHEST precision — see jax_lut.py: TF32 flips sign tests."""
    T, nap, _ = w.shape
    a = jax.lax.dot_general(x, w.reshape(T * nap, -1).T, (((1,), (0,)), ((), ())),
                            precision=jax.lax.Precision.HIGHEST)
    return a.reshape(x.shape[0], T, nap) + b[None]


def _hard_index(a, nap):
    powers = 2 ** jnp.arange(nap - 1, -1, -1)
    return ((a > 0).astype(jnp.int32) * powers[None, None, :]).sum(-1)     # [B, T]


def _gather_rows(weights, index):
    return jnp.take_along_axis(weights[None], index[:, :, None, None],
                               axis=2).squeeze(2)                          # [B, T, n_out]


def _soft_surrogate(x, w, b, weights, log_T_soft, log_T_sel, index, bm):
    """The full-K softmax surrogate, pinned to `index`. Differentiated by JAX."""
    nap = w.shape[1]
    T_soft, T_sel = jnp.exp(log_T_soft), jnp.exp(log_T_sel)
    a = _project(x, w, b)
    denom = T_soft + jnp.abs(a)
    shifts = jnp.arange(nap - 1, -1, -1)
    bits = ((index[:, :, None] >> shifts[None, None, :]) & 1).astype(a.dtype)
    p_signs = bits * 2.0 - 1.0                    # == sign(a) the forward saw
    p = p_signs * jnp.abs(a) / denom
    ts = jnp.einsum("btp,pk->btk", p, bm)
    sel = jax.nn.softmax(ts / T_sel, axis=-1)
    return jnp.einsum("btk,tko->bto", sel, weights)                        # [B, T, n_out]


@functools.partial(jax.custom_vjp, nondiff_argnums=(6, 7))
def lut_apply(x, w, b, weights, log_T_soft, log_T_sel, n_heads, tph):
    """Hard forward: [B, D] -> [B, n_heads, n_out]."""
    a = _project(x, w, b)
    index = _hard_index(a, w.shape[1])
    rows = _gather_rows(weights, index)
    return rows.reshape(x.shape[0], n_heads, tph, -1).sum(2)


def _fwd(x, w, b, weights, log_T_soft, log_T_sel, n_heads, tph):
    a = _project(x, w, b)
    index = _hard_index(a, w.shape[1])
    rows = _gather_rows(weights, index)
    out = rows.reshape(x.shape[0], n_heads, tph, -1).sum(2)
    return out, (x, w, b, weights, log_T_soft, log_T_sel, index)


def _bwd(n_heads, tph, res, g):
    x, w, b, weights, log_T_soft, log_T_sel, index = res
    T, nap, _ = w.shape
    n_out = weights.shape[2]
    B = x.shape[0]
    bm = bit_matrix_msb(nap)

    # grad_out is shared by every table inside a head (the head sums over tph).
    grad_pt = jnp.broadcast_to(g[:, :, None, :], (B, n_heads, tph, n_out)
                               ).reshape(B, T, n_out)

    # --- table weights: HARD 1-row scatter at the selected row ---------------
    flat_idx = (index + (jnp.arange(T) * weights.shape[1])[None, :]).reshape(-1)
    gw = jnp.zeros((T * weights.shape[1], n_out), grad_pt.dtype)
    gw = gw.at[flat_idx].add(grad_pt.reshape(-1, n_out))
    grad_weights = gw.reshape(weights.shape)

    # --- x, w, b, temperatures: SOFT full-K surrogate, weights held constant --
    wt = jax.lax.stop_gradient(weights)
    f = lambda x_, w_, b_, ts_, tl_: _soft_surrogate(x_, w_, b_, wt, ts_, tl_, index, bm)
    _, vjp = jax.vjp(f, x, w, b, log_T_soft, log_T_sel)
    gx, gw_hp, gb, gts, gtl = vjp(grad_pt)

    return gx, gw_hp, gb, grad_weights, gts, gtl


lut_apply.defvjp(_fwd, _bwd)


def policy(params, obs):
    """Walker2d head: standardise -> LUT -> sum heads -> clip (env clips identically)."""
    x = (obs - params["obs_mean"]) / (params["obs_std"] + 1e-6)
    y = lut_apply(x, params["w"], params["b"], params["weights"],
                  params["log_T_soft"], params["log_T_sel"],
                  params["n_heads"], params["tph"])
    return jnp.clip(y.sum(1), -1.0, 1.0)


def init(key, nap, tph, n_heads, input_dim, n_outputs, obs_mean, obs_std,
         table_std=0.05):
    kw, kb, kv = jax.random.split(key, 3)
    T = n_heads * tph
    return dict(w=jax.random.normal(kw, (T, nap, input_dim)) * 0.5,
                b=jax.random.normal(kb, (T, nap)) * 0.1,
                weights=jax.random.normal(kv, (T, 2 ** nap, n_outputs)) * table_std,
                log_T_soft=jnp.log(jnp.asarray(0.5)),
                log_T_sel=jnp.log(jnp.asarray(0.5)),
                n_heads=n_heads, tph=tph,
                obs_mean=obs_mean, obs_std=obs_std)
