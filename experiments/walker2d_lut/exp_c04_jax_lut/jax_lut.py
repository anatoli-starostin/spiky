"""exp_c04 — HyperplaneMultiHeadLUT forward, ported to JAX (#75).

A faithful, jittable port of the *hard* forward of
`spiky.lutorch.hyperplane_multi_head_lut.HyperplaneMultiHeadLUT`, so a LUT policy can
be evaluated inside the MJX rollout loop (gradient-free search needs only the forward).

The torch reference, for line-by-line comparison
(`hyperplane_multi_head_lut.py:_hyperplane_project` + `_hyperplane_lut_fwd_body`):

    a     = x @ w.reshape(T*NAP, D).T -> [B, T, NAP]; a += b
    bits  = (a > 0)
    index = sum_i bits_i * 2^(NAP-1-i)                 # MSB-first
    out   = embedding_bag(weights[table, index], bags of tph, mode='sum')
          -> [B, n_heads, n_outputs]

Two details that decide bit-exactness and are easy to get subtly wrong:
  * the packing is **MSB-first** (`powers[i] = 2^(NAP-1-i)`), not LSB-first;
  * the threshold is a **strict** `a > 0`, so an exact zero gives bit 0.

The `embedding_bag(mode='sum')` reduce is just a sum over the tables_per_head axis
after gathering one row per table, which is what the reshape+sum below does.
"""
import jax
import jax.numpy as jnp

# =============================================================================
# The forward — this is the whole thing
# =============================================================================


def lut_forward(params, x):
    """x: [B, D] -> [B, n_heads, n_outputs].

    params: dict(w [T,NAP,D], b [T,NAP], weights [T,K,n_out], n_heads, tph)
    """
    w, b, weights = params["w"], params["b"], params["weights"]
    T, nap, _ = w.shape
    # HIGHEST precision is REQUIRED, not a nicety. JAX defaults to TF32 for fp32
    # matmuls on GPU; torch (here) does not. TF32's ~10-bit mantissa perturbs the
    # pre-activation a_i, and near a decision boundary that FLIPS the sign bit
    # 1[a_i > 0] — which selects a different table row and changes the output by a
    # whole table entry, not by an epsilon. Measured: TF32 gave max|Δ| = 1.4e-1
    # against torch; HIGHEST gives exact equality.
    a = jax.lax.dot_general(x, w.reshape(T * nap, -1).T, (((1,), (0,)), ((), ())),
                            precision=jax.lax.Precision.HIGHEST)   # [B, T*NAP]
    a = a.reshape(x.shape[0], T, nap) + b[None]           # [B, T, NAP]
    powers = 2 ** jnp.arange(nap - 1, -1, -1)             # MSB-first
    index = ((a > 0).astype(jnp.int32) * powers[None, None, :]).sum(-1)   # [B, T]
    rows = jnp.take_along_axis(                           # gather one row per table
        weights[None], index[:, :, None, None], axis=2).squeeze(2)        # [B, T, n_out]
    return rows.reshape(x.shape[0], params["n_heads"], params["tph"], -1).sum(2)


def lut_policy(params, obs):
    """Walker2d policy head: standardise -> LUT -> sum heads -> clip to the action box."""
    x = (obs - params["obs_mean"]) / (params["obs_std"] + 1e-6)
    return jnp.clip(lut_forward(params, x).sum(1), -1.0, 1.0)


# =============================================================================
# Flat-vector <-> params, for gradient-free search
# =============================================================================

def params_to_flat(params):
    return jnp.concatenate([params["w"].ravel(), params["b"].ravel(),
                            params["weights"].ravel()])


def make_unflatten(shape_w, shape_b, shape_weights, n_heads, tph,
                   obs_mean, obs_std):
    nw = int(jnp.prod(jnp.array(shape_w)))
    nb = int(jnp.prod(jnp.array(shape_b)))

    def unflatten(flat):
        return dict(w=flat[:nw].reshape(shape_w),
                    b=flat[nw:nw + nb].reshape(shape_b),
                    weights=flat[nw + nb:].reshape(shape_weights),
                    n_heads=n_heads, tph=tph,
                    obs_mean=obs_mean, obs_std=obs_std)
    return unflatten


def n_flat_params(shape_w, shape_b, shape_weights):
    return (int(jnp.prod(jnp.array(shape_w))) + int(jnp.prod(jnp.array(shape_b)))
            + int(jnp.prod(jnp.array(shape_weights))))


# =============================================================================
# Import a torch LUTPolicy checkpoint (saved by exp_c03) into JAX params
# =============================================================================

def from_npz(path):
    """Load params exported by `export_torch_lut.py`.

    torch and JAX live in SEPARATE venvs here (JAX ships its own CUDA stack, and
    mixing it with torch's risked breaking a running baseline), so the handoff is an
    .npz written by the torch side rather than an in-process conversion.
    """
    import numpy as np
    z = np.load(path)
    return dict(w=jnp.asarray(z["w"]), b=jnp.asarray(z["b"]),
                weights=jnp.asarray(z["weights"]),
                n_heads=int(z["n_heads"]), tph=int(z["tph"]),
                obs_mean=jnp.asarray(z["obs_mean"]),
                obs_std=jnp.asarray(z["obs_std"]))
