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
import subprocess

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

def pairs_to_wb(idx_a, idx_b, input_dim):
    """(a, b) index pairs -> the frozen hyperplane w[t,i] = e_a - e_b, b = 0.

    This is the whole "no new forward needed" claim in three lines: bit_i is
    1[w_i . x + b_i > 0] = 1[x[a] - x[b] > 0], which is exactly an anchor comparator.
    Verified bit-exact against FastMultiHeadLut by verify_ext.py check B.
    """
    idx_a, idx_b = np.asarray(idx_a), np.asarray(idx_b)
    n_tables, nap = idx_a.shape
    if (idx_a == idx_b).any():
        raise ValueError("degenerate anchor pair with a == b: that bit would collapse "
                         "to a constant, not a comparator")
    w = np.zeros((n_tables, nap, input_dim), np.float32)
    t_i = np.arange(n_tables)[:, None]
    p_i = np.arange(nap)[None, :]
    w[t_i, p_i, idx_a] = 1.0
    w[t_i, p_i, idx_b] = -1.0
    return jnp.asarray(w), jnp.zeros((n_tables, nap), jnp.float32)


def anchor_pair_wb_lutorch(n_tables, nap, input_dim, seed=0, policy="balanced",
                           heads=1, device="cpu", cache_dir=None, generate=True):
    """Anchor pairs drawn by LUTORCH'S OWN sampler, returned as w = e_a - e_b, b = 0.

    The draw happens in torch (see gen_anchors.py) and is handed over as cached
    indices, because the trainer's venv has no torch and torch's RNG stream is not
    reproducible in numpy — a numpy "port" could match the ALGORITHM but never the
    actual draw, and matching the actual draw is the point.

    policy: any AnchorSamplingPolicy value. Note `balanced` is what this task
    specified, but FastMultiHeadLut itself uses `canonical_full_coverage` and REJECTS
    `balanced` — so pick canonical_full_coverage if the goal is FastMHL semantics.

    device: which torch generator draws. CPU and CUDA generators give DIFFERENT draws
    from the same seed, so reproducing a GPU-built torch module needs device="cuda".
    """
    cache_dir = cache_dir or os.path.expanduser("~/.cache/spiky_anchors")
    name = (f"anchors_{policy}_t{n_tables}_nap{nap}_d{input_dim}"
            f"_h{heads}_s{seed}_{device}.npz")
    path = os.path.join(cache_dir, name)
    if not os.path.exists(path):
        if not generate:
            raise FileNotFoundError(f"no cached anchors at {path}")
        _generate(n_tables, nap, input_dim, seed, policy, heads, device, cache_dir)
    z = np.load(path)
    return pairs_to_wb(z["anchor_a"], z["anchor_b"], input_dim)


def _generate(n_tables, nap, input_dim, seed, policy, heads, device, cache_dir):
    """Shell out to the SPIKY venv (which has torch) to draw and cache the pairs."""
    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.abspath(os.path.join(here, "..", "..", ".."))
    py = os.path.join(repo, ".venv", "bin", "python")
    if not os.path.exists(py):
        raise FileNotFoundError(
            f"need the spiky venv at {py} to draw lutorch anchors (this venv has no "
            f"torch); or pre-generate with gen_anchors.py and pass cache_dir")
    cmd = [py, os.path.join(here, "gen_anchors.py"),
           "--n-tables", str(n_tables), "--nap", str(nap),
           "--input-dim", str(input_dim), "--heads", str(heads),
           "--seed", str(seed), "--policy", policy, "--device", device,
           "--cache-dir", cache_dir]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"gen_anchors.py failed:\n{r.stdout}\n{r.stderr}")
    print(f"  [anchors] {r.stdout.strip()}", flush=True)


def anchor_pair_wb(rng, n_tables, nap, input_dim):
    """DEPRECATED home-grown sampler. Kept because it produced published numbers.

    The exp_c11 2x2 anchors cells and the ENTIRE exp_c12 capacity sweep were run with
    this draw, so deleting it would make those results unreproducible. New runs should
    use `anchor_pair_wb_lutorch`, which uses lutorch's real sampler.

    How it differs from lutorch BALANCED: this balances WITHIN a table (every table's
    2*NAP endpoints are drawn without replacement, so no coordinate is reused inside a
    table and a == b is impossible), whereas lutorch balances GLOBALLY across the whole
    n_tables*nap stream, allows a coordinate to repeat inside a table, draws a and b
    independently, and repairs a == b by rejection. Opposite guarantees.

    It also has a latent defect the lutorch policies do not: the 2*nap > input_dim
    branch below tops up WITH replacement and does not exclude already-picked
    coordinates, so a collision silently degenerates a comparator into a
    single-coordinate sign test. Never triggered at input_dim=17, nap<=8.

    w[t,i] = e_a - e_b, b = 0 — i.e. bit_i = 1[x[a] - x[b] > 0].
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
