"""exp_c32 — BucketLIFDetectorsMHL ported to JAX, for the MJX Walker2d SAC actor (#75).

Fourth LIF front-end in this line (c30 dense-P, c30b factorised-P, c31 PureLIF TTFS). It is
the biggest departure of the four, and two of the differences change the *shape* of the
model, not just its parameterisation:

  1. NO ANCHOR PAIRS. There is no `n_anchor_pairs`, no per-table bit vector, and no
     `_prow` product over bits. Each table has exactly ONE LIF neuron. The constructor
     signature drops `n_anchor_pairs` entirely -- passing our usual nap=6 is not just
     unnecessary, it is a TypeError.

  2. THE ROW IS A TIME BUCKET. PureLIF turned the first-spike time into `nap` independent
     bits by comparing it against `nap` deadlines, giving 2**nap rows. Bucket LIF compares
     the single spike time against `n_buckets - 1` ordered BOUNDARIES and uses the index of
     the interval it falls into. Rows per table = `n_buckets`, not `2**nap`. At 16 buckets
     that is 16 rows against PureLIF's 64 -- the table shrinks 4x on top of everything else.

     The addressing is therefore a MONOTONE quantisation of one scalar, where every other
     model in this chapter addressed with a set of independent sign tests. Rows are
     ordered: row 5 is "fired a bit later than row 4". Neighbouring rows are neighbours in
     time, which no previous front-end here could say.

  3. BOUNDARIES ARE TRAINABLE AND KEPT SORTED by construction:
         boundaries = beta_base + cumsum(softplus(beta_raw))
     softplus is strictly positive, so the cumulative sum is strictly increasing whatever
     the optimiser does. There is no projection step and no way to produce a crossed pair.

  4. NO `temp_bit`. The bit temperature is replaced by `T_bkt`, the softness of the bucket
     partition. Same role -- how sharply a continuous time becomes a discrete row -- but it
     is per-LUT and it smooths a partition of unity rather than an independent sigmoid.

WHAT CARRIES OVER UNCHANGED: the membrane and first-spike machinery is character-for-
character PureLIF's (sorted arrivals, leaky integration in arrival order, fixed
theta_mem = 1.0, smooth first-success for the soft path), and so is the decoupled
straight-through decode

    y = y_hard + y_addr - stop_gradient(y_addr)

with `y_hard = onehot(hard bucket) @ table` and `y_addr = g_soft @ table.detach()`. The
surrogate is written into the forward as an additive, value-cancelling term, so plain
autodiff reproduces the torch semantics and NO custom VJP is needed.

THE SOFT BUCKET IS A PARTITION OF UNITY, which is the part worth checking rather than
trusting. With S_k = sigmoid((t - b_k) / T_bkt):

    g_0 = 1 - S_0,   g_m = S_{m-1} - S_m,   g_last = S_{last}

telescopes to sum_m g_m = 1 exactly, for any boundaries and any T_bkt. The parity harness
asserts that directly, because a partition that does not sum to one would still train --
it would just silently scale the addressed row.

`eps` IS UNUSED, as in PureLIF. Asserted in the parity dump rather than taken from the
docstring.

Parameters (T = n_tables, N = input_dim, Mb = n_buckets):
    delay (T,N)  w (T,N)  tau_raw (T,)  log_T_cross (T,)  log_T_bkt (T,)
    beta_base (T,)  beta_raw (T, Mb-1)  table (T, Mb, n_outputs)
"""
import jax
import jax.numpy as jnp

jax.config.update("jax_default_matmul_precision", "highest")

T_WINDOW = 32.0
LATENCY_C = 16.0
LATENCY_ALPHA = 3.0
THETA_MEM = 1.0          # a fixed buffer in the reference, NOT a parameter


def _softplus_pos(raw):
    return jax.nn.softplus(raw) + 1e-3


def boundaries(p):
    """[T, n_buckets-1], strictly increasing along the last axis by construction."""
    return p["beta_base"][:, None] + jnp.cumsum(jax.nn.softplus(p["beta_raw"]), axis=-1)


def first_spike(p, x, n_tables, t_window=T_WINDOW, latency_c=LATENCY_C,
                latency_alpha=LATENCY_ALPHA):
    """One LIF neuron per table. x:[B,N] -> (t_hard, t_soft), each [B,T].

    Identical to exp_c31's membrane, with `n_tables` neurons instead of
    `n_tables * nap` detectors -- see jax_pure_lif.spike_bits for the step-by-step.
    """
    t = jnp.clip(latency_c - latency_alpha * x, 0.0, t_window)          # [B,N]
    a = t[:, None, :] + p["delay"][None]                                # [B,T,N]

    idx = jnp.argsort(a, axis=-1, stable=True)
    a_srt = jnp.take_along_axis(a, idx, axis=-1)
    w_srt = jnp.take_along_axis(jnp.broadcast_to(p["w"][None], a.shape), idx, axis=-1)

    dt = a_srt[:, :, :, None] - a_srt[:, :, None, :]                    # [B,T,N,N]
    tau = _softplus_pos(p["tau_raw"])[None, :, None, None]
    contrib = jnp.where(dt >= 0.0,
                        w_srt[:, :, None, :] * jnp.exp(-jax.nn.relu(dt) / tau),
                        0.0)
    V = contrib.sum(-1)                                                 # [B,T,N]

    crossed = V >= THETA_MEM
    kstar = jnp.argmax(crossed.astype(x.dtype), axis=-1)
    t_hard = jnp.take_along_axis(a_srt, kstar[:, :, None], axis=-1)[:, :, 0]
    t_hard = jnp.where(crossed.any(-1), t_hard, t_window)               # [B,T]

    T_cross = jnp.exp(p["log_T_cross"])[None, :, None]
    c = jax.nn.sigmoid((V - THETA_MEM) / T_cross)
    surv = jnp.cumprod(1.0 - c, axis=-1)
    surv_prev = jnp.concatenate([jnp.ones_like(surv[..., :1]), surv[..., :-1]], axis=-1)
    prob = c * surv_prev
    t_soft = (prob * a_srt).sum(-1) + surv[..., -1] * t_window          # [B,T]
    return t_hard, t_soft


def bucket_soft(p, t, n_tables):
    """Partition-of-unity soft one-hot. t:[B,T] -> [B,T,n_buckets], sums to 1 exactly."""
    b = boundaries(p)                                                   # [T, Mb-1]
    T_bkt = jnp.exp(p["log_T_bkt"]).reshape(1, n_tables, 1)
    S = jax.nn.sigmoid((t[:, :, None] - b[None]) / T_bkt)               # [B,T,Mb-1]
    g0 = 1.0 - S[..., :1]
    gmid = S[..., :-1] - S[..., 1:]                                     # empty if Mb == 2
    glast = S[..., -1:]
    return jnp.concatenate([g0, gmid, glast], axis=-1)


def bucket_hard(p, t, n_buckets):
    """Hard bucket one-hot. m* = #{boundaries <= t}; t_window folds into the last bucket."""
    b = boundaries(p)
    mstar = (t[:, :, None] >= b[None]).sum(-1)                          # [B,T]
    return jax.nn.one_hot(mstar, n_buckets, dtype=t.dtype)


def _rows(onehot, table):
    return jnp.einsum("btm,tmo->bto", onehot, table)


def apply(p, x, eps, n_heads, tph, n_buckets, mode="st"):
    """x:[B,input_dim] -> [B,n_heads,n_outputs].

    `eps` is accepted and IGNORED, exactly as the torch reference does. Signature keeps it
    so this module is drop-in for jax_pure_lif / jax_lif_mhl; `n_buckets` takes the slot
    that `nap` occupied in those, and means something different -- rows = n_buckets, not
    2**nap.
    """
    del eps
    n_tables = n_heads * tph
    B = x.shape[0]
    t_hard, t_soft = first_spike(p, x, n_tables)

    if mode == "hard":
        rows = _rows(bucket_hard(p, t_hard, n_buckets), p["table"])
    elif mode == "soft":
        rows = _rows(bucket_soft(p, t_soft, n_tables), p["table"])
    elif mode == "st":
        y_hard = _rows(bucket_hard(p, t_hard, n_buckets), p["table"])
        y_addr = _rows(bucket_soft(p, t_soft, n_tables),
                       jax.lax.stop_gradient(p["table"]))
        rows = y_hard + y_addr - jax.lax.stop_gradient(y_addr)
    else:
        raise ValueError(f"mode must be 'st'|'hard'|'soft', got {mode!r}")
    return rows.reshape(B, n_heads, tph, -1).sum(2)


def address(p, x, eps, n_heads, tph, n_buckets):
    """Hard bucket index per (sample, table): [B, n_tables] int32. The coverage diagnostic.

    Unlike every earlier front-end here this index is ORDERED -- adjacent values mean
    adjacent spike times -- so a coverage histogram over it is readable as a distribution,
    not just as an occupancy count.
    """
    del eps, n_buckets
    n_tables = n_heads * tph
    t_hard, _ = first_spike(p, x, n_tables)
    return (t_hard[:, :, None] >= boundaries(p)[None]).sum(-1).astype(jnp.int32)


def init(key, n_buckets, tph, n_heads, input_dim, n_outputs):
    """Mirrors the torch module's own init, distribution for distribution.

    torch: delay=0, w=0.2*randn, tau_raw=1, log_T_cross=0, log_T_bkt=0, beta_base=0,
    beta_raw=inv_softplus(t_window/n_buckets), table=0.1*randn. The beta_raw value is the
    one that matters: it makes the initial boundaries evenly spaced at
    step, 2*step, ..., (n_buckets-1)*step across (0, t_window).
    """
    n_tables = n_heads * tph
    kw, kt = jax.random.split(key, 2)
    step = T_WINDOW / n_buckets
    inv_softplus_step = float(jnp.log(jnp.expm1(jnp.asarray(step, jnp.float32))))
    return dict(
        delay=jnp.zeros((n_tables, input_dim), jnp.float32),
        w=jax.random.normal(kw, (n_tables, input_dim), jnp.float32) * 0.2,
        tau_raw=jnp.ones((n_tables,), jnp.float32),
        log_T_cross=jnp.zeros((n_tables,), jnp.float32),
        log_T_bkt=jnp.zeros((n_tables,), jnp.float32),
        beta_base=jnp.zeros((n_tables,), jnp.float32),
        beta_raw=jnp.full((n_tables, n_buckets - 1), inv_softplus_step, jnp.float32),
        table=jax.random.normal(kt, (n_tables, n_buckets, n_outputs), jnp.float32) * 0.1,
    )


def n_params(p):
    """(front-end, table). Everything is learnable -- no frozen anchor buffers."""
    det = sum(int(v.size) for k, v in p.items() if k != "table")
    return det, int(p["table"].size)
