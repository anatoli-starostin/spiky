"""exp_c32b — the CORRECTED BucketLIFDetectorsMHL ported to JAX (#75).

Re-port against `exp/lif-detectors-mhl` @ 2a2795e0, which fixes three flaws in the version
exp_c32 ran (9e4dad08). exp_c32 scored 1174.4 +/- 517.2 with 0/300 full episodes; the
mechanism we reported for that failure -- every run pinned near the top bucket, low buckets
unused -- is precisely what fix (1) below addresses.

THE THREE FIXES, and what each one does mechanically:

  1. SYNAPSES ARE NOW BOUNDED AND EXCITATORY. `w = w_max * sigmoid(w_raw)` with w_max = 2,
     so 0 <= w <= 2 and the trainable parameter is `w_raw`, not `w`. Previously
     `w = 0.2 * randn`: FREE-SIGNED. That is the flaw behind everything exp_c32 showed.
     With signed weights the membrane is a sum of 17 terms of mean zero, so it rarely
     reaches the fixed theta_mem = 1.0, the neuron does not spike, and a non-spiking neuron
     folds into the LAST bucket by construction. Hence exp_c32's ~97% no-spike mass at
     init, its bucket index pinned near 15, and its permanently unused low buckets. Now
     every synapse is positive, so ~17 terms averaging ~0.2 accumulate rather than cancel.

  2. HOT INIT: `w_raw ~ N(-2.2, 0.5)`. sigmoid(-2.2) ~ 0.0998, so the effective weight
     still starts near 0.2 -- the same SCALE as before, but all of one sign. The reference
     docstring reports this puts the init no-spike mass at ~0.03 instead of ~0.97. The fix
     is the sign structure, not the magnitude.

  3. TAU FLOOR RAISED 1e-3 -> 1.0: `tau = softplus(tau_raw) + 1.0`. The membrane time
     constant can no longer drop into the regime where the trace decays inside a single
     arrival step, which would make the leaky integrator a memoryless one and discard the
     arrival-order information the whole model is built on. At init tau is now 2.313 rather
     than 1.314.

Parameter COUNT is unchanged at 7,840 -- `w_raw` has the same shape as the old `w`, and
`w_max` is a plain float attribute, not a Parameter.

The rest of this file is exp_c32's port verbatim; see below for the model description.

---

BucketLIFDetectorsMHL ported to JAX, for the MJX Walker2d SAC actor (#75).

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
    delay (T,N)  w_raw (T,N)  tau_raw (T,)  log_T_cross (T,)  log_T_bkt (T,)
    beta_base (T,)  beta_raw (T, Mb-1)  table (T, Mb, n_outputs)
"""
import jax
import jax.numpy as jnp

jax.config.update("jax_default_matmul_precision", "highest")

T_WINDOW = 32.0
LATENCY_C = 16.0
LATENCY_ALPHA = 3.0
THETA_MEM = 1.0          # a fixed buffer in the reference, NOT a parameter
W_MAX = 2.0              # a plain attribute in the reference, NOT a parameter
W_RAW_MEAN, W_RAW_STD = -2.2, 0.5          # the "hot" init
TAU_FLOOR = 1.0          # raised from 1e-3 by the fix


def synapses(p):
    """Bounded excitatory weights in [0, W_MAX]. The fix that matters."""
    return W_MAX * jax.nn.sigmoid(p["w_raw"])


def _tau(raw):
    return jax.nn.softplus(raw) + TAU_FLOOR


# =============================================================================
# The membrane, in LINEAR time and memory
# =============================================================================
# The torch reference (and our exp_c30/c31/c32 ports) computes
#
#     V_k = sum_{j <= k} w_j * exp(-(a_k - a_j) / tau)
#
# by materialising the full pairwise matrix dt[k, j] = a_k - a_j, shape (B, T, N, N),
# masking it causal and summing. That is O(N^2) in both time and memory and it is the
# single most expensive thing in this model family -- it is why exp_c31 cost ~4x exp_c30
# per iteration and held ~0.25 GB of activations.
#
# It is also unnecessary. Because `a_srt` is sorted ASCENDING, the same quantity satisfies
# an exact first-order linear recurrence:
#
#     V_k = w_k + exp(-(a_k - a_{k-1}) / tau) * V_{k-1},      V_0 = w_0
#
# Proof: split the sum at j = k, then factor exp(-(a_k - a_j)/tau) =
# exp(-(a_k - a_{k-1})/tau) * exp(-(a_{k-1} - a_j)/tau) for every j <= k-1.
#
# This is EXACT, not an approximation, and it is numerically safe: consecutive gaps are
# non-negative, so every decay factor lies in (0, 1] and nothing can overflow. (The
# tempting alternative -- factor out exp(a_k/tau) and use a cumulative sum -- is NOT safe:
# exp(a_j/tau) reaches e^32 at the window edge and overflows once tau is small. That is
# why this was not done for exp_c31, and it was the wrong reason: the recurrence form
# below was available all along.)
#
# `jax.lax.associative_scan` evaluates the recurrence in O(log N) depth by composing the
# affine maps x -> A_k x + B_k, so this is not a sequential Python loop.
#
# Result: the (B, T, N, N) tensor is GONE. Peak activation drops by a factor of ~N.


def _affine_compose(l, r):
    """Compose x -> A1 x + B1 then x -> A2 x + B2, i.e. (A1*A2, A2*B1 + B2)."""
    (a1, b1), (a2, b2) = l, r
    return (a1 * a2, a2 * b1 + b2)


def membrane_linear(a_srt, w_srt, tau):
    """[B,T,N] sorted arrivals + matching weights -> membrane at each arrival, [B,T,N].

    O(N) memory, O(log N) depth. No pairwise tensor is ever formed."""
    gaps = jnp.diff(a_srt, axis=-1)                                     # [B,T,N-1] >= 0
    decay = jnp.concatenate([jnp.zeros_like(a_srt[..., :1]),
                             jnp.exp(-gaps / tau)], axis=-1)            # [B,T,N]
    _, V = jax.lax.associative_scan(_affine_compose, (decay, w_srt), axis=-1)
    return V


def membrane_cumsum(a_srt, w_srt, tau):
    """nucstar's prefix factorisation (bucket_lif_detectors_mhl @ 0024b81f), in JAX.

        V_k = exp(-a_k/tau) * cumsum_{j<=k}( w_j * exp(a_j/tau) )

    Algebraically identical to `membrane_linear` and also O(N). Differs in conditioning:
    the intermediate reaches exp(t_window/tau) before the rescale, so it OVERFLOWS in
    float32 once t_window/tau > 88.7 (measured: at t_window=32 it dies below tau≈0.36; at
    tau=1.0 it dies above t_window≈88). Safe at the shipped config and measured equally
    accurate there — see compare_membranes.py. Kept selectable so the two can be
    benchmarked head to head."""
    return jnp.exp(-a_srt / tau) * jnp.cumsum(w_srt * jnp.exp(a_srt / tau), axis=-1)


def membrane_quadratic(a_srt, w_srt, tau):
    """The reference O(N^2) formulation. Kept ONLY so `bench_linear.py` can measure what
    the recurrence saves and assert the two agree. Never called in training."""
    dt = a_srt[:, :, :, None] - a_srt[:, :, None, :]                    # [B,T,N,N]
    contrib = jnp.where(dt >= 0.0,
                        w_srt[:, :, None, :] * jnp.exp(-jax.nn.relu(dt) / tau[..., None]),
                        0.0)
    return contrib.sum(-1)


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
    w = synapses(p)                                                     # [T,N], all >= 0
    w_srt = jnp.take_along_axis(jnp.broadcast_to(w[None], a.shape), idx, axis=-1)

    tau = _tau(p["tau_raw"])[None, :, None]                             # [1,T,1]
    V = membrane_linear(a_srt, w_srt, tau)                              # [B,T,N], O(N)

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

    torch: delay=0, w_raw=-2.2+0.5*randn (the HOT init), tau_raw=1, log_T_cross=0,
    log_T_bkt=0, beta_base=0, beta_raw=inv_softplus(t_window/n_buckets), table=0.1*randn.

    Two of these carry the fixes. `w_raw` centred at -2.2 gives an effective weight
    w = 2*sigmoid(-2.2) ~ 0.2 -- the same scale as the old signed init, but strictly
    positive, so 17 synapses accumulate toward theta_mem instead of cancelling. And
    beta_raw makes the initial boundaries evenly spaced at step, 2*step, ...,
    (n_buckets-1)*step across (0, t_window).
    """
    n_tables = n_heads * tph
    kw, kt = jax.random.split(key, 2)
    step = T_WINDOW / n_buckets
    inv_softplus_step = float(jnp.log(jnp.expm1(jnp.asarray(step, jnp.float32))))
    return dict(
        delay=jnp.zeros((n_tables, input_dim), jnp.float32),
        w_raw=(W_RAW_MEAN + W_RAW_STD
               * jax.random.normal(kw, (n_tables, input_dim), jnp.float32)),
        tau_raw=jnp.ones((n_tables,), jnp.float32),
        log_T_cross=jnp.zeros((n_tables,), jnp.float32),
        log_T_bkt=jnp.zeros((n_tables,), jnp.float32),
        beta_base=jnp.zeros((n_tables,), jnp.float32),
        beta_raw=jnp.full((n_tables, n_buckets - 1), inv_softplus_step, jnp.float32),
        table=jax.random.normal(kt, (n_tables, n_buckets, n_outputs), jnp.float32) * 0.1,
    )


def set_boundaries(p, target):
    """Rewrite beta_base/beta_raw so `boundaries(p)` equals `target` [T, n_buckets-1].

    Inverts the monotone parameterisation `b = beta_base + cumsum(softplus(beta_raw))`:
    with beta_base = 0, beta_raw_0 = inv_softplus(b_0) and beta_raw_k =
    inv_softplus(b_k - b_{k-1}). Requires every b_0 > 0 and every gap > 0, so the caller
    must have enforced MIN_GAP -- see `quantile_boundaries`.

    Note this preserves the invariant by construction: whatever we write, the forward pass
    still recovers a strictly increasing sequence, because softplus is strictly positive.
    """
    target = jnp.sort(target, axis=-1)
    first = target[:, :1]
    gaps = jnp.diff(target, axis=-1)
    raw = jnp.concatenate([first, gaps], axis=-1)
    # inv_softplus(y) = log(expm1(y)); guard y -> 0 where expm1 underflows.
    raw = jnp.log(jnp.expm1(jnp.clip(raw, 1e-4, None)))
    return dict(p, beta_base=jnp.zeros_like(p["beta_base"]), beta_raw=raw)


MIN_GAP = 0.05


def quantile_boundaries(t_hard, n_buckets, min_gap=MIN_GAP, t_window=T_WINDOW):
    """Per-table equal-mass boundaries from observed first-spike times.

    t_hard: [S, T] -> [T, n_buckets-1].

    THE POINT. Uniform boundaries spread n_buckets-1 cuts evenly over (0, t_window), but
    the measured spike distribution is narrow -- on a trained exp_c32b actor the middle
    50% of spikes spans 3.41 of 32 time units (11% of the window), and only 4.7 of 16
    buckets are effectively used (entropy 2.06 of 4.00 bits). Equal-mass cuts put every
    boundary where samples actually are, so the occupancy entropy starts at its ceiling.

    TWO PRACTICAL CORRECTIONS, both measured rather than assumed:
      * 16.3% of raw quantile gaps come out degenerate (zero width) because a large mass
        sits exactly at t_window (the no-spike fold-in). Zero gaps are not representable
        by the softplus parameterisation and would collapse buckets, so a `min_gap` floor
        is enforced by a cumulative running maximum.
      * the result is clipped into (0, t_window + margin] so the last bucket still catches
        the no-spike mass at t = t_window.
    """
    q = jnp.linspace(0.0, 100.0, n_buckets + 1)[1:-1]
    b = jnp.percentile(t_hard, q, axis=0).T                    # [T, n_buckets-1]
    b = jnp.sort(b, axis=-1)
    # Enforce a minimum gap: b_k >= b_{k-1} + min_gap, via a running max.
    def _step(prev, cur):
        nxt = jnp.maximum(cur, prev + min_gap)
        return nxt, nxt
    first = jnp.maximum(b[:, 0], min_gap)
    _, rest = jax.lax.scan(_step, first, b[:, 1:].T)
    return jnp.concatenate([first[:, None], rest.T], axis=-1)


def n_params(p):
    """(front-end, table). Everything is learnable -- no frozen anchor buffers."""
    det = sum(int(v.size) for k, v in p.items() if k != "table")
    return det, int(p["table"].size)
