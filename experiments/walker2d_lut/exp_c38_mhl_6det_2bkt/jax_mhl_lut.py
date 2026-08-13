"""exp_c38 — LIFMultiHeadLUT ported to JAX (branch exp/lif-detectors-mhl @ 24c0e60a).

WHAT THIS CLASS IS. nucstar has collapsed the whole LIF-detector line into ONE module.
Where we previously ported four separate torch classes (c30 dense-P LIFDetectorsMHL,
c30b factorised-P, c31 PureLIFDetectorsMHL, c32-c37 BucketLIFDetectorsMHL), there is now
`LIFMultiHeadLUT` with three nested structural levels:

    n_heads          output heads; forward always returns (B, n_heads, n_outputs)
    tables_per_head  tables SUMMED within each head          (n_tables = heads * tph)
    n_det            LIF detectors per table, each emitting one of n_buckets digits;
                     the n_det digits combine MIXED-RADIX into an index over
                     n_buckets**n_det cells

That third level is the genuinely new axis. Everything else in the class is machinery we
have already ported and validated.

HOW THE OLD VARIANTS FALL OUT OF IT:

  n_det = 1                     -> exactly BucketLIFDetectorsMHL (c32b/c33/c34/c35/c36/c37).
                                   One LIF per table, M buckets, rows = M.
  n_det = D, n_buckets = 2      -> D independent LIF detectors per table, each thresholded
                                   against ONE trainable boundary, packed MSB-first into
                                   2**D rows. This is structurally the PureLIF (c31) row
                                   layout -- 2**nap rows from nap binary tests -- except
                                   that c31 compared ONE neuron's spike time against nap
                                   deadlines, whereas here each of the D tests comes from
                                   its OWN LIF cell with its own delays, synapses and tau.
                                   It is therefore closer to c30's per-bit detector bank,
                                   but with the bucket/boundary parameterisation and
                                   without the pairwise P channel.
  n_det > 1, n_buckets > 2      -> the retired ProductBucketLIFMHL: a mixed-radix product
                                   of several ordered time quantisations.

THE CONFIGURATION THIS EXPERIMENT RUNS is n_heads=1, tph=32, n_det=6, n_buckets=2:
32 tables, each addressed by 6 independent LIF detectors, 2**6 = 64 cells per table. So it
lands on c31's row count and c36's parameter scale by an entirely different route.

WHAT IS NEW RELATIVE TO OUR c32b PORT, item by item (each one is a place a port can be
silently wrong, so each is asserted in parity_check.py):

  1. THE DETECTOR AXIS IS ALWAYS PRESENT. `delay` and `w_raw` are (T, D, N), `tau_raw` is
     (T, D), `beta_base` is (T, D, 1) and `beta_raw` is (T, D, M-1). In c32b these were
     (T, N), (T,), (T,) and (T, M-1). The reference deliberately keeps D=1 as a real axis
     rather than special-casing it, and so does this port.

  2. THE TEMPERATURES STAYED PER-TABLE. `log_T_cross` and `log_T_bkt` are (T,), NOT
     (T, D) -- all D detectors of a table share one crossing sharpness and one partition
     softness. Broadcasting them as (1, T, 1, 1) is the only correct reading and a port
     that gave them a detector axis would still train.

  3. LATENCY CODING IS FIXED, NOT PARAMETERISED, and the slope is pinned at alpha=3:
         t = clip(t_window * (0.5 - 3x/32), 0, t_window)
     At t_window = 32 this is `clip(16 - 3x, 0, 32)`, i.e. NUMERICALLY IDENTICAL to what
     c32b used. The change is that the constants are no longer free -- there is no
     latency_c and no latency_alpha. Saturation is at x = +-16/3 ~ +-5.33 regardless of
     t_window.

  4. THE DELAY IS CLAMPED TO [0, t_window] INSIDE THE FORWARD. c32b let `delay` roam. Two
     consequences: delays are causal (non-negative), and arrivals stay in [0, 2*t_window]
     so the reference's cumsum membrane stays float32-safe. `delay_init_std > 0` seeds it
     from a HALF-NORMAL (abs of a normal draw) instead of zeros.

  5. THE ROW INDEX IS MIXED-RADIX over the D bucket digits, MSB-first:
         idx = sum_d b_d * M**(D-1-d)
     and the table is (T, M**D, n_outputs).

  6. THE SOFT ADDRESS READOUT IS A RANK-1 TENSOR CONTRACTION, not a dense outer product.
     The reference peels one detector axis at a time with sequential einsums against the
     DETACHED table, which is what keeps memory at O(M**D) rather than materialising a
     (B, T, M, ..., M) joint distribution. This port reproduces that shape exactly; see
     `soft_read`.

  7. `eps` IS GONE FROM THE SIGNATURE ENTIRELY. In c31/c32b it was accepted and ignored;
     here it does not exist. This module therefore takes no eps at all, and the trainer
     stores no eval_eps.

  8. `freeze_temperature` fixes log_T_cross / log_T_bkt at 0.0 (T = 1.0) and makes them
     non-trainable. In torch that is `requires_grad=False`; in JAX there is no such flag,
     so the trainer zeroes those two gradients (the same device the c35 boundary freeze
     used). They remain in the parameter dict and in the parameter count, exactly as
     `param_count()` counts them on the torch side.

  9. THE `mode=` STRING IS GONE. The c30/c31/c32b torch modules took
     `forward(x, eps=..., mode="st"|"hard"|"soft")`. LIFMultiHeadLUT takes `forward(x)`
     and BRANCHES ON `self.training`: train mode is the straight-through path, eval mode
     (module.eval()) is a no_grad hard path with NO softmax and NO temperatures whose
     value equals the training value. There is no "soft" forward at all any more -- the
     soft readout survives only as the internal address-gradient term `_soft_read`, and
     it is never a forward mode.

     JAX has no module-level train/eval flag, so this port keeps ONE argument to select
     the path -- but it is named after the reference's own semantics, `mode="train"` and
     `mode="eval"`, NOT after the retired st/hard/soft vocabulary, and "soft" is not an
     accepted value. parity_check asserts mode="train" against `m.train(); m(x)` and
     mode="eval" against `m.eval(); m(x)`.

WHAT CARRIES OVER UNCHANGED from c32b, and is therefore already validated: bounded
excitatory synapses w = w_max * sigmoid(w_raw) with the hot init w_raw ~ N(-2.2, 0.5);
tau = softplus(tau_raw) + 1.0; the fixed theta_mem = 1.0 buffer; the smooth first-success
soft spike time; strictly-increasing boundaries by softplus-cumsum; and the decoupled
straight-through decode

    y = y_hard + y_addr - stop_gradient(y_addr)

which needs no custom VJP -- plain autodiff reproduces the torch semantics.

THE MEMBRANE. The reference uses the prefix factorisation

    V_k = exp(-a_k/tau) * cumsum_{j<=k}( w_j exp(a_j/tau) )

which is O(N) but conditioned badly: the intermediate reaches exp(a_max/tau), and because
the delay clamp now lets arrivals reach 2*t_window = 64, that is exp(64) ~ 6.2e27 at the
tau floor of 1.0. Still inside float32, but only by ten orders of magnitude, and it dies
outright if tau ever drops below ~0.72. This port uses the algebraically identical
recurrence

    V_k = w_k + exp(-(a_k - a_{k-1})/tau) * V_{k-1},     V_0 = w_0

evaluated by `jax.lax.associative_scan` in O(log N) depth. Every factor lies in (0, 1], so
it CANNOT overflow for any tau or any window. Both forms are provided and
`compare_membranes.py` measures their agreement (they matched to 6.2e-07 relative in
exp_c32b); `membrane_linear` is what training uses.

Parameters (T = n_tables, D = n_det, N = input_dim, M = n_buckets, O = n_outputs):
    delay (T,D,N)  w_raw (T,D,N)  tau_raw (T,D)  beta_base (T,D,1)  beta_raw (T,D,M-1)
    log_T_cross (T,)  log_T_bkt (T,)  table (T, M**D, O)
"""
import jax
import jax.numpy as jnp

jax.config.update("jax_default_matmul_precision", "highest")

T_WINDOW = 32.0
LATENCY_ALPHA = 3.0      # pinned in the reference; no longer a parameter
THETA_MEM = 1.0          # a fixed buffer in the reference, NOT a parameter
W_MAX = 2.0              # a plain attribute in the reference, NOT a parameter
W_RAW_MEAN, W_RAW_STD = -2.2, 0.5          # the "hot" init
TAU_FLOOR = 1.0
MAX_CELLS = 65536        # the reference's cap on n_buckets**n_det

# Letters for the per-detector axes in `soft_read`'s einsums. Must avoid b, t, o.
_AXES = "cdefghijkl"


def synapses(p):
    """Bounded excitatory weights in [0, W_MAX]. Shape (T, D, N)."""
    return W_MAX * jax.nn.sigmoid(p["w_raw"])


def _tau(raw):
    return jax.nn.softplus(raw) + TAU_FLOOR


def latency(x, t_window=T_WINDOW):
    """FIXED latency code, slope alpha=3. clip(t_window*(0.5 - 3x/32), 0, t_window).

    At t_window=32 this is clip(16 - 3x, 0, 32), identical to c32b's hard-coded map, but
    the constants are structural now rather than trainable."""
    return jnp.clip(t_window * (0.5 - LATENCY_ALPHA * x / 32.0), 0.0, t_window)


def boundaries(p):
    """(T, D, M-1), strictly increasing along the last axis by construction.

    beta_base is (T, D, 1) here, not (T,) as in c32b -- every DETECTOR carries its own
    offset, so the D detectors of one table quantise time differently."""
    return p["beta_base"] + jnp.cumsum(jax.nn.softplus(p["beta_raw"]), axis=-1)


def radix(n_det, n_buckets):
    """MSB-first mixed-radix weights: M**(D-1-d). Matches the reference's buffer."""
    return n_buckets ** (n_det - 1 - jnp.arange(n_det))


# =============================================================================
# The membrane
# =============================================================================
def _affine_compose(l, r):
    """Compose x -> A1 x + B1 then x -> A2 x + B2, i.e. (A1*A2, A2*B1 + B2)."""
    (a1, b1), (a2, b2) = l, r
    return (a1 * a2, a2 * b1 + b2)


def membrane_linear(a_srt, w_srt, tau):
    """(B,T,D,N) sorted arrivals + matching weights -> membrane at each arrival.

    O(N) memory, O(log N) depth, and it cannot overflow: every decay factor is in (0, 1]
    because consecutive gaps of a sorted sequence are non-negative."""
    gaps = jnp.diff(a_srt, axis=-1)
    decay = jnp.concatenate([jnp.zeros_like(a_srt[..., :1]),
                             jnp.exp(-gaps / tau)], axis=-1)
    _, V = jax.lax.associative_scan(_affine_compose, (decay, w_srt), axis=-1)
    return V


def membrane_cumsum(a_srt, w_srt, tau):
    """The reference's own prefix factorisation, in JAX. Algebraically identical to
    `membrane_linear` and also O(N), but the intermediate reaches exp(a_max/tau) and so
    overflows in float32 once a_max/tau > 88.7. With the delay clamp letting arrivals
    reach 2*t_window = 64, that means it dies below tau ~ 0.72 -- and the tau floor is
    1.0, so the shipped config is safe by a factor of ~1.4 in tau and nothing more.
    Kept selectable so compare_membranes.py can measure the agreement."""
    return jnp.exp(-a_srt / tau) * jnp.cumsum(w_srt * jnp.exp(a_srt / tau), axis=-1)


# =============================================================================
# Spike -> bucket digits -> cell
# =============================================================================
# =============================================================================
# ORDERING THE ARRIVALS — the whole cost of this model, and it is not a sort
# =============================================================================
# Putting the N arrivals of each (sample, table, detector) into time order is ~95% of this
# actor's cost. Everything downstream is nearly free: the full membrane adds 0.6 ms and
# the entire train forward, including the six-einsum mixed-radix soft readout, adds 0.02.
#
# There are three ways to spell the ordering, and the choice interacts with
# `XLA_FLAGS=--xla_gpu_deterministic_ops=true`, which every run in this chapter sets for
# reproducibility. All three produce BIT-IDENTICAL output -- max|diff| exactly 0.0 on both
# the arrivals and the weights -- so this is purely about lowering. Measured at the shipped
# shape (512 x 32 tables x 6 detectors x 17 synapses = 1.67M arrivals), under the flag:
#
#   form            membrane fwd   GRADIENT      note
#   "argsort"          25.7 ms      23.9 ms      what c30-c37 all use
#   "lax_sort"          0.97 ms     HANGS        >400 s, no result
#   "rank"              5.6 ms       0.82 ms     <- default
#
# WHY `lax_sort` HANGS, since it cost a 45-minute stalled sweep to find out. The FORWARD is
# fine under the flag (0.55 ms). It is the VJP that dies: the transpose of a permutation
# un-permutes the incoming gradient, which XLA emits as a SCATTER, and determinism forces
# that scatter to be serialised. Training calls the gradient twice per update, so the run
# reached iteration 500 and then failed to reach 1,000 in over 45 minutes, looking ~50x
# slower than its own microbenchmark with none of the slowdown in the model. Diagnosing it
# by the forward benchmark alone is impossible -- the forward looks 26x FASTER.
#
# The same reasoning condemns `argsort`, for two independent reasons: `jnp.argsort` itself
# costs 19-22 ms (the gathers that follow are free at 0.18 ms, so the sort IS the cost),
# and `take_along_axis`'s VJP is likewise a scatter-add.
#
# WHY "rank" IS FAST, AND WHY IT NEEDS NO SORT AT ALL. The permutation is recovered by
# counting: rank_k = #{j : a_j < a_k}, ties broken by index, which is exactly the stable
# sort position of element k. That is a pairwise comparison over an N=17 axis -- 289
# comparisons per (sample, table, detector), trivially parallel, no sort primitive. It is
# then APPLIED as a contraction against the one-hot P[r,k] = 1[rank_k == r], i.e. a matmul.
# A matmul's transpose is a matmul, so the backward is a matmul too: no scatter anywhere,
# nothing for the determinism flag to serialise, and the gradient comes out 29x faster
# than the argsort form and faster than the forward.
#
# Gradients are correct because P is piecewise-constant in `a` (it is built from integer
# comparisons), exactly as torch.sort's permutation is treated as constant. The parity gate
# asserts this against the torch reference rather than taking it on trust.
#
# The cost is memory: P is (B, T, D, N, N) = 28.4M floats = 114 MB at the shipped shape.
# That is affordable because N=17. At large N the quadratic term would dominate and
# "lax_sort" (with the flag dropped) becomes the right choice instead.
SORT_FORM = "rank"          # "rank" | "argsort" | "lax_sort"


def _sorted_arrivals(a, w):
    """(B,T,D,N) arrivals and per-synapse weights -> both permuted into arrival order.

    All three forms are bit-identical; see the note above for why the default is neither
    of the two that use a sort primitive."""
    wb = jnp.broadcast_to(w[None], a.shape)
    if SORT_FORM == "lax_sort":
        return jax.lax.sort((a, wb), dimension=-1, num_keys=1, is_stable=True)
    if SORT_FORM == "argsort":
        idx = jnp.argsort(a, axis=-1, stable=True)
        return (jnp.take_along_axis(a, idx, axis=-1),
                jnp.take_along_axis(wb, idx, axis=-1))
    if SORT_FORM != "rank":
        raise ValueError(f"SORT_FORM must be 'rank'|'argsort'|'lax_sort', "
                         f"got {SORT_FORM!r}")
    n = a.shape[-1]
    ii = jnp.arange(n)
    # earlier[k, j] = "j comes before k", with ties broken by index == a STABLE order.
    earlier = ((a[..., None, :] < a[..., :, None])
               | ((a[..., None, :] == a[..., :, None]) & (ii[None, :] < ii[:, None])))
    rank = earlier.sum(-1)                                  # (B,T,D,N) stable position
    P = (rank[..., None, :] == ii[:, None]).astype(a.dtype)  # (B,T,D,N_r,N_k) one-hot
    return ((P * a[..., None, :]).sum(-1), (P * wb[..., None, :]).sum(-1))


def membrane(p, x, t_window=T_WINDOW, form="linear"):
    """x:(B,N) -> (a_srt, V), both (B, T, D, N)."""
    lat = latency(x, t_window)                                       # (B,N)
    d = jnp.clip(p["delay"], 0.0, t_window)                          # (T,D,N), causal
    a = lat[:, None, None, :] + d[None]                              # (B,T,D,N)
    a_srt, w_srt = _sorted_arrivals(a, synapses(p))

    tau = _tau(p["tau_raw"])[None, :, :, None]                       # (1,T,D,1)
    fn = membrane_linear if form == "linear" else membrane_cumsum
    return a_srt, fn(a_srt, w_srt, tau)


def first_spike(p, x, t_window=T_WINDOW, form="linear"):
    """-> (t_hard, t_soft), each (B, T, D). One first-spike time PER DETECTOR."""
    a_srt, V = membrane(p, x, t_window, form)

    crossed = V >= THETA_MEM
    kstar = jnp.argmax(crossed.astype(a_srt.dtype), axis=-1)
    t_hard = jnp.take_along_axis(a_srt, kstar[..., None], axis=-1)[..., 0]
    t_hard = jnp.where(crossed.any(-1), t_hard, t_window)            # (B,T,D)

    # T_cross is PER TABLE, shared across the D detectors -> (1,T,1,1).
    T_cross = jnp.exp(p["log_T_cross"])[None, :, None, None]
    c = jax.nn.sigmoid((V - THETA_MEM) / T_cross)
    surv = jnp.cumprod(1.0 - c, axis=-1)
    surv_prev = jnp.concatenate([jnp.ones_like(surv[..., :1]), surv[..., :-1]], axis=-1)
    prob = c * surv_prev
    t_soft = (prob * a_srt).sum(-1) + surv[..., -1] * t_window       # (B,T,D)
    return t_hard, t_soft


def bucket(p, t_hard, t_soft):
    """-> (b_hard (B,T,D) int32, g_soft (B,T,D,M) partition of unity).

    g telescopes to 1 exactly for any boundaries and any T_bkt; at M=2 the middle term is
    empty and g reduces to [1-S, S], a plain soft bit."""
    b = boundaries(p)[None]                                          # (1,T,D,M-1)
    T_bkt = jnp.exp(p["log_T_bkt"])[None, :, None, None]             # per TABLE
    S = jax.nn.sigmoid((t_soft[..., None] - b) / T_bkt)              # (B,T,D,M-1)
    g = jnp.concatenate([1.0 - S[..., :1], S[..., :-1] - S[..., 1:], S[..., -1:]],
                        axis=-1)
    b_hard = (t_hard[..., None] >= b).sum(-1).astype(jnp.int32)      # (B,T,D)
    return b_hard, g


def cell_index(b_hard, n_det, n_buckets):
    """(B,T,D) digits -> (B,T) mixed-radix cell index, MSB-first."""
    return (b_hard * radix(n_det, n_buckets)[None, None, :]).sum(-1)


ONEHOT_MAX_CELLS = 4096


def hard_read(p, b_hard, n_det, n_buckets):
    """(B,T,O). Full table gradient, scattered to the one selected cell per (sample,table).

    TWO SPELLINGS OF THE SAME READ, and the choice is not cosmetic -- it cost 45 minutes of
    a stalled sweep to find out.

    The torch reference writes this as advanced indexing, `self.table[tt, idx]`. Spelled
    that way in JAX the VJP is a SCATTER-ADD into the table, and this chapter's runs all
    set `XLA_FLAGS=--xla_gpu_deterministic_ops=true` for reproducibility. Under that flag
    XLA must serialise scatter-add to make the reduction order deterministic, and the
    serialised version is catastrophically slow here: 500 training iterations in over 45
    minutes, against 1.0 minute for the identical run without the flag. The model looked
    ~50x slower than its own microbenchmark, and none of the slowdown was in the model.

    The one-hot contraction below is the same function -- selecting row c is
    onehot(c) @ table, and the gradient of a matmul is a matmul, which is deterministic by
    construction and needs no serialisation. It is also what every earlier front-end in
    this chapter (c30-c37) used, which is why none of them ever hit this.

    The one-hot is materialised at (B, T, cells), so it is only cheaper while `cells` is
    small. At the shipped 64 cells it is 512x32x64 floats and the contraction is 12.6
    MFLOP -- nothing. Above ONEHOT_MAX_CELLS the one-hot would dominate, so the gather is
    used instead; a configuration up there must either drop the determinism flag or accept
    the serialised scatter. The reference's own cap is 65,536 cells.
    """
    idx = cell_index(b_hard, n_det, n_buckets)                       # (B,T)
    cells = n_buckets ** n_det
    if cells <= ONEHOT_MAX_CELLS:
        oh = jax.nn.one_hot(idx, cells, dtype=p["table"].dtype)      # (B,T,cells)
        return jnp.einsum("btc,tco->bto", oh, p["table"])
    T = p["table"].shape[0]
    return p["table"][jnp.arange(T)[None, :], idx]


def soft_read(p, g, n_det, n_buckets):
    """(B,T,O) against a DETACHED table -- the address-gradient path.

    Peels one detector axis per einsum exactly as the reference does. The dense joint
    distribution over M**D cells is never formed: after step d the running tensor is
    (B, T, M**(D-1-d), O)."""
    T, _, O = p["table"].shape
    tab = jax.lax.stop_gradient(p["table"]).reshape(
        (T,) + (n_buckets,) * n_det + (O,))
    ax = _AXES[:n_det]
    cur = jnp.einsum(f"t{ax}o,bt{ax[0]}->bt{ax[1:]}o", tab, g[:, :, 0, :])
    for d in range(1, n_det):
        rem = ax[d:]
        cur = jnp.einsum(f"bt{rem}o,bt{rem[0]}->bt{rem[1:]}o", cur, g[:, :, d, :])
    return cur


def apply(p, x, n_heads, tph, n_buckets, n_det, mode="train", form="linear"):
    """x:(B,input_dim) -> (B, n_heads, n_outputs).

    `mode` names the branch the reference takes on `self.training`. It is NOT the retired
    st/hard/soft mode string of the c30/c31/c32b modules, and "soft" is not a value:

      mode="train"  the reference's TRAINING forward (module.train()) -- straight-through.
                    Value is the hard winner, weight gradient goes to that winner, address
                    gradient flows through the soft distribution against a DETACHED table.
      mode="eval"   the reference's EVAL forward (module.eval()) -- the no_grad hard path,
                    no softmax and no temperatures. Its value must equal the train value,
                    and parity_check asserts that on both sides.

    There is no eps. The reference dropped it from the signature entirely."""
    B = x.shape[0]
    t_hard, t_soft = first_spike(p, x, form=form)
    b_hard, g = bucket(p, t_hard, t_soft)

    if mode == "eval":
        rows = hard_read(p, b_hard, n_det, n_buckets)
    elif mode == "train":
        y_hard = hard_read(p, b_hard, n_det, n_buckets)
        y_addr = soft_read(p, g, n_det, n_buckets)
        rows = y_hard + y_addr - jax.lax.stop_gradient(y_addr)
    else:
        raise ValueError(f"mode must be 'train'|'eval' (the reference branches on "
                         f"self.training; there is no 'soft' forward), got {mode!r}")
    return rows.reshape(B, n_heads, tph, -1).sum(2)


def address(p, x, n_det, n_buckets):
    """(B, n_tables) int32 joint cell index -- the coverage diagnostic.

    Unlike the n_det=1 bucket index this is NOT ordered: it is a mixed-radix packing of D
    independently-ordered digits, so adjacent values are not adjacent in time. A coverage
    histogram over it is an occupancy count, not a distribution."""
    t_hard, t_soft = first_spike(p, x)
    b_hard, _ = bucket(p, t_hard, t_soft)
    return cell_index(b_hard, n_det, n_buckets)


def digits(p, x):
    """(B, n_tables, n_det) per-detector bucket digits. For the per-detector diagnostics."""
    t_hard, t_soft = first_spike(p, x)
    return bucket(p, t_hard, t_soft)[0]


def init(key, n_buckets, n_det, tph, n_heads, input_dim, n_outputs,
         delay_init_std=0.0):
    """Mirrors the torch module's init, distribution for distribution.

    torch: delay = abs(delay_init_std * randn) if std > 0 else zeros (half-normal,
    non-negative = causal); w_raw = -2.2 + 0.5*randn; tau_raw = 1; beta_base = 0;
    beta_raw = inv_softplus(t_window / n_buckets); log_T_cross = log_T_bkt = 0;
    table = 0.1*randn.

    At n_buckets = 2 the single initial boundary sits at inv_softplus(16) -> softplus ->
    16.0, the middle of the window: each detector starts as an unbiased soft bit."""
    n_tables = n_heads * tph
    cells = n_buckets ** n_det
    if cells > MAX_CELLS:
        raise ValueError(f"n_buckets**n_det = {n_buckets}**{n_det} = {cells} exceeds "
                         f"MAX_CELLS={MAX_CELLS}")
    kw, kt, kd = jax.random.split(key, 3)
    step = T_WINDOW / n_buckets
    inv_softplus_step = float(jnp.log(jnp.expm1(jnp.asarray(step, jnp.float32))))
    dsh = (n_tables, n_det, input_dim)
    if float(delay_init_std) > 0.0:
        delay = jnp.abs(float(delay_init_std)
                        * jax.random.normal(kd, dsh, jnp.float32))
    else:
        delay = jnp.zeros(dsh, jnp.float32)
    return dict(
        delay=delay,
        w_raw=(W_RAW_MEAN + W_RAW_STD * jax.random.normal(kw, dsh, jnp.float32)),
        tau_raw=jnp.ones((n_tables, n_det), jnp.float32),
        beta_base=jnp.zeros((n_tables, n_det, 1), jnp.float32),
        beta_raw=jnp.full((n_tables, n_det, n_buckets - 1), inv_softplus_step,
                          jnp.float32),
        log_T_cross=jnp.zeros((n_tables,), jnp.float32),
        log_T_bkt=jnp.zeros((n_tables,), jnp.float32),
        table=jax.random.normal(kt, (n_tables, cells, n_outputs), jnp.float32) * 0.1,
    )


def n_params(p):
    """(front-end, table). Counts log_T_* even when frozen, matching torch's
    param_count(), which sums over parameters regardless of requires_grad."""
    det = sum(int(v.size) for k, v in p.items() if k != "table")
    return det, int(p["table"].size)
