"""exp_c31 — PureLIFDetectorsMHL ported to JAX, for the MJX Walker2d SAC actor (#75).

The third LIF front-end in this line, after exp_c30 (dense `P`) and exp_c30b (factorised
`P`). `spiky.lutorch.pure_lif_detectors_mhl.PureLIFDetectorsMHL` on branch
`exp/lif-detectors-mhl` is not a trimmed LIFDetectorsMHL -- it is a DIFFERENT readout, and
the differences are worth stating before the code, because two of them change what the
trainer can even do:

  1. TIME TO FIRST SPIKE, not a membrane threshold. LIFDetectorsMHL evaluates one
     smoothed membrane V at a single learned read time `r` and compares it to `theta`.
     PureLIF integrates the leaky membrane across the arrivals IN ARRIVAL ORDER (hence the
     sort), finds the FIRST arrival at which V crosses a FIXED theta_mem = 1.0, and uses
     that crossing TIME as the detector's output. The learned quantity is the deadline
     `L`: the bit is "did this cell fire before L".

  2. THE BIT IS FLIPPED. `bit = 1[t* < L]` -- EARLY spike means detected. In
     LIFDetectorsMHL a LARGER membrane meant detected. The sign convention is inverted, so
     a port that transcribed the comparison from the older module would train a mirror
     image of the intended detector and still pass a shape check.

  3. NO ORDERED-PAIR CHANNEL. No `P` at all -- the (M,N,N) block that was 55,488 of
     exp_c30's 87,361 params, and the thing exp_c30b spent its whole existence
     factorising. What replaces it is not a cheaper pair term but the arrival ORDER
     itself: the sort makes the membrane path order-dependent, so ordering information
     enters through the dynamics rather than through a learned pairwise matrix.

  4. `eps` IS UNUSED. The module accepts it for API parity and ignores it (its docstring
     says so outright): TTFS has no gate temperature. The soft path is smoothed by
     `T_cross` (crossing sharpness) and `temp_bit` (bit sharpness), and BOTH ARE TRAINABLE
     per LUT. So there is no annealing knob on this model -- see `pure_lif_sac.py`, which
     keeps the eps plumbing so the trainer stays a one-line diff from exp_c30's but does
     not pretend the schedule does anything.

  5. THE PER-TABLE PARAMS ARE DIFFERENT ONES. LIFDetectorsMHL shares `r, tau_s, tau_p,
     theta` per table and has a SCALAR `log_temp_bit`. PureLIF shares `tau, T_cross,
     log_temp_bit` per table (log_temp_bit is now a VECTOR of length n_tables) and makes
     `L` per DETECTOR. Anything indexing log_temp_bit as a scalar will silently broadcast.

WHAT IS UNCHANGED, and is the reason the port needs no custom VJP: the decode. It is
`_prow` + the decoupled straight-through block, character for character the same as
LIFDetectorsMHL:

    y = y_hard + y_addr - stop_gradient(y_addr)

with `y_hard = prow(hard_bits) @ table` and `y_addr = prow(soft_bits) @ table.detach()`.
The surrogate is written INTO the forward as an additive, value-cancelling term, so plain
autodiff of the transcribed expression reproduces the torch semantics term for term. (By
contrast `jax_lut_grad` needs a `custom_vjp` because its surrogate is a genuinely
different function from its forward.)

THE SORT is the one new gradient path. `jnp.take_along_axis(a, argsort(a), -1)` matches
torch's `sort` both in value and in backward (a permutation scatters its cotangent), and
`w` is gathered by the SAME indices, exactly as the reference does. Ties are the caveat:
`latency = clip(c - alpha*x, 0, T)` saturates, so two clamped channels arrive at the same
instant and the tie-break decides which weight lands first. `argsort(..., stable=True)`
pins ours to ascending index order; the parity test avoids the clamp region so the
comparison measures the port and not two libraries' tie-breaks. The trainer is
self-consistent either way -- only cross-framework comparison is affected.

PRECISION. `jax_default_matmul_precision=highest`, for the reason jax_lut_grad documents:
TF32 in an einsum is enough to flip a comparison and select a WHOLE WRONG ROW.

Parameters (M = n_tables * nap detectors, N = input_dim, T = n_tables):
    delay (M,N)  w (M,N)  L (M,)  tau_raw (T,)  log_T_cross (T,)  log_temp_bit (T,)
    table (T, 2**nap, n_outputs)
Detector m belongs to table m // nap, bit m % nap -- the flat ordering the torch module
uses when it reshapes to (B, n_tables, nap). The parity test pins that ordering.
"""
import jax
import jax.numpy as jnp

jax.config.update("jax_default_matmul_precision", "highest")

# torch PureLIFDetectorsMHL constructor defaults; named here so the trainer and the parity
# dump cannot drift apart.
T_WINDOW = 32.0
LATENCY_C = 16.0
LATENCY_ALPHA = 3.0
TEMP_BIT_INIT = 1.0
THETA_MEM = 1.0          # a fixed buffer in the reference, NOT a parameter


def bit_matrix(nap):
    """[2**nap, nap] of 0/1, MSB-first: bit_matrix[c, k] = (c >> (nap-1-k)) & 1."""
    codes = jnp.arange(1 << nap)[:, None]
    shifts = jnp.arange(nap - 1, -1, -1)[None, :]
    return ((codes >> shifts) & 1).astype(jnp.float32)


def _softplus_pos(raw):
    return jax.nn.softplus(raw) + 1e-3


def spike_bits(p, x, n_tables, nap, t_window=T_WINDOW, latency_c=LATENCY_C,
               latency_alpha=LATENCY_ALPHA):
    """The TTFS front-end. x:[B,N] -> (hard_bits, soft_bits, t_hard).

    hard_bits/soft_bits are [B, n_tables, nap]; t_hard is [B, M] (diagnostic).

    Step by step, mirroring the reference:
      a[b,m,i]    arrival of input i at detector m = latency(x) + delay
      a_srt       arrivals ascending; w gathered by the SAME permutation
      V[b,m,k]    membrane at the k-th arrival = sum over ALREADY-ARRIVED j of
                  w_j * exp(-(a_k - a_j)/tau).  Causal mask includes j == k, so a
                  detector can be pushed over threshold by the arrival being read.
      t_hard      the first a_srt[k] with V_k >= theta_mem, else the window edge.
      t_soft      a smooth first-success over the same sorted arrivals: with
                  c_k = sigmoid((V_k - theta)/T_cross) the probability that k is the
                  first crossing is c_k * prod_{j<k}(1 - c_j); the leftover survival
                  mass is assigned to the window edge, so t_soft -> t_hard as
                  T_cross -> 0 AND as the never-fires case is approached.
      bits        FLIPPED: fired before the deadline L counts as detected.
    """
    B, n = x.shape
    M = n_tables * nap

    t = jnp.clip(latency_c - latency_alpha * x, 0.0, t_window)          # [B,N]
    a = t[:, None, :] + p["delay"][None]                                # [B,M,N]

    # Stable ascending sort, and w permuted with it. `stable` pins the tie-break; see the
    # module docstring on why that matters at the clamp boundaries.
    idx = jnp.argsort(a, axis=-1, stable=True)                          # [B,M,N]
    a_srt = jnp.take_along_axis(a, idx, axis=-1)
    w_srt = jnp.take_along_axis(jnp.broadcast_to(p["w"][None], a.shape), idx, axis=-1)

    # dt[b,m,k,j] = a_srt[k] - a_srt[j]   (torch: a_srt.unsqueeze(-1) - a_srt.unsqueeze(-2))
    dt = a_srt[:, :, :, None] - a_srt[:, :, None, :]                    # [B,M,N,N]
    tau = jnp.repeat(_softplus_pos(p["tau_raw"]), nap)[None, :, None, None]
    contrib = jnp.where(dt >= 0.0,
                        w_srt[:, :, None, :] * jnp.exp(-jax.nn.relu(dt) / tau),
                        0.0)
    V = contrib.sum(-1)                                                 # [B,M,N]

    # --- hard first-spike time ---------------------------------------------
    crossed = V >= THETA_MEM
    kstar = jnp.argmax(crossed.astype(x.dtype), axis=-1)                # first True
    t_hard = jnp.take_along_axis(a_srt, kstar[:, :, None], axis=-1)[:, :, 0]
    t_hard = jnp.where(crossed.any(-1), t_hard, t_window)               # [B,M]

    # --- soft first-spike time ---------------------------------------------
    T_cross = jnp.repeat(jnp.exp(p["log_T_cross"]), nap)[None, :, None]
    c = jax.nn.sigmoid((V - THETA_MEM) / T_cross)                       # [B,M,N]
    surv = jnp.cumprod(1.0 - c, axis=-1)
    surv_prev = jnp.concatenate([jnp.ones_like(surv[..., :1]), surv[..., :-1]], axis=-1)
    prob = c * surv_prev
    t_soft = (prob * a_srt).sum(-1) + surv[..., -1] * t_window          # [B,M]

    # --- FLIPPED detection: early spike (t < L) is a set bit ----------------
    s_hard = (p["L"][None] - t_hard).reshape(B, n_tables, nap)
    s_soft = (p["L"][None] - t_soft).reshape(B, n_tables, nap)
    hard_bits = (s_hard > 0).astype(x.dtype)
    temp_bit = jnp.exp(p["log_temp_bit"]).reshape(1, n_tables, 1)
    soft_bits = jnp.clip(jax.nn.sigmoid(s_soft / temp_bit), 1e-6, 1 - 1e-6)
    return hard_bits, soft_bits, t_hard


def _prow(bits, nap):
    """[B,T,nap] bit values -> [B,T,2**nap] per-table cell distribution.

    prow[...,c] = prod_k [bits_k if code c has bit k set else 1-bits_k]. One-hot for hard
    0/1 bits; the exact softmax over cells for independent Bernoulli soft bits."""
    bm = bit_matrix(nap)[None, None]                                    # [1,1,rows,nap]
    b = bits[:, :, None, :]                                             # [B,T,1,nap]
    return jnp.prod(bm * b + (1.0 - bm) * (1.0 - b), axis=-1)


def _rows(prow, table):
    return jnp.einsum("btc,tco->bto", prow, table)


def apply(p, x, eps, n_heads, tph, nap, mode="st"):
    """x:[B,input_dim] -> [B,n_heads,n_outputs].

    `eps` is accepted and IGNORED, exactly as the torch reference does -- TTFS has no gate
    temperature. It stays in the signature so this module is drop-in for jax_lif_mhl.

    mode 'st'   : decoupled straight-through -- TRAIN with this. Forward value is the hard
                  single cell; the table gradient is the one-row scatter, the detector
                  gradient is the full-K softmax.
    mode 'hard' : pure argmax inference; identical forward value to 'st', no grad path.
    mode 'soft' : the differentiable blend, reference only -- do NOT train with it.
    """
    del eps
    n_tables = n_heads * tph
    B = x.shape[0]
    hard_bits, soft_bits, _ = spike_bits(p, x, n_tables, nap)

    if mode == "hard":
        rows = _rows(_prow(hard_bits, nap), p["table"])
    elif mode == "soft":
        rows = _rows(_prow(soft_bits, nap), p["table"])
    elif mode == "st":
        y_hard = _rows(_prow(hard_bits, nap), p["table"])
        y_addr = _rows(_prow(soft_bits, nap), jax.lax.stop_gradient(p["table"]))
        rows = y_hard + y_addr - jax.lax.stop_gradient(y_addr)
    else:
        raise ValueError(f"mode must be 'st'|'hard'|'soft', got {mode!r}")
    return rows.reshape(B, n_heads, tph, -1).sum(2)


def address(p, x, eps, n_heads, tph, nap):
    """Hard packed address per (sample, table): [B, n_tables] int32, MSB-first.

    The coverage diagnostic, and the analogue of `jax_lut_grad._hard_index`."""
    del eps
    n_tables = n_heads * tph
    hard_bits, _, _ = spike_bits(p, x, n_tables, nap)
    powers = 2 ** jnp.arange(nap - 1, -1, -1)
    return (hard_bits.astype(jnp.int32) * powers[None, None, :]).sum(-1)


def init(key, nap, tph, n_heads, input_dim, n_outputs):
    """Mirrors the torch module's own init, distribution for distribution.

    torch: delay=0, w=0.2*randn, L=0.5*T_window, tau_raw=1, log_T_cross=0,
    log_temp_bit=log(1)=0, table=0.1*randn. Reproduced rather than re-tuned, so any
    difference from the reference is attributable to the port and not to the init."""
    n_tables = n_heads * tph
    M = n_tables * nap
    kw, kt = jax.random.split(key, 2)
    return dict(
        delay=jnp.zeros((M, input_dim), jnp.float32),
        w=jax.random.normal(kw, (M, input_dim), jnp.float32) * 0.2,
        L=jnp.full((M,), 0.5 * T_WINDOW, jnp.float32),
        tau_raw=jnp.ones((n_tables,), jnp.float32),
        log_T_cross=jnp.zeros((n_tables,), jnp.float32),
        log_temp_bit=jnp.full((n_tables,), float(jnp.log(jnp.asarray(TEMP_BIT_INIT))),
                              jnp.float32),
        table=jax.random.normal(kt, (n_tables, 1 << nap, n_outputs), jnp.float32) * 0.1,
    )


def n_params(p):
    """Learnable parameter count. Everything here IS learnable -- unlike the LUT actor,
    the LIF front-end has no frozen anchor buffers."""
    det = sum(int(v.size) for k, v in p.items() if k != "table")
    return det, int(p["table"].size)
