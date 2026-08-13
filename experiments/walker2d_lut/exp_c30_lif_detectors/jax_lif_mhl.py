"""exp_c30 — LIFDetectorsMHL ported to JAX, for the MJX Walker2d SAC actor (#75).

WHY A PORT AND NOT A DROP-IN. `spiky.lutorch.lif_detectors_mhl.LIFDetectorsMHL` (branch
`exp/lif-detectors-mhl`) is a `torch.nn.Module`. Our Walker2d SAC is JAX end to end --
the ENVIRONMENT itself is MJX -- and the update is one jitted function. A torch module
cannot live inside that without a host round-trip per step, which would break `jit` and
the determinism this chapter depends on. So the module is reimplemented here, in JAX, and
held to the torch reference by an explicit parity test (`torch_ref_dump.py` ->
`parity_check.py`) rather than by inspection.

THE ONE THING THAT HAD TO BE GOT RIGHT is `mode="st"`. The torch module writes it as

    y = y_hard + y_addr - y_addr.detach()

with ``y_hard = prow(hard_bits) @ table`` and ``y_addr = prow(soft_bits) @ table.detach()``.
That single line encodes the whole decoupling:

  * ``prow(hard_bits)`` is a CONSTANT one-hot (the bits come from a `>` comparison, which
    carries no gradient), so d y_hard / d table is exactly a one-row scatter at the
    addressed row -- the same honest hard gradient our `jax_lut_grad._bwd` builds by hand
    with `.at[].add()`.
  * ``y_addr`` detaches the table, so its gradient reaches ONLY the detector parameters,
    through the full-K softmax over all 2**nap cells (a product of independent per-bit
    Bernoullis over the outcomes IS that softmax).
  * subtracting ``stop_gradient(y_addr)`` cancels its value, leaving the forward exactly
    equal to the hard single cell.

Which means this port needs NO custom VJP: transcribing that expression and letting JAX
differentiate it reproduces the torch semantics term for term. `jax_lut_grad` needs a
`custom_vjp` because its surrogate is a DIFFERENT function from its forward (a pinned
softmax standing in for a hard index); here the surrogate is already written into the
forward as an additive, value-cancelling term, so plain autodiff is both simpler and less
likely to drift from the reference than a hand-transcribed backward.

PRECISION. `jax_default_matmul_precision=highest`, for the reason jax_lut_grad documents:
TF32 in an einsum is enough to flip a sign test and select a WHOLE WRONG ROW. Here the
membrane is elementwise (no matmul), but the row einsum and the parity gradients need it.

Parameters, flat over M = n_tables * n_anchor_pairs detectors, N = input_dim:
    d (M,N)  w (M,N)  r (M,)  tau_s_raw (M,)  P (M,N,N)  tau_p_raw (M,)  theta (M,)
    log_temp_bit ()   table (n_tables, 2**nap, n_outputs)
Detector m belongs to table m // nap, bit m % nap -- the flat ordering the torch module
uses when it reshapes V to (B, n_tables, nap). The parity test pins that ordering.
"""
import jax
import jax.numpy as jnp

jax.config.update("jax_default_matmul_precision", "highest")

# torch LIFDetectorsMHL constructor defaults; named here so the trainer and the parity
# dump cannot drift apart.
T_WINDOW = 32.0
LATENCY_C = 16.0
LATENCY_ALPHA = 3.0
PAIR_INIT = 0.01
TEMP_BIT_INIT = 1.0


def bit_matrix(nap):
    """[2**nap, nap] of 0/1, MSB-first: bit_matrix[c, k] = (c >> (nap-1-k)) & 1."""
    codes = jnp.arange(1 << nap)[:, None]
    shifts = jnp.arange(nap - 1, -1, -1)[None, :]
    return ((codes >> shifts) & 1).astype(jnp.float32)


def _softplus_pos(raw):
    return jax.nn.softplus(raw) + 1e-3


def membrane(p, x, eps, t_window=T_WINDOW, latency_c=LATENCY_C,
             latency_alpha=LATENCY_ALPHA):
    """Combined LIF membrane V per detector. x:[B,N] -> [B,M].

    Vself is the magnitude channel, Vpair the order/contrast channel; `eps` is the gate
    sharpness shared by both (smaller = closer to a hard step)."""
    t = jnp.clip(latency_c - latency_alpha * x, 0.0, t_window)          # [B,N]
    a = t[:, None, :] + p["d"][None]                                    # [B,M,N]
    dts = p["r"][None, :, None] - a
    tau_s = _softplus_pos(p["tau_s_raw"])[None, :, None]
    vself = (p["w"][None] * jnp.exp(-jax.nn.relu(dts) / tau_s)
             * jax.nn.sigmoid(dts / eps)).sum(-1)                       # [B,M]
    # D[b,m,i,j] = a[b,m,j] - a[b,m,i]  (torch: a.unsqueeze(-2) - a.unsqueeze(-1))
    D = a[:, :, None, :] - a[:, :, :, None]                             # [B,M,N,N]
    tau_p = _softplus_pos(p["tau_p_raw"])[None, :, None, None]
    g = jnp.exp(-jax.nn.relu(D) / tau_p) * jax.nn.sigmoid(D / eps)
    n = x.shape[1]
    offdiag = 1.0 - jnp.eye(n, dtype=x.dtype)
    vpair = ((p["P"] * offdiag)[None] * g).sum((-1, -2))                # [B,M]
    return vself + vpair


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

    mode 'st'   : decoupled straight-through -- TRAIN with this. Forward value is the hard
                  single cell; the table gradient is the one-row scatter, the detector
                  gradient is the full-K softmax.
    mode 'hard' : pure argmax inference; identical forward value to 'st', no grad path.
    mode 'soft' : the differentiable blend, reference only -- do NOT train with it.
    """
    n_tables = n_heads * tph
    B = x.shape[0]
    V = membrane(p, x, eps).reshape(B, n_tables, nap)
    th = p["theta"].reshape(1, n_tables, nap)
    hard_bits = (V > th).astype(x.dtype)

    if mode == "hard":
        rows = _rows(_prow(hard_bits, nap), p["table"])
    elif mode in ("soft", "st"):
        temp_bit = jnp.exp(p["log_temp_bit"])
        soft_bits = jnp.clip(jax.nn.sigmoid((V - th) / temp_bit), 1e-6, 1 - 1e-6)
        if mode == "soft":
            rows = _rows(_prow(soft_bits, nap), p["table"])
        else:
            y_hard = _rows(_prow(hard_bits, nap), p["table"])
            y_addr = _rows(_prow(soft_bits, nap),
                           jax.lax.stop_gradient(p["table"]))
            rows = y_hard + y_addr - jax.lax.stop_gradient(y_addr)
    else:
        raise ValueError(f"mode must be 'st'|'hard'|'soft', got {mode!r}")
    return rows.reshape(B, n_heads, tph, -1).sum(2)


def address(p, x, eps, n_heads, tph, nap):
    """Hard packed address per (sample, table): [B, n_tables] int32, MSB-first.

    The coverage diagnostic, and the analogue of `jax_lut_grad._hard_index`."""
    n_tables = n_heads * tph
    V = membrane(p, x, eps).reshape(x.shape[0], n_tables, nap)
    bits = (V > p["theta"].reshape(1, n_tables, nap)).astype(jnp.int32)
    powers = 2 ** jnp.arange(nap - 1, -1, -1)
    return (bits * powers[None, None, :]).sum(-1)


def init(key, nap, tph, n_heads, input_dim, n_outputs):
    """Mirrors the torch module's own init, distribution for distribution.

    torch: d=0, w=0.2*randn, r=0.9*T, tau_*_raw=1, P=pair_init*randn, theta=0,
    log_temp_bit=log(1), table=0.1*randn. Reproduced rather than re-tuned, so any
    difference from the reference is attributable to the port and not to the init."""
    n_tables = n_heads * tph
    M = n_tables * nap
    kw, kp, kt = jax.random.split(key, 3)
    return dict(
        d=jnp.zeros((M, input_dim), jnp.float32),
        w=jax.random.normal(kw, (M, input_dim), jnp.float32) * 0.2,
        r=jnp.full((M,), 0.9 * T_WINDOW, jnp.float32),
        tau_s_raw=jnp.ones((M,), jnp.float32),
        P=jax.random.normal(kp, (M, input_dim, input_dim), jnp.float32) * PAIR_INIT,
        tau_p_raw=jnp.ones((M,), jnp.float32),
        theta=jnp.zeros((M,), jnp.float32),
        log_temp_bit=jnp.log(jnp.asarray(TEMP_BIT_INIT, jnp.float32)),
        table=jax.random.normal(kt, (n_tables, 1 << nap, n_outputs),
                                jnp.float32) * 0.1,
    )


def n_params(p):
    """Learnable parameter count. Everything here IS learnable -- unlike the LUT actor,
    the LIF front-end has no frozen anchor buffers."""
    det = sum(int(v.size) for k, v in p.items() if k != "table")
    return det, int(p["table"].size)
