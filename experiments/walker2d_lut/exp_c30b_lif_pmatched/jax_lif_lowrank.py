"""exp_c30b — the exp_c30 LIF-detector actor with a PARAM-MATCHED ordered-pair channel.

exp_c30 showed the LIF front-end drives the Walker2d SAC actor about as well as the
hyperplane sign-tests (3931 +/- 586 vs exp_c18's 4308 +/- 500) -- but at 87,361 params
against 49,152, so it said nothing about per-parameter quality. The ordered-pair channel
`P` was 55,488 of that, 88% of the detector bank. This module shrinks P alone; everything
else is byte-for-byte the exp_c30 model.

THE REPLACEMENT. Dense P is (M, N, N) -- a free ordered-pair weight per detector. Here:

    P[m] = Pu[m] @ Pv[m].T  +  Pb[m] @ 1.T          Pu, Pv: (N, 2)   Pb: (N,)

i.e. a rank-2 factorisation plus a rank-1 term whose right factor is pinned to the
all-ones vector. The pinned term is not filler: `Pb[m, i]` is a per-SOURCE-channel weight
applied to all of detector m's outgoing comparisons -- "how much does channel i's arrival
order matter to this detector at all", independent of which channel it is compared
against. So the reduced P is a structured rank-3 matrix, one of whose factors is frozen.

WHY THIS SHAPE AND NOT ANOTHER. The budget for P is 49,152 - 31,873 = 17,279 params, and
the clean alternatives all miss it or cost too much:

    per-detector rank 2                13,056   44,929   -8.6%   2x pair-channel cost
    per-detector rank 3                19,584   51,457   +4.7%   3x
    per-table-shared V, rank 4         15,232   47,105   -4.2%   4x
    pair channel on 9 of 17 channels   15,552   47,425   -3.5%   0.28x
    shared CP dictionary, C=76         17,176   49,049   -0.2%   76x   <- infeasible
    THIS: rank 2 + source bias         16,320   48,193   -1.95%  2x

The CP dictionary lands closest on paper and is unusable in practice: its contraction is
C times the pair-channel cost rather than k times, because the shared basis cannot be
folded into the per-detector contraction. Dropping 8 of 17 channels lands at -3.5% but
removes those channels from ordered comparison ENTIRELY -- and exp_c29 spent a whole wave
learning that per-channel liveness is exactly where these models fail. This shape keeps
all 17 channels, keeps every detector's pair matrix independent, costs 2x, and lands
1.95% UNDER the target. Under rather than over is deliberate: if the LIF actor holds up
at 98% of the hyperplane budget, the per-parameter claim is conservative.

INIT preserves the reference's "pair channel starts near zero, so each detector begins as
a pure value/range unit". Dense torch init is P ~ N(0, 0.01^2). Here the product of two
N(0, s^2) factors summed over k has std s^2 sqrt(k), and the bias adds t, so the variance
is split evenly between the two terms: 2 s^4 = t^2 = 0.5e-4.

Everything else -- membrane, latency map, `_prow`, the mode="st" decoupling, `address`,
`n_params` -- is unchanged from `exp_c30/jax_lif_mhl.py`, which is itself parity-checked
against the torch reference 13/13. `check_lowrank.py` pins THIS module to that one by
materialising P and comparing, so the dense path stays the oracle.
"""
import jax
import jax.numpy as jnp

jax.config.update("jax_default_matmul_precision", "highest")

T_WINDOW = 32.0
LATENCY_C = 16.0
LATENCY_ALPHA = 3.0
TEMP_BIT_INIT = 1.0
PAIR_RANK = 2
# 2*s^4 = t^2 = 0.5e-4  ->  dense-equivalent P std ~ 0.01, matching the torch reference.
PU_INIT = (0.5e-4 / 2.0) ** 0.25          # 0.0707
PB_INIT = (0.5e-4) ** 0.5                 # 0.00707


def bit_matrix(nap):
    codes = jnp.arange(1 << nap)[:, None]
    shifts = jnp.arange(nap - 1, -1, -1)[None, :]
    return ((codes >> shifts) & 1).astype(jnp.float32)


def _softplus_pos(raw):
    return jax.nn.softplus(raw) + 1e-3


def dense_P(p):
    """Materialise the (M, N, N) ordered-pair matrix this factorisation represents.

    Only for checking and inspection -- the forward never builds it, which is the point."""
    return (jnp.einsum("mik,mjk->mij", p["Pu"], p["Pv"])
            + p["Pb"][:, :, None])


def membrane(p, x, eps, t_window=T_WINDOW, latency_c=LATENCY_C,
             latency_alpha=LATENCY_ALPHA):
    """Combined LIF membrane V per detector. x:[B,N] -> [B,M].

    The pair term is contracted through the factors, never through a dense P: the
    (B,M,N,N) gate tensor is reduced against Pv (rank-k), then against Pu, so the cost is
    k times the dense contraction rather than N times."""
    t = jnp.clip(latency_c - latency_alpha * x, 0.0, t_window)
    a = t[:, None, :] + p["d"][None]                                    # [B,M,N]
    dts = p["r"][None, :, None] - a
    tau_s = _softplus_pos(p["tau_s_raw"])[None, :, None]
    vself = (p["w"][None] * jnp.exp(-jax.nn.relu(dts) / tau_s)
             * jax.nn.sigmoid(dts / eps)).sum(-1)                       # [B,M]

    D = a[:, :, None, :] - a[:, :, :, None]                             # [B,M,N,N]
    tau_p = _softplus_pos(p["tau_p_raw"])[None, :, None, None]
    g = jnp.exp(-jax.nn.relu(D) / tau_p) * jax.nn.sigmoid(D / eps)
    n = x.shape[1]
    gm = g * (1.0 - jnp.eye(n, dtype=x.dtype))                          # off-diagonal
    # sum_ij [ sum_k Pu[m,i,k] Pv[m,j,k] + Pb[m,i] ] * gm[b,m,i,j]
    tmp = jnp.einsum("bmij,mjk->bmik", gm, p["Pv"])
    vpair = (jnp.einsum("mik,bmik->bm", p["Pu"], tmp)
             + jnp.einsum("mi,bmi->bm", p["Pb"], gm.sum(-1)))
    return vself + vpair


def _prow(bits, nap):
    bm = bit_matrix(nap)[None, None]
    b = bits[:, :, None, :]
    return jnp.prod(bm * b + (1.0 - bm) * (1.0 - b), axis=-1)


def _rows(prow, table):
    return jnp.einsum("btc,tco->bto", prow, table)


def apply(p, x, eps, n_heads, tph, nap, mode="st"):
    """x:[B,input_dim] -> [B,n_heads,n_outputs]. Modes as in exp_c30/jax_lif_mhl."""
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
            y_addr = _rows(_prow(soft_bits, nap), jax.lax.stop_gradient(p["table"]))
            rows = y_hard + y_addr - jax.lax.stop_gradient(y_addr)
    else:
        raise ValueError(f"mode must be 'st'|'hard'|'soft', got {mode!r}")
    return rows.reshape(B, n_heads, tph, -1).sum(2)


def address(p, x, eps, n_heads, tph, nap):
    n_tables = n_heads * tph
    V = membrane(p, x, eps).reshape(x.shape[0], n_tables, nap)
    bits = (V > p["theta"].reshape(1, n_tables, nap)).astype(jnp.int32)
    powers = 2 ** jnp.arange(nap - 1, -1, -1)
    return (bits * powers[None, None, :]).sum(-1)


def init(key, nap, tph, n_heads, input_dim, n_outputs, rank=PAIR_RANK):
    """Identical to exp_c30's init except that dense P becomes (Pu, Pv, Pb)."""
    n_tables = n_heads * tph
    M = n_tables * nap
    kw, ku, kv, kb, kt = jax.random.split(key, 5)
    return dict(
        d=jnp.zeros((M, input_dim), jnp.float32),
        w=jax.random.normal(kw, (M, input_dim), jnp.float32) * 0.2,
        r=jnp.full((M,), 0.9 * T_WINDOW, jnp.float32),
        tau_s_raw=jnp.ones((M,), jnp.float32),
        Pu=jax.random.normal(ku, (M, input_dim, rank), jnp.float32) * PU_INIT,
        Pv=jax.random.normal(kv, (M, input_dim, rank), jnp.float32) * PU_INIT,
        Pb=jax.random.normal(kb, (M, input_dim), jnp.float32) * PB_INIT,
        tau_p_raw=jnp.ones((M,), jnp.float32),
        theta=jnp.zeros((M,), jnp.float32),
        log_temp_bit=jnp.log(jnp.asarray(TEMP_BIT_INIT, jnp.float32)),
        table=jax.random.normal(kt, (n_tables, 1 << nap, n_outputs),
                                jnp.float32) * 0.1,
    )


def n_params(p):
    det = sum(int(v.size) for k, v in p.items() if k != "table")
    return det, int(p["table"].size)
