"""The power-of-two two-cell LUT read: the ONE shared definition used by LightMultiHeadLUT's quantised training
forward (straight-through), by its integer eval read, and by the exported artefact (quantised_light_ffn.py).

Reference: doc/research/lut_ablation/quantisation_simple.tex (the simplified, no-mantissa note). Every formula here is the
note's, and where the ablation prototype (lut_ablation/*/pow2_blend_read.py) computed the same quantity another way this
module follows the NOTE. The differences are ulp-level and are counted and bounded in test_light_quant_mode.py:

  * log2 s is evaluated in the log domain,  log2 S + (g + gamma * sum_i logsigmoid(beta u_i)) / ln 2  (note Section 3;
    g = 0 in the frozen-g rows the note describes). The prototype formed s and took log2(s).
  * q = clamp(floor(2 u* / (tau ln 2) + 1/2), 0, J)  (note eq. 5). The prototype counted J thresholds with bucketize.
  * c_q = log2(1 + 2^-q) for q < C, else 0  (note eq. 5). The prototype read an 8-entry table (bit-identical values).
  * round(xi) = floor(xi + 1/2), half UP, for k' and for the integer weights (note Section 3). The prototype used
    torch.round (half to even).

Constants: J = 64 (bound on q) and C = 8 (c_q cut-off) are fixed; they are the trained values and J is not inert for
training (a smaller bound changes the straight-through gradients of dropped cells). Q (second-cell drop), L (window width)
and kmax (window top) come from a preset and may be overridden for research.

Integer read (note Section 6): the head sum is kept in units of 2^-6 in int32; each table adds
(W_hat[c1] << (k' + 6)) + (W_hat[c2] << (k' + 6 - q)), both shifts in [0, 10] because k' >= -3 and q <= 3 for any cell that
is read. int8_accumulate gathers the int8 rows as bytes, shifts each left by its cell's shift and sums: shift-and-add, no
multiply. Under torch.compile this is one generated kernel over the int8 bytes (QuantisedLightFFN's torch read; with
the CUDA extension the fused kernel in csrc/pow2_int8_read.cu computes the same integers).

Table width: only int8 is implemented. The quantiser is generic in bits / offset; packing and the integer read dispatch on
`bits` (pack_tables -> pack_int8_rows, int_blend_read -> int8_blend_read), so int4 is an added preset, packer and reader.
"""
import math
from typing import Dict, Optional

import torch
import torch.nn.functional as F

LN2 = math.log(2.0)
J = 64                      # bound on q
C = 8                       # c_q = 0 for q >= C
FIXED_POINT_SHIFT = 6       # the integer head sum is in units of 2^-6 (note Section 6)
N_SHIFTS = 11               # shifts k' + 6 - q, k' + 6 lie in [0, 10] for every cell that is read

# Presets. Only int8 is implemented; int4 (the note's b=4, o=-1 row) was deliberately deferred. The quantiser
# (head_chan_exponents / quantise_tables) is parameterised by bit width and offset, and packing / the integer read dispatch on
# `bits`, so int4 is a later preset plus a nibble packer and reader beside pack_int8_rows / int8_blend_read, not a redesign.
PRESETS: Dict[str, Dict[str, int]] = {
    "p2_int8": dict(bits=8, offset=0, Q=3, L=8, kmax=4),
}
_OVERRIDABLE = ("Q", "L", "kmax")


def resolve_quant_config(quant_mode: Optional[str], quant_overrides: Optional[dict] = None) -> Optional[dict]:
    """The validated quantisation constants for `quant_mode` (None -> None). Overrides may set Q, L and kmax only."""
    if quant_mode is None:
        if quant_overrides:
            raise ValueError("quant_overrides given without quant_mode")
        return None
    if quant_mode not in PRESETS:
        raise ValueError(f"quant_mode must be None or one of {sorted(PRESETS)}, got {quant_mode!r}")
    cfg = dict(PRESETS[quant_mode])
    for k, v in (quant_overrides or {}).items():
        if k not in _OVERRIDABLE:
            raise ValueError(f"quant_overrides may set only {_OVERRIDABLE}; got {k!r} (J={J} and C={C} are fixed, "
                             "bits and offset follow the preset)")
        if isinstance(v, bool) or not isinstance(v, int):
            raise ValueError(f"quant_overrides[{k!r}] must be an int, got {v!r}")
        cfg[k] = v
    if not 0 <= cfg["Q"] <= 3:
        # Q > 3 would allow k' + 6 - q < 0 (a right shift) for k' = -3, breaking the integer form of note Section 6.
        raise ValueError(f"Q must be in [0, 3] (the second cell's shift k' + 6 - q must stay >= 0), got {cfg['Q']}")
    if not 1 <= cfg["L"] <= 8:
        raise ValueError(f"L must be in [1, 8], got {cfg['L']}")
    if not -3 <= cfg["kmax"] - cfg["L"] + 1 or not cfg["kmax"] <= 4:
        # window [kmax - L + 1, kmax] must stay inside [-3, 4] so both shifts fit [0, 10] and the int32 bound holds.
        raise ValueError(f"the window [kmax - L + 1, kmax] must lie inside [-3, 4], got kmax={cfg['kmax']}, L={cfg['L']}")
    cfg["mode"] = quant_mode
    cfg["lo"] = cfg["kmax"] - cfg["L"] + 1
    cfg["hi"] = cfg["kmax"]
    cfg["J"], cfg["C"] = J, C
    return cfg


def round_half_up(x: torch.Tensor) -> torch.Tensor:
    """round(xi) = floor(xi + 1/2), the note's rounding (ties go up, unlike torch.round)."""
    return torch.floor(x + 0.5)


# ------------------------------------------------------------------ the per-table scalars (no gradient) -----------------
def log2_score(m: torch.Tensor, g: torch.Tensor, beta: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
    """log2 s of the learned_margin score s = (sum_i u_i) exp(g + gamma sum_i logsigmoid(beta u_i)), in the log domain.
    m = |d| [..., NAP] -> [...]. -inf where all margins are 0 (such a table is skipped)."""
    return torch.log2(m.sum(dim=-1)) + (g + gamma * F.logsigmoid(beta * m).sum(dim=-1)) / LN2


def q_exponent(mv: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
    """q = clamp(floor(2 u* / (tau ln 2) + 1/2), 0, J) as a float tensor of integers. mv = u* (the smallest margin)."""
    return torch.clamp(torch.floor(mv * (2.0 / (tau * LN2)) + 0.5), 0, J)


def c_q(q: torch.Tensor) -> torch.Tensor:
    """c_q = log2(1 + 2^-q) for q < C, else 0."""
    return torch.where(q < C, torch.log2(1.0 + torch.pow(2.0, -q)), torch.zeros_like(q))


def blend_candidates(d: torch.Tensor, index: torch.Tensor, powers: torch.Tensor):
    """The two cells a table reads. d [..., NAP] margins (differentiable), index [...] packed address (MSB-first).
    Returns m = |d| [..., NAP], mv = min margin [..., 1] (differentiable), idx [..., 2] = [c1, c2] where c2 flips the
    least certain bit."""
    m = d.abs()
    mv, mj = m.min(dim=-1, keepdim=True)
    bsel = (torch.gather(d, -1, mj) > 0).to(torch.int64)
    idx = torch.cat([index.unsqueeze(-1), index.unsqueeze(-1) + powers[mj] * (1 - 2 * bsel)], dim=-1)
    return m, mv, idx


@torch.no_grad()
def blend_exponents(m: torch.Tensor, mv: torch.Tensor, tau: torch.Tensor, g: torch.Tensor, beta: torch.Tensor,
                    gamma: torch.Tensor, cfg: dict):
    """The note's per-table integers. m [..., NAP], mv [..., 1]. Returns float tensors of integers q, k [...] (k already
    clamped into the window [lo, hi]) and bool tensors skip (k' below the window), drop (second cell not read)."""
    q = q_exponent(mv.squeeze(-1), tau)
    kr = round_half_up(log2_score(m, g, beta, gamma) - c_q(q))
    skip = kr < cfg["lo"]
    k = torch.clamp(kr, cfg["lo"], cfg["hi"])
    drop = q > cfg["Q"]
    return q, k, skip, drop


# ------------------------------------------------------------------ training: straight-through weights ------------------
def ste_blend_weights(score: torch.Tensor, mv: torch.Tensor, tau: torch.Tensor, q: torch.Tensor, k: torch.Tensor,
                      skip: torch.Tensor, drop: torch.Tensor) -> torch.Tensor:
    """Per-cell weights [..., 2] with value exactly (2^k', 2^(k'-q)) (0 when skipped / dropped) and the gradient of the
    exact blend weights (s v, s (1 - v)) times the same ratios (note eq. 6). The ratio uses the clamped k' also for skipped
    and dropped cells, so they keep a gradient and can come back (the prototype's semantics)."""
    x = 2.0 * mv / tau                                             # [..., 1]
    e = score.unsqueeze(-1) * torch.cat([torch.sigmoid(x), torch.sigmoid(-x)], dim=-1)   # [..., 2]: s v, s (1 - v)
    b = torch.pow(2.0, torch.stack([k, k - q], dim=-1)).to(e.dtype)                      # clamped values
    keep = torch.stack([~skip, ~(skip | drop)], dim=-1)
    val = torch.where(keep, b, torch.zeros_like(b))
    s = e * (b / e.detach().clamp_min(1e-30))
    return val + (s - s.detach())


# ------------------------------------------------------------------ table weights: exponents, integers, packing ---------
@torch.no_grad()
def head_chan_exponents(W: torch.Tensor, n_heads: int, bits: int, offset: int) -> torch.Tensor:
    """e[h, c] = ceil(log2 max |W over head h's tables and cells, channel c|) - (bits - 1) + offset  (note eq. 7).
    W [n_heads * T, K, D] head-major. Returns a float tensor of integers [n_heads, D]."""
    D = W.shape[-1]
    A = W.abs().reshape(n_heads, -1, D).amax(dim=1)
    return torch.ceil(torch.log2(A.clamp_min(1e-30))) - (bits - 1) + offset


def _scale(e: torch.Tensor, n_tables: int, dtype) -> torch.Tensor:
    H, D = e.shape
    return torch.pow(2.0, e).to(dtype).reshape(H, 1, 1, D).expand(H, n_tables // H, 1, D).reshape(n_tables, 1, D)


@torch.no_grad()
def quantise_tables(W: torch.Tensor, e: torch.Tensor, bits: int) -> torch.Tensor:
    """W_hat = clamp(round(W / 2^e), -2^(b-1), 2^(b-1) - 1)  (note eq. 8), integers in W's float dtype, [n_tables, K, D]."""
    lo, hi = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
    return torch.clamp(round_half_up(W / _scale(e, W.shape[0], W.dtype)), lo, hi)


def ste_tables(W: torch.Tensor, n_heads: int, bits: int, offset: int) -> torch.Tensor:
    """Training tables: value exactly W_hat * 2^e, gradient the identity to the float master W (note Section 5)."""
    e = head_chan_exponents(W, n_heads, bits, offset)
    with torch.no_grad():
        Wq = quantise_tables(W, e, bits) * _scale(e, W.shape[0], W.dtype)
    return Wq + (W - W.detach())


@torch.no_grad()
def pack_tables(Wint: torch.Tensor, bits: int) -> torch.Tensor:
    """Integer tables [..., D] -> packed storage for `bits`. Dispatches to the per-width packer; only int8 exists."""
    if bits == 8:
        return pack_int8_rows(Wint)
    raise NotImplementedError(f"no packer for bits={bits}: int{bits} is not implemented (add a packer beside pack_int8_rows)")


@torch.no_grad()
def pack_int8_rows(Wint: torch.Tensor) -> torch.Tensor:
    """int8 packing: one signed byte per table entry, [..., D] -> int8 [..., D]. Refuses values outside [-128, 127]."""
    q = Wint.to(torch.int16)
    if bool((q < -128).any()) or bool((q > 127).any()):
        raise ValueError("integer tables out of the int8 range")
    return q.to(torch.int8)


# ------------------------------------------------------------------ eval: integer shift-add read ------------------------
def shift_groups(q: torch.Tensor, k: torch.Tensor, skip: torch.Tensor, drop: torch.Tensor) -> torch.Tensor:
    """Per cell read [..., 2]: the left shift k' + 6 (first cell) and k' + 6 - q (second cell), or the discard group
    N_SHIFTS for a cell that is not read. The range [0, 10] of every read shift is guaranteed by resolve_quant_config
    (window inside [-3, 4], Q <= 3)."""
    sh = torch.stack([k + FIXED_POINT_SHIFT, k + FIXED_POINT_SHIFT - q], dim=-1)
    keep = torch.stack([~skip, ~(skip | drop)], dim=-1)
    return torch.where(keep, sh, torch.full_like(sh, float(N_SHIFTS))).to(torch.int64)


@torch.no_grad()
def int_blend_read(packed: torch.Tensor, bits: int, D: int, flat_idx: torch.Tensor, q: torch.Tensor, k: torch.Tensor,
                   skip: torch.Tensor, drop: torch.Tensor, chunk_bags: Optional[int] = 4096) -> torch.Tensor:
    """The note's Section 6 integer read, dispatched on the table width. Only int8 exists (int8_blend_read).

    packed   packed rows [n_tables * K, D'] as written by pack_tables(.., bits)
    flat_idx [N, H, T, 2] row indices into `packed` (table offsets already added)
    q, k     [N, H, T] float tensors of integers (k clamped into the window), skip / drop [N, H, T] bool
    Returns the head sums in units of 2^-6, int32 [N, H, D]."""
    if bits == 8:
        return int8_blend_read(packed, D, flat_idx, shift_groups(q, k, skip, drop), chunk_bags)
    raise NotImplementedError(f"no integer reader for bits={bits}: int{bits} is not implemented "
                              "(add a reader beside int8_blend_read)")


def int8_accumulate(packed: torch.Tensor, flat_idx: torch.Tensor, group: torch.Tensor) -> torch.Tensor:
    """int8 shift-add accumulation, one expression (the form torch.compile fuses into a single kernel that reads the int8
    bytes, shifts and sums without materialising the gathered rows). packed int8 [n_tables * K, D]; flat_idx / group
    [N, H, T, 2]. Returns int32 [N, H, D] = sum over the 2T cells of (row << shift), cells in the discard group excluded.
    |sum| <= 2T * 128 * 2^10, inside int32."""
    N, H, T, _ = flat_idx.shape
    rows = packed[flat_idx.reshape(N * H, 2 * T)].to(torch.int32)                      # bytes gathered, widened
    g = group.reshape(N * H, 2 * T, 1)
    keep = g < N_SHIFTS
    shifted = torch.bitwise_left_shift(rows, torch.where(keep, g, torch.zeros_like(g)).to(torch.int32))
    return torch.where(keep, shifted, torch.zeros_like(shifted)).sum(dim=1, dtype=torch.int32).view(N, H, -1)


@torch.no_grad()
def int8_blend_read(packed: torch.Tensor, D: int, flat_idx: torch.Tensor, group: torch.Tensor,
                    chunk_bags: Optional[int] = 4096) -> torch.Tensor:
    """int8 shift-add read over int8 rows [n_tables * K, D] (no unpack step). Each bag (token, head) sums its tables' two
    rows shifted left by k' + 6 and k' + 6 - q (int8_accumulate). Eager callers chunk the bags so the gathered int32 rows
    stay bounded; compiled callers pass chunk_bags=None and get one fused kernel."""
    if packed.dtype != torch.int8 or packed.shape[-1] != D:
        raise ValueError(f"int8_blend_read needs int8 rows of width {D}, got {packed.dtype} {tuple(packed.shape)}")
    N = flat_idx.shape[0]
    if chunk_bags is None or N * flat_idx.shape[1] <= chunk_bags:
        return int8_accumulate(packed, flat_idx, group)
    step = max(1, chunk_bags // flat_idx.shape[1])
    return torch.cat([int8_accumulate(packed, flat_idx[i:i + step], group[i:i + step]) for i in range(0, N, step)])
