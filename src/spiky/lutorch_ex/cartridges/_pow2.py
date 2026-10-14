"""Power-of-two int8 LUT read: the torch definition of the power-of-two read. QuantisedConfidenceLUT (the reference
class), its forward_int, DeployedQuantisedConfidenceLUT's integer read, and export all use it;
FusedQuantisedConfidenceLUT's p2_scalars op (``_pow2_int8``) computes the same integers and recomputes its backward with
:func:`ste_cell_weights` (the agreement is tested in tests/test_fused_vs_reference.py).

It is pure PyTorch (no CUDA / no extension dependency) and holds the whole p2_int8 quantisation
math — the power-of-two scale, the log-domain score exponent, the two-cell blend exponents, the
straight-through weights and tables, and the int8 shift-add read. Keeping it a separate module
keeps Gen-1/Gen-2/ConfidenceLUT untouched.

The formulas: log2 s is in the log domain, log2 S + (g + gamma*sum_i logsigmoid(beta u_i))/ln2 (g=0 in the
frozen-g rows); q = clamp(floor(2 u*/(tau ln2) + 1/2), 0, J); c_q = log2(1+2^-q) for q<C else 0;
round(xi) = floor(xi + 1/2) (half up). Integer read keeps the head sum in units of 2^-6 in int32,
each table adding (W_hat[c1] << (k'+6)) + (W_hat[c2] << (k'+6-q)); int8 only.
"""
from __future__ import annotations

import math
from typing import Dict, Optional

import torch
import torch.nn.functional as F

LN2 = math.log(2.0)
J = 64                      # bound on q
C = 8                       # c_q = 0 for q >= C
FIXED_POINT_SHIFT = 6       # the integer head sum is in units of 2^-6 (note Section 6)
N_SHIFTS = 11               # shifts k' + 6 - q, k' + 6 lie in [0, 10] for every cell that is read

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
        raise ValueError(f"Q must be in [0, 3] (the second cell's shift k' + 6 - q must stay >= 0), got {cfg['Q']}")
    if not 1 <= cfg["L"] <= 8:
        raise ValueError(f"L must be in [1, 8], got {cfg['L']}")
    if not -3 <= cfg["kmax"] - cfg["L"] + 1 or not cfg["kmax"] <= 4:
        raise ValueError(f"the window [kmax - L + 1, kmax] must lie inside [-3, 4], "
                         f"got kmax={cfg['kmax']}, L={cfg['L']}")
    cfg["mode"] = quant_mode
    cfg["lo"] = cfg["kmax"] - cfg["L"] + 1
    cfg["hi"] = cfg["kmax"]
    cfg["J"], cfg["C"] = J, C
    return cfg


def round_half_up(x: torch.Tensor) -> torch.Tensor:
    """round(xi) = floor(xi + 1/2), the note's rounding (ties go up, unlike torch.round)."""
    return torch.floor(x + 0.5)


def log2_score(m: torch.Tensor, g: torch.Tensor, beta: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
    """log2 s of the learned_margin score s = (sum_i u_i) exp(g + gamma sum_i logsigmoid(beta u_i)), log domain.
    m = |d| [..., NAP] -> [...]. -inf where all margins are 0 (such a table is skipped)."""
    return torch.log2(m.sum(dim=-1)) + (g + gamma * F.logsigmoid(beta * m).sum(dim=-1)) / LN2


def c_q(q: torch.Tensor) -> torch.Tensor:
    """c_q = log2(1 + 2^-q) for q < C, else 0."""
    return torch.where(q < C, torch.log2(1.0 + torch.pow(2.0, -q)), torch.zeros_like(q))


def _left_fold(t: torch.Tensor) -> torch.Tensor:
    """sum over the last dim as ((t0 + t1) + t2) + ..., the kernel's order (not torch's .sum reduction tree)."""
    s = t[..., 0]
    for i in range(1, t.shape[-1]):
        s = s + t[..., i]
    return s


@torch.no_grad()
def blend_exponents(m: torch.Tensor, mv: torch.Tensor, tau: torch.Tensor, g: torch.Tensor, beta: torch.Tensor,
                    gamma: torch.Tensor, cfg: dict):
    """The note's per-table integers. m [..., NAP], mv [..., 1]. Returns float tensors of integers q, k [...] (k already
    clamped into the window [lo, hi]) and bool tensors skip (k' below the window), drop (second cell not read).

    Written in the CUDA kernel's operation order (csrc/pow2_scalars.cuh), one elementwise op at a time: the nap terms
    of S and of the logsigmoid sum left-folded, and every division a true fp32 tensor division (a Python-scalar divisor
    would make torch multiply by its reciprocal instead, and 2.0 / t is reciprocal(t) * 2). Each integer is a floor of
    these fp32 values, so the same operations in the same order give the kernel's integers, at every nap.
    Trade-off, accepted on purpose: (1) the reference now partly encodes the kernel's operation order, so agreement
    between the two is a weaker form of independence than with torch's own reduction; (2) a left fold is less accurate
    than torch's pairwise reduction: relative error of S, mean 3.40e-8 vs 2.97e-8, max 2.7e-7 vs 1.6e-7 (measured at
    the champion geometry); (3) the bit-agreement relies on the same libdevice log2 / log1p / exp in torch's build and
    in the extension's nvcc build -- measured identical on this toolkit, not guaranteed across a CUDA or torch upgrade.
    tests/test_fused_vs_reference.py therefore compares the fused class with this reference BIT FOR BIT, with no
    tolerance and no gate: an upgrade that changes libdevice shows up as a test failure rather than being silently
    absorbed by a tolerance."""
    ln2 = torch.full((), LN2, dtype=m.dtype, device=m.device)
    two = torch.full((), 2.0, dtype=m.dtype, device=m.device)
    q = torch.clamp(torch.floor(mv.squeeze(-1) * torch.div(two, tau * ln2) + 0.5), 0, J)      # kernel: 2/(tau ln2)
    S = _left_fold(m)
    ls = _left_fold(F.logsigmoid(beta * m))                         # min(0, x) - log1p(exp(-|x|)), as in the kernel
    kr = torch.floor(torch.log2(S) + torch.div(g + gamma * ls, ln2) - c_q(q) + 0.5)
    skip = kr < cfg["lo"]
    k = torch.clamp(kr, cfg["lo"], cfg["hi"])
    drop = q > cfg["Q"]
    return q, k, skip, drop


@torch.no_grad()
def single_exponent(m: torch.Tensor, g: torch.Tensor, beta: torch.Tensor, gamma: torch.Tensor, cfg: dict):
    """The read_top_n = 1 integers: k' = round(log2 s) = floor(log2 S + (g + gamma ls)/ln2 + 1/2) (no blend, so no c_q),
    clamped into [lo, hi]; skip = k' below the window. m [..., NAP] -> float k, bool skip [...]. Written in the kernel's
    operation order (csrc/pow2_scalars.cuh, p2::single_scalars), as blend_exponents is for n = 2: the nap terms of S and
    of the logsigmoid sum left-folded, the division by ln2 a true fp32 tensor division."""
    ln2 = torch.full((), LN2, dtype=m.dtype, device=m.device)
    S = _left_fold(m)
    ls = _left_fold(F.logsigmoid(beta * m))
    kr = torch.floor(torch.log2(S) + torch.div(g + gamma * ls, ln2) + 0.5)
    return torch.clamp(kr, cfg["lo"], cfg["hi"]), kr < cfg["lo"]


def ste_blend_weights(score: torch.Tensor, mv: torch.Tensor, tau: torch.Tensor, q: torch.Tensor, k: torch.Tensor,
                      skip: torch.Tensor, drop: torch.Tensor) -> torch.Tensor:
    """Per-cell weights [..., 2] with value exactly (2^k', 2^(k'-q)) (0 when skipped / dropped) and the gradient of the
    exact blend weights (s v, s (1 - v)) times the same ratios (note eq. 6). The ratio uses the clamped k' also for
    skipped and dropped cells, so they keep a gradient and can come back (the prototype's semantics)."""
    x = 2.0 * mv / tau                                             # [..., 1]
    e = score.unsqueeze(-1) * torch.cat([torch.sigmoid(x), torch.sigmoid(-x)], dim=-1)   # [..., 2]: s v, s (1 - v)
    b = torch.pow(2.0, torch.stack([k, k - q], dim=-1)).to(e.dtype)                      # clamped values
    keep = torch.stack([~skip, ~(skip | drop)], dim=-1)
    val = torch.where(keep, b, torch.zeros_like(b))
    s = e * (b / e.detach().clamp_min(1e-30))
    return val + (s - s.detach())


def score_from_margins(m: torch.Tensor, g: torch.Tensor, beta: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
    """learned_margin score s = (sum_i m_i) exp(g + gamma sum_i logsigmoid(beta m_i)), m = |d| [..., NAP] -> [...].
    ConfidenceLUT._score plus an additive log-gain g (always 0 in lutorch_ex), with beta = exp(log_beta),
    gamma = exp(log_gamma) given."""
    return m.sum(dim=-1) * torch.exp(g + gamma * F.logsigmoid(beta * m).sum(dim=-1))


def ste_cell_weights(d: torch.Tensor, tau: torch.Tensor, g: torch.Tensor, beta: torch.Tensor, gamma: torch.Tensor,
                     q: torch.Tensor, k: torch.Tensor, skip: torch.Tensor, drop: torch.Tensor) -> torch.Tensor:
    """The straight-through per-cell weights [..., 2] for given integers: value (2^k', 2^(k'-q)) or 0, gradient through
    s v, s (1 - v) times the ratios. THE torch expression: the reference read uses it, and the fused op's backward
    recomputes it with the op's integers held fixed."""
    m = d.abs()
    mv = m.min(dim=-1, keepdim=True).values
    return ste_blend_weights(score_from_margins(m, g, beta, gamma), mv, tau, q, k, skip, drop)


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
    """W_hat = clamp(round(W / 2^e), -2^(b-1), 2^(b-1) - 1)  (note eq. 8), integers in W's float dtype,
    [n_tables, K, D]."""
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
    raise NotImplementedError(f"no packer for bits={bits}: int{bits} is not implemented "
                              "(add a packer beside pack_int8_rows)")


@torch.no_grad()
def pack_int8_rows(Wint: torch.Tensor) -> torch.Tensor:
    """int8 packing: one signed byte per table entry, [..., D] -> int8 [..., D]. Refuses values outside [-128, 127]."""
    q = Wint.to(torch.int16)
    if bool((q < -128).any()) or bool((q > 127).any()):
        raise ValueError("integer tables out of the int8 range")
    return q.to(torch.int8)


def shift_groups(q: torch.Tensor, k: torch.Tensor, skip: torch.Tensor, drop: torch.Tensor) -> torch.Tensor:
    """Per cell read [..., 2]: the left shift k' + 6 (first cell) and k' + 6 - q (second cell), or the discard group
    N_SHIFTS for a cell that is not read."""
    sh = torch.stack([k + FIXED_POINT_SHIFT, k + FIXED_POINT_SHIFT - q], dim=-1)
    keep = torch.stack([~skip, ~(skip | drop)], dim=-1)
    return torch.where(keep, sh, torch.full_like(sh, float(N_SHIFTS))).to(torch.int64)


@torch.no_grad()
def int_blend_read(packed: torch.Tensor, bits: int, D: int, flat_idx: torch.Tensor, q: torch.Tensor, k: torch.Tensor,
                   skip: torch.Tensor, drop: torch.Tensor, chunk_bags: Optional[int] = 4096) -> torch.Tensor:
    """The note's Section 6 integer read, dispatched on the table width. Only int8 exists (int8_blend_read).
    flat_idx [N, H, T, 2] row indices into `packed` (table offsets already added). Returns int32 [N, H, D]
    head sums in units of 2^-6."""
    if bits == 8:
        return int8_blend_read(packed, D, flat_idx, shift_groups(q, k, skip, drop), chunk_bags)
    raise NotImplementedError(f"no integer reader for bits={bits}: int{bits} is not implemented "
                              "(add a reader beside int8_blend_read)")


@torch.no_grad()
def int_single_read(packed: torch.Tensor, bits: int, D: int, flat_idx: torch.Tensor, k: torch.Tensor,
                    skip: torch.Tensor, chunk_bags: Optional[int] = 4096) -> torch.Tensor:
    """The read_top_n = 1 integer read: each table's one row shifted left by k' + 6 (nothing when skipped). flat_idx
    [N, H, T, 1] row indices into `packed`. Returns int32 [N, H, D] head sums in units of 2^-6."""
    if bits == 8:
        sh = torch.where(skip, torch.full_like(k, float(N_SHIFTS)), k + FIXED_POINT_SHIFT).to(torch.int64)
        return int8_blend_read(packed, D, flat_idx, sh.unsqueeze(-1), chunk_bags)
    raise NotImplementedError(f"no integer reader for bits={bits}: int{bits} is not implemented "
                              "(add a reader beside int8_blend_read)")


def int8_accumulate(packed: torch.Tensor, flat_idx: torch.Tensor, group: torch.Tensor) -> torch.Tensor:
    """int8 shift-add accumulation, one expression (the form torch.compile fuses into a single kernel that reads the
    int8 bytes, shifts and sums without materialising the gathered rows). packed int8 [n_tables * K, D]; flat_idx /
    group [N, H, T, n] (n cells read per table: 2, or 1 for read_top_n = 1). Returns int32 [N, H, D]."""
    N, H, T, n = flat_idx.shape
    rows = packed[flat_idx.reshape(N * H, n * T)].to(torch.int32)                      # bytes gathered, widened
    g = group.reshape(N * H, n * T, 1)
    keep = g < N_SHIFTS
    shifted = torch.bitwise_left_shift(rows, torch.where(keep, g, torch.zeros_like(g)).to(torch.int32))
    return torch.where(keep, shifted, torch.zeros_like(shifted)).sum(dim=1, dtype=torch.int32).view(N, H, -1)


@torch.no_grad()
def int8_blend_read(packed: torch.Tensor, D: int, flat_idx: torch.Tensor, group: torch.Tensor,
                    chunk_bags: Optional[int] = 4096) -> torch.Tensor:
    """int8 shift-add read over int8 rows [n_tables * K, D] (no unpack step). Each bag (token, head) sums its tables'
    rows shifted left by their group's shift (two rows, k' + 6 and k' + 6 - q, for n = 2; one, k' + 6, for n = 1). Eager
    callers chunk the bags; compiled callers pass chunk_bags=None."""
    if packed.dtype != torch.int8 or packed.shape[-1] != D:
        raise ValueError(f"int8_blend_read needs int8 rows of width {D}, got {packed.dtype} {tuple(packed.shape)}")
    N = flat_idx.shape[0]
    if chunk_bags is None or N * flat_idx.shape[1] <= chunk_bags:
        return int8_accumulate(packed, flat_idx, group)
    step = max(1, chunk_bags // flat_idx.shape[1])
    return torch.cat([int8_accumulate(packed, flat_idx[i:i + step], group[i:i + step]) for i in range(0, N, step)])
