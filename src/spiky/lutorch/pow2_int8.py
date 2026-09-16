"""The int8 power-of-two read on the CUDA extension, usable from torch: extension build / load and device gating,
row striding, the fused inference read, and the spiky_lutorch::p2_scalars custom op. Hardware-gated, never raises.

EXTENSION

csrc/pow2_int8_read.cu provides

  * read_fused  the inference read (QuantisedLightFFN): for every (token, head), in ONE launch, the per-table integers from
                the compressed code (calling p2::table_scalars, csrc/pow2_scalars.cuh) and the note's Section 6 integer
                accumulation -- each cell's int8 row loaded 16 bytes at a time (int4 vector loads), every byte
                sign-extended to int32 in registers, shifted by k' + 6 or k' + 6 - q and added into int32 accumulators that
                stay in registers across the table loop; skipped tables and dropped second cells excluded; the
                accumulators converted to float once. `cells_out` exposes the kernel's integers for tests.
  * scalars     the forward of the spiky_lutorch::p2_scalars custom op (below): p2::table_scalars for every
                table, used by the training forward -- so training and inference take their integers from one function.
  * read_cells  the same accumulation on integers supplied by the caller (packed by pack_cells). A REFERENCE for tests:
                bit-identical to pow2_read.int8_blend_read on the same integers, it pins read_fused's accumulation
                independently of how the integers were computed.

Cell width D is a runtime parameter. The kernel reads rows with a STRIDE = ceil(D / 16) * 16 bytes (the vector load width):
`stride_tables` pads the stored int8 rows with zero bytes up to it -- nothing at D = 48, at most 15 bytes per row otherwise.
The padding lanes are ALSO predicated off inside the kernel (masked at accumulate time), so the result does not depend on
the padding bytes; the tests fill them with garbage to prove it.

Built lazily with torch.utils.cpp_extension on first use, for the visible device's architecture, and ONLY on an architecture
in VALIDATED_ARCHES: an explicit allowlist of the compute capabilities on which the kernel's test matrix (tests/
test_pow2_int8.py) has passed. Nothing in the kernel is architecture-specific; the allowlist records validation, not a
hardware requirement, and an architecture joins it once that matrix is green on it. Everything else -- no CUDA, an
architecture not on the list, no nvcc, a failed build, SPIKY_P2_CUDA_DISABLE=1 -- makes `load()` return None and callers use
the torch implementation (pow2_read).

CUSTOM OP -- the per-table integers of the power-of-two read, from ONE forward implementation for training and eval.

With the CUDA extension available (a validated architecture) and CUDA fp32 margins, the integers come from the
registered custom op `spiky_lutorch::p2_scalars`, whose forward is p2::table_scalars (csrc/pow2_scalars.cuh) -- the same
function the inference kernel calls inline:

  * LightMultiHeadLUT quant_mode training forward      -> cell_weights (the op, differentiable)
  * LightMultiHeadLUT.forward_int, QuantisedLightFFN's torch read -> table_integers (the op, no grad)
  * QuantisedLightFFN on CUDA                          -> read_fused, calling the same function in-kernel

so the training forward, forward_int and the exported artefact take identical integers by construction.

Backward: RECOMPUTE with the torch expression. The op's backward re-evaluates the straight-through form of pow2_read (score,
smallest margin, ste_blend_weights) with the op's integers held fixed and returns autograd's gradient of it: the same code as
the torch path, so the same gradient semantics, including the ratio gradient of skipped tables and dropped second cells.
Without the extension, training and eval alike use the torch implementation in pow2_read.
"""
import os
from typing import Tuple

import torch
import torch.nn.functional as F

from . import pow2_read

BLOCK_NS = (32, 64, 128)
DEFAULT_BLOCK_N = 64
LOAD_WIDTH = 16
DISCARD = 15                # shift code of a cell that is not read (csrc/pow2_scalars.cuh)
_CSRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "csrc")
# Compute capabilities the kernel is enabled on -- only those whose test matrix has passed on real hardware:
#   (12, 0) sm_120  RTX 5090 (Blackwell), validated on gpustar
#   (9, 0)  sm_90   H100 (Hopper), enabled for validation on nebius-h100: tests/test_pow2_int8.py must be green there
#                   before its results are trusted (see the lut_ablation exp_n_abl_47 package's validation gate)
VALIDATED_ARCHES = ((12, 0), (9, 0))

_ext = None
_error = None
_tried = False


def load():
    """Build / load the extension once. Returns the module or None (never raises)."""
    global _ext, _error, _tried
    if _tried:
        return _ext
    _tried = True
    if os.environ.get("SPIKY_P2_CUDA_DISABLE") == "1":
        _error = "disabled by SPIKY_P2_CUDA_DISABLE=1"
        return None
    try:
        cap = torch.cuda.get_device_capability() if torch.cuda.is_available() else None
        if cap not in VALIDATED_ARCHES:
            _error = f"device compute capability {cap} is not in VALIDATED_ARCHES {VALIDATED_ARCHES}"
            return None
        from torch.utils.cpp_extension import load as _load
        _ext = _load(name="spiky_lutorch_pow2_int8_read", sources=[os.path.join(_CSRC, "pow2_int8_read.cu")],
                     extra_cuda_cflags=["-O3", "-std=c++20", "--fmad=false"], extra_cflags=["-O3", "-std=c++20"],
                     verbose=False)                          # cpp_extension targets the visible device's architecture
    except Exception as e:                                   # no nvcc, compile error, ...: fall back silently
        _ext = None
        msg = str(e).strip()
        _error = f"{type(e).__name__}: {msg.splitlines()[-1][:200]}" if msg else type(e).__name__
    return _ext


def available():
    """(bool, message); never raises."""
    ext = load()
    return ext is not None, ("pow2 int8 CUDA kernel ready" if ext is not None else f"pow2 int8 CUDA kernel unavailable ({_error})")


def _reset_for_tests():
    global _ext, _error, _tried
    _ext, _error, _tried = None, None, False


def row_stride(D: int) -> int:
    """Stored row stride for the kernel: D padded up to a multiple of the 16-byte load width."""
    return (D + LOAD_WIDTH - 1) // LOAD_WIDTH * LOAD_WIDTH


def stride_tables(packed: torch.Tensor, D: int) -> torch.Tensor:
    """int8 rows [R, D] -> [R, row_stride(D)], zero-padded and contiguous (the same tensor when D is 16-byte aligned)."""
    if packed.dtype != torch.int8 or packed.dim() != 2 or packed.shape[1] != D:
        raise ValueError(f"expected int8 rows [R, {D}], got {packed.dtype} {tuple(packed.shape)}")
    pad = row_stride(D) - D
    return packed.contiguous() if pad == 0 else F.pad(packed, (0, pad)).contiguous()


def pack_cells(idx: torch.Tensor, q: torch.Tensor, k: torch.Tensor, skip: torch.Tensor, drop: torch.Tensor) -> torch.Tensor:
    """Cells for read_cells: uint8 [N, H, T, 3] = (c1, c2, sh1 | sh2 << 4), sh = discard code 15 for a cell that is not
    read. idx [N, H, T, 2] are cell numbers inside their table (0..2^NAP - 1, no table offset). Compile-friendly."""
    group = pow2_read.shift_groups(q, k, skip, drop)                       # [.., 2], discard = N_SHIFTS (11)
    sh = torch.where(group == pow2_read.N_SHIFTS, torch.full_like(group, DISCARD), group)
    return torch.stack([idx[..., 0], idx[..., 1], sh[..., 0] | (sh[..., 1] << 4)], dim=-1).to(torch.uint8)


def _check(tables_stride, D):
    ext = load()
    if ext is None:
        raise RuntimeError(f"pow2 int8 CUDA kernel unavailable: {_error}")
    if tables_stride.shape[1] != row_stride(D):
        raise ValueError(f"tables must have stride {row_stride(D)} for D={D} (use stride_tables)")
    return ext


def read_cells(tables_stride: torch.Tensor, cells: torch.Tensor, n_anchor_pairs: int, D: int, lo: int, hi: int, Q: int,
               block_n: int = DEFAULT_BLOCK_N, load16: bool = True) -> torch.Tensor:
    """REFERENCE (tests): the int32 accumulators (as float32, units of 2^-6) [N, H, D] from caller-supplied cells."""
    ext = _check(tables_stride, D)
    N, H, T, _ = cells.shape
    e = torch.empty(0, device=tables_stride.device)
    return ext.read(e, e, e, cells.contiguous(), tables_stride, N, H, T, n_anchor_pairs, 1 << n_anchor_pairs, 0, D,
                    lo, hi, Q, e, e, e, e, block_n, False, load16)


def read_fused(z: torch.Tensor, anchor_a32: torch.Tensor, anchor_b32: torch.Tensor, tables_stride: torch.Tensor,
               scalars, n_anchor_pairs: int, D: int, lo: int, hi: int, Q: int, block_n: int = DEFAULT_BLOCK_N,
               load16: bool = True, cells_out: torch.Tensor = None) -> torch.Tensor:
    """The int32 accumulators (as float32, units of 2^-6) [N, H, D], per-table integers computed in the kernel.
    z [N, H, din] fp32; anchors int32 [H, T, NAP] (column indices inside the head); scalars = (tau, g, beta, gamma) as
    one-element fp32 CUDA tensors -- the same values the op receives."""
    ext = _check(tables_stride, D)
    N, H, din = z.shape
    T = anchor_a32.shape[1]
    e = torch.empty(0, device=tables_stride.device, dtype=torch.uint8) if cells_out is None else cells_out
    return ext.read(z.contiguous(), anchor_a32, anchor_b32, e, tables_stride, N, H, T, n_anchor_pairs,
                    1 << n_anchor_pairs, din, D, lo, hi, Q, *scalars, block_n, True, load16)


# ======================================================================================================================
# the spiky_lutorch::p2_scalars custom op
def score_from_margins(m: torch.Tensor, g: torch.Tensor, beta: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
    """learned_margin score s = (sum_i m_i) exp(g + gamma sum_i logsigmoid(beta m_i)), m = |d| [..., NAP] -> [...].
    The expression of fast_multi_head_lut._confidence_score with beta = exp(log_beta), gamma = exp(log_gamma) given."""
    return m.sum(dim=-1) * torch.exp(g + gamma * F.logsigmoid(beta * m).sum(dim=-1))


def ste_cell_weights(d: torch.Tensor, tau: torch.Tensor, g: torch.Tensor, beta: torch.Tensor, gamma: torch.Tensor,
                     q: torch.Tensor, k: torch.Tensor, skip: torch.Tensor, drop: torch.Tensor) -> torch.Tensor:
    """The straight-through per-cell weights [..., 2] for given integers: value (2^k', 2^(k'-q)) or 0, gradient through
    s v, s (1 - v) times the ratios. THE torch expression, used by the torch forward and by the op's recompute backward."""
    m = d.abs()
    mv = m.min(dim=-1, keepdim=True).values
    return pow2_read.ste_blend_weights(score_from_margins(m, g, beta, gamma), mv, tau, q, k, skip, drop)


# ------------------------------------------------------------------ the custom op --------------------------------------
_registered = False
_enabled = False        # True once the op is registered; set_enabled(False) forces the torch definition (tests)


def ensure_registered() -> bool:
    """Build / load the extension and register spiky_lutorch::p2_scalars, once and eagerly (call before any torch.compile'd
    forward). False when the extension is unavailable (load never raises)."""
    global _registered
    if _registered:
        return True
    ext = load()
    if ext is None:
        return False

    @torch.library.custom_op("spiky_lutorch::p2_scalars", mutates_args=(), device_types="cuda")
    def p2_scalars(d: torch.Tensor, tau: torch.Tensor, g: torch.Tensor, beta: torch.Tensor, gamma: torch.Tensor,
                   lo: int, hi: int, Q: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return tuple(ext.scalars(d, tau.reshape(1), g.reshape(1), beta.reshape(1), gamma.reshape(1), lo, hi, Q))

    @p2_scalars.register_fake
    def _(d, tau, g, beta, gamma, lo, hi, Q):
        lead = d.shape[:-1]
        return (d.new_empty(*lead, 2), d.new_empty(*lead, 3, dtype=torch.uint8), d.new_empty(*lead, 2, dtype=torch.int8))

    def setup_context(ctx, inputs, output):
        d, tau, g, beta, gamma, lo, hi, Q = inputs
        _psw, cells, kq = output
        ctx.save_for_backward(d, tau, g, beta, gamma, cells, kq)
        ctx.Q = Q

    def backward(ctx, grad_psw, _grad_cells, _grad_kq):
        d, tau, g, beta, gamma, cells, kq = ctx.saved_tensors
        q, k, skip, drop = _integers_from(cells, kq, ctx.Q)
        leaves = [x.detach().requires_grad_(nd) for x, nd in zip((d, tau, g, beta, gamma), ctx.needs_input_grad[:5])]
        with torch.enable_grad():
            psw = ste_cell_weights(*leaves, q, k, skip, drop)
            grads = iter(torch.autograd.grad(psw, [x for x in leaves if x.requires_grad], grad_psw, allow_unused=True))
        out = [next(grads) if x.requires_grad else None for x in leaves]
        return (*out, None, None, None)

    p2_scalars.register_autograd(backward, setup_context=setup_context)
    _registered = True
    set_enabled(True)
    return True


def _integers_from(cells: torch.Tensor, kq: torch.Tensor, Q: int):
    """(q, k, skip, drop) as the torch path's float / bool tensors, from the op's cells and kq outputs."""
    k = kq[..., 0].to(torch.float32)
    q = kq[..., 1].to(torch.float32)
    skip = (cells[..., 2] & 15) == DISCARD
    drop = q > Q
    return q, k, skip, drop


def set_enabled(flag: bool) -> None:
    """Use the op (True, the default once registered) or force the torch definition (False) for every consumer."""
    global _enabled
    _enabled = bool(flag) and _registered


def op_available(d: torch.Tensor) -> bool:
    """True when this call can use the custom op (CUDA fp32 margins, extension built, op registered). Registration itself
    is done eagerly by ensure_registered(); inside a compiled region this only reads module flags (no graph break)."""
    return _enabled and d.is_cuda and d.dtype == torch.float32


# ------------------------------------------------------------------ the consumers' entry points -------------------------
def cell_weights(d: torch.Tensor, index: torch.Tensor, powers: torch.Tensor, tau, g, beta, gamma, cfg: dict):
    """Training forward: (psw [..., 2] straight-through cell weights, idx [..., 2] cell numbers c1, c2)."""
    if op_available(d):
        psw, cells, _kq = torch.ops.spiky_lutorch.p2_scalars(d, tau, g, beta, gamma, cfg["lo"], cfg["hi"], cfg["Q"])
        return psw, cells[..., :2].to(torch.int64)
    m, mv, idx = pow2_read.blend_candidates(d, index, powers)
    q, k, skip, drop = pow2_read.blend_exponents(m, mv, tau, g, beta, gamma, cfg)
    return ste_cell_weights(d, tau, g, beta, gamma, q, k, skip, drop), idx


@torch.no_grad()
def table_integers(d: torch.Tensor, index: torch.Tensor, powers: torch.Tensor, tau, g, beta, gamma, cfg: dict):
    """Eval: (idx [..., 2] cell numbers, q, k, skip, drop), from the op when available, else from pow2_read."""
    if op_available(d):
        _psw, cells, kq = torch.ops.spiky_lutorch.p2_scalars(d, tau, g, beta, gamma, cfg["lo"], cfg["hi"], cfg["Q"])
        return (cells[..., :2].to(torch.int64), *_integers_from(cells, kq, cfg["Q"]))
    m, mv, idx = pow2_read.blend_candidates(d, index, powers)
    return (idx, *pow2_read.blend_exponents(m, mv, tau, g, beta, gamma, cfg))
