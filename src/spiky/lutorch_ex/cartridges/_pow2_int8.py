"""The CUDA side of the power-of-two int8 read: extension build / load and device gating, and the lutorch_ex::p2_scalars
custom op. Hardware-gated, never raises.

EXTENSION

csrc/pow2_int8_read.cu provides one function, `scalars`: the forward of the lutorch_ex::p2_scalars custom op (below),
p2::table_scalars (csrc/pow2_scalars.cuh) for every table. lutorch_ex reads int8 tables with the torch definition
(pow2_read.int_blend_read, compiled on CUDA); the extension supplies only the per-table integers.

Built lazily with torch.utils.cpp_extension on first use, for the visible device's architecture, and ONLY on an architecture
in VALIDATED_ARCHES: an explicit allowlist of the compute capabilities on which the kernel has been validated on real
hardware. Nothing in it is architecture-specific; the allowlist records validation, not a hardware requirement. Everything
else -- no CUDA, an architecture not on the list, no nvcc, a failed build, SPIKY_P2_CUDA_DISABLE=1 -- makes `load()` return
None and callers use the torch implementation (pow2_read).

CUSTOM OP -- the per-table integers of the power-of-two read.

With the extension available and CUDA fp32 margins, QuantisedConfidenceLUT's read_top_n=2 reads -- the training read
(_quant_read_monolith, pairs mode) and the eval read (_quant_grp_out) -- take their per-table integers and straight-through
cell weights from the op, through cell_weights. Every other read (read_top_n=1, single-anchor training,
QuantisedConfidenceLUT.forward_int, DeployedQuantisedConfidenceLUT) uses the torch definition in pow2_read. The two must
agree on the integers; tests/test_quantised_confidence.py checks the op-backed eval read against forward_int.

Backward: RECOMPUTE with the torch expression. The op's backward re-evaluates the straight-through form of pow2_read (score,
smallest margin, ste_blend_weights) with the op's integers held fixed and returns autograd's gradient of it: the same code as
the torch path, so the same gradient semantics, including the ratio gradient of skipped tables and dropped second cells.
Without the extension, training and eval alike use the torch implementation in pow2_read.
"""
import os
from typing import Tuple

import torch
import torch.nn.functional as F

from . import _pow2 as pow2_read  # vendored pow2_read (lutorch_ex copy)

DISCARD = 15                # shift code of a cell that is not read (csrc/pow2_scalars.cuh)
_CSRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "csrc")
# Compute capabilities the kernel is enabled on -- only those it has been validated on, on real hardware:
#   (12, 0) sm_120  RTX 5090 (Blackwell), validated on gpustar
#   (9, 0)  sm_90   H100 (Hopper), enabled for validation on nebius-h100 (see the lut_ablation exp_n_abl_47
#                   package's validation gate)
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
        _ext = _load(name="lutorch_ex_pow2_int8_read", sources=[os.path.join(_CSRC, "pow2_int8_read.cu")],
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


# ======================================================================================================================
# the lutorch_ex::p2_scalars custom op
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
    """Build / load the extension and register lutorch_ex::p2_scalars, once and eagerly (call before any torch.compile'd
    forward). False when the extension is unavailable (load never raises)."""
    global _registered
    if _registered:
        return True
    ext = load()
    if ext is None:
        return False

    @torch.library.custom_op("lutorch_ex::p2_scalars", mutates_args=(), device_types="cuda")
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
        psw, cells, _kq = torch.ops.lutorch_ex.p2_scalars(d, tau, g, beta, gamma, cfg["lo"], cfg["hi"], cfg["Q"])
        return psw, cells[..., :2].to(torch.int64)
    m, mv, idx = pow2_read.blend_candidates(d, index, powers)
    q, k, skip, drop = pow2_read.blend_exponents(m, mv, tau, g, beta, gamma, cfg)
    return ste_cell_weights(d, tau, g, beta, gamma, q, k, skip, drop), idx
