"""The per-table integers of the power-of-two read, from ONE forward implementation for training and eval.

With the CUDA extension available (pow2_int8_cuda, compute capability 12.x) and CUDA fp32 margins, the integers come from the
registered custom op `spiky_lutorch::p2_scalars`, whose forward is p2::table_scalars (csrc/pow2_scalars.cuh) -- the same
function the inference kernel calls inline:

  * LightMultiHeadLUT quant_mode training forward      -> cell_weights (the op, differentiable)
  * LightMultiHeadLUT.forward_int, QuantisedLightFFN's torch read -> table_integers (the op, no grad)
  * QuantisedLightFFN on CUDA                          -> pow2_int8_cuda.read_fused, calling the same function in-kernel

so the training forward, forward_int and the exported artefact take identical integers by construction.

Backward: RECOMPUTE with the torch expression. The op's backward re-evaluates the straight-through form of pow2_read (score,
smallest margin, ste_blend_weights) with the op's integers held fixed and returns autograd's gradient of it: the same code as
the torch path, so the same gradient semantics, including the ratio gradient of skipped tables and dropped second cells.

Without the extension (CPU, other GPUs, no nvcc, SPIKY_P2_CUDA_DISABLE=1) everything uses the torch implementation in
pow2_read, for training and eval alike.
"""
from typing import Tuple

import torch
import torch.nn.functional as F

from . import pow2_int8_cuda
from . import pow2_read

DISCARD = 15                                 # shift code of a cell that is not read (csrc/pow2_scalars.cuh)


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
    forward). False when the extension is unavailable (pow2_int8_cuda.load never raises)."""
    global _registered
    if _registered:
        return True
    ext = pow2_int8_cuda.load()
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
