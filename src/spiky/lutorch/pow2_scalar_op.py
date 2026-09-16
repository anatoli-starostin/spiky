"""The per-table integers of the power-of-two read, from ONE forward implementation for training and eval.

With the CUDA extension available (pow2_int8_cuda, compute capability 12.x) and CUDA fp32 margins, every consumer takes its
integers from the registered custom op `spiky_lutorch::p2_scalars`, whose forward is p2::table_scalars
(csrc/pow2_scalars.cuh) -- the same function the fused eval kernel calls inline:

  * LightMultiHeadLUT training forward (quant_mode)            -> table_integers / cell_weights
  * LightMultiHeadLUT.forward_int, QuantisedLightFFN "off"/"cells" -> table_integers
  * QuantisedLightFFN "fused"                                  -> the kernel, calling the same function

so train == int and fused == cells hold by construction.

Backward (Stage A): RECOMPUTE with the existing torch expression. The op's backward re-evaluates the straight-through form of
pow2_read (score, smallest margin, ste_blend_weights) with the op's integers held fixed and returns autograd's gradient of it:
the same code as the torch path, so the same gradient semantics, including the ratio gradient of skipped tables and dropped
second cells.

Without the extension (CPU, other GPUs, no nvcc, SPIKY_P2_CUDA_DISABLE=1) everything falls back to the torch implementation in
pow2_read (the PR 1 path), for training and eval alike.
"""
import os
from typing import Tuple

import torch
import torch.nn.functional as F

from . import pow2_read

OP_NAME = "spiky_lutorch::p2_scalars"
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
_enabled = False        # True once the op is registered; set_enabled(False) forces the torch path (tests, A/B timing)
_BACKWARD = os.environ.get("SPIKY_P2_BACKWARD", "recompute")   # "analytic": STAGE B experimental CUDA backward


def _ext():
    from . import pow2_int8_cuda
    return pow2_int8_cuda.load()


def _register():
    """Register spiky_lutorch::p2_scalars (once), only when the extension is available."""
    global _registered
    if _registered:
        return True
    if _ext() is None:
        return False

    @torch.library.custom_op(OP_NAME, mutates_args=(), device_types="cuda")
    def p2_scalars(d: torch.Tensor, tau: torch.Tensor, g: torch.Tensor, beta: torch.Tensor, gamma: torch.Tensor,
                   lo: int, hi: int, Q: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        psw, cells, kq = _ext().scalars(d, tau.reshape(1), g.reshape(1), beta.reshape(1), gamma.reshape(1),
                                                       lo, hi, Q)
        return psw, cells, kq

    @p2_scalars.register_fake
    def _(d, tau, g, beta, gamma, lo, hi, Q):
        lead = d.shape[:-1]
        return (d.new_empty(*lead, 2), d.new_empty(*lead, 3, dtype=torch.uint8), d.new_empty(*lead, 2, dtype=torch.int8))

    def setup_context(ctx, inputs, output):
        d, tau, g, beta, gamma, lo, hi, Q = inputs
        _psw, cells, kq = output
        ctx.save_for_backward(d, tau, g, beta, gamma, cells, kq)
        ctx.Q = Q

    @torch.library.custom_op(OP_NAME + "_backward", mutates_args=(), device_types="cuda")
    def p2_scalars_backward(d: torch.Tensor, grad_psw: torch.Tensor, kq: torch.Tensor, tau: torch.Tensor, g: torch.Tensor,
                            beta: torch.Tensor, gamma: torch.Tensor
                            ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return tuple(_ext().scalars_backward(d, grad_psw, kq, tau.reshape(1), g.reshape(1), beta.reshape(1),
                                             gamma.reshape(1), 0))

    @p2_scalars_backward.register_fake
    def _(d, grad_psw, kq, tau, g, beta, gamma):
        lead = d.shape[:-1]
        return (torch.empty_like(d), d.new_empty(lead), d.new_empty(lead), d.new_empty(lead), torch.empty_like(d))

    def backward(ctx, grad_psw, _grad_cells, _grad_kq):
        d, tau, g, beta, gamma, cells, kq = ctx.saved_tensors
        if _BACKWARD == "analytic":                                   # STAGE B (experimental)
            gd, gtau, gg, ggamma, gbeta = torch.ops.spiky_lutorch.p2_scalars_backward(
                d, grad_psw.contiguous(), kq, tau, g, beta, gamma)
            need = ctx.needs_input_grad
            return (gd if need[0] else None, gtau.sum().reshape(tau.shape) if need[1] else None,
                    gg.sum().reshape(g.shape) if need[2] else None, gbeta.sum().reshape(beta.shape) if need[3] else None,
                    ggamma.sum().reshape(gamma.shape) if need[4] else None, None, None, None)
        q, k, skip, drop = _integers_from(cells, kq, ctx.Q)
        leaves = [x.detach().requires_grad_(nd) for x, nd in zip((d, tau, g, beta, gamma), ctx.needs_input_grad[:5])]
        with torch.enable_grad():
            psw = ste_cell_weights(*leaves, q, k, skip, drop)
            wanted = [x for x in leaves if x.requires_grad]
            grads = iter(torch.autograd.grad(psw, wanted, grad_psw, allow_unused=True) if wanted else ())
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


def ensure_registered() -> bool:
    """Build / load the extension and register the op, eagerly (call before any torch.compile'd forward). Never raises."""
    try:
        return _register()
    except Exception:
        return False


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


@torch.no_grad()
def table_cells(d: torch.Tensor, index: torch.Tensor, powers: torch.Tensor, tau, g, beta, gamma, cfg: dict) -> torch.Tensor:
    """Eval, cells regime: uint8 [..., 3] (c1, c2, sh1 | sh2 << 4) for the CUDA accumulation."""
    if op_available(d):
        return torch.ops.spiky_lutorch.p2_scalars(d, tau, g, beta, gamma, cfg["lo"], cfg["hi"], cfg["Q"])[1]
    from . import pow2_int8_cuda
    m, mv, idx = pow2_read.blend_candidates(d, index, powers)
    return pow2_int8_cuda.pack_cells(idx, *pow2_read.blend_exponents(m, mv, tau, g, beta, gamma, cfg))
