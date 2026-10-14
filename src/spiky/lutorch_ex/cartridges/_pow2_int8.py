"""The CUDA side of FusedQuantisedConfidenceLUT: the p2 extension's build / load and device gating, the
lutorch_ex::p2_scalars custom op and the fused read. Used ONLY by the fused class; the reference QuantisedConfidenceLUT
never calls it (importing this module builds nothing: ensure_registered() loads the extension).

EXTENSION

csrc/pow2_int8_read.cu provides `scalars` (the forward of the lutorch_ex::p2_scalars custom op below: p2::table_scalars,
csrc/pow2_scalars.cuh, for every table) and FusedQuantisedConfidenceLUT's read: `table_exponents`, `quant_fwd` (the
fused forward) and `quant_bwd` (the fused backward), wrapped by FusedQuantRead.

Built lazily with torch.utils.cpp_extension on first use, for the visible device's architecture, and ONLY on an
architecture in VALIDATED_ARCHES: an explicit allowlist of the compute capabilities on which the kernel has been
validated on real hardware. Nothing in it is architecture-specific; the allowlist records validation, not a hardware
requirement. Anything else -- no CUDA, an architecture not on the list, no nvcc, a failed build -- makes `load()` return
None and records the classified cause (cartridges/_fallback.py), which FusedQuantisedConfidenceLUT raises with. There is
no fallback and no switch: to run without the extension, construct QuantisedConfidenceLUT.

CUSTOM OP -- the per-table integers of the power-of-two read.

Its integers must agree with the torch definition in _pow2 (blend_exponents). That is tested, not guaranteed by
construction: tests/test_fused_vs_reference.py compares the fused class with the reference class.

Backward: RECOMPUTE with the torch expression. The op's backward re-evaluates the straight-through form
(_pow2.ste_cell_weights) with the op's integers held fixed and returns autograd's gradient of it: the same code as the
reference, so the same gradient semantics, including the ratio gradient of skipped tables and dropped second cells.
"""
from __future__ import annotations

import os
from typing import Tuple

import torch

from . import _fallback
from ._pow2 import ste_cell_weights

EXT_NAME = "lutorch_ex_pow2_int8_read"
DISCARD = 15                # shift code of a cell that is not read (csrc/pow2_scalars.cuh)
_CSRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "csrc")
# Compute capabilities the kernel is enabled on -- only those it has been validated on, on real hardware:
#   (12, 0) sm_120  RTX 5090 (Blackwell), validated on gpustar
#   (9, 0)  sm_90   H100 (Hopper), enabled for validation on nebius-h100 (see the lut_ablation exp_n_abl_47
#                   package's validation gate)
VALIDATED_ARCHES = ((12, 0), (9, 0))

_ext = None
_tried = False


def load():
    """Build / load the extension once. Returns the module or None (never raises; the cause is recorded)."""
    global _ext, _tried
    if _tried:
        return _ext
    _tried = True
    if not torch.cuda.is_available():
        _fallback.record_build_failure(EXT_NAME)                       # -> no_cuda_device
        return None
    cap = torch.cuda.get_device_capability()
    if cap not in VALIDATED_ARCHES:
        _fallback.record_cause(EXT_NAME, _fallback.Cause(
            "unvalidated_arch", f"device compute capability {cap} is not in VALIDATED_ARCHES {VALIDATED_ARCHES}"))
        return None
    try:
        from torch.utils.cpp_extension import load as _load
        _ext = _load(name=EXT_NAME, sources=[os.path.join(_CSRC, "pow2_int8_read.cu")],
                     extra_cuda_cflags=["-O3", "-std=c++20", "--fmad=false"], extra_cflags=["-O3", "-std=c++20"],
                     verbose=False)                          # cpp_extension targets the visible device's architecture
    except Exception as e:                                   # no nvcc, compile error, ...: recorded; the class raises
        _ext = None
        _fallback.record_build_failure(EXT_NAME, e)
    return _ext


def _reset_for_tests():
    global _ext, _tried
    _ext, _tried = None, False


# ======================================================================================================================
# the lutorch_ex::p2_scalars custom op
_registered = False


def ensure_registered() -> bool:
    """Build / load the extension and register lutorch_ex::p2_scalars, once and eagerly (before any torch.compile'd
    forward). False when the extension is unavailable (never raises; the cause is recorded)."""
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
        return (d.new_empty(*lead, 2), d.new_empty(*lead, 3, dtype=torch.uint8),
                d.new_empty(*lead, 2, dtype=torch.int8))

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
    return True


def _integers_from(cells: torch.Tensor, kq: torch.Tensor, Q: int):
    """(q, k, skip, drop) as the torch definition's float / bool tensors, from the op's cells and kq outputs."""
    k = kq[..., 0].to(torch.float32)
    q = kq[..., 1].to(torch.float32)
    skip = (cells[..., 2] & 15) == DISCARD
    drop = q > Q
    return q, k, skip, drop


# ======================================================================================================================
# the fused forward (csrc/pow2_int8_read.cu: table_exponents + quant_fwd)


class FusedQuantRead(torch.autograd.Function):
    """FusedQuantisedConfidenceLUT's read, ``z [B, G, d_in]`` -> ``[B, G, d_out]``.

    Forward: three launches -- the input-independent table exponents (a row-split partial max, then the max over the
    splits) and the fused read (addressing, the p2 integers and straight-through weights in registers, the fake-quant
    gather); the margins never reach global memory.

    Backward: one launch of the hand-written backward (csrc/pow2_int8_read.cu, quant_bwd), plus the sum of its per-CTA
    scalar partials. Saved: z, the parameters, the exponents, the dropout factors and each table's integers (c1, c2,
    clamped k', q, skip: one packed int32 per table, written by the forward when a gradient is needed); the kernel
    re-stages z for the margins and reads the integers instead of recomputing them. The gradient semantics are the
    reference's (_pow2.ste_cell_weights, ste_tables): the identity to the fp32 master table and the straight-through
    ratios b / e (clamped k', also for skipped and dropped cells)."""

    @staticmethod
    def forward(ctx, z, weights, log_beta, log_gamma, log_tau, anc_a16, anc_b16, zoff, zent, dmask, cfg, n, grad_mode):
        # n = read_top_n, the kernels' template parameter N (log_tau is None at n = 1: no blend, no tau)
        ext = load()
        W2 = weights.reshape(-1, weights.shape[-1])
        E = ext.table_exponents(W2, weights.shape[0], float(cfg["bits"] - 1 - cfg["offset"]))
        qlo, qhi = -(1 << (cfg["bits"] - 1)), (1 << (cfg["bits"] - 1)) - 1
        args = (W2, E, dmask, log_beta, log_gamma, log_tau, n, cfg["lo"], cfg["hi"], cfg["Q"], float(qlo), float(qhi))
        # the integers are saved exactly when a backward can follow: the caller's grad mode (grad mode is off inside
        # forward) and an input that needs a gradient -- autograd state, not a user switch
        save = grad_mode and any(ctx.needs_input_grad[:5])
        out, tabs = ext.quant_fwd(z, anc_a16, anc_b16, *args, save)
        ctx.save_for_backward(z, weights, E, log_beta, log_gamma, log_tau, anc_a16, anc_b16, zoff, zent, dmask, tabs)
        ctx.cfg = (n, cfg["lo"], cfg["hi"], cfg["Q"], float(qlo), float(qhi))
        return out

    @staticmethod
    def backward(ctx, go):
        z, weights, E, log_beta, log_gamma, log_tau, anc_a16, anc_b16, zoff, zent, dmask, tabs = ctx.saved_tensors
        W2 = weights.reshape(-1, weights.shape[-1])
        gz, gW, gscal = load().quant_bwd(go.contiguous(), z, anc_a16, anc_b16, zoff, zent, tabs, W2, E, dmask,
                                         log_beta, log_gamma, log_tau, *ctx.cfg)
        gs = gscal.sum(0)                      # over CTAs (a fixed-shape reduction: deterministic)
        need = ctx.needs_input_grad
        return (gz if need[0] else None, gW.view_as(weights) if need[1] else None,
                gs[0].view_as(log_beta) if need[2] else None, gs[1].view_as(log_gamma) if need[3] else None,
                gs[2].view_as(log_tau) if need[4] else None, None, None, None, None, None, None, None, None)
