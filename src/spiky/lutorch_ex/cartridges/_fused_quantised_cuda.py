"""The CUDA implementation of FusedQuantisedConfidenceLUT and DeployedQuantisedConfidenceLUT: the p2 extension's
build / load and device gating, the fused read, and the checks that a call is a case the kernels cover; plus the
lutorch_ex::p2_scalars custom op, a test oracle. Used ONLY by those two classes and the tests; the reference
QuantisedConfidenceLUT never calls it (importing this module builds nothing: ensure_registered() loads the extension).

EXTENSION

csrc/pow2_int8_read.cu provides the production kernels -- FusedQuantisedConfidenceLUT's read: `table_exponents`,
`quant_fwd` (the fused forward) and `quant_bwd` (the fused backward), wrapped by FusedQuantRead; and
DeployedQuantisedConfidenceLUT's `deploy_fwd` -- and `scalars`, the forward of the lutorch_ex::p2_scalars op below
(p2::table_scalars, csrc/pow2_scalars.cuh, for every table), which no class calls.

Built lazily with torch.utils.cpp_extension on first use, for the visible device's architecture, and ONLY on an
architecture in VALIDATED_ARCHES: an explicit allowlist of the compute capabilities on which the kernel has been
validated on real hardware. Nothing in it is architecture-specific; the allowlist records validation, not a hardware
requirement. Anything else -- no CUDA, an architecture not on the list, no nvcc, a failed build -- makes `EXT.get()`
return None and records the classified cause (cartridges/_fallback.py), which the fused classes raise with. There is
no fallback and no switch: to run without the extension, construct QuantisedConfidenceLUT.

CUSTOM OP -- the per-table integers of the power-of-two read: a test oracle, not a production path.

lutorch_ex::p2_scalars exposes p2::table_scalars -- the device function the production kernels compute their
integers with -- on its own, so tests/test_fused_vs_reference.py can compare those integers with the torch definition
in _pow2 (blend_exponents) on every table. No class calls it: FusedQuantisedConfidenceLUT runs
FusedQuantRead (quant_fwd / quant_bwd) and DeployedQuantisedConfidenceLUT runs deploy_fwd. It is registered by
ensure_registered(), which also builds the extension the production kernels need.

Backward: RECOMPUTE with the torch expression. The op's backward re-evaluates the straight-through form
(_pow2.ste_cell_weights) with the op's integers held fixed and returns autograd's gradient of it: the same code as the
reference, so the same gradient semantics, including the ratio gradient of skipped tables and dropped second cells.
"""
from __future__ import annotations

from typing import Tuple

import torch

from . import _fallback
from ._cuda_ext import Extension, device_problem, not_covered, row_problem, row_threads
from ._dtypes import _LOW_PRECISION, unsupported_dtype
from ._pow2 import ste_cell_weights

DISCARD = 15                # shift code of a cell that is not read (csrc/pow2_scalars.cuh)
#: Compute capabilities the kernel is enabled on -- only those it has been validated on, on real hardware:
#: (12, 0) sm_120 (RTX 5090, Blackwell) and (9, 0) sm_90 (H100, Hopper).
VALIDATED_ARCHES = ((12, 0), (9, 0))
#: --fmad=false: no fused multiply-add contraction, so every add of the fixed reduction order is a separate float32 add.
EXT = Extension("lutorch_ex_pow2_int8_read", ["pow2_int8_read.cu"], extra_cuda_cflags=["--fmad=false"],
                validated_arches=VALIDATED_ARCHES)
REFERENCE = "QuantisedConfidenceLUT"
#: The quantised family's fused CUDA kernels (quant_fwd / quant_bwd, deploy_fwd) take float32 input and a float32 table
#: and write float32 output only.
_FUSED_DTYPES = (torch.float32,)
_FUSED_OUTPUT = "the output is torch.float32"


def input_dtype_problem(dtype: torch.dtype, note: str = "") -> str:
    """The refusal of an input dtype the quantised kernels do not take (with the output dtype, and an optional note)."""
    return unsupported_dtype("input", dtype, _FUSED_DTYPES, _FUSED_OUTPUT + (f"; {note}" if note else ""))


def require_cuda(module, x: torch.Tensor) -> None:
    """FusedQuantisedConfidenceLUT: raise unless the kernel can run this call: a RuntimeError naming every case it does
    not cover (single anchors, nap > 8, a CPU input, an input or table other than float32, a d_out one CTA cannot
    cover), or the classified NativeUnavailableError when only the extension is missing. The reference class is offered
    only for the cases it runs: not for a bfloat16 / float16 input or table, which it does not take either."""
    problems = []
    if module.single:
        problems.append("anchor_mode='single' (pairs only)")
    if module.spec.nap > 8:
        problems.append(f"nap={module.spec.nap} (max 8)")
    problems += [p for p in (device_problem(x),) if p]
    if x.dtype != torch.float32:
        problems.append(input_dtype_problem(x.dtype))
    if module.weights.dtype != torch.float32:
        problems.append(unsupported_dtype("table", module.weights.dtype, _FUSED_DTYPES))
    problems += [p for p in (row_problem(module.spec.d_out, *row_threads(module.spec.d_out, (4, 2, 1))),) if p]
    if problems:
        lowp = x.dtype in _LOW_PRECISION or module.weights.dtype in _LOW_PRECISION
        raise RuntimeError(not_covered(type(module).__name__, problems, None if lowp else REFERENCE))
    if not ensure_registered():
        _fallback.raise_unavailable(type(module).__name__, EXT.name)


def require_deploy_cuda(module, x: torch.Tensor):
    """DeployedQuantisedConfidenceLUT on a CUDA input: the extension module, or raise: a RuntimeError naming every case
    the deploy kernel does not cover (an input other than float32, nap > 8, d_in beyond int16 anchors, a d_out one CTA
    cannot cover at the kernel's widths 8 / 4 / 2 / 1), or the classified NativeUnavailableError when only the extension
    is missing. The deployed class has no reference twin to offer."""
    problems = []
    if x.dtype != torch.float32:
        problems.append(input_dtype_problem(x.dtype))
    if module.spec.nap > 8:
        problems.append(f"nap={module.spec.nap} (max 8)")
    if module.spec.d_in > torch.iinfo(torch.int16).max:
        problems.append(f"d_in={module.spec.d_in} (max {torch.iinfo(torch.int16).max})")
    problems += [p for p in (row_problem(module.spec.d_out, *row_threads(module.spec.d_out, (8, 4, 2, 1))),) if p]
    if problems:
        raise RuntimeError(not_covered(type(module).__name__, problems, None))
    ext = EXT.get()
    if ext is None:
        _fallback.raise_unavailable(type(module).__name__, EXT.name)
    return ext


# ======================================================================================================================
# the lutorch_ex::p2_scalars custom op (a test oracle; see the module docstring)
_registered = False


def ensure_registered() -> bool:
    """Build / load the extension, once, and register the lutorch_ex::p2_scalars test op with it. False when the
    extension is unavailable (never raises; the cause is recorded)."""
    global _registered
    if _registered:
        return True
    ext = EXT.get()
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
    reference's (_pow2.ste_cell_weights, ste_tables): the identity to the float32 master table and the straight-through
    ratios b / e (clamped k', also for skipped and dropped cells)."""

    @staticmethod
    def forward(ctx, z, weights, log_beta, log_gamma, log_tau, anc_a16, anc_b16, zoff, zent, dmask, cfg, n, grad_mode):
        # n = read_top_n, the kernels' template parameter N (log_tau is None at n = 1: no blend, no tau)
        ext = EXT.get()
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
        gz, gW, gscal = EXT.get().quant_bwd(go.contiguous(), z, anc_a16, anc_b16, zoff, zent, tabs, W2, E, dmask,
                                         log_beta, log_gamma, log_tau, *ctx.cfg)
        gs = gscal.sum(0)                      # over CTAs (a fixed-shape reduction: deterministic)
        need = ctx.needs_input_grad
        return (gz if need[0] else None, gW.view_as(weights) if need[1] else None,
                gs[0].view_as(log_beta) if need[2] else None, gs[1].view_as(log_gamma) if need[3] else None,
                gs[2].view_as(log_tau) if need[4] else None, None, None, None, None, None, None, None, None)
