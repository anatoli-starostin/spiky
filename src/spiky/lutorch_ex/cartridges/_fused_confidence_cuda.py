"""The CUDA implementation of FusedConfidenceLUT, its only one: hand-written forward + backward kernels
(``csrc/fused_confidence.cu``), the extension that builds them, the autograd Function around them, and the check that a
call is a case they cover. The maths and the precision contract are in :mod:`.fused_confidence`.
"""
from __future__ import annotations

import torch

from . import _fallback
from ._cuda_ext import Extension, _launch, _row_launch, device_problem, not_covered, row_problem
from ._dtypes import _CUDA_INPUT_DTYPES, _CUDA_TABLE_DTYPES, unsupported_dtype

EXT = Extension("lutorch_ex_fused_confidence", ["fused_confidence.cu"])
REFERENCE = "ConfidenceLUT"


def require_cuda(module, x: torch.Tensor) -> None:
    """Raise unless the kernels can run this input: a RuntimeError naming every case they do not cover (no CUDA
    device or a CPU input, an input other than float32 / bfloat16 / float16, a table other than float32 /
    bfloat16, nap > 16, a d_out one CTA cannot cover), or the classified NativeUnavailableError when only the
    extension is missing."""
    problems = [p for p in (device_problem(x),) if p]
    if x.dtype not in _CUDA_INPUT_DTYPES:
        problems.append(unsupported_dtype("input", x.dtype, _CUDA_INPUT_DTYPES))
    if module.weights.dtype not in _CUDA_TABLE_DTYPES:
        problems.append(unsupported_dtype("table", module.weights.dtype, _CUDA_TABLE_DTYPES))
    if module.spec.nap > 16:
        problems.append(f"nap={module.spec.nap} (max 16)")
    problems += [p for p in (row_problem(module.spec.d_out, *_row_launch(module.knobs, module.spec.d_out,
                                                                        module.weights.dtype)),) if p]
    if problems:
        raise RuntimeError(not_covered(type(module).__name__, problems, REFERENCE))
    if EXT.get() is None:
        _fallback.raise_unavailable(type(module).__name__, EXT.name)


class FusedConfidence(torch.autograd.Function):
    @staticmethod
    def forward(ctx, z, weights, log_beta, log_gamma, log_tau, anc_a, anc_b, keep, keep_scale, nap, n, knobs,
                out_bf16):
        # z: float32 or bfloat16 (upconverted in-kernel). log_beta / log_gamma / log_tau: float32. weights: float32 or
        # bfloat16 (the kernels read it in place). The output is float32, or bfloat16 when out_bf16 (bfloat16 z only).
        ext = EXT.get()
        W2 = weights.reshape(-1, weights.shape[-1])
        # The knobs made legal for this d_out, as for the Manifesto kernels: the vector width is halved until it divides
        # d_out and each thread count is raised to cover one table row (a no-op at the default knobs for d_out <= 256
        # divisible by 4, so those launches are unchanged).
        fwd_threads, bwd_threads, vec = _launch(knobs, W2.shape[1], weights.dtype)
        out = ext.confidence_fwd(z, anc_a, anc_b, W2, keep, keep_scale, log_beta, log_gamma, log_tau,
                                 nap, n, fwd_threads, knobs.rows_per_cta, vec, out_bf16)
        ctx.save_for_backward(z, weights, log_beta, log_gamma, log_tau, anc_a, anc_b, keep)
        ctx.cfg = (keep_scale, nap, n, knobs, vec, bwd_threads)
        return out

    @staticmethod
    def backward(ctx, go):
        z, weights, log_beta, log_gamma, log_tau, anc_a, anc_b, keep = ctx.saved_tensors
        keep_scale, nap, n, knobs, vec, bwd_threads = ctx.cfg
        W2 = weights.reshape(-1, weights.shape[-1])
        # The kernel reads go in float32 (a bfloat16 go from the bfloat16-output mode is upcast: exact).
        # gz comes back in z's dtype.
        gW, gz, gscal = EXT.get().confidence_bwd(
            go.float().contiguous(), z, anc_a, anc_b, W2, keep, keep_scale, log_beta, log_gamma, log_tau,
            nap, n, bwd_threads, knobs.rows_per_cta, vec, knobs.vec_atomics)
        g = gscal.sum(0)
        g_tau = g[2] if n == 2 else None
        # gW is accumulated in float32 whatever the table dtype; a bfloat16 Parameter must receive a bfloat16 .grad, so
        # it is cast once here (one rounding per element, after the whole accumulation; a no-op for a float32 table).
        return (gz, gW.view_as(weights).to(weights.dtype), g[0], g[1], g_tau,
                None, None, None, None, None, None, None, None)
