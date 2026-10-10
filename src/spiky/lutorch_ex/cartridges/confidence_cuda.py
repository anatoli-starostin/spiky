"""ConfidenceLUTCuda — ConfidenceLUT with its forward and backward in hand-written CUDA (``csrc/confidence_cuda.cu``).

Same parameters, numerics and interface as :class:`ConfidenceLUT` (read_top_n 1 and 2, pairs and single anchors,
table dropout, learnable or frozen β / γ / τ); only the implementation differs. Instead of torch.compile + Inductor
(3 forward and 9 backward kernels at the d24 geometry), one forward and one backward kernel each run the whole
cartridge: addressing, confidence score, blend weight, score-weighted read, and on the backward the weight-gradient
scatter, the score / blend / margin gradients into z, and the β / γ / τ gradients. The per-table address, score and
margins never touch global memory; the backward recomputes them from z.

Every launch decision Inductor makes heuristically is an explicit knob here (:class:`CudaKnobs`): threads per CTA
(forward and backward separately), rows per CTA, the vector width of the table loads / gradient scatters, and
vector atomics. Defaults come from ``LUTORCH_EX_CONF_CUDA_*`` environment variables and can be changed per instance
through ``cartridge.knobs``.

fp32 CUDA only. Anything else (CPU, fp64, the extension failing to build, ``LUTORCH_EX_NO_CUDA_EXT=1``) runs the
inherited ConfidenceLUT path, so the module always works. ``fused_read`` is ignored by the CUDA path.

Table dropout draws its keep flags with ``torch.rand`` (one bool per (b, g, t), like Inductor's separate RNG kernel)
and the kernels apply them; the stream of random numbers differs from the compiled ConfidenceLUT's, the
distribution does not.
"""
from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, field

import torch

from ._native_ops import _CSRC
from .confidence import ConfidenceLUT


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


@dataclass
class CudaKnobs:
    """Launch configuration for the CUDA kernels (all explicit; sweep them with ``bench_confidence_cuda.py``)."""
    # Defaults = the sweep winner on both the RTX 5090 (sm_120) and the H100 (sm_90) at the d24 geometry.
    fwd_threads: int = field(default_factory=lambda: _env_int("LUTORCH_EX_CONF_CUDA_FWD_THREADS", 64))
    bwd_threads: int = field(default_factory=lambda: _env_int("LUTORCH_EX_CONF_CUDA_BWD_THREADS", 64))
    rows_per_cta: int = field(default_factory=lambda: _env_int("LUTORCH_EX_CONF_CUDA_ROWS_PER_CTA", 4))
    vec: int = field(default_factory=lambda: _env_int("LUTORCH_EX_CONF_CUDA_VEC", 4))
    vec_atomics: bool = field(default_factory=lambda: _env_int("LUTORCH_EX_CONF_CUDA_VEC_ATOMICS", 1) == 1)
    # L2 handling of the table W (prototype). l2_hint: per-load L2 eviction priority on the W reads,
    # 0 = none, 1 = evict_last, 2 = evict_first. l2_window: hitRatio of a persisting access-policy window over W
    # on the launch stream (0 = off); the persisting carve-out is sized to min(W bytes, the device maximum).
    l2_hint: int = field(default_factory=lambda: _env_int("LUTORCH_EX_CONF_CUDA_L2_HINT", 0))
    l2_window: float = field(default_factory=lambda: float(os.environ.get("LUTORCH_EX_CONF_CUDA_L2_WINDOW", "0")))
    # Which table the BACKWARD window covers: 0 = W (the reads), 1 = grad W (the atomic scatter target).
    l2_window_target: int = field(default_factory=lambda: _env_int("LUTORCH_EX_CONF_CUDA_L2_WINDOW_TARGET", 0))


_EXT = None
_TRIED = False


def confidence_cuda_ext():
    """Build/load the ``lutorch_ex_confidence_cuda`` JIT extension; None if unavailable (never raises)."""
    global _EXT, _TRIED
    if _TRIED:
        return _EXT
    _TRIED = True
    if os.environ.get("LUTORCH_EX_NO_CUDA_EXT", "0") == "1" or not torch.cuda.is_available():
        return None
    try:
        from torch.utils.cpp_extension import load
        std = os.environ.get("SPIKY_CXX_STD", "c++20")
        _EXT = load(name="lutorch_ex_confidence_cuda", sources=[os.path.join(_CSRC, "confidence_cuda.cu")],
                    extra_cflags=[f"-std={std}", "-O3"], extra_cuda_cflags=[f"-std={std}", "-O3"], verbose=False)
    except Exception as e:
        warnings.warn(f"lutorch_ex: the confidence_cuda extension could not be built/loaded; ConfidenceLUTCuda uses "
                      f"the ConfidenceLUT path instead. {type(e).__name__}: {e}", RuntimeWarning, stacklevel=2)
        _EXT = None
    return _EXT


class _ConfidenceCuda(torch.autograd.Function):
    @staticmethod
    def forward(ctx, z, weights, log_beta, log_gamma, log_tau, anc_a, anc_b, keep, keep_scale, nap, eps, n, knobs):
        ext = confidence_cuda_ext()
        W2 = weights.reshape(-1, weights.shape[-1])
        out = ext.confidence_fwd(z, anc_a, anc_b, W2, keep, keep_scale, log_beta, log_gamma, log_tau,
                                 nap, eps, n, knobs.fwd_threads, knobs.rows_per_cta, knobs.vec,
                                 knobs.l2_hint, knobs.l2_window)
        ctx.save_for_backward(z, weights, log_beta, log_gamma, log_tau, anc_a, anc_b, keep)
        ctx.cfg = (keep_scale, nap, eps, n, knobs)
        return out

    @staticmethod
    def backward(ctx, go):
        z, weights, log_beta, log_gamma, log_tau, anc_a, anc_b, keep = ctx.saved_tensors
        keep_scale, nap, eps, n, knobs = ctx.cfg
        W2 = weights.reshape(-1, weights.shape[-1])
        gW, gz, gscal = confidence_cuda_ext().confidence_bwd(
            go.contiguous(), z, anc_a, anc_b, W2, keep, keep_scale, log_beta, log_gamma, log_tau,
            nap, eps, n, knobs.bwd_threads, knobs.rows_per_cta, knobs.vec, knobs.vec_atomics,
            knobs.l2_hint, knobs.l2_window, knobs.l2_window_target)
        g = gscal.sum(0)
        g_tau = g[2] if n == 2 else None
        return (gz, gW.view_as(weights), g[0], g[1], g_tau, None, None, None, None, None, None, None, None)


class ConfidenceLUTCuda(ConfidenceLUT):
    """:class:`ConfidenceLUT` with a hand-written CUDA forward/backward; see the module docstring.

    Args: those of :class:`ConfidenceLUT`, plus ``knobs`` (a :class:`CudaKnobs`; default from the environment).
    """

    _COMPILE_TRAIN = False      # the CUDA path is not compiled; the fallback path then runs eager in training

    def __init__(self, spec, *, knobs: CudaKnobs | None = None, **kw):
        super().__init__(spec, **kw)
        self.knobs = knobs if knobs is not None else CudaKnobs()
        if spec.d_in > torch.iinfo(torch.int16).max:
            raise ValueError(f"ConfidenceLUTCuda needs d_in <= 32767, got {spec.d_in}")
        # int16 copies of the anchors for the kernels (the int64 buffers stay for the fallback path and state_dict).
        self.register_buffer("_anc_a16", self.anchor_a.to(torch.int16).contiguous(), persistent=False)
        if self.anchor_b is not None:
            self.register_buffer("_anc_b16", self.anchor_b.to(torch.int16).contiguous(), persistent=False)
        else:
            self._anc_b16 = None

    def _cuda_ok(self, x: torch.Tensor) -> bool:
        return (x.is_cuda and x.dtype == torch.float32 and self.weights.dtype == torch.float32
                and self.spec.nap <= 16 and confidence_cuda_ext() is not None)

    def _keep_flags(self, B: int, device):
        """Table-dropout keep flags ``[B, G, tph]`` bool (TRAIN + grad only, else None); kernels scale by 1/keep."""
        if not (self.training and self.table_dropout_rate > 0.0 and torch.is_grad_enabled()):
            return None
        return torch.rand(B, self.spec.n_groups, self.spec.tph, device=device) < (1.0 - self.table_dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self._cuda_ok(x):
            return super().forward(x)
        self._check_input(x)
        z = x[:, self.in_head, :].contiguous()                  # [B, G, d_in]
        keep = self._keep_flags(x.shape[0], x.device)
        keep_scale = 1.0 / (1.0 - self.table_dropout_rate)
        log_tau = self.log_read_tau if self.read_top_n == 2 else self.confidence_log_beta
        grp = _ConfidenceCuda.apply(z, self.weights, self.confidence_log_beta, self.confidence_log_gamma, log_tau,
                                    self._anc_a16, self._anc_b16, keep, keep_scale, self.spec.nap, self.cmp_eps,
                                    self.read_top_n, self.knobs)
        return self._route(grp, x)
