"""FusedConfidenceLUT — ConfidenceLUT with its forward and backward in hand-written CUDA (``csrc/fused_confidence.cu``).

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

Dispatch (``backend`` forces a path; 'auto' is the hybrid):
* 'cuda' -- the hand-written kernels: a CUDA input, an fp32 or bf16 table and the extension loaded;
* 'pure' -- the inherited ConfidenceLUT path (compiled on CUDA, eager on CPU): CPU, fp64, an fp16 table, the
  extension failing to build or ``LUTORCH_EX_NO_CUDA_EXT=1``;
* 'auto' -- 'cuda' whenever it can run, else 'pure'.

bf16/fp16 (like the other fused cartridges): addressing runs in fp32 (so the discrete bit decisions don't drift),
every read and reduction accumulates in fp32, and the output is cast back to the input dtype once. A bf16 table
stays bf16 in memory and is upconverted in registers on the 'cuda' path; its weight gradient accumulates in an fp32
buffer and is cast to bf16 once, at the end. On the 'pure' path low precision runs the ConfidenceLUT math in fp32 on
an fp32 view of the parameters. ``fused_read`` is ignored by the CUDA path.

Table dropout draws its keep flags with ``torch.rand`` (one bool per (b, g, t), like Inductor's separate RNG kernel)
and the kernels apply them; the stream of random numbers differs from the compiled ConfidenceLUT's, the
distribution does not.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import torch
from torch.nn.utils.stateless import _reparametrize_module

from . import _fallback
from ._fused_ops import validate_backend
from ._native_ops import _CSRC
from .confidence import ConfidenceLUT

EXT_NAME = "lutorch_ex_fused_confidence"

_LOW_PREC = (torch.bfloat16, torch.float16)
_CUDA_TABLE_DTYPES = (torch.float32, torch.bfloat16)


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


@dataclass
class CudaKnobs:
    """Launch configuration for the CUDA kernels (all explicit; sweep them with ``bench_fused_confidence.py``)."""
    # Defaults = the sweep winner on both the RTX 5090 (sm_120) and the H100 (sm_90) at the d24 geometry.
    fwd_threads: int = field(default_factory=lambda: _env_int("LUTORCH_EX_CONF_CUDA_FWD_THREADS", 64))
    bwd_threads: int = field(default_factory=lambda: _env_int("LUTORCH_EX_CONF_CUDA_BWD_THREADS", 64))
    rows_per_cta: int = field(default_factory=lambda: _env_int("LUTORCH_EX_CONF_CUDA_ROWS_PER_CTA", 4))
    vec: int = field(default_factory=lambda: _env_int("LUTORCH_EX_CONF_CUDA_VEC", 4))
    vec_atomics: bool = field(default_factory=lambda: _env_int("LUTORCH_EX_CONF_CUDA_VEC_ATOMICS", 1) == 1)
    # Vector width for a bf16 table (1, 2, 4 or 8; 8 bf16 = the same 16-byte load as 4 fp32). 4 is the RTX 5090 sweep
    # winner: 8 is slower, mostly in the backward (two float4 grad-W atomics per thread instead of one).
    vec_bf16: int = field(default_factory=lambda: _env_int("LUTORCH_EX_CONF_CUDA_VEC_BF16", 4))


_EXT = None
_TRIED = False


def fused_confidence_ext():
    """Build/load the ``lutorch_ex_fused_confidence`` JIT extension; None if unavailable (never raises)."""
    global _EXT, _TRIED
    if _TRIED:
        return _EXT
    _TRIED = True
    if os.environ.get("LUTORCH_EX_NO_CUDA_EXT", "0") == "1":
        _fallback.record_build_failure(EXT_NAME, disabled=True)
        return None
    if not torch.cuda.is_available():
        _fallback.record_build_failure(EXT_NAME)
        return None
    try:
        from torch.utils.cpp_extension import load
        std = os.environ.get("SPIKY_CXX_STD", "c++20")
        _EXT = load(name=EXT_NAME, sources=[os.path.join(_CSRC, "fused_confidence.cu")],
                    extra_cflags=[f"-std={std}", "-O3"], extra_cuda_cflags=[f"-std={std}", "-O3"], verbose=False)
    except Exception as e:
        # Never raise here; classify WHY. FusedConfidenceLUT reports it loudly where it actually falls back.
        _fallback.record_build_failure(EXT_NAME, e)
        _EXT = None
    return _EXT


class _FusedConfidence(torch.autograd.Function):
    @staticmethod
    def forward(ctx, z, weights, log_beta, log_gamma, log_tau, anc_a, anc_b, keep, keep_scale, nap, eps, n, knobs):
        # z, log_beta / log_gamma / log_tau: fp32. weights: fp32 or bf16 (the kernels read it in place).
        ext = fused_confidence_ext()
        W2 = weights.reshape(-1, weights.shape[-1])
        vec = knobs.vec if weights.dtype == torch.float32 else knobs.vec_bf16
        out = ext.confidence_fwd(z, anc_a, anc_b, W2, keep, keep_scale, log_beta, log_gamma, log_tau,
                                 nap, eps, n, knobs.fwd_threads, knobs.rows_per_cta, vec)
        ctx.save_for_backward(z, weights, log_beta, log_gamma, log_tau, anc_a, anc_b, keep)
        ctx.cfg = (keep_scale, nap, eps, n, knobs, vec)
        return out                                          # fp32

    @staticmethod
    def backward(ctx, go):
        z, weights, log_beta, log_gamma, log_tau, anc_a, anc_b, keep = ctx.saved_tensors
        keep_scale, nap, eps, n, knobs, vec = ctx.cfg
        W2 = weights.reshape(-1, weights.shape[-1])
        gW, gz, gscal = fused_confidence_ext().confidence_bwd(
            go.contiguous(), z, anc_a, anc_b, W2, keep, keep_scale, log_beta, log_gamma, log_tau,
            nap, eps, n, knobs.bwd_threads, knobs.rows_per_cta, vec, knobs.vec_atomics)
        g = gscal.sum(0)
        g_tau = g[2] if n == 2 else None
        # gW is accumulated in fp32 whatever the table dtype; a bf16 Parameter must receive a bf16 .grad, so it is
        # cast once here (one rounding per element, after the whole accumulation; a no-op for an fp32 table).
        return (gz, gW.view_as(weights).to(weights.dtype), g[0], g[1], g_tau,
                None, None, None, None, None, None, None, None)


class FusedConfidenceLUT(ConfidenceLUT):
    """:class:`ConfidenceLUT` with a hand-written CUDA forward/backward; see the module docstring.

    Args: those of :class:`ConfidenceLUT`, plus ``backend`` ('auto' / 'cuda' / 'pure') and ``knobs`` (a
    :class:`CudaKnobs`; default from the environment).
    """

    _COMPILE_TRAIN = False      # the CUDA path is not compiled; the 'pure' path then runs eager in training

    #: Every backend forward dispatches on ('auto' picks one of the others per call).
    _BACKENDS = ("auto", "cuda", "pure")

    def __init__(self, spec, *, backend: str = "auto", knobs: CudaKnobs | None = None, **kw):
        validate_backend(type(self).__name__, backend, self._BACKENDS)
        super().__init__(spec, **kw)
        self.backend = backend
        self.knobs = knobs if knobs is not None else CudaKnobs()
        if spec.d_in > torch.iinfo(torch.int16).max:
            raise ValueError(f"FusedConfidenceLUT needs d_in <= 32767, got {spec.d_in}")
        # int16 copies of the anchors for the kernels (the int64 buffers stay for the fallback path and state_dict).
        self.register_buffer("_anc_a16", self.anchor_a.to(torch.int16).contiguous(), persistent=False)
        if self.anchor_b is not None:
            self.register_buffer("_anc_b16", self.anchor_b.to(torch.int16).contiguous(), persistent=False)
        else:
            self._anc_b16 = None

    def _supports_low_precision(self) -> bool:
        return True  # bf16/fp16: fp32 addressing + fp32-accumulated reads ('cuda' and 'pure')

    def _shape_ok(self, x: torch.Tensor) -> bool:
        """Everything the CUDA kernels need EXCEPT the extension itself (a CUDA input, fp32/bf16 table, nap <= 16).
        Failing this is a deliberate design fallback (no kernel for that case), not a broken fast path."""
        return (x.is_cuda and x.dtype in (torch.float32,) + _LOW_PREC and self.weights.dtype in _CUDA_TABLE_DTYPES
                and self.spec.nap <= 16)

    def _cuda_ok(self, x: torch.Tensor) -> bool:
        return self._shape_ok(x) and fused_confidence_ext() is not None

    def _keep_flags(self, B: int, device):
        """Table-dropout keep flags ``[B, G, tph]`` bool (TRAIN + grad only, else None); kernels scale by 1/keep."""
        if not (self.training and self.table_dropout_rate > 0.0 and torch.is_grad_enabled()):
            return None
        return torch.rand(B, self.spec.n_groups, self.spec.tph, device=device) < (1.0 - self.table_dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        be = self.backend
        if be == "auto":
            if self._cuda_ok(x):
                be = "cuda"
            else:
                if self._shape_ok(x):   # the case the kernels exist for, but the extension is missing: involuntary
                    _fallback.report_involuntary_fallback(type(self).__name__, EXT_NAME, "auto", "pure")
                be = "pure"
        elif be == "cuda" and not self._cuda_ok(x):
            if self._shape_ok(x):       # only the extension is missing: name the classified cause
                cause = _fallback.recorded(EXT_NAME) or _fallback.Cause(
                    "unclassified", "the extension loader did not record a cause",
                    remedy=_fallback.CAUSES["unclassified"][1])
                raise _fallback.NativeUnavailableError(
                    _fallback.banner(type(self).__name__, EXT_NAME, "cuda", "(none: explicit request)", cause))
            raise RuntimeError(
                f"{type(self).__name__}: backend='cuda' needs a CUDA input, an fp32 or bf16 table, nap <= 16 and the "
                f"fused_confidence extension; got input {x.dtype} on {x.device}, table {self.weights.dtype}, "
                f"nap {self.spec.nap}, extension {'loaded' if fused_confidence_ext() is not None else 'unavailable'}")
        self.last_backend = be                                   # provenance for benchmarks (what actually ran)
        if be == "pure":
            return self._pure_forward(x)
        self._check_input(x)
        # Addressing in fp32 (a no-op for fp32 input); grad x comes back through this cast in the input dtype.
        z = x[:, self.in_head, :].float().contiguous()          # [B, G, d_in]
        keep = self._keep_flags(x.shape[0], x.device)
        keep_scale = 1.0 / (1.0 - self.table_dropout_rate)
        log_tau = self.log_read_tau if self.read_top_n == 2 else self.confidence_log_beta
        grp = _FusedConfidence.apply(z, self.weights, self.confidence_log_beta.float(),
                                     self.confidence_log_gamma.float(), log_tau.float(),
                                     self._anc_a16, self._anc_b16, keep, keep_scale, self.spec.nap, self.cmp_eps,
                                     self.read_top_n, self.knobs)
        return self._route(grp, x).to(x.dtype)

    def _pure_forward(self, x: torch.Tensor) -> torch.Tensor:
        """The inherited ConfidenceLUT path. Full precision: unchanged (compiled on CUDA). bf16/fp16 input or
        parameters: the same math in fp32 on an fp32 view of the parameters / floating buffers (eager), output cast
        back to the input dtype; gradients flow back through the casts (fp32 accumulation, one cast per leaf)."""
        low = x.dtype in _LOW_PREC or any(t.dtype in _LOW_PREC for t in self.parameters())
        if not low:
            return super().forward(x)
        f32 = {n: t.float() for n, t in list(self.named_parameters()) + list(self.named_buffers())
               if t.is_floating_point() and t.dtype in _LOW_PREC}
        with _reparametrize_module(self, f32):
            out = ConfidenceLUT._forward_impl(self, x.float())
        return out.to(x.dtype)
