"""FusedConfidenceLUT — the fused twin of ConfidenceLUT: its forward and backward in hand-written CUDA
(``csrc/fused_confidence.cu``).

Same parameters, numerics, state_dict and interface as :class:`ConfidenceLUT` (read_top_n 1 and 2, pairs and single
anchors, table dropout, learnable or frozen β / γ / τ); both build on
:class:`~spiky.lutorch_ex.cartridges.confidence_base.ConfidenceBase`, neither inherits the other's forward. Instead of torch.compile + Inductor
(3 forward and 9 backward kernels at the d24 geometry), one forward and one backward kernel each run the whole
cartridge: addressing, confidence score, blend weight, score-weighted read, and on the backward the weight-gradient
scatter, the score / blend / margin gradients into z, and the β / γ / τ gradients. The per-table address, score and
margins never touch global memory; the backward recomputes them from z.

Every launch decision Inductor makes heuristically is an explicit knob here (:class:`CudaKnobs`): threads per CTA
(forward and backward separately), rows per CTA, the vector width of the table loads / gradient scatters, and
vector atomics. Defaults come from ``LUTORCH_EX_CONF_CUDA_*`` environment variables and can be changed per instance
through ``cartridge.knobs``.

The kernels are its only implementation (spec rule 3). A case they do not cover (a CPU input, an fp64 input, an fp16
or fp64 table, nap > 16, d_in beyond int16 anchors, a d_out one CTA cannot cover) raises, naming the case, and so does a
missing extension, naming the cause; :class:`ConfidenceLUT` runs everywhere else.

bf16/fp16 (like the other fused cartridges): addressing runs in fp32 (so the discrete bit decisions don't drift),
every read and reduction accumulates in fp32, and the output is rounded to the input dtype once. A
bf16 input is read by the kernels directly (upconverted while each row is staged) and, when groups map 1:1 onto
output heads, the kernel stores the bf16 output itself (a fan-in keeps the fp32 output, sums the groups, then casts);
both are bitwise identical to casting outside the kernel. A bf16 table stays bf16 in memory and is upconverted in
registers; its weight gradient accumulates in an fp32 buffer and is cast to bf16 once, at the end.

Table dropout draws its keep flags with ``torch.rand`` (one bool per (b, g, t), like Inductor's separate RNG kernel)
and the kernels apply them; the stream of random numbers differs from the compiled ConfidenceLUT's, the
distribution does not.
"""
from __future__ import annotations

import torch

from . import _fallback
from ._cuda_ext import _MAX_CTA_THREADS, CudaKnobs, _launch, _row_launch, load_extension  # noqa: F401
from .confidence_base import ConfidenceBase

EXT_NAME = "lutorch_ex_fused_confidence"

_LOW_PREC = (torch.bfloat16, torch.float16)
_CUDA_TABLE_DTYPES = (torch.float32, torch.bfloat16)

_EXT = None
_TRIED = False


def fused_confidence_ext():
    """Build/load the ``lutorch_ex_fused_confidence`` JIT extension; None if unavailable (never raises; the cause is
    recorded for the error the cartridge raises)."""
    global _EXT, _TRIED
    if _TRIED:
        return _EXT
    _TRIED = True
    _EXT = load_extension(EXT_NAME, ["fused_confidence.cu"])
    return _EXT


class _FusedConfidence(torch.autograd.Function):
    @staticmethod
    def forward(ctx, z, weights, log_beta, log_gamma, log_tau, anc_a, anc_b, keep, keep_scale, nap, eps, n, knobs,
                out_bf16):
        # z: fp32 or bf16 (upconverted in-kernel). log_beta / log_gamma / log_tau: fp32. weights: fp32 or bf16 (the
        # kernels read it in place). The output is fp32, or bf16 when out_bf16 (bf16 z only).
        ext = fused_confidence_ext()
        W2 = weights.reshape(-1, weights.shape[-1])
        # The knobs made legal for this d_out, as for the Manifesto kernels: the vector width is halved until it divides
        # d_out and each thread count is raised to cover one table row (a no-op at the default knobs for d_out <= 256
        # divisible by 4, so those launches are unchanged).
        fwd_threads, bwd_threads, vec = _launch(knobs, W2.shape[1], weights.dtype)
        out = ext.confidence_fwd(z, anc_a, anc_b, W2, keep, keep_scale, log_beta, log_gamma, log_tau,
                                 nap, eps, n, fwd_threads, knobs.rows_per_cta, vec, out_bf16)
        ctx.save_for_backward(z, weights, log_beta, log_gamma, log_tau, anc_a, anc_b, keep)
        ctx.cfg = (keep_scale, nap, eps, n, knobs, vec, bwd_threads)
        return out

    @staticmethod
    def backward(ctx, go):
        z, weights, log_beta, log_gamma, log_tau, anc_a, anc_b, keep = ctx.saved_tensors
        keep_scale, nap, eps, n, knobs, vec, bwd_threads = ctx.cfg
        W2 = weights.reshape(-1, weights.shape[-1])
        # The kernel reads go in fp32 (a bf16 go from the bf16-output mode is upcast: exact). gz comes back in z's dtype.
        gW, gz, gscal = fused_confidence_ext().confidence_bwd(
            go.float().contiguous(), z, anc_a, anc_b, W2, keep, keep_scale, log_beta, log_gamma, log_tau,
            nap, eps, n, bwd_threads, knobs.rows_per_cta, vec, knobs.vec_atomics)
        g = gscal.sum(0)
        g_tau = g[2] if n == 2 else None
        # gW is accumulated in fp32 whatever the table dtype; a bf16 Parameter must receive a bf16 .grad, so it is
        # cast once here (one rounding per element, after the whole accumulation; a no-op for an fp32 table).
        return (gz, gW.view_as(weights).to(weights.dtype), g[0], g[1], g_tau,
                None, None, None, None, None, None, None, None, None)


class FusedConfidenceLUT(ConfidenceBase):
    """The fused twin of :class:`ConfidenceLUT`; see the module docstring.

    Args: those of :class:`~spiky.lutorch_ex.cartridges.confidence_base.ConfidenceBase`, plus ``knobs`` (a
    :class:`CudaKnobs`; default from the environment).
    """

    def __init__(self, spec, *, knobs: CudaKnobs | None = None, **kw):
        super().__init__(spec, **kw)
        self.knobs = knobs if knobs is not None else CudaKnobs()
        if spec.d_in > torch.iinfo(torch.int16).max:
            raise ValueError(f"FusedConfidenceLUT needs d_in <= 32767, got {spec.d_in}")
        # int16 copies of the anchors for the kernels (the int64 buffers stay the state_dict's source of truth).
        self.register_buffer("_anc_a16", self.anchor_a.to(torch.int16).contiguous(), persistent=False)
        if self.anchor_b is not None:
            self.register_buffer("_anc_b16", self.anchor_b.to(torch.int16).contiguous(), persistent=False)
        else:
            self._anc_b16 = None

    def _require_cuda(self, x: torch.Tensor) -> None:
        """Raise unless the kernels can run this input: a RuntimeError naming the unsupported case (a CPU input, an
        input other than fp32 / bf16 / fp16, a table other than fp32 / bf16, nap > 16, a d_out one CTA cannot cover),
        or the classified NativeUnavailableError when only the extension is missing."""
        vec, need = _row_launch(self.knobs, self.spec.d_out, self.weights.dtype)
        problems = []
        if not x.is_cuda:
            problems.append(f"the input is on {x.device}, the kernels run on CUDA only")
        if x.dtype not in (torch.float32,) + _LOW_PREC:
            problems.append(f"input dtype {x.dtype} (supported: float32, bfloat16, float16)")
        if self.weights.dtype not in _CUDA_TABLE_DTYPES:
            problems.append(f"table dtype {self.weights.dtype} (supported: float32, bfloat16)")
        if self.spec.nap > 16:
            problems.append(f"nap={self.spec.nap} (max 16)")
        if need > _MAX_CTA_THREADS:
            problems.append(f"d_out={self.spec.d_out} needs {need} threads per CTA at vec {vec} (limit {_MAX_CTA_THREADS})")
        if problems:
            raise RuntimeError(f"{type(self).__name__}: not a case the fused kernels cover: {'; '.join(problems)}. "
                               f"Construct ConfidenceLUT (the reference class) for this case.")
        if fused_confidence_ext() is None:
            _fallback.raise_unavailable(type(self).__name__, EXT_NAME)

    def _keep_flags(self, B: int, device):
        """Table-dropout keep flags ``[B, G, tph]`` bool (TRAIN + grad only, else None); kernels scale by 1/keep."""
        if not (self.training and self.table_dropout_rate > 0.0 and torch.is_grad_enabled()):
            return None
        return torch.rand(B, self.spec.n_groups, self.spec.tph, device=device) < (1.0 - self.table_dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward_impl(x)

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        self._check_input(x)
        self._require_cuda(x)
        # [B, G, d_in]; when every group reads its own head (h_in == n_groups) the in_head gather is the identity and
        # is skipped (no copy of the input). Addressing runs in fp32: a bf16 input goes to the kernels as is and is
        # upconverted while each row is staged (exact); fp16 is cast here. grad x comes back in the input dtype.
        # A gather (h_in != n_groups) can send one input head to several groups: cast a low-precision input to fp32
        # BEFORE it, so those groups' input gradients sum in fp32 and are rounded to the input dtype once.
        zin = x if self.spec.h_in == self.spec.n_groups else \
            (x if x.dtype == torch.float32 else x.float())[:, self.in_head, :]
        z = (zin if zin.dtype == torch.bfloat16 else zin.float()).contiguous()
        # bf16 output straight from the kernel (one rounding, as the cast it replaces) when groups map 1:1 onto output
        # heads; a fan-in sums the groups in _route first, so it keeps the fp32 output and casts once after the sum.
        out_bf16 = z.dtype == torch.bfloat16 and self.spec.h_out == self.spec.n_groups
        keep = self._keep_flags(x.shape[0], x.device)
        keep_scale = 1.0 / (1.0 - self.table_dropout_rate)
        log_tau = self.log_read_tau if self.read_top_n == 2 else self.confidence_log_beta
        grp = _FusedConfidence.apply(z, self.weights, self.confidence_log_beta.float(),
                                     self.confidence_log_gamma.float(), log_tau.float(),
                                     self._anc_a16, self._anc_b16, keep, keep_scale, self.spec.nap, self.cmp_eps,
                                     self.read_top_n, self.knobs, out_bf16)
        return self._route(grp, x).to(x.dtype)
