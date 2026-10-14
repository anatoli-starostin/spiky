"""FusedConfidenceLUT — the fused twin of ConfidenceLUT: its forward and backward in hand-written CUDA
(``csrc/fused_confidence.cu``).

Same parameters, numerics, state_dict and interface as :class:`ConfidenceLUT` (read_top_n 1 and 2, pairs and single
anchors, table dropout, learnable or frozen β / γ / τ); both build on
:class:`~spiky.lutorch_ex.cartridges.confidence_base.ConfidenceBase`, neither inherits the other's forward. Instead of
torch.compile + Inductor (3 forward and 9 backward kernels at the d24 geometry), one forward and one backward kernel
each run the whole cartridge: addressing, confidence score, blend weight, score-weighted read, and on the backward the
weight-gradient scatter, the score / blend / margin gradients into z, and the β / γ / τ gradients. The per-table
address, score and margins never touch global memory; the backward recomputes them from z.

Every launch decision Inductor makes heuristically is an explicit knob here (:class:`CudaKnobs`): threads per CTA
(forward and backward separately), rows per CTA, the vector width of the table loads / gradient scatters, and
vector atomics. Defaults come from ``LUTORCH_EX_CONF_CUDA_*`` environment variables and can be changed per instance
through ``cartridge.knobs``.

The kernels are its only implementation. A case they do not cover (a CPU input, a float64 input, a float16
or float64 table, nap > 16, d_in beyond int16 anchors, a d_out one CTA cannot cover) raises, naming the case, and so
does a missing extension, naming the cause; :class:`ConfidenceLUT` runs everywhere else.

bfloat16 / float16 (like the other fused cartridges): addressing runs in float32 (so the discrete bit decisions don't
drift), every read and reduction accumulates in float32, and the output is rounded to the input dtype once. A
bfloat16 input is read by the kernels directly (upconverted while each row is staged) and, when groups map 1:1 onto
output heads, the kernel stores the bfloat16 output itself (a fan-in keeps the float32 output, sums the groups, then
casts); both are bitwise identical to casting outside the kernel. A bfloat16 table stays bfloat16 in memory and is
upconverted in registers; its weight gradient accumulates in a float32 buffer and is cast to bfloat16 once, at the end.

Table dropout draws its keep flags with ``torch.rand`` (one bool per (b, g, t), like Inductor's separate RNG kernel)
and the kernels apply them; the stream of random numbers differs from the compiled ConfidenceLUT's, the
distribution does not.
"""
from __future__ import annotations

import torch

from . import _fallback
from ._cuda_ext import (_MAX_CTA_THREADS, CudaKnobs, Extension, _launch, _row_launch,  # noqa: F401
                        device_problem, not_covered, row_problem)
from ._dtypes import _CUDA_INPUT_DTYPES, _CUDA_TABLE_DTYPES, unsupported_dtype
from .confidence_base import ConfidenceBase

EXT = Extension("lutorch_ex_fused_confidence", ["fused_confidence.cu"])


class _FusedConfidence(torch.autograd.Function):
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
        """Raise unless the kernels can run this input: a RuntimeError naming every case they do not cover (no CUDA
        device or a CPU input, an input other than float32 / bfloat16 / float16, a table other than float32 /
        bfloat16, nap > 16, a d_out one CTA cannot cover), or the classified NativeUnavailableError when only the
        extension is missing."""
        problems = [p for p in (device_problem(x),) if p]
        if x.dtype not in _CUDA_INPUT_DTYPES:
            problems.append(unsupported_dtype("input", x.dtype, _CUDA_INPUT_DTYPES))
        if self.weights.dtype not in _CUDA_TABLE_DTYPES:
            problems.append(unsupported_dtype("table", self.weights.dtype, _CUDA_TABLE_DTYPES))
        if self.spec.nap > 16:
            problems.append(f"nap={self.spec.nap} (max 16)")
        problems += [p for p in (row_problem(self.spec.d_out, *_row_launch(self.knobs, self.spec.d_out,
                                                                            self.weights.dtype)),) if p]
        if problems:
            raise RuntimeError(not_covered(type(self).__name__, problems, "ConfidenceLUT"))
        if EXT.get() is None:
            _fallback.raise_unavailable(type(self).__name__, EXT.name)

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
        # is skipped (no copy of the input). Addressing runs in float32: a bfloat16 input goes to the kernels as is and
        # is upconverted while each row is staged (exact); float16 is cast here. grad x comes back in the input dtype.
        # A gather (h_in != n_groups) can send one input head to several groups: cast a low-precision input to float32
        # BEFORE it, so those groups' input gradients sum in float32 and are rounded to the input dtype once.
        zin = x if self.spec.h_in == self.spec.n_groups else \
            (x if x.dtype == torch.float32 else x.float())[:, self.in_head, :]
        z = (zin if zin.dtype == torch.bfloat16 else zin.float()).contiguous()
        # bfloat16 output straight from the kernel (one rounding, as the cast it replaces) when groups map 1:1 onto
        # output heads; a fan-in sums the groups in _route first, so it keeps the float32 output and casts once after
        # the sum.
        out_bf16 = z.dtype == torch.bfloat16 and self.spec.h_out == self.spec.n_groups
        keep = self._keep_flags(x.shape[0], x.device)
        keep_scale = 1.0 / (1.0 - self.table_dropout_rate)
        log_tau = self.log_read_tau if self.read_top_n == 2 else self.confidence_log_beta
        grp = _FusedConfidence.apply(z, self.weights, self.confidence_log_beta.float(),
                                     self.confidence_log_gamma.float(), log_tau.float(),
                                     self._anc_a16, self._anc_b16, keep, keep_scale, self.spec.nap,
                                     self.read_top_n, self.knobs, out_bf16)
        return self._route(grp, x).to(x.dtype)
