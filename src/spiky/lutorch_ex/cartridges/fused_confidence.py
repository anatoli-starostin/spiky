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

from ._cuda_ext import CudaKnobs
from ._fused_confidence_cuda import FusedConfidence, require_cuda
from .confidence_base import ConfidenceBase

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

    def _keep_flags(self, B: int, device):
        """Table-dropout keep flags ``[B, G, tph]`` bool (TRAIN + grad only, else None); kernels scale by 1/keep."""
        if not (self.training and self.table_dropout_rate > 0.0 and torch.is_grad_enabled()):
            return None
        return torch.rand(B, self.spec.n_groups, self.spec.tph, device=device) < (1.0 - self.table_dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward_impl(x)

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        self._check_input(x)
        require_cuda(self, x)
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
        grp = FusedConfidence.apply(z, self.weights, self.confidence_log_beta.float(),
                                    self.confidence_log_gamma.float(), log_tau.float(),
                                    self._anc_a16, self._anc_b16, keep, keep_scale, self.spec.nap,
                                    self.read_top_n, self.knobs, out_bf16)
        return self._route(grp, x).to(x.dtype)
