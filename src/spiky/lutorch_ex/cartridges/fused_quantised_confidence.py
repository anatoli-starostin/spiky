"""FusedQuantisedConfidenceLUT -- the fused twin of QuantisedConfidenceLUT: its forward and backward in hand-written
CUDA (``csrc/pow2_int8_read.cu`` through :mod:`._fused_quantised_cuda`). The quantised family, its maths and its
deployment export are described in :mod:`.quantised_confidence`.
"""
from __future__ import annotations

import torch

from . import _fused_quantised_cuda
from .confidence_base import ConfidenceBase
from .quantised_confidence import QuantisedScaffold


class FusedQuantisedConfidenceLUT(QuantisedScaffold, ConfidenceBase):
    """The fused twin of :class:`QuantisedConfidenceLUT`: one fused CUDA forward; see the module docstring.

    Covers read_top_n 1 and 2, pairs anchors, nap <= 8, d_in <= 32767, a CUDA float32 input and a float32 table.
    Anything else raises, naming the case and, where it runs the case, the reference class (not for a bfloat16 /
    float16 input or table, which the reference does not take either): d_in > 32767 as a ValueError at construction,
    every other case as a RuntimeError per call. A missing extension raises
    :class:`~spiky.lutorch_ex.cartridges._fallback.NativeUnavailableError` with its classified cause. Same parameters,
    state_dict and deployment export as the reference.

    Forward (``_fused_quantised_cuda.FusedQuantRead``): the table exponents, then ONE kernel for the whole read --
    addressing, the p2 integers and straight-through weights in registers, the fake-quant gather and the fixed-order
    reduction (the order is documented in ``csrc/pow2_int8_read.cu``). The margins never reach global memory. Backward:
    ONE hand-written kernel that re-stages z per CTA for the margins, as FusedConfidenceLUT's does, and reads the
    forward's per-table integers (one packed int32 per table) instead of recomputing them; its reduction orders are
    documented beside it."""

    def __init__(self, spec, *, read_top_n: int, quant_mode: str = "p2_int8", quant_overrides=None, **kw):
        super().__init__(spec, read_top_n=read_top_n, **kw)
        self._init_quant(quant_mode, quant_overrides)
        # Build the extension eagerly (this also registers the p2_scalars test op); never raises -- require_cuda raises
        # per call.
        _fused_quantised_cuda.ensure_registered()
        # int16 copies of the anchors for the kernel (the int64 buffers stay the state_dict's source of truth).
        if spec.d_in > torch.iinfo(torch.int16).max:
            raise ValueError(f"FusedQuantisedConfidenceLUT needs d_in <= 32767, got {spec.d_in}. "
                             "Construct QuantisedConfidenceLUT (the reference class) for this case.")
        self.register_buffer("_anc_a16", self.anchor_a.to(torch.int16).contiguous(), persistent=False)
        if self.anchor_b is not None:
            self.register_buffer("_anc_b16", self.anchor_b.to(torch.int16).contiguous(), persistent=False)
            # The backward's input-dim -> anchor-entry map (per group, CSR): entry e < tph * nap is anchor_a's e-th,
            # else anchor_b's (e - tph * nap)-th; increasing within each dim, the fixed order of the z gradient's sum.
            G, an = self.spec.n_groups, self.spec.tph * self.spec.nap
            dims = torch.cat([self.anchor_a.reshape(G, an), self.anchor_b.reshape(G, an)], dim=1)
            ent = torch.sort(dims, dim=1, stable=True).indices
            cnt = torch.stack([torch.bincount(r, minlength=self.spec.d_in) for r in dims])
            off = torch.cat([torch.zeros(G, 1, dtype=cnt.dtype, device=cnt.device), cnt.cumsum(1)], dim=1)
            self.register_buffer("_zoff", off.to(torch.int32).contiguous(), persistent=False)
            self.register_buffer("_zent", ent.to(torch.int32).contiguous(), persistent=False)
        else:
            self._anc_b16 = self._zoff = self._zent = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._check_input(x)
        _fused_quantised_cuda.require_cuda(self, x)
        return self._forward_impl(x)

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        z = (x if self.spec.h_in == self.spec.n_groups else x[:, self.in_head, :]).contiguous()
        dmask = self._table_dropout_mask(x.shape[0], x.device, x.dtype)
        grp_out = _fused_quantised_cuda.FusedQuantRead.apply(
            z, self.weights, self.confidence_log_beta.float(), self.confidence_log_gamma.float(),
            self.log_read_tau.float() if self.read_top_n == 2 else None, self._anc_a16, self._anc_b16, self._zoff,
            self._zent, None if dmask is None else dmask.contiguous(), self._quant, self.read_top_n,
            torch.is_grad_enabled())
        return self._route(grp_out, x)
