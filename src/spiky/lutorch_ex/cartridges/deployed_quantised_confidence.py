"""DeployedQuantisedConfidenceLUT -- the inference-only int8 form of a QuantisedConfidenceLUT, built from its deployment
export (``QuantisedConfidenceLUT.from_deployment``). On CUDA one kernel (``csrc/pow2_int8_read.cu``, deploy_fwd, through
:mod:`._fused_quantised_cuda`); on the CPU the torch definition (``_pow2``). The quantised family is described in
:mod:`.quantised_confidence`.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from ..lut_spec import LUTSpec
from . import _fused_quantised_cuda, _pow2
from ._cuda_ext import forward_group_major
from ._dtypes import _LOW_PRECISION
from ._fused_ops import _global_cells
from .manifesto_base import resolve_index_dtype
from .quantised_confidence import QuantisedConfidenceLUT


class DeployedQuantisedConfidenceLUT(QuantisedConfidenceLUT):
    """Inference-only p2_int8 cartridge: holds the int8 ``packed`` table (NO float32 master Parameter),
    the anchors and the β/γ/τ scalars as buffers, and returns the RAW int32 accumulator (units 2^-6) as a
    float tensor; the per-(group,channel) output scale lives, pre-folded, in the ProjectionMHL decompress
    weight. The read width is the export's: ``meta["read_top_n"]`` (1 or 2), never a user choice.

    On CUDA the whole read is ONE kernel (``_fused_quantised_cuda`` / ``csrc/pow2_int8_read.cu``, deploy_fwd, the
    FusedQuantisedConfidenceLUT forward template specialised for the stored int8 rows and for the read width):
    the margins and integers in registers, int8 rows gathered and shifted, an exact int32 sum. It needs a float32
    input, nap <= 8 and d_in <= 32767; anything else raises, naming the case, and a missing extension raises
    the classified NativeUnavailableError (no fallback). On CPU the same integers are computed by the torch
    definition (``_pow2``: blend_exponents / single_exponent, the int8 shift-add read), float32 or float64.

    Lives next to :class:`QuantisedConfidenceLUT` so the general deployment module (``deploy.py``)
    holds no cartridge-specific knowledge; it is reached only via the format-tag registry (see
    :meth:`QuantisedConfidenceLUT.from_deployment`)."""

    _HAS_TRAINABLE_TABLE = False     # int8 buffers only: no regulariser applies (cell_tv_penalty skips it)

    def __init__(self, spec: LUTSpec, tensors: dict, meta: dict, device=None,
                 index_dtype: Optional[torch.dtype] = None):
        nn.Module.__init__(self)   # skip the float32-weight-allocating ManifestoLUT.__init__
        self.spec = spec
        self.index_dtype = resolve_index_dtype(index_dtype, spec)   # what ManifestoLUT.__init__ would set
        self._index_dtype_auto = index_dtype is None
        self.read_top_n = int(meta["read_top_n"])
        if self.read_top_n not in (1, 2):
            raise ValueError(f"DeployedQuantisedConfidenceLUT reads 1 or 2 cells per table; the export says "
                             f"read_top_n={self.read_top_n}")
        self.single = (spec.anchor_mode == "single")
        # Exports written before cmp_eps was removed carry meta["cmp_eps"]. Like every other meta key this class does
        # not use (cartridge, quant_cfg, spec, the wrapper's dims), it is ignored -- but only at 0.0, the only value
        # ever written: a non-zero threshold would select different cells, which this read no longer implements.
        if float(meta.get("cmp_eps", 0.0)) != 0.0:
            raise ValueError(f"this export has cmp_eps={meta['cmp_eps']}; the address threshold was removed (bits are "
                             "u > 0), so it cannot be loaded faithfully")
        self._quant = _pow2.resolve_quant_config(meta.get("quant_mode", "p2_int8"))
        self.table_dropout_rate = 0.0
        self._compiled_train = None
        dev = torch.device(device) if device is not None else None
        # int8 [n_tables, K, d_out]
        self.register_buffer("packed", tensors["packed"].to(dev) if dev else tensors["packed"])
        names = ["anchor_a", "powers", "in_head", "out_head", "confidence_log_beta", "confidence_log_gamma"]
        if self.read_top_n == 2:
            names.append("log_read_tau")                 # the blend temperature (n = 1 has none)
        if self.single:
            self.anchor_b = None                         # single-anchor mode: each bit tests one coordinate vs zero
        else:
            names.append("anchor_b")
        for b in names:
            t = tensors[b]
            self.register_buffer(b, t.to(dev) if dev else t)
        # int16 copies of the anchors for the kernel (the int64 buffers stay the source of truth).
        if spec.d_in <= torch.iinfo(torch.int16).max:
            self.register_buffer("_anc_a16", self.anchor_a.to(torch.int16).contiguous(), persistent=False)
            self.register_buffer("_anc_b16", None if self.single else self.anchor_b.to(torch.int16).contiguous(),
                                 persistent=False)

    def cell_tv(self) -> torch.Tensor:
        raise TypeError(f"{type(self).__name__} is inference-only: it holds int8 tables as buffers and no trainable "
                        "table, so the cell-TV regulariser does not apply to it (cell_tv_penalty skips it)")

    def decompress_scale(self) -> Optional[torch.Tensor]:
        return None   # the pow2 output scale is already folded (statically) into the stored decompress weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # inference
        # No float32 ``weights`` Parameter here (int8 ``packed`` instead), so the reference forward's low-precision
        # guard (which reads ``self.weights.dtype``) cannot run on it -- hence the override, rejecting bfloat16 /
        # float16 on the input. On CUDA this is a fused CUDA cartridge: float32 in / out only (float64 is refused by
        # require_deploy_cuda); the CPU path runs the torch definition, which also takes float64. The message is true on
        # both devices.
        if x.dtype in _LOW_PRECISION:
            raise TypeError(f"{type(self).__name__}: " + _fused_quantised_cuda.input_dtype_problem(
                x.dtype, "on the CPU the torch definition also takes torch.float64"))
        if x.is_cuda:
            self._check_input(x)
            ext = _fused_quantised_cuda.require_deploy_cuda(self, x)
            return self._forward_kernel(x, ext)
        return self._forward_impl(x)

    def _forward_kernel(self, x: torch.Tensor, ext) -> torch.Tensor:
        cfg = self._quant
        z = (x if self.spec.h_in == self.spec.n_groups else x[:, self.in_head, :]).contiguous()
        grp_out = ext.deploy_fwd(z, self._anc_a16, self._anc_b16, self.packed.reshape(-1, self.spec.d_out),
                                 self.confidence_log_beta.float().reshape(1),
                                 self.confidence_log_gamma.float().reshape(1),
                                 self.log_read_tau.float().reshape(1) if self.read_top_n == 2 else None,
                                 self.read_top_n, cfg["lo"], cfg["hi"], cfg["Q"],
                                 forward_group_major("QUANT", z.device))
        return self._route(grp_out, x)   # RAW (units 2^-6); scale folded into decompress

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        """The torch definition of the same read (the CPU path)."""
        cfg = self._quant
        spec = self.spec
        G, tph, K, d_out = spec.n_groups, spec.tph, spec.n_cells, spec.d_out
        B = x.shape[0]
        z, u, c, j_star, u_abs_star, c_alt = self._addresses(x)
        beta, gamma = self._betagamma(u.dtype)
        g0 = torch.zeros((), dtype=u.dtype, device=u.device)
        m = u.abs()
        gc = _global_cells(c, G, tph, K)
        packed = self.packed.reshape(G * tph * K, d_out)
        if self.read_top_n == 2:
            tau = self.log_read_tau.to(u.dtype).exp()
            q, k, skip, drop = _pow2.blend_exponents(m, u_abs_star.unsqueeze(-1), tau, g0, beta, gamma, cfg)
            gca = _global_cells(c_alt, G, tph, K)
            flat_idx = torch.stack([gc, gca], dim=-1).reshape(B, G, tph, 2)
            acc = _pow2.int_blend_read(packed, cfg["bits"], d_out, flat_idx, q, k, skip, drop)  # int32 [B,G,d_out]
        else:
            k, skip = _pow2.single_exponent(m, g0, beta, gamma, cfg)
            acc = _pow2.int_single_read(packed, cfg["bits"], d_out, gc.reshape(B, G, tph, 1), k, skip)
        return self._route(acc.to(x.dtype), x)   # RAW (units 2^-6); scale folded into decompress
