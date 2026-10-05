"""Fused, GPU-efficient twins of the Gen-2 soft-sign cartridges.

``FusedSoftSignHardLUT`` / ``FusedSoftSignSmoothLUT`` are numerically equivalent to the pure
``SoftSignHardLUT`` / ``SoftSignSmoothLUT`` (the oracle) but dispatch the training step to the native
lprojection kernels, reusing the Gen-1 fused read / weight-grad / carrier machinery and
adding only the soft-sign-specific margin/temperature gradient (see ``_native_softsign``). The
native backward forms the weight gradient and the two carriers ``grad.W[c]`` / ``grad.W[c']`` in
one fused pass WITHOUT materialising the ``[B, G, tph, d_out]`` cell tensors — the whole point of
Option 2.

Dispatch (``backend`` forces a path; 'auto' is the hybrid):
* eval  -> pure compiled gather read (fastest on the inference path, inherited);
* train -> 'native' whenever native_available(): a CUDA device with the native ops loaded (any
  architecture the extension builds for -- e.g. H100 or RTX 5090), else 'tier1' (pure embedding_bag path,
  CPU / no ext). FusedSoftSignSmoothLUT overrides this: its 'auto' always trains on 'tier1' (native is no
  faster there); 'native' stays reachable with backend="native".

bf16/fp16: addressing runs in fp32 (so the discrete bit decisions don't drift), reads/reductions
accumulate in fp32, output cast back once. Kept only if it is a real H100 speedup (evaluated in
the bf16 grid); otherwise ``_supports_low_precision`` is flipped to False and these raise like the
pure cartridges. fp32/fp64 are the correctness oracle.
"""
from __future__ import annotations

import torch

from ._fused_ops import _acc_dtype, _global_cells, fused_blend_read, fused_hard_read, validate_backend
from ._native_ops import native_available
from ._native_softsign import NativeSoftSignHard, NativeSoftSignSmooth
from .softsign_base import SoftSignLUT

_LOW_PREC = (torch.bfloat16, torch.float16)


class FusedSoftSignLUT(SoftSignLUT):
    """Shared dispatch/addressing for the fused soft-sign twins. Abstract (no ``_forward_impl``)."""

    #: bf16/fp16 support — a real H100 speedup keeps it True, otherwise flipped to False (drop).
    _SUPPORTS_LOW_PRECISION = True

    #: Every backend the twins' _forward_impl dispatches on ('auto' picks one of the others per call).
    _BACKENDS = ("auto", "pure_eval", "tier1", "native")

    def __init__(self, spec, *, backend: str = "auto", **kw):
        validate_backend(type(self).__name__, backend, self._BACKENDS)
        super().__init__(spec, **kw)
        self.backend = backend

    def _supports_low_precision(self) -> bool:
        return self._SUPPORTS_LOW_PRECISION

    def _needs_alt(self) -> bool:
        return self.training

    def _combine(self, y_hard, y_alt, u_abs_star):  # pragma: no cover - forward is overridden
        raise NotImplementedError

    def _gc(self, c):
        """Flat per-(group,table) cell indices into W.reshape(G*tph*K, d_out)."""
        G, tph, K = self.spec.n_groups, self.spec.tph, self.spec.n_cells
        return _global_cells(c, G, tph, K)

    def _read_both(self, W, c, c_alt):
        """(W[c], W[c']) each [B,G,tph,d_out] from a reshaped weight table W [G,tph,K,d_out]."""
        W2 = W.reshape(-1, W.shape[-1])
        return W2[self._gc(c)], W2[self._gc(c_alt)]

    def _star(self, z, u, j_star):
        """(a_local, b_local, a_global, b_global, u_signed_star) at the deciding bit j*.

        Single mode: b is a placeholder (= a); the input grad scatters to a only (honoured by
        the native tail via ``single``)."""
        G, tph, nap, d_in = self.spec.n_groups, self.spec.tph, self.spec.nap, self.spec.d_in
        B = z.shape[0]
        je = j_star.unsqueeze(-1)
        u_signed = u.gather(-1, je).squeeze(-1)
        al = self.anchor_a.unsqueeze(0).expand(B, G, tph, nap).gather(-1, je).squeeze(-1)
        bl = al if self.single else (
            self.anchor_b.unsqueeze(0).expand(B, G, tph, nap).gather(-1, je).squeeze(-1)
        )
        off = torch.arange(G, device=z.device).view(1, G, 1) * d_in
        return al, bl, al + off, bl + off, u_signed

    def _pick(self, x: torch.Tensor) -> str:
        if not self.training:
            return "pure_eval"
        if native_available(x.device):
            return "native"
        return "tier1"

    def _temps(self, dtype):
        """The two exp'd temperatures in ``dtype`` (addressing dtype), differentiable wrt the
        log-parametrised leaves — passed to the native Function so it owns their gradient."""
        return (self.log_soft_score_temp.to(dtype).exp(), self.log_select_temp.to(dtype).exp())


class FusedSoftSignHardLUT(FusedSoftSignLUT):
    """Native fused twin of SoftSignHardLUT (variant 2.3)."""

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        low = x.dtype in _LOW_PREC
        z, u, c, j_star, u_abs_star, c_alt = self._addresses(x.float() if low else x)
        be = self._pick(x) if self.backend == "auto" else self.backend
        dmask = self._table_dropout_mask(x.shape[0], self.weights.device, _acc_dtype(self.weights.dtype))
        if be == "pure_eval":
            W = self.weights.float() if low else self.weights
            rd = self._read(c) if not low else None
            grp_out = (rd.sum(dim=2, dtype=_acc_dtype(self.weights.dtype))
                       if not low else fused_hard_read(W, c))
        elif be == "native":
            al, bl, ag, bg, us = self._star(z, u, j_star)
            zc = x[:, self.in_head, :] if low else z
            t_soft, t_select = self._temps(us.dtype)
            grp_out = NativeSoftSignHard.apply(self.weights, zc, c, c_alt, us, ag, bg,
                                               self.single, t_soft, t_select, dmask)
        else:  # tier1: pure embedding_bag straight-through (CPU / no native)
            W = self.weights.float() if low else self.weights
            w = self._blend_w(u_abs_star)
            y_c = fused_hard_read(W, c, drop_mask=dmask)
            y_hard_pt, y_alt_pt = self._read_both(W, c, c_alt)
            diff = (y_alt_pt - y_hard_pt).detach()
            sw = w.unsqueeze(-1) * diff
            if dmask is not None:
                sw = sw * dmask.unsqueeze(-1)
            surr = sw.sum(dim=2)
            grp_out = y_c + (surr - surr.detach())
        return self._route(grp_out, x).to(x.dtype)


class FusedSoftSignSmoothLUT(FusedSoftSignLUT):
    """Native fused twin of SoftSignSmoothLUT (variant 2.4)."""

    def _pick(self, x: torch.Tensor) -> str:
        # Smooth override: the pure embedding_bag train read is already fused and lighter on peak
        # memory at large batch, and the native soft-sign path is a wash (no speedup) for smooth —
        # so "auto" prefers tier1 (embedding_bag). The native path stays reachable via an explicit
        # backend="native". (Hard keeps the base _pick: its native path is a clear win.)
        return "pure_eval" if not self.training else "tier1"

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        low = x.dtype in _LOW_PREC
        z, u, c, j_star, u_abs_star, c_alt = self._addresses(x.float() if low else x)
        w = self._blend_w(u_abs_star)
        be = self._pick(x) if self.backend == "auto" else self.backend
        dmask = self._table_dropout_mask(x.shape[0], self.weights.device, _acc_dtype(self.weights.dtype))
        if be == "pure_eval":
            W = self.weights.float() if low else self.weights
            y_hard, y_alt = self._read_both(W, c, c_alt)
            blend = y_hard + w.unsqueeze(-1) * (y_alt - y_hard)
            if dmask is not None:
                blend = blend * dmask.unsqueeze(-1)
            grp_out = blend.sum(dim=2, dtype=_acc_dtype(self.weights.dtype))
        elif be == "native":
            al, bl, ag, bg, us = self._star(z, u, j_star)
            zc = x[:, self.in_head, :] if low else z
            t_soft, t_select = self._temps(us.dtype)
            grp_out = NativeSoftSignSmooth.apply(self.weights, zc, c, c_alt, us, ag, bg,
                                                 self.single, t_soft, t_select, w.detach(), dmask)
        else:  # tier1: pure fused blend read (CPU / no native)
            W = self.weights.float() if low else self.weights
            grp_out = fused_blend_read(W, c, c_alt, w, drop_mask=dmask)
        return self._route(grp_out, x).to(x.dtype)
