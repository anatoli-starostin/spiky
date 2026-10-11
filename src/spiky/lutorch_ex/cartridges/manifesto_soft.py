"""ManifestoSoftLUT — the soft Manifesto reference cartridge (Gen-1 variant 1.2 with ``weight="rational"``, Gen-2
variant 2.4 with ``weight="softsign"``).

"Soft forward, two-alternative blend." The soft counterpart of
:class:`~spiky.lutorch_ex.cartridges.manifesto_hard.ManifestoHardLUT`.

All shared structure (anchor pairs, MSB-first addressing, head routing, the two-cell read, the neighbour weight
``w``) lives in :class:`~spiky.lutorch_ex.cartridges.manifesto_base.ManifestoLUT`.

The math
========
Where the hard cartridge returns ``W[c_t]`` and only *back-propagates* through the blend, this cartridge makes the
blend itself the output, at both train and eval:

    y = sum_t [ (1 - w_t) * W_t[c_t] + w_t * W_t[c_t'] ]

with ``c_t`` the MSB-first sign-bit address, ``c_t'`` its least-confident-bit-flip neighbour
(``j* = argmin_j |u_j|``), and the neighbour weight ``w_t = w(|u_{j*}|)`` in ``(0, 1/2]`` (``weight="rational"``:
``0.5 / (1 + |u_{j*}|)``; ``weight="softsign"``: the learned-temperature sigmoid).

The blend is an ordinary differentiable expression, so gradients flow into **both** cell rows (``(1 - w)`` into
``W[c_t]`` and ``w`` into ``W[c_t']``) and into the input (and, with ``softsign``, the two temperatures) via ``w``.
Only the discrete indices ``c_t, c_t', j*`` are non-differentiable. The value is continuous across a bit flip (at
``u = 0`` the two cells carry equal weight ``1/2`` and simply swap roles).

Read path: training uses one ``F.embedding_bag`` with ``per_sample_weights = [1-w, w]`` (read, scale and tph sum in one
kernel); eval uses the plain two-cell read and blend, compiled on CUDA (ReferenceForward). Backward is plain autograd.
fp32/fp64 only: the forward raises on bf16/fp16 (low precision is the fused cartridges' feature).
"""
from __future__ import annotations

import torch

from ._fused_ops import fused_blend_read
from .manifesto_base import ManifestoLUT
from .reference_base import ReferenceForward


class ManifestoSoftLUT(ReferenceForward, ManifestoLUT):
    """Soft Manifesto reference cartridge; see module docstring for the math."""

    def _pure_blend(self, c, c_alt, w):
        y_hard, y_alt = self._read_pair(c, c_alt)
        return (y_hard + w.unsqueeze(-1) * (y_alt - y_hard)).sum(dim=2)

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        z, u, c, j_star, u_abs_star, c_alt = self._addresses(x)
        w = self._blend_w(u_abs_star)
        if self.training:
            dmask = self._table_dropout_mask(x.shape[0], self.weights.device, self.weights.dtype)
            grp_out = fused_blend_read(self.weights, c, c_alt, w, drop_mask=dmask)
        else:
            grp_out = self._pure_blend(c, c_alt, w)
        return self._route(grp_out, x)
