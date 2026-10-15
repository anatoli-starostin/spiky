"""ManifestoSoftLUT — the soft Manifesto reference cartridge, with either uncertainty function
(``uncertainty_func="rational"`` or ``"softsign"``).

"Soft forward, two-alternative blend." The soft counterpart of
:class:`~spiky.lutorch_ex.cartridges.manifesto_hard.ManifestoHardLUT`.

All shared structure (anchor pairs, MSB-first addressing, head routing, the two-cell read, the uncertainty ``U``)
lives in :class:`~spiky.lutorch_ex.cartridges.manifesto_base.ManifestoLUT`.

The math
========
Where the hard cartridge returns ``W[c_t]`` and only *back-propagates* through the blend, this cartridge makes the
blend itself the output, at both train and eval:

    y = sum_t [ (1 - U_t) * W_t[c_t] + U_t * W_t[c_t'] ]

with ``c_t`` the MSB-first sign-bit address, ``c_t'`` its least-confident-bit-flip neighbour
(``j* = argmin_j |u_j|`` over the margins ``u_j``), and the uncertainty ``U_t = U(|u_{j*}|)`` in ``(0, 1/2]``
(``uncertainty_func="rational"``: ``0.5 / (1 + |u_{j*}|)``; ``uncertainty_func="softsign"``: the learned-temperature
sigmoid).

The blend is an ordinary differentiable expression, so gradients flow into **both** cell rows (``(1 - U)`` into
``W[c_t]`` and ``U`` into ``W[c_t']``) and into the input (and, with ``softsign``, the two temperatures) via ``U``.
Only the discrete indices ``c_t, c_t', j*`` are non-differentiable. The value is continuous across a bit flip (at
``u = 0`` the two cells carry an equal share ``1/2`` and simply swap roles).

Read path: training uses one ``F.embedding_bag`` with ``per_sample_weights = [1-U, U]`` (read, scale and tph sum in one
kernel); eval uses the plain two-cell read and blend, compiled on CUDA (ReferenceForward). Backward is plain autograd.
bfloat16 / float16 run as float32 maths with one rounding per leaf (ReferenceForward).
"""
from __future__ import annotations

import torch

from ._fused_ops import fused_blend_read
from .manifesto_base import ManifestoLUT
from .reference_base import ReferenceForward


class ManifestoSoftLUT(ReferenceForward, ManifestoLUT):
    """Soft Manifesto reference cartridge; see module docstring for the math."""

    def _pure_blend(self, c, c_alt, U):
        y_hard, y_alt = self._read_pair(c, c_alt)
        return (y_hard + U.to(y_hard.dtype).unsqueeze(-1) * (y_alt - y_hard)).sum(dim=2)   # in the table's dtype

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        z, u, c, j_star, u_abs_star, c_alt = self._addresses(x)
        U = self._uncertainty(u_abs_star)
        if self.training:
            dmask = self._table_dropout_mask(x.shape[0], self.weights.device, self.weights.dtype)
            grp_out = fused_blend_read(self.weights, c, c_alt, U, drop_mask=dmask)
        else:
            grp_out = self._pure_blend(c, c_alt, U)
        return self._route(grp_out, x)
