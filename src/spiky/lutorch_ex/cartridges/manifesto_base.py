"""Shared structure for the Manifesto cartridge family (hard and soft variants).

Both Manifesto cartridges share everything except the final step — how the addressed
cell ``c_t`` is combined with its least-confident-bit-flip neighbour ``c_t'``:

* :class:`~spiky.lutorch_ex.cartridges.manifesto_hard.ManifestoHardLUT` — hard value with
  a straight-through surrogate gradient (gen-1 variant 1.1, ``smooth_mode=False``).
* :class:`~spiky.lutorch_ex.cartridges.manifesto_soft.ManifestoSoftLUT` — the ``(1-U)``/``U``
  two-cell blend as both value and gradient (gen-1 variant 1.2, ``smooth_mode=True``).

Everything else lives here and is identical between them:

- **Anchors**: canonical full coverage. In ``anchor_mode == "pairs"`` (default) frozen
  index buffers ``anchor_a``/``anchor_b`` of shape ``[G, tph, nap]``
  (``anchors.canonical_full_coverage_pairs``); in ``anchor_mode == "single"`` only
  ``anchor_a`` (``anchors.canonical_full_coverage_singles``, ``anchor_b is None``).
- **Addressing**: MSB-first sign-bit packing (``addressing.msb_first_powers``). Each table's
  ``nap`` margins — ``u_j = z[a_j] - z[b_j]`` in pairs mode, ``u_j = z[a_j]`` (single anchor
  vs zero) in single mode — give sign bits ``[u_j > eps]`` packed (bit 0 = high bit) into
  the cell index ``c_t``; the neighbour ``c_t' = c_t`` with the bit of the least-confident
  margin ``j* = argmin_j |u_j|`` flipped. Only the margin changes between modes; the whole
  two-cell structure and routing below are identical, so every cartridge supports both.
- **Head routing** (the shape-contract invariant): ``G = max(h_in, h_out)`` groups, group
  ``g`` reads input head ``g % h_in`` and writes output head ``g % h_out``, groups summed
  into the output (fan-in sums all groups into head 0).

Subclasses implement only :meth:`_combine`.
"""
from __future__ import annotations

import os
from abc import abstractmethod
from typing import Optional

import torch
import torch.nn as nn

from ..addressing import msb_first_powers
from ..anchors import canonical_full_coverage_pairs, canonical_full_coverage_singles
from ..lut_base import MultiHeadLUT
from ..lut_spec import LUTSpec
from ._fused_ops import _global_cells

# Compile the hot forward on CUDA by default (the convention for all cartridges); eager on
# CPU, where torch.compile overhead isn't worth it. LUTORCH_EX_NO_COMPILE=1 disables it.
_COMPILE_ENABLED = os.environ.get("LUTORCH_EX_NO_COMPILE", "0") != "1" and hasattr(torch, "compile")

_LOW_PRECISION = (torch.bfloat16, torch.float16)


class ManifestoLUT(MultiHeadLUT):
    """Base for the Manifesto cartridges: addressing + two-cell structure + routing.

    Abstract: subclasses supply :meth:`_combine` to turn the two addressed cells
    (``y_hard`` = ``W[c_t]``, ``y_alt`` = ``W[c_t']``) and the deciding margin ``u_star``
    into the per-table output.

    Dtype contract: this base (the pure cartridges' path) carries **no mixed-precision
    handling** — it reads, reduces, and routes in the weight/input dtype, and supports only
    float32/float64. Handed bf16/fp16 params or inputs it **raises** (see :meth:`forward`)
    rather than run lossy low-precision math. Low precision (fp32 addressing + fp32-accumulated
    reads) is a fused-cartridge feature
    (:class:`~spiky.lutorch_ex.cartridges.fused_manifesto_hard.FusedManifestoHardLUT` /
    :class:`~spiky.lutorch_ex.cartridges.fused_manifesto_soft.FusedManifestoSoftLUT`), which
    override :meth:`_supports_low_precision`.
    """

    def __init__(
        self,
        spec: LUTSpec,
        *,
        seed: int = 0,
        weight_init_std: float = 1e-3,
        cmp_eps: float = 0.0,
        device: Optional[torch.device] = None,
        **unused,
    ):
        super().__init__(spec)
        G, tph, nap, d_in, d_out = (
            spec.n_groups, spec.tph, spec.nap, spec.d_in, spec.d_out,
        )
        self.single = spec.anchor_mode == "single"
        min_d = 1 if self.single else 2
        if d_in < min_d:
            raise ValueError(
                f"Manifesto cartridges need d_in >= {min_d} for anchor_mode={spec.anchor_mode!r}, "
                f"got {d_in}"
            )
        self.cmp_eps = float(cmp_eps)

        # Fixed anchors per (group, table): canonical full-coverage policy. In "pairs" mode
        # distinct canonical a<b pairs per table (covering the C(d_in,2) pool); in "single"
        # mode distinct single coordinates per table (covering the d_in coordinates). Frozen
        # buffers. anchor_b is None in single mode (each bit tests one coordinate vs zero).
        if self.single:
            a = canonical_full_coverage_singles(d_in, G, tph, nap, seed=seed)
            self.register_buffer("anchor_a", a)
            self.anchor_b = None
        else:
            a, b = canonical_full_coverage_pairs(d_in, G, tph, nap, seed=seed)
            self.register_buffer("anchor_a", a)
            self.register_buffer("anchor_b", b)
        # MSB-first bit weights: pair 0 -> high bit 2**(nap-1).
        self.register_buffer("powers", msb_first_powers(nap))
        # Group -> input/output head maps (the routing invariant).
        self.register_buffer("in_head", torch.arange(G, dtype=torch.long) % spec.h_in)
        self.register_buffer("out_head", torch.arange(G, dtype=torch.long) % spec.h_out)

        # Learnable cell tables: W[g, t, c, :], c in [0, K).
        wgen = torch.Generator().manual_seed(seed)
        w = torch.randn(G, tph, spec.n_cells, d_out, generator=wgen) * weight_init_std
        self.weights = nn.Parameter(w)

        # Lazily-built torch.compile of the forward, used only on CUDA (see forward()).
        self._compiled = None
        # Lazily-built torch.compile of the addressing, for the native TRAIN path (see _addr()).
        self._compiled_addr = None

    def _read(self, idx: torch.Tensor) -> torch.Tensor:
        """Gather one cell row per table: ``W[g, t, idx[b,g,t]]`` -> ``[B, G, tph, d_out]``.

        Flat advanced-index gather into the reshaped weight table: the backward scatters the
        gradient into grad_W of shape [G*tph*K, d_out] (small), with NO [B,G,tph,K,d_out]
        intermediate — so peak memory is O(B*G*tph*d_out), independent of K (was the OOM).

        Operates in the weight dtype as-is — this pure path carries no mixed-precision handling
        (bf16/fp16 support lives in the fused cartridges). See the class docstring.
        """
        G, tph, K = self.spec.n_groups, self.spec.tph, self.spec.n_cells
        W2 = self.weights.reshape(G * tph * K, self.spec.d_out)
        return W2[_global_cells(idx, G, tph, K)]              # [B, G, tph, d_out]

    def _read_pair(self, c: torch.Tensor, c_alt: torch.Tensor):
        """Return ``(W[c_t], W[c_t'])``, each ``[B,G,tph,d_out]`` — two flat gathers, same as _read."""
        G, tph, K = self.spec.n_groups, self.spec.tph, self.spec.n_cells
        W2 = self.weights.reshape(G * tph * K, self.spec.d_out)
        return W2[_global_cells(c, G, tph, K)], W2[_global_cells(c_alt, G, tph, K)]

    def _needs_alt(self) -> bool:
        """Whether the alternative cell ``c_t'`` (and the uncertainty) is needed this call.

        Default ``True`` — the soft blend always needs it. The hard cartridge overrides this
        to ``self.training`` so eval reads only the single addressed cell.
        """
        return True

    def _route(self, grp_out: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Map per-group outputs ``[B, G, d_out]`` to ``[B, h_out, d_out]`` per the invariant."""
        spec = self.spec
        if spec.h_out == spec.n_groups:
            return grp_out                                    # bijection (per-head / fan-out): no scatter
        if spec.h_out == 1:
            return grp_out.sum(dim=1, keepdim=True)           # fan-in: plain sum over groups
        # Unreachable for valid specs (validated in LUTSpec); kept correct as a fallback.
        y = grp_out.new_zeros(grp_out.shape[0], spec.h_out, spec.d_out)
        return y.index_add_(1, self.out_head, grp_out)

    @abstractmethod
    def _combine(
        self, y_hard: torch.Tensor, y_alt: torch.Tensor, u_abs_star: torch.Tensor
    ) -> torch.Tensor:  # pragma: no cover - abstract
        """Combine the two addressed cells into the per-table output ``[B, G, tph, d_out]``.

        Called only when the alternative is needed (soft always; hard in training).

        Args:
            y_hard: ``W[c_t]`` — the hard-addressed cell, ``[B, G, tph, d_out]``.
            y_alt: ``W[c_t']`` — the least-confident-bit-flip neighbour, ``[B, G, tph, d_out]``.
            u_abs_star: magnitude ``|u_{j*}|`` of the deciding margin, ``[B, G, tph]`` (the
                rational uncertainty depends only on the magnitude).
        """
        raise NotImplementedError

    def _supports_low_precision(self) -> bool:
        """Whether this cartridge supports bf16/fp16 params/inputs. False on the pure base
        (see the class docstring); the fused cartridges override it to True."""
        return False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Low precision (bf16/fp16) is a fused-cartridge feature; the pure cartridges reject it
        # with a clear error rather than silently running lossy bf16 math.
        if not self._supports_low_precision() and (
            x.dtype in _LOW_PRECISION or self.weights.dtype in _LOW_PRECISION
        ):
            raise TypeError(
                f"{type(self).__name__} does not support low precision: got input dtype "
                f"{x.dtype} and weight dtype {self.weights.dtype}. bf16/fp16 is supported only by "
                "the fused cartridges (FusedManifestoHardLUT / FusedManifestoSoftLUT); keep this "
                "cartridge (and ProjectionMHL wrapping it) in float32/float64."
            )
        # Compile ONLY the EVAL forward on CUDA (built lazily on first such call, per instance).
        # torch.compile helps the eval/inference path, but hurts the train+backward step at the
        # training batch (memory-bound there — eager is fastest), so training forward and its
        # backward run eager. CPU is always plain eager (keeps the CPU tests unaffected).
        if _COMPILE_ENABLED and x.is_cuda and not self.training:
            if self._compiled is None:
                self._compiled = torch.compile(self._forward_impl, dynamic=True)
            return self._compiled(x)
        return self._forward_impl(x)

    def _addr(self, x: torch.Tensor):
        """Addressing for the native TRAIN path. Identical result to :meth:`_addresses`, but
        compiled with ``torch.compile`` on CUDA so inductor fuses the margin / sign-bit-pack /
        argmin steps instead of materialising the big ``[B, G, tph, nap]`` intermediates eagerly
        (~10x faster at the training batch: 4.4 ms -> 0.4 ms on the champion shape). Eager on CPU
        or when compile is disabled. NOT used on the eval path (the base ``forward`` already
        compiles the whole eval ``_forward_impl``), so there is no nested compile."""
        if _COMPILE_ENABLED and x.is_cuda:
            if self._compiled_addr is None:
                self._compiled_addr = torch.compile(self._addresses, dynamic=True)
            return self._compiled_addr(x)
        return self._addresses(x)

    def _addresses(self, x: torch.Tensor):
        """Shared addressing: input routing + per-table sign-bit address and its neighbour.

        Returns ``(z, u, c, j_star, u_abs_star, c_alt)``:
          z          [B, G, d_in]      per-group input slice (x routed by in_head),
          u          [B, G, tph, nap]  signed anchor-pair margins u_j = z[a_j] - z[b_j],
          c          [B, G, tph]       MSB-first sign-bit address,
          j_star     [B, G, tph]       least-confident pair (argmin|u_j|),
          u_abs_star [B, G, tph]       |u_{j*}|,
          c_alt      [B, G, tph]       c with the j* bit flipped.
        Used by both the pure forward and the fused cartridges.
        """
        self._check_input(x)  # [B, h_in, d_in]
        G, tph, nap = self.spec.n_groups, self.spec.tph, self.spec.nap
        B = x.shape[0]
        z = x[:, self.in_head, :]  # route each group to its input head -> [B, G, d_in]
        idx_a = self.anchor_a.reshape(1, G, tph * nap).expand(B, G, tph * nap)
        z_a = z.gather(2, idx_a).reshape(B, G, tph, nap)
        if self.single:
            # Single anchor vs zero: the margin is the coordinate itself (no partner).
            u = z_a
        else:
            idx_b = self.anchor_b.reshape(1, G, tph * nap).expand(B, G, tph * nap)
            z_b = z.gather(2, idx_b).reshape(B, G, tph, nap)
            u = z_a - z_b
        c = ((u > self.cmp_eps).to(torch.long) * self.powers).sum(dim=-1)  # MSB-first, stop-grad
        u_abs_star, j_star = u.abs().min(dim=-1)                           # |u_{j*}| and j*
        c_alt = c ^ self.powers[j_star]
        return z, u, c, j_star, u_abs_star, c_alt

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        z, u, c, j_star, u_abs_star, c_alt = self._addresses(x)
        if self._needs_alt():
            # Both cells read in ONE fused gather; combine per the cartridge.
            y_hard, y_alt = self._read_pair(c, c_alt)
            per_table = self._combine(y_hard, y_alt, u_abs_star)   # [B, G, tph, d_out]
        else:
            # Eval shortcut (hard cartridge): only the addressed cell matters — one gather.
            per_table = self._read(c)
        grp_out = per_table.sum(dim=2)  # sum over tph -> [B, G, d_out]
        return self._route(grp_out, x)
