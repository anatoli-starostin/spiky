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

    # Whether to torch.compile the TRAIN forward too (not just eval). Default False: for the
    # Gen-1/Gen-2 cartridges the train step is memory-bound and eager is fastest. The Gen-3
    # confidence cartridges set this True -- their score/blend read-out fans out into ~10 tiny
    # elementwise kernels (abs/logsigmoid/exp/sum/sigmoid/gather/cat/psw) that inductor folds into
    # 1-2 (graph break at embedding_bag), cutting the train step AND its peak memory substantially
    # (matches the OLD LightMHL blend-compile lever). CUDA-only; see :meth:`forward`.
    _COMPILE_TRAIN: bool = False
    # `dynamic=` for the TRAIN compile. True (default) traces one shape-agnostic graph (no recompiles
    # across batch sizes). A cartridge whose hot train path needs the reference's exact fusion can set
    # None (torch.compile's auto: specialise on the first shape, matching OLD LightMHL's compile) --
    # that produces a cheaper fused backward for the quant monolith. Only consulted when _COMPILE_TRAIN.
    _COMPILE_TRAIN_DYNAMIC = True

    def __init__(
        self,
        spec: LUTSpec,
        *,
        seed: int = 0,
        weight_init_std: float = 1e-3,
        cmp_eps: float = 0.0,
        table_dropout_rate: float = 0.0,
        device: Optional[torch.device] = None,
        **unused,
    ):
        super().__init__(spec)
        G, tph, nap, d_in, d_out = (
            spec.n_groups, spec.tph, spec.nap, spec.d_in, spec.d_out,
        )
        # Table-level (whole-table) inverted dropout rate; 0 = off (default, every existing cartridge
        # byte-identical). See :meth:`_table_dropout_mask` / :meth:`_drop_tables`.
        self.table_dropout_rate = float(table_dropout_rate)
        if not 0.0 <= self.table_dropout_rate < 1.0:
            raise ValueError(f"table_dropout_rate must be in [0, 1), got {self.table_dropout_rate}")
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
        self._compiled = None          # eval forward
        self._compiled_train = None    # train forward (only when _COMPILE_TRAIN; Gen-3)
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

    # -- shared regularisation / dropout (all cartridges inherit) ----------------------------

    def cell_tv(self) -> torch.Tensor:
        """Hamming-1 adjacent-cell total-variation penalty on the cell table ``[G, tph, K, d_out]``.

        View the ``K = 2**nap`` cells of each table as the vertices of an nap-cube
        (``[G*tph, 2, 2, ..., 2, d_out]``) and sum ``(v_c - v_c')**2`` over every Hamming-1 pair
        (each counted once, via a size-2 ``diff`` along each of the nap bit-axes), then divide by the
        pair count ``n_tables * nap * 2**(nap-1)`` -- the mean over pairs (and tables) of
        ``||v_c - v_c'||**2`` (d_out summed in, not averaged). Bit-for-bit the reference
        ``LightMultiHeadLUT.cell_tv``. Differentiable w.r.t. ``weights``; adds no nodes to the forward
        (called only by the trainer when its TV lambda > 0)."""
        G, tph, K, d_out = self.weights.shape
        nap = self.spec.nap
        t = self.weights.reshape(G * tph, *([2] * nap), d_out)   # [T, 2,..,2, d_out] hypercube
        tv = t.new_zeros(())
        for ax in range(1, nap + 1):                             # one cube axis per bit
            d = t.diff(dim=ax)
            tv = tv + (d * d).sum()
        n_pairs = (G * tph) * nap * (1 << (nap - 1))             # total Hamming-1 pairs
        return tv / n_pairs

    def to_deployment(self) -> dict:
        """SupportsDeploymentExport (deployment-compaction axis): base no-op pass-through. A cartridge
        that cannot compactify reports ``format='dense'`` and export_deployment serialises its params
        via the normal (backbone) state_dict, unchanged. Compactifiable cartridges (e.g.
        :class:`QuantisedConfidenceLUT`) override this to emit a compact payload. Never raises."""
        return {"format": "dense", "tensors": {}, "meta": {"cartridge": type(self).__name__}}

    def _table_dropout_mask(self, B: int, device, dtype) -> Optional[torch.Tensor]:
        """Inverted table-dropout keep-mask ``[B, G, tph]`` (TRAIN + grad only; ``None`` otherwise).

        One independent Bernoulli per (sample, group, table): keep with prob ``1 - rate``, survivors
        scaled by ``1/(1-rate)`` so the expected read is unchanged; dropped tables contribute 0. Matches
        the reference ``head_dropout`` (whole-table granularity on the ``[B, G=heads, tph]`` axis).
        Uses ``torch.rand`` so it is compile-friendly and seed-reproducible through the global RNG."""
        if not (self.training and self.table_dropout_rate > 0.0 and torch.is_grad_enabled()):
            return None
        keep_prob = 1.0 - self.table_dropout_rate
        G, tph = self.spec.n_groups, self.spec.tph
        return (torch.rand(B, G, tph, device=device, dtype=dtype) < keep_prob).to(dtype) / keep_prob

    def _drop_tables(self, per_table: torch.Tensor) -> torch.Tensor:
        """Apply table dropout to a per-table tensor ``[B, G, tph, ...]`` (mask broadcast over the
        trailing dims). No-op at eval / rate 0. For reads whose per-table contribution is a tensor;
        reads that carry a per-sample-weight fold the mask into that weight instead (same effect)."""
        mask = self._table_dropout_mask(per_table.shape[0], per_table.device, per_table.dtype)
        if mask is None:
            return per_table
        return per_table * mask.reshape(mask.shape + (1,) * (per_table.dim() - mask.dim()))

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
        # Compile the forward on CUDA (built lazily on first such call, per instance). EVAL is
        # always compiled. TRAIN is compiled only for cartridges that opt in via _COMPILE_TRAIN
        # (the Gen-3 confidence line): there the score/blend read-out is a long chain of tiny
        # elementwise kernels that inductor fuses, cutting both the train step and its peak memory.
        # For the Gen-1/Gen-2 cartridges (_COMPILE_TRAIN False) the train step is memory-bound and
        # eager is fastest, so training stays eager. CPU is always plain eager (keeps CPU tests
        # eager / bit-identical). Eval and train use separate compiled objects (each traces its own
        # branch of _forward_impl).
        if _COMPILE_ENABLED and x.is_cuda:
            if not self.training:
                if self._compiled is None:
                    self._compiled = torch.compile(self._forward_impl, dynamic=True)
                return self._compiled(x)
            if self._COMPILE_TRAIN:
                if self._compiled_train is None:
                    self._compiled_train = torch.compile(self._forward_impl,
                                                         dynamic=self._COMPILE_TRAIN_DYNAMIC)
                return self._compiled_train(x)
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
        per_table = self._drop_tables(per_table)   # table dropout (no-op at eval / rate 0)
        grp_out = per_table.sum(dim=2)  # sum over tph -> [B, G, d_out]
        return self._route(grp_out, x)


def cell_tv_penalty(module: nn.Module) -> torch.Tensor:
    """Model-level Hamming-1 cell-TV penalty: the MEAN of :meth:`ManifestoLUT.cell_tv` over every
    ``ManifestoLUT`` cartridge inside ``module`` (equivalent to the reference model's
    ``lut_tv_penalty()``, which averages ``cell_tv()`` over its LUT layers). Scale by lambda and add
    to the loss, matching the reference ergonomics: ``(lam * cell_tv_penalty(model)).backward()``.
    Returns a 0-dim zero (no grad) when the module has no such cartridge."""
    tvs = [m.cell_tv() for m in module.modules() if isinstance(m, ManifestoLUT)]
    if not tvs:
        return torch.zeros(())
    return torch.stack(tvs).mean()
