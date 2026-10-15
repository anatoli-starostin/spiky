"""Shared structure for the Manifesto cartridge family (hard and soft variants, two uncertainty functions).

The Manifesto cartridges read, per table, the addressed cell ``c_t`` and its least-confident-bit-flip neighbour
``c_t'`` and blend them, the neighbour's share being the uncertainty ``w``, a function of the deciding margin
``|u_{j*}|`` chosen by ``uncertainty_func``:

* ``uncertainty_func="rational"``: ``w = U = 0.5 / (1 + |u_{j*}|)``, no parameters;
* ``uncertainty_func="softsign"``: ``rho = |u_{j*}| / (T_soft + |u_{j*}|)``, ``w = sigmoid(-2 rho / T_select)``, with
  two learned per-layer temperatures stored log-parametrised (``log_soft_score_temp``, ``log_select_temp``;
  initialised at 0.5).

Both are in ``(0, 1/2]``, ``1/2`` at ``u = 0`` and tending to 0 as the margin grows. The variants:

* :class:`~spiky.lutorch_ex.cartridges.manifesto_hard.ManifestoHardLUT` — hard value with a straight-through
  surrogate gradient (the gradient of the blend);
* :class:`~spiky.lutorch_ex.cartridges.manifesto_soft.ManifestoSoftLUT` — the ``(1-w)`` / ``w`` two-cell blend as both
  value and gradient.

Everything else lives here and is identical between them:

- **Anchors**: canonical full coverage. In ``anchor_mode == "pairs"`` (default) frozen
  index buffers ``anchor_a``/``anchor_b`` of shape ``[G, tph, nap]``
  (``anchors.canonical_full_coverage_pairs``); in ``anchor_mode == "single"`` only
  ``anchor_a`` (``anchors.canonical_full_coverage_singles``, ``anchor_b is None``).
- **Addressing**: MSB-first sign-bit packing (``addressing.msb_first_powers``). Each table's
  ``nap`` margins — ``u_j = z[a_j] - z[b_j]`` in pairs mode, ``u_j = z[a_j]`` (single anchor
  vs zero) in single mode — give sign bits ``[u_j > 0]`` packed (bit 0 = high bit) into
  the cell index ``c_t``; the neighbour ``c_t' = c_t`` with the bit of the least-confident
  margin ``j* = argmin_j |u_j|`` flipped. Only the margin changes between modes; the whole
  two-cell structure and routing below are identical, so every cartridge supports both.
- **Head routing** (the shape-contract invariant): ``G = max(h_in, h_out)`` groups, group
  ``g`` reads input head ``g % h_in``. A head always sums its own tables; across heads the
  outputs are routed one-to-one and concatenated, and summed only in fan-in. Concretely,
  ``per_table.sum(dim=2)`` gives ``[B, G, d_out]``; :meth:`_route` returns it unchanged when
  ``h_out == G`` (per-head; fan-out, where every group reads input head 0) and
  ``.sum(dim=1, keepdim=True)`` -> ``[B, 1, d_out]`` when ``h_out == 1`` (fan-in);
  ``ProjectionMHL.forward`` then reshapes ``[B, h_out, d_out]`` to ``[B, h_out * d_out]``,
  the concatenation.

Subclasses implement :meth:`_forward_impl`.
"""
from __future__ import annotations

import math
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
from .uncertainty import rational_uncertainty

# Compile the hot forward on CUDA by default (the convention for all cartridges); eager on
# CPU, where torch.compile overhead isn't worth it. LUTORCH_EX_NO_COMPILE=1 disables it.
_COMPILE_ENABLED = os.environ.get("LUTORCH_EX_NO_COMPILE", "0") != "1" and hasattr(torch, "compile")

#: The uncertainty functions (see the module docstring).
UNCERTAINTY_FUNCS = ("rational", "softsign")

# Cell-index dtypes (``index_dtype``, every cartridge). The index the reads consume is the FLAT row index into the
# [n_groups * tph * 2**nap, d_out] table, up to n_groups * tph * 2**nap - 1 (262,143 at h=16, tph=64, nap=8), not the
# within-table cell id < 2**nap; and torch's index ops (embedding_bag, index_select, index_add_, indexing) accept
# only int32 / int64. So int32 is the only narrowing; int16/int8 cannot represent the flat index and are rejected.
INDEX_DTYPES = (torch.int64, torch.int32)


def check_index_dtype(index_dtype, spec) -> torch.dtype:
    if index_dtype not in INDEX_DTYPES:
        raise ValueError(f"index_dtype must be one of {INDEX_DTYPES}, got {index_dtype!r}")
    max_flat = spec.n_groups * spec.tph * spec.n_cells - 1
    if max_flat > torch.iinfo(index_dtype).max:
        raise ValueError(f"index_dtype={index_dtype} cannot hold the flat cell index (max {max_flat})")
    return index_dtype


def resolve_index_dtype(index_dtype: Optional[torch.dtype], spec) -> torch.dtype:
    """``None`` (the default) -> int32 when the flat cell index fits it, else int64. An explicit dtype is honoured
    exactly (an explicit int32 that does not fit raises, from :func:`check_index_dtype`). Only the STORED cell-index
    tensors use this dtype; offset arithmetic and the powers buffer stay int64."""
    if index_dtype is None:
        fits = spec.n_groups * spec.tph * spec.n_cells - 1 <= torch.iinfo(torch.int32).max
        return torch.int32 if fits else torch.int64
    return check_index_dtype(index_dtype, spec)


# Auto mode also widens per call: some PyTorch kernels use the index dtype for POSITIONS, not just values -- the CUDA
# embedding_bag forward device-asserts above 2**31 - 1 index entries with int32 (measured, torch 2.9.1), so an
# auto-int32 cartridge reads in int64 for a call whose read could exceed it (B * n_groups * tph * 2 entries, 2 = the
# most cells any read takes per table). The separate, much lower embedding_bag BACKWARD size limit is a documented
# upstream constraint, not enforced here (_fused_ops.py: "PyTorch's embedding_bag backward size limit").
_INT32_MAX_ENTRIES = torch.iinfo(torch.int32).max


class ManifestoLUT(MultiHeadLUT):
    """Base for the Manifesto cartridges: addressing + two-cell structure + routing.

    Scaffolding only: state (tables, anchors, the soft-sign temperatures), validation, the
    addressing and the uncertainty (:meth:`_blend_w`), shared by the reference and the fused classes. It has NO
    runnable forward: a reference class gets its forward from
    :class:`~spiky.lutorch_ex.cartridges.reference_base.ReferenceForward`, a fused class defines its own kernel call, so
    a fused class has nothing to fall back to. ``uncertainty_func`` selects the uncertainty function; the temperature
    arguments exist only with ``uncertainty_func="softsign"``.
    """
    # Whether the cartridge has the uncertainty function (uncertainty_func=, the soft-sign temperatures). False for the
    # confidence family, which scores its cells instead.
    _HAS_UNCERTAINTY_FUNC: bool = True
    # Whether the cartridge has a trainable float table, the one the regularisers act on. False for the inference-only
    # DeployedQuantisedConfidenceLUT; cell_tv_penalty skips such a cartridge.
    _HAS_TRAINABLE_TABLE: bool = True

    def __init__(
        self,
        spec: LUTSpec,
        *,
        seed: int = 0,
        weight_init_std: float = 1e-3,
        table_dropout_rate: float = 0.0,
        index_dtype: Optional[torch.dtype] = None,
        uncertainty_func: str = "rational",
        soft_score_temp: Optional[float] = None,
        select_temp: Optional[float] = None,
        learnable_temps: Optional[bool] = None,
        **unknown,
    ):
        # Every keyword is consumed by this class or a subclass on the way here; anything left is a typo or an option
        # this cartridge does not have (silently ignoring it once turned a misspelt table_dropout_rate into no dropout).
        if not self._HAS_UNCERTAINTY_FUNC:     # ConfidenceLUT and its family score their cells; no uncertainty function
            given = dict(uncertainty_func=uncertainty_func if uncertainty_func != "rational" else None,
                         soft_score_temp=soft_score_temp, select_temp=select_temp, learnable_temps=learnable_temps)
            unknown.update({k: v for k, v in given.items() if v is not None})
        if unknown:
            raise TypeError(f"{type(self).__name__}: unexpected keyword argument(s) {sorted(unknown)}")
        if uncertainty_func not in UNCERTAINTY_FUNCS:
            raise ValueError(f"{type(self).__name__}: uncertainty_func must be one of {UNCERTAINTY_FUNCS}, "
                             f"got {uncertainty_func!r}")
        temps = dict(soft_score_temp=soft_score_temp, select_temp=select_temp, learnable_temps=learnable_temps)
        if uncertainty_func == "rational" and any(v is not None for v in temps.values()):
            given = sorted(k for k, v in temps.items() if v is not None)
            raise ValueError(f"{type(self).__name__}: {given} apply only to uncertainty_func='softsign'")
        super().__init__(spec)
        self.uncertainty_func = uncertainty_func
        G, tph, nap, d_in, d_out = (
            spec.n_groups, spec.tph, spec.nap, spec.d_in, spec.d_out,
        )
        self.index_dtype = resolve_index_dtype(index_dtype, spec)
        self._index_dtype_auto = index_dtype is None     # auto may widen per call (see _INT32_MAX_ENTRIES)
        # Table-level (whole-table) inverted dropout rate; 0 = off (default, every existing cartridge
        # byte-identical). See :meth:`_table_dropout_mask`.
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
        # out_head is not read by any forward (_route needs only h_out); it stays a buffer because it is in every
        # saved state_dict and in the deployment export.
        self.register_buffer("out_head", torch.arange(G, dtype=torch.long) % spec.h_out)

        # Learnable cell tables W[g, t, c, :], c in [0, K). Default init: per-head CPU generator seeded
        # (seed + h + 1), Uniform[-weight_init_std, +weight_init_std] (realised std = weight_init_std /
        # sqrt(3) ~= 5.77e-4 at the 1e-3 default), so a given seed reproduces the same tables bit-for-bit.
        blocks = [(torch.rand(tph, spec.n_cells, d_out,
                              generator=torch.Generator().manual_seed(seed + h + 1)) - 0.5) * (2.0 * weight_init_std)
                  for h in range(G)]
        self.weights = nn.Parameter(torch.stack(blocks, dim=0))   # [G, tph, K, d_out]

        # uncertainty_func="softsign": the two learned temperatures, log-parametrised so they stay positive.
        if uncertainty_func == "softsign":
            ls = math.log(float(0.5 if soft_score_temp is None else soft_score_temp))
            lx = math.log(float(0.5 if select_temp is None else select_temp))
            if learnable_temps is None or learnable_temps:
                self.log_soft_score_temp = nn.Parameter(torch.tensor(ls))
                self.log_select_temp = nn.Parameter(torch.tensor(lx))
            else:
                self.register_buffer("log_soft_score_temp", torch.tensor(ls))
                self.register_buffer("log_select_temp", torch.tensor(lx))

        # The reference forward's lazily-built torch.compile objects (ReferenceForward; unused by the fused classes).
        self._compiled = None          # eval forward
        self._compiled_train = None    # train forward (only when _COMPILE_TRAIN)
        self._compiled_tv = None       # cell_tv (every cartridge, CUDA)

    def _read(self, idx: torch.Tensor) -> torch.Tensor:
        """Gather one cell row per table: ``W[g, t, idx[b,g,t]]`` -> ``[B, G, tph, d_out]``.

        Flat advanced-index gather into the reshaped weight table: the backward scatters the
        gradient into grad_W of shape [G*tph*K, d_out] (small), with NO [B,G,tph,K,d_out]
        intermediate — so peak memory is O(B*G*tph*d_out), independent of K (was the OOM).

        Operates in the weight dtype as-is — this pure path carries no mixed-precision handling
        (bfloat16 / float16 support lives in the fused cartridges). See the class docstring.
        """
        G, tph, K = self.spec.n_groups, self.spec.tph, self.spec.n_cells
        W2 = self.weights.reshape(G * tph * K, self.spec.d_out)
        return W2[_global_cells(idx, G, tph, K)]              # [B, G, tph, d_out]

    def _read_pair(self, c: torch.Tensor, c_alt: torch.Tensor):
        """Return ``(W[c_t], W[c_t'])``, each ``[B,G,tph,d_out]`` — two flat gathers, same as _read."""
        G, tph, K = self.spec.n_groups, self.spec.tph, self.spec.n_cells
        W2 = self.weights.reshape(G * tph * K, self.spec.d_out)
        return W2[_global_cells(c, G, tph, K)], W2[_global_cells(c_alt, G, tph, K)]

    def _blend_w(self, u_abs_star: torch.Tensor) -> torch.Tensor:
        """The uncertainty ``w`` (the neighbour cell ``c'``'s share of the blend), in ``(0, 1/2]``, from ``|u_{j*}|``
        ``[B, G, tph]``.

        ``rational``: ``U = 0.5 / (1 + |u*|)``. ``softsign``: ``rho = |u*| / (T_s + |u*|)``,
        ``w = sigmoid(-2 rho / T_sel)``; the temperatures are taken in the margin's dtype (the addressing
        precision: float32 for a bfloat16 fused cartridge), so the blend runs there."""
        if self.uncertainty_func == "rational":
            return rational_uncertainty(u_abs_star)
        t_s = self.log_soft_score_temp.to(u_abs_star.dtype).exp()
        t_sel = self.log_select_temp.to(u_abs_star.dtype).exp()
        rho = u_abs_star / (t_s + u_abs_star)
        return torch.sigmoid(-2.0 * rho / t_sel)

    def _softsign_temps(self, dtype):
        """``(T_soft, T_select)`` in ``dtype``, differentiable wrt the log-parametrised leaves
        (``uncertainty_func="softsign"``):
        for the fused CUDA backward, which owns the temperatures' gradient."""
        return self.log_soft_score_temp.to(dtype).exp(), self.log_select_temp.to(dtype).exp()

    def _route(self, grp_out: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Map per-group outputs ``[B, G, d_out]`` to ``[B, h_out, d_out]`` per the invariant."""
        # LUTSpec allows only these two cases: h_out == n_groups (per-head, fan-out) or h_out == 1 (fan-in).
        if self.spec.h_out == self.spec.n_groups:
            return grp_out                                    # bijection (per-head / fan-out): no scatter
        return grp_out.sum(dim=1, keepdim=True)               # fan-in: plain sum over groups

    @abstractmethod
    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - abstract
        """``[B, h_in, d_in]`` -> ``[B, h_out, d_out]``: the cartridge's read (compiled on CUDA by :meth:`forward`)."""
        raise NotImplementedError

    # -- shared regularisation / dropout (all cartridges inherit) ----------------------------

    def cell_tv(self) -> torch.Tensor:
        """Hamming-1 adjacent-cell total-variation penalty on the cell table ``[G, tph, K, d_out]``.

        The mean over Hamming-1 cell pairs (and tables) of ``||v_c - v_c'||**2`` (d_out summed in, not averaged); the
        formula is :func:`_cell_tv_eager`. Differentiable w.r.t. ``weights``; adds no nodes to the forward (called only
        by the trainer when its TV lambda > 0).

        The single dispatch point for the penalty, as :meth:`ReferenceForward._dispatch` is for the reference forward:
        on CUDA it runs :func:`_cell_tv_eager` through a lazily built per-instance ``torch.compile`` (static shapes),
        which fuses the nap per-axis passes and does not materialise the per-axis differences (eager holds nap of
        them, 4x the table at nap=8, for the backward); on the CPU, or with ``LUTORCH_EX_NO_COMPILE=1``, the plain
        eager formula. ``LUTORCH_EX_NO_COMPILE=1`` means "pure PyTorch, no acceleration" here."""
        nap = self.spec.nap
        w = self.weights.reshape(-1, *self.weights.shape[2:])       # [n_tables, K, d_out]
        if _COMPILE_ENABLED and w.is_cuda:
            if self._compiled_tv is None:
                # Static shapes, unlike the reference forward's dynamic=True: on symbolic shapes Inductor's per-axis
                # reductions are ~4x slower (0.86 vs 0.21 ms of GPU time at h16 tph64 nap8 d48, float32). A table never
                # changes shape, so this costs one compile per (table shape, nap, dtype), shared by every instance.
                self._compiled_tv = torch.compile(_cell_tv_eager, dynamic=False)
            return self._compiled_tv(w, nap)
        return _cell_tv_eager(w, nap)

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Abstract: the shared base never runs a read. Construct a reference class
        (ManifestoHardLUT / ManifestoSoftLUT / ConfidenceLUT) or a fused one (Fused*)."""
        raise NotImplementedError(f"{type(self).__name__} is the shared scaffolding base and has no forward")

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
        # The one place every cartridge's cell index is produced: in index_dtype (int32 by default when the table fits,
        # else int64; or an explicit choice), so every downstream read (_global_cells keeps c's dtype) and the saved
        # tensors use it. powers stays an int64 buffer; only the stored indices narrow.
        dt = self.index_dtype
        if dt == torch.int32 and getattr(self, "_index_dtype_auto", False) and \
                B * G * tph * 2 > _INT32_MAX_ENTRIES:            # auto: widen a call too large for int32 positions
            dt = torch.int64
        powers = self.powers.to(dt)
        c = ((u > 0).to(dt) * powers).sum(dim=-1, dtype=dt)                # MSB-first, stop-grad
        u_abs_star, j_star = u.abs().min(dim=-1)                           # |u_{j*}| and j*
        c_alt = c ^ powers[j_star]
        return z, u, c, j_star, u_abs_star, c_alt



def _cell_tv_eager(weights: torch.Tensor, nap: int) -> torch.Tensor:
    """The cell-TV formula in plain PyTorch (see :meth:`ManifestoLUT.cell_tv`), ``weights [n_tables, 2**nap, d_out]``.

    View the ``K = 2**nap`` cells of each table as the vertices of an nap-cube (``[n_tables, 2, 2, ..., 2, d_out]``)
    and sum ``(v_c - v_c')**2`` over every Hamming-1 pair (each counted once, via a size-2 ``diff`` along each of the
    nap bit-axes), then divide by the pair count ``n_tables * nap * 2**(nap-1)``."""
    n_tables, K, d_out = weights.shape
    t = weights.reshape(n_tables, *([2] * nap), d_out)       # [T, 2,..,2, d_out] hypercube
    tv = t.new_zeros(())
    for ax in range(1, nap + 1):                             # one cube axis per bit
        d = t.diff(dim=ax)
        tv = tv + (d * d).sum()
    n_pairs = n_tables * nap * (1 << (nap - 1))              # total Hamming-1 pairs
    return tv / n_pairs


def cell_tv_penalty(module: nn.Module) -> torch.Tensor:
    """Model-level Hamming-1 cell-TV penalty: the MEAN of :meth:`ManifestoLUT.cell_tv` over every
    ``ManifestoLUT`` cartridge inside ``module`` that has a trainable table (an inference-only
    DeployedQuantisedConfidenceLUT is skipped) (equivalent to the reference model's
    ``lut_tv_penalty()``, which averages ``cell_tv()`` over its LUT layers). Scale by lambda and add
    to the loss, matching the reference ergonomics: ``(lam * cell_tv_penalty(model)).backward()``.
    Returns a 0-dim zero (no grad) when the module has no such cartridge."""
    tvs = [m.cell_tv() for m in module.modules() if isinstance(m, ManifestoLUT) and m._HAS_TRAINABLE_TABLE]
    if not tvs:
        return torch.zeros(())
    return torch.stack(tvs).mean()
