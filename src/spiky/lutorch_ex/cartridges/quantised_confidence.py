"""QuantisedConfidenceLUT — the p2_int8 quant-aware sibling of ConfidenceLUT.

Same learned-margin confidence score and τ-blend as :class:`ConfidenceLUT` (inherited), but the
read is **quant-aware**: the cell tables are fake-quantised to int8 with a per-(group, channel)
power-of-two scale, and the score / blend weights are rounded to powers of two — both with a
straight-through estimator, so the forward value is the quantised read while gradients flow to the
float master weights, β, γ (and τ for n=2) through the exact float surrogate. It is the
int8-deployable form of ConfidenceLUT, at read_top_n = 1 or 2.

The quantisation math lives in :mod:`._pow2`, so this cartridge is a thin wiring layer over it:

  s_t      = (Σ_j|u_j|)·exp(γ·Σ_j logσ(β|u_j|))             # the ConfidenceLUT score (g dropped)
  W_hat·2^e= ste_tables(W)                                  # int8 fake-quant, per-(group,channel) 2^e
  read_top_n = 2:  y = Σ_t [ 2^k'·W_hat[c] + 2^(k'-q)·W_hat[c'] ]·2^e     (STE blend weights)
  read_top_n = 1:  y = Σ_t 2^k'·W_hat[c]·2^e                               (STE single-cell weight)

where k' = round(log2 s_t − c_q) and q = round(2|u_{j*}|/(τ ln2)) are the note's power-of-two
exponents (stop-grad; the STE carries the exact (s·v, s·(1−v)) / s gradient). An integer int8
shift-add read (``forward_int``, units 2^-6) is provided for inference / cross-checking and equals
the fake-quant value to round-off.

Scope: constant cells only, both anchor modes (inherited), read_top_n ∈ {1,2}, no hard forward, no
trainable anchors. float32 / float64 only (bfloat16 / float16 raise TypeError via ReferenceForward). p2_int8 is
the only preset; n=1 quant is a lutorch_ex extension of the reference (which was n=2-only).

Implementations (the class is the implementation, there is no selector):
* :class:`QuantisedConfidenceLUT` -- the reference: the torch definition (``_pow2``) only, every case, CPU and CUDA,
  compiled on CUDA. It loads no extension.
* :class:`FusedQuantisedConfidenceLUT` -- the fused twin: one fused CUDA forward
  (``_fused_quantised_cuda.FusedQuantRead``: addressing, the per-table integers and straight-through weights, the
  fake-quant gather, in one kernel after the table exponents) and one hand-written CUDA backward that reads the
  forward's per-table integers back; read_top_n 1 and 2 are two instantiations of the same kernels (template parameter
  N, as in the deployed kernel). Cases the kernel does not cover (single anchors, nap > 8, a CPU input, a non-float32
  input or table) raise, naming the case, and so does a missing extension.
* :class:`DeployedQuantisedConfidenceLUT` -- the inference-only int8 form, read_top_n 1 or 2 as exported: on CUDA ONE
  kernel (``csrc/pow2_int8_read.cu``, deploy_fwd: addressing, the integers, the int8 shift-add read), float32 input; on
  CPU the torch definition (``_pow2``), the same integers.

Both trainable classes share the quant scaffolding (:class:`QuantisedScaffold`: config, fake-quant tables, the integer
read, deployment export) and the state_dict; neither inherits the other's forward.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from ..lut_spec import LUTSpec
from . import _pow2
from ._fused_ops import _global_cells
from .confidence import ConfidenceLUT


class QuantisedScaffold:
    """The quant scaffolding shared by :class:`QuantisedConfidenceLUT` and :class:`FusedQuantisedConfidenceLUT` (no
    forward): the p2_int8 config, the fake-quant tables, the integer read and the deployment export."""

    def _init_quant(self, quant_mode, quant_overrides):
        self._quant = _pow2.resolve_quant_config(quant_mode, quant_overrides)
        if self._quant is None:
            raise ValueError(f"{type(self).__name__} requires a quant_mode (e.g. 'p2_int8')")

    def _betagamma(self, dtype):
        return (self.confidence_log_beta.to(dtype).exp(), self.confidence_log_gamma.to(dtype).exp())

    def _fake_quant_tables(self):
        """Fake-quant cell tables (STE): value W_hat·2^e, gradient identity to the float master.
        Returns the flat ``[G*tph*K, d_out]`` table used by the embedding_bag read."""
        G, tph, K, d_out = self.weights.shape
        W = _pow2.ste_tables(self.weights.reshape(G * tph, K, d_out),
                             n_heads=G, bits=self._quant["bits"], offset=self._quant["offset"])
        return W.reshape(G * tph * K, d_out)

    def _blend_read(self, psw2, c, c_alt, W2):
        """The n=2 read: cells c and c_alt of every table, weighted by psw2 [B, G, tph, 2] -> [B, G, d_out].

        EXACT IN FP32, so the order of embedding_bag's sum does not matter (and FusedQuantisedConfidenceLUT's
        different, documented order gives the same bits). Per output (group g, channel c) every term is
        W_hat * 2^e * 2^k' or W_hat * 2^e * 2^(k' - q), with |W_hat| <= 128 (int8: [-128, 127]), k' in [-3, 4] and
        q <= 3 (Q): an integer multiple of 2^(e - 6) of magnitude at most 128 * 2^10. A bag holds 2 tph = 128 terms at
        the p2_int8 preset, so every partial sum is an integer multiple of 2^(e - 6) bounded by 128 * 128 * 2^10 = 2^24:
        float32 represents every integer up to 2^24 exactly, and no addition rounds. The argument FAILS (sums can round,
        and two orders can then differ):
        * tph > 64: 2 tph * 128 * 2^10 exceeds 2^24;
        * a read window wider than [-3, 4] (or a larger Q): more bits per term;
        * table dropout: the weights carry 1 / (1 - rate), not a power of two, so the products already round."""
        G, tph, K, d_out = self.weights.shape
        B = c.shape[0]
        gc = _global_cells(c, G, tph, K)
        gca = _global_cells(c_alt, G, tph, K)
        idx = torch.cat([gc, gca], dim=2).reshape(B * G, 2 * tph)
        psw = torch.cat([psw2[..., 0], psw2[..., 1]], dim=2).reshape(B * G, 2 * tph).to(W2.dtype)
        # embedding_bag's backward has a size limit: _fused_ops.py
        return F.embedding_bag(idx, W2, per_sample_weights=psw, mode="sum").reshape(B, G, d_out)

    # -- integer int8 shift-add read (inference / cross-check), equals the fake-quant value --------

    @torch.no_grad()
    def forward_int(self, x: torch.Tensor) -> torch.Tensor:
        """p2_int8 integer eval read (note §6): int8 rows, int32 shift-add in units 2^-6, scaled by
        2^(e-6). Equals the fake-quant forward value to round-off. read_top_n 2 (two cells, shifts k'+6 and
        k'+6-q) and 1 (one cell, shift k'+6) -- the integer form DeployedQuantisedConfidenceLUT runs."""
        cfg = self._quant
        G, tph, K, d_out = self.weights.shape
        B = x.shape[0]
        z, u, c, j_star, u_abs_star, c_alt = self._addresses(x)
        beta, gamma = self._betagamma(u.dtype)
        g0 = torch.zeros((), dtype=u.dtype, device=u.device)
        m = u.abs(); mv = u_abs_star.unsqueeze(-1)
        e = _pow2.head_chan_exponents(self.weights.reshape(G * tph, K, d_out), G, cfg["bits"], cfg["offset"])
        Wint = _pow2.quantise_tables(self.weights.reshape(G * tph, K, d_out), e, cfg["bits"])
        packed = _pow2.pack_tables(Wint, cfg["bits"]).reshape(G * tph * K, d_out)
        gc = _global_cells(c, G, tph, K)
        if self.read_top_n == 2:
            tau = self.log_read_tau.to(u.dtype).exp()
            q, k, skip, drop = _pow2.blend_exponents(m, mv, tau, g0, beta, gamma, cfg)
            gca = _global_cells(c_alt, G, tph, K)
            flat_idx = torch.stack([gc, gca], dim=-1).reshape(B, G, tph, 2)     # [N=B, H=G, T=tph, 2]
            acc = _pow2.int_blend_read(packed, cfg["bits"], d_out, flat_idx, q, k, skip, drop)  # int32 [B,G,d_out]
        else:
            k, skip = _pow2.single_exponent(m, g0, beta, gamma, cfg)
            acc = _pow2.int_single_read(packed, cfg["bits"], d_out, gc.reshape(B, G, tph, 1), k, skip)
        scale = torch.pow(2.0, e.to(x.dtype) - _pow2.FIXED_POINT_SHIFT)       # [G, d_out], per-(group,channel)
        grp_out = acc.to(x.dtype) * scale.unsqueeze(0)
        return self._route(grp_out, x)

    # -- deployment compaction (SupportsDeploymentExport) -------------------------------------

    @torch.no_grad()
    def to_deployment(self) -> dict:
        """Compact int8 payload for deployment export (the `SupportsDeploymentExport` axis).

        Emits the int8 `packed` table (float32 master + raw exponent buffer are NOT included) plus a
        per-out-feature output scale ``2^(e - FIXED_POINT_SHIFT)`` that ``export_deployment`` folds
        EXACTLY (power-of-two -> only the float32 exponent field changes) into a copy of the ProjectionMHL
        decompress weight. The deployed cartridge then returns the RAW int32 accumulator (units 2^-6).
        read_top_n 1 or 2 (the export records it; the deployed class runs that read), h_out==n_groups only
        (the per-(group,channel) scale maps 1:1 to decompress out_features; fan-in would need a pre-route scale). NOTE:
        this is a deploy-only transform; the trained cartridge deliberately does NOT implement `decompress_scale`,
        so ProjectionMHL never folds anything on the live (fake-quant, already-2^e-scaled) forward."""
        cfg = self._quant
        G, tph, K, d_out = self.weights.shape
        if self.spec.h_out != G:
            raise NotImplementedError(
                f"to_deployment supports h_out == n_groups only (got h_out={self.spec.h_out}, n_groups={G}); "
                "the per-(group,channel) output scale folds 1:1 into decompress out_features in that case.")
        Wflat = self.weights.reshape(G * tph, K, d_out)
        e = _pow2.head_chan_exponents(Wflat, G, cfg["bits"], cfg["offset"])                 # [G, d_out]
        # [n_tables, K, d_out] int8
        packed = _pow2.pack_tables(_pow2.quantise_tables(Wflat, e, cfg["bits"]), cfg["bits"]).contiguous()
        # [out_features]
        decompress_scale = torch.pow(2.0, e.float() - _pow2.FIXED_POINT_SHIFT).reshape(-1).contiguous()
        tensors = {
            "packed": packed,
            "confidence_log_beta": self.confidence_log_beta.detach().clone(),
            "confidence_log_gamma": self.confidence_log_gamma.detach().clone(),
            "anchor_a": self.anchor_a.clone(),
            "powers": self.powers.clone(), "in_head": self.in_head.clone(), "out_head": self.out_head.clone(),
        }
        if self.read_top_n == 2:                         # the blend temperature (n = 1 reads one cell, no blend)
            tensors["log_read_tau"] = self.log_read_tau.detach().clone()
        if self.anchor_b is not None:                    # single-anchor mode has no partner coordinate
            tensors["anchor_b"] = self.anchor_b.clone()
        return {
            "format": "p2_int8",
            "decompress_scale": decompress_scale,
            "tensors": tensors,
            "meta": {
                "cartridge": "QuantisedConfidenceLUT", "quant_mode": "p2_int8",
                "quant_cfg": {k: int(v) for k, v in cfg.items() if isinstance(v, (int, bool))},
                "read_top_n": int(self.read_top_n),
                "spec": {"h_in": self.spec.h_in, "h_out": self.spec.h_out, "tph": self.spec.tph,
                         "nap": self.spec.nap, "d_in": self.spec.d_in, "d_out": self.spec.d_out,
                         "anchor_mode": self.spec.anchor_mode},
            },
        }

    @classmethod
    def from_deployment(cls, spec: LUTSpec, tensors: dict, meta: dict, device=None) -> "MultiHeadLUT":
        """Reconstruct the inference-only deployed cartridge from stored tensors + meta (the load-side
        complement of :meth:`to_deployment`). Registered under the ``"p2_int8"`` format tag so
        :func:`~spiky.lutorch_ex.deploy.load_deployment` can dispatch without naming a concrete class."""
        from .deployed_quantised_confidence import DeployedQuantisedConfidenceLUT   # it subclasses the reference
        return DeployedQuantisedConfidenceLUT(spec, tensors, meta, device=device)


class QuantisedConfidenceLUT(QuantisedScaffold, ConfidenceLUT):
    """p2_int8 quant-aware twin of :class:`ConfidenceLUT`, the reference implementation; see module docstring."""

    _ACCEPTS_LOW_PRECISION = False   # float32 / float64 only: the quantised read has no low-precision contract
    #: A scheduling hint, not maths. _pow2.blend_exponents left-folds the nap terms of S and of the logsigmoid sum (the
    #: kernel's operation order, what makes this reference bit-exact with FusedQuantisedConfidenceLUT): each fold reads
    #: the [B, G, tph, nap] margins as nap slices, which crosses Inductor's realize_reads_threshold (default 4), and the
    #: folded producer then exceeds realize_opcount_threshold (default 30), so Inductor writes that ~1 GiB producer (at
    #: 32,768 vectors) to global memory and re-reads it in a separate kernel. With both raised it keeps the margins in
    #: registers and still emits the same in-order add chain (the eager maths, which layer 1 pins, is untouched).
    #: Measured on the RTX 5090 (read_top_n=2, 32,768 vectors, float32; commit 89814465), eval ms / peak GiB: defaults
    #: 11.4-11.7 / 2.03; reads 8 alone 11.4 / 1.44; reads 8 + op count 60 11.5 / 1.50; reads 8 + op count 100 9.9 / 0.90
    #: (also at 1000). Train total unchanged.
    #: Version-dependent: a torch upgrade can rename the options or change the heuristic; a bench_snapshot run
    #: (benchmarks/) is the regression detector. DeployedQuantisedConfidenceLUT is never compiled: on CUDA it runs the
    #: deploy_fwd kernel, on the CPU the eager torch definition.
    _COMPILE_OPTIONS = {"realize_reads_threshold": 8, "realize_opcount_threshold": 100}

    def __init__(self, spec, *, read_top_n: int, quant_mode: str = "p2_int8", quant_overrides=None, **kw):
        # read_top_n is required (no default), as for every confidence cartridge. Both widths train, evaluate and
        # export (FusedQuantisedConfidenceLUT covers both too): to_deployment records meta["read_top_n"] and
        # DeployedQuantisedConfidenceLUT reads that many cells (deploy_fwd_kernel<VEC, N>).
        super().__init__(spec, read_top_n=read_top_n, **kw)
        self._init_quant(quant_mode, quant_overrides)

    # -- quant-aware building blocks -------------------------------------------------------

    def _ste_score_weight(self, s, k1, skip):
        """Single-cell (n=1) STE weight: value 2^k' (0 if skipped), gradient the exact score s."""
        b = torch.pow(2.0, k1).to(s.dtype)
        val = torch.where(skip, torch.zeros_like(b), b)
        st = s * (b / s.detach().clamp_min(1e-30))
        return val + (st - st.detach())

    def _quant_grp_out(self, u, c, c_alt, u_abs_star, frozen_coef=None):
        """The quant-aware (fake-quant + STE weights) score-weighted read -> [B, G, d_out].

        ``frozen_coef`` (test/analysis only) injects the STE's stop-grad per-cell ratio
        ``b / (s·σ)_base`` captured at a base point, and replaces the straight-through weight by the
        SMOOTH ``(s·σ)·ratio``. With the ratio frozen the forward is smooth and its gradient equals
        the real STE backward exactly — so it is the function float64 gradcheck validates (the
        staircase STE value itself is piecewise-constant and not directly gradcheckable)."""
        G, tph, K, d_out = self.weights.shape
        B = c.shape[0]
        cfg = self._quant
        beta, gamma = self._betagamma(u.dtype)
        g0 = torch.zeros((), dtype=u.dtype, device=u.device)       # g dropped (== 0)
        W2 = self._fake_quant_tables()
        m = u.abs()
        mv = u_abs_star.unsqueeze(-1)                              # [B, G, tph, 1]
        dmask = self._table_dropout_mask(B, u.device, u.dtype)    # [B,G,tph] keep-mask or None
        if self.read_top_n == 2:
            tau = self.log_read_tau.to(u.dtype).exp()
            if frozen_coef is None:
                # Straight-through per-cell blend weights from the torch definition (blend_exponents +
                # ste_cell_weights -- the `val + (st - st.detach())` STE), train and eval alike.
                q, k, skip, drop = _pow2.blend_exponents(m, mv, tau, g0, beta, gamma, cfg)
                psw2 = _pow2.ste_cell_weights(u, tau, g0, beta, gamma, q, k, skip, drop)
            else:
                s = self._score(u)
                x2 = 2.0 * mv / tau
                ex = s.unsqueeze(-1) * torch.cat([torch.sigmoid(x2), torch.sigmoid(-x2)], dim=-1)
                psw2 = ex * frozen_coef                                          # smooth, frozen ratio
            if dmask is not None:
                psw2 = psw2 * dmask.unsqueeze(-1)       # table dropout (one Bernoulli/table, both cells)
            return self._blend_read(psw2, c, c_alt, W2)
        # read_top_n == 1: single scored cell, score rounded to a power of two (STE). One logsigmoid pass is shared
        # between the linear score s and its log2 (k'), instead of computing the score twice.
        msum = m.sum(dim=-1)
        lse = F.logsigmoid(beta * m).sum(dim=-1)
        s = msum * torch.exp(gamma * lse)                          # == self._score(u)
        if frozen_coef is None:
            k1, skip = _pow2.single_exponent(m, g0, beta, gamma, cfg)   # the deployed kernel's integers and order
            psw = self._ste_score_weight(s, k1, skip).reshape(B * G, tph).to(W2.dtype)
        else:
            psw = (s * frozen_coef).reshape(B * G, tph).to(W2.dtype)            # smooth, frozen ratio
        if dmask is not None:
            psw = psw * dmask.reshape(B * G, tph).to(psw.dtype)                 # table dropout
        # EXACT IN FP32 too, re-derived for the single-cell read (not inherited from _blend_read's n = 2 argument): per
        # output (group g, channel c) each term is W_hat * 2^e * 2^k' with |W_hat| <= 128 (int8: [-128, 127]) and k' in
        # the window [lo, hi] = [-3, 4] (a skipped table adds 0; no q at n = 1): an integer multiple of 2^(e + lo) of
        # magnitude at most 128 * 2^(hi - lo) = 2^14. A bag holds tph terms, so every partial sum is an integer multiple
        # of 2^(e + lo) bounded by tph * 2^14, at most 2^24 -- exactly representable, no addition rounds -- for
        # tph <= 1024 (in general tph <= 2^(17 - (hi - lo))), and FusedQuantisedConfidenceLUT's order gives the same
        # bits. It FAILS for tph > 1024, a wider window, or table dropout (the 1 / (1 - rate) weights round).
        gc = _global_cells(c, G, tph, K).reshape(B * G, tph)
        # embedding_bag's backward has a size limit: _fused_ops.py
        return F.embedding_bag(gc, W2, per_sample_weights=psw, mode="sum").reshape(B, G, d_out)

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        # Quant-aware read for both train (STE grads) and eval (no_grad -> the pure quantised value;
        # ste_tables' W - W.detach() term is 0 under no_grad). The base forward compiles the WHOLE
        # _forward_impl on CUDA (eval always; train via _COMPILE_TRAIN, inherited from ConfidenceLUT).
        z, u, c, j_star, u_abs_star, c_alt = self._addresses(x)
        grp_out = self._quant_grp_out(u, c, c_alt, u_abs_star)
        return self._route(grp_out, x)

    @torch.no_grad()
    def frozen_ratio(self, x: torch.Tensor):
        """The STE's per-cell stop-grad ratio ``b / (s·σ)_base`` at ``x`` (test/analysis). For the
        frozen-surrogate gradcheck: returned detached, it holds the power-of-two magnitudes fixed so
        the surrogate is smooth and gradcheckable."""
        z, u, c, j_star, u_abs_star, c_alt = self._addresses(x)
        cfg = self._quant
        beta, gamma = self._betagamma(u.dtype)
        g0 = torch.zeros((), dtype=u.dtype, device=u.device)
        s = self._score(u)
        m = u.abs(); mv = u_abs_star.unsqueeze(-1)
        if self.read_top_n == 2:
            tau = self.log_read_tau.to(u.dtype).exp()
            q, k, skip, drop = _pow2.blend_exponents(m, mv, tau, g0, beta, gamma, cfg)
            x2 = 2.0 * mv / tau
            ex = s.unsqueeze(-1) * torch.cat([torch.sigmoid(x2), torch.sigmoid(-x2)], dim=-1)
            b = torch.pow(2.0, torch.stack([k, k - q], dim=-1)).to(ex.dtype)
            return (b / ex.clamp_min(1e-30)).detach()
        k1 = torch.clamp(_pow2.round_half_up(_pow2.log2_score(m, g0, beta, gamma)), cfg["lo"], cfg["hi"])
        return (torch.pow(2.0, k1).to(s.dtype) / s.clamp_min(1e-30)).detach()

    def _forward_surrogate(self, x: torch.Tensor, frozen_coef) -> torch.Tensor:
        """Smooth-surrogate forward with a frozen ratio (test/analysis): same gradient as the STE
        forward at the base point, smooth value. Used for float64 gradcheck of the quant backward."""
        z, u, c, j_star, u_abs_star, c_alt = self._addresses(x)
        return self._route(self._quant_grp_out(u, c, c_alt, u_abs_star, frozen_coef=frozen_coef), x)


from ..deploy_registry import register_rebuilder  # noqa: E402  (bottom import avoids a cycle)

register_rebuilder("p2_int8", QuantisedConfidenceLUT.from_deployment)
