"""QuantisedConfidenceLUT — the p2_int8 quant-aware sibling of ConfidenceLUT (Gen-3 LightMHL).

Same learned-margin confidence score and τ-blend as :class:`ConfidenceLUT` (inherited), but the
read is **quant-aware**: the cell tables are fake-quantised to int8 with a per-(group, channel)
power-of-two scale, and the score / blend weights are rounded to powers of two — both with a
straight-through estimator, so the forward value is the quantised read while gradients flow to the
float master weights, β, γ (and τ for n=2) through the exact float surrogate. This reproduces the
quantised champion (ablation row 3.2 int8 / abl_48).

The quantisation math is the reference's, vendored verbatim in :mod:`._pow2` (from
``spiky.lutorch.pow2_read`` on ``research/ffn_replacement_fix``), so this cartridge is a thin wiring
layer over it:

  s_t      = (Σ_j|u_j|)·exp(γ·Σ_j logσ(β|u_j|))             # the ConfidenceLUT score (g dropped)
  W_hat·2^e= ste_tables(W)                                  # int8 fake-quant, per-(group,channel) 2^e
  read_top_n = 2:  y = Σ_t [ 2^k'·W_hat[c] + 2^(k'-q)·W_hat[c'] ]·2^e     (STE blend weights)
  read_top_n = 1:  y = Σ_t 2^k'·W_hat[c]·2^e                               (STE single-cell weight)

where k' = round(log2 s_t − c_q) and q = round(2|u_{j*}|/(τ ln2)) are the note's power-of-two
exponents (stop-grad; the STE carries the exact (s·v, s·(1−v)) / s gradient). An integer int8
shift-add read (``forward_int``, units 2^-6) is provided for inference / cross-checking and equals
the fake-quant value to round-off.

Scope: constant cells only, both anchor modes (inherited), read_top_n ∈ {1,2}, no hard forward, no
trainable anchors. fp32/fp64 only (bf16 raises via the base). p2_int8 is the only preset; n=1 quant
is a lutorch_ex extension of the reference (which was n=2-only).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from . import _pow2, _pow2_int8
from ._fused_ops import _global_cells
from .confidence import ConfidenceLUT


class QuantisedConfidenceLUT(ConfidenceLUT):
    """p2_int8 quant-aware twin of :class:`ConfidenceLUT`; see module docstring."""

    def __init__(self, spec, *, quant_mode: str = "p2_int8", quant_overrides=None, **kw):
        super().__init__(spec, **kw)
        self._quant = _pow2.resolve_quant_config(quant_mode, quant_overrides)
        if self._quant is None:
            raise ValueError("QuantisedConfidenceLUT requires a quant_mode (e.g. 'p2_int8')")
        # Eagerly build/register the native p2_scalars op (standalone JIT; returns False and falls
        # back to the pure _pow2 path on CPU / no nvcc / build failure — never raises).
        _pow2_int8.ensure_registered()

    # -- quant-aware building blocks -------------------------------------------------------

    def _betagamma(self, dtype):
        return (self.confidence_log_beta.to(dtype).exp(), self.confidence_log_gamma.to(dtype).exp())

    def _fake_quant_tables(self):
        """Fake-quant cell tables (STE): value W_hat·2^e, gradient identity to the float master.
        Returns the flat ``[G*tph*K, d_out]`` table used by the embedding_bag read."""
        G, tph, K, d_out = self.weights.shape
        W = _pow2.ste_tables(self.weights.reshape(G * tph, K, d_out),
                             n_heads=G, bits=self._quant["bits"], offset=self._quant["offset"])
        return W.reshape(G * tph * K, d_out)

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
        the real STE backward exactly — so it is the function fp64 gradcheck validates (the
        staircase STE value itself is piecewise-constant and not directly gradcheckable)."""
        G, tph, K, d_out = self.weights.shape
        B = c.shape[0]
        cfg = self._quant
        beta, gamma = self._betagamma(u.dtype)
        g0 = torch.zeros((), dtype=u.dtype, device=u.device)       # g dropped (== 0)
        W2 = self._fake_quant_tables()
        m = u.abs()
        mv = u_abs_star.unsqueeze(-1)                              # [B, G, tph, 1]
        if self.read_top_n == 2:
            tau = self.log_read_tau.to(u.dtype).exp()
            if frozen_coef is None:
                # Native p2_scalars op when available (fused q/k'/skip/drop + STE weights), else the
                # pure _pow2 path. Both give bit-identical psw; we pair it with our own c/c_alt
                # (the op's returned cell indices are discarded, so no bit-convention coupling).
                psw2 = _pow2_int8.cell_weights(u, c, self.powers, tau, g0, beta, gamma, cfg)[0]
            else:
                s = self._score(u)
                x2 = 2.0 * mv / tau
                ex = s.unsqueeze(-1) * torch.cat([torch.sigmoid(x2), torch.sigmoid(-x2)], dim=-1)
                psw2 = ex * frozen_coef                                          # smooth, frozen ratio
            gc = _global_cells(c, G, tph, K)
            gca = _global_cells(c_alt, G, tph, K)
            idx = torch.cat([gc, gca], dim=2).reshape(B * G, 2 * tph)
            psw = torch.cat([psw2[..., 0], psw2[..., 1]], dim=2).reshape(B * G, 2 * tph).to(W2.dtype)
            return F.embedding_bag(idx, W2, per_sample_weights=psw, mode="sum").reshape(B, G, d_out)
        # read_top_n == 1: single scored cell, score rounded to a power of two (STE)
        s = self._score(u)                                         # [B, G, tph]
        if frozen_coef is None:
            k1 = _pow2.round_half_up(_pow2.log2_score(m, g0, beta, gamma))
            skip = k1 < cfg["lo"]
            k1 = torch.clamp(k1, cfg["lo"], cfg["hi"])
            psw = self._ste_score_weight(s, k1, skip).reshape(B * G, tph).to(W2.dtype)
        else:
            psw = (s * frozen_coef).reshape(B * G, tph).to(W2.dtype)            # smooth, frozen ratio
        gc = _global_cells(c, G, tph, K).reshape(B * G, tph)
        return F.embedding_bag(gc, W2, per_sample_weights=psw, mode="sum").reshape(B, G, d_out)

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        # Quant-aware read for both train (STE grads) and eval (no_grad -> the pure quantised value;
        # ste_tables' W - W.detach() term is 0 under no_grad). The TRAIN addressing uses the base's
        # compiled _addr (fuses the eager margin/sign-bit/argmin materialisation on CUDA, ~10x);
        # eval uses plain _addresses inside the base forward's whole-forward compile.
        z, u, c, j_star, u_abs_star, c_alt = self._addr(x) if self.training else self._addresses(x)
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
        forward at the base point, smooth value. Used for fp64 gradcheck of the quant backward."""
        z, u, c, j_star, u_abs_star, c_alt = self._addresses(x)
        return self._route(self._quant_grp_out(u, c, c_alt, u_abs_star, frozen_coef=frozen_coef), x)

    # -- integer int8 shift-add read (inference / cross-check), equals the fake-quant value --------

    @torch.no_grad()
    def forward_int(self, x: torch.Tensor) -> torch.Tensor:
        """p2_int8 integer eval read (note §6): int8 rows, int32 shift-add in units 2^-6, scaled by
        2^(e-6). Equals the fake-quant forward value to round-off. read_top_n==2 only (the reference
        integer form); n=1 uses the fake-quant eval (same value)."""
        if self.read_top_n != 2:
            self.eval()
            with torch.no_grad():
                return self(x)
        cfg = self._quant
        G, tph, K, d_out = self.weights.shape
        B = x.shape[0]
        z, u, c, j_star, u_abs_star, c_alt = self._addresses(x)
        beta, gamma = self._betagamma(u.dtype)
        g0 = torch.zeros((), dtype=u.dtype, device=u.device)
        tau = self.log_read_tau.to(u.dtype).exp()
        m = u.abs(); mv = u_abs_star.unsqueeze(-1)
        e = _pow2.head_chan_exponents(self.weights.reshape(G * tph, K, d_out), G, cfg["bits"], cfg["offset"])
        Wint = _pow2.quantise_tables(self.weights.reshape(G * tph, K, d_out), e, cfg["bits"])
        packed = _pow2.pack_tables(Wint, cfg["bits"]).reshape(G * tph * K, d_out)
        q, k, skip, drop = _pow2.blend_exponents(m, mv, tau, g0, beta, gamma, cfg)
        gc = _global_cells(c, G, tph, K); gca = _global_cells(c_alt, G, tph, K)
        flat_idx = torch.stack([gc, gca], dim=-1).reshape(B, G, tph, 2)     # [N=B, H=G, T=tph, 2]
        acc = _pow2.int_blend_read(packed, cfg["bits"], d_out, flat_idx, q, k, skip, drop)  # int32 [B,G,d_out]
        scale = torch.pow(2.0, e.to(x.dtype) - _pow2.FIXED_POINT_SHIFT)       # [G, d_out], per-(group,channel)
        grp_out = acc.to(x.dtype) * scale.unsqueeze(0)
        return self._route(grp_out, x)
