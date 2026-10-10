"""ConfidenceLUT — the Gen-3 confidence-scored (LookupFFN-inspired) cartridge, scored-only.

Same sign-bit addressing as the Manifesto family — per table, ``n = nap`` anchor-pair margins
``u_j`` (``u_j = z[a_j] - z[b_j]`` in pairs mode, ``u_j = z[a_j]`` in single mode), a cell
``c_t = pack(sign(u))`` (the address is DETACHED / integer). What differs is the combine: instead
of a soft blend, the hard read is gated by a differentiable **confidence score** of the margin
magnitudes:

    s_t = (Σ_j |u_j|) · exp( γ · Σ_j logσ(β·|u_j|) )          # LookupFFN's learned-margin score
    y   = Σ_t s_t · W_t[c_t]                                  # read_top_n = 1  (ablation row 3.1)

with ``β = exp(confidence_log_beta)``, ``γ = exp(confidence_log_gamma)`` learned per layer (there
is no additive log-gain ``g``: frozen at 0, ``exp(0)=1``, it would do nothing). At ``read_top_n = 2`` (the Gen-3 champion, ablation row 3.2) the read is a
two-cell τ-blend between the addressed cell ``c`` and its least-confident-bit neighbour ``c'``
(``j* = argmin_j |u_j|``), still score-gated:

    v   = σ(-2·|u_{j*}| / τ)  ∈ (0, ½]                        # τ = exp(log_read_tau), learned
    y   = Σ_t s_t · [ (1 - v) W_t[c_t] + v W_t[c_t'] ]

The address is non-differentiable, so the input gradient reaches ``x`` ONLY through the score
``s_t`` (all ``nap`` margins) and, at ``n = 2``, the blend weight ``v`` (the deciding margin
``|u_{j*}|``) — there is no directional routing gradient.
Gradients flow to the weight tables, ``β``, ``γ`` and (``n = 2``) ``τ``.

Scope (locked): **scored-only** (no hard-forward mode / STE stitch / score rescale), **constant
cells** only (``w_c ∈ R^{d_out}``), both anchor modes (inherited from the base). Precision:
fp32/fp64 only — like the other pure cartridges it carries no mixed-precision handling, so the
base ``forward`` raises ``TypeError`` on bf16/fp16.

Read path mirrors the pure SoftSign cartridges: the train read is one ``F.embedding_bag`` (fuses
the per-table gather + the score-scaled tph-sum, no ``[B,G,tph,d_out]`` tensor), eval reads the
plain gather, and torch.compile wraps the eval forward on CUDA only (inherited from the base).
Backward is plain autograd — ``s_t`` / ``v`` (and their temperatures) are fully differentiable.
``fused_read=True`` replaces both reads with the fused gather read (``cartridges/_fused_read.py``).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ._fused_ops import _global_cells
from ._fused_read import fused_scored_read
from .manifesto_base import ManifestoLUT


class ConfidenceLUT(ManifestoLUT):
    """Gen-3 confidence-scored cartridge; see module docstring.

    Args (beyond the shared :class:`ManifestoLUT` ones):
        read_top_n: 1 (single cell, row 3.1) or 2 (two-cell τ-blend, row 3.2 champion).
        beta_init / gamma_init: initial β, γ (stored log-parametrised; champion (β,γ)=(2,1)).
        read_tau_init: initial τ for read_top_n=2 (champion 0.5).
        read_tau_learnable: whether τ is a learned nn.Parameter (default True) or a frozen buffer.
        learnable_score: whether β, γ are learned nn.Parameters (default True) or frozen buffers.
        fused_read: read via the fused gather -> x score -> sum (cartridges/_fused_read.py) instead of the
            default embedding_bag read; replaces embedding_bag's forward and score-gradient kernels (default False).
    """

    # Compile the TRAIN forward too (not just eval): the score/blend read-out folds into a handful
    # of fused kernels, cutting the training step and peak memory (see ManifestoLUT.forward). The
    # whole _forward_impl is compiled, so the addressing is included -- _addresses is called
    # directly (no separate compiled _addr) to avoid a nested compile.
    _COMPILE_TRAIN = True

    def __init__(
        self,
        spec,
        *,
        read_top_n: int = 1,
        beta_init: float = 2.0,
        gamma_init: float = 1.0,
        read_tau_init: float = 0.5,
        read_tau_learnable: bool = True,
        learnable_score: bool = True,
        fused_read: bool = False,
        **kw,
    ):
        super().__init__(spec, **kw)
        # Fused read (see cartridges/_fused_read.py): gather -> x score -> sum from the fp32 master, replacing
        # embedding_bag's forward and score-gradient kernels. False (default) = the unchanged embedding_bag read.
        self.fused_read = bool(fused_read)
        if read_top_n not in (1, 2):
            raise ValueError(f"read_top_n must be 1 or 2, got {read_top_n}")
        if not (beta_init > 0 and gamma_init > 0 and read_tau_init > 0):
            raise ValueError(
                f"beta_init, gamma_init, read_tau_init must be > 0, got "
                f"({beta_init}, {gamma_init}, {read_tau_init})"
            )
        self.read_top_n = int(read_top_n)

        # Learned margin-score temperatures, log-parametrised so β, γ stay positive.
        lb, lg = math.log(float(beta_init)), math.log(float(gamma_init))
        if learnable_score:
            self.confidence_log_beta = nn.Parameter(torch.tensor(lb))
            self.confidence_log_gamma = nn.Parameter(torch.tensor(lg))
        else:
            self.register_buffer("confidence_log_beta", torch.tensor(lb))
            self.register_buffer("confidence_log_gamma", torch.tensor(lg))

        # Two-cell blend temperature τ (only for read_top_n == 2).
        if self.read_top_n == 2:
            lt = math.log(float(read_tau_init))
            if read_tau_learnable:
                self.log_read_tau = nn.Parameter(torch.tensor(lt))
            else:
                self.register_buffer("log_read_tau", torch.tensor(lt))

    # -- the Gen-3-specific pieces: the confidence score and the blend weight ---------------

    def _score(self, u: torch.Tensor) -> torch.Tensor:
        """Confidence score ``s_t = (Σ_j|u_j|)·exp(γ·Σ_j logσ(β|u_j|))`` from margins ``u``
        ``[B, G, tph, nap]`` -> ``[B, G, tph]``. Taken in the margins' dtype."""
        m = u.abs()
        beta = self.confidence_log_beta.to(m.dtype).exp()
        gamma = self.confidence_log_gamma.to(m.dtype).exp()
        return m.sum(dim=-1) * torch.exp(gamma * F.logsigmoid(beta * m).sum(dim=-1))

    def _blend_v(self, u_abs_star: torch.Tensor) -> torch.Tensor:
        """Two-cell blend weight ``v = σ(-2|u_{j*}|/τ) ∈ (0, ½]`` (read_top_n == 2)."""
        tau = self.log_read_tau.to(u_abs_star.dtype).exp()
        return torch.sigmoid(-2.0 * u_abs_star / tau)

    # -- score-weighted reads (train = fused embedding_bag; eval = plain gather) ------------

    def _scored_read(self, c: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """``Σ_t s_t·W_t[c_t]`` via one embedding_bag (per_sample_weights = s); no
        ``[B,G,tph,d_out]`` tensor. -> ``[B, G, d_out]``."""
        G, tph, K, d_out = self.weights.shape
        B = c.shape[0]
        W2 = self.weights.reshape(G * tph * K, d_out)
        gc = _global_cells(c, G, tph, K).reshape(B * G, tph)
        psw = s.reshape(B * G, tph).to(W2.dtype)
        return F.embedding_bag(gc, W2, per_sample_weights=psw, mode="sum").reshape(B, G, d_out)

    def _scored_blend_read(self, c, c_alt, s, v) -> torch.Tensor:
        """``Σ_t s_t·[(1-v)W[c] + v W[c']]`` via one embedding_bag over the two cells with
        per_sample_weights ``[s(1-v), s v]``. -> ``[B, G, d_out]``."""
        G, tph, K, d_out = self.weights.shape
        B = c.shape[0]
        W2 = self.weights.reshape(G * tph * K, d_out)
        gc = _global_cells(c, G, tph, K)
        gca = _global_cells(c_alt, G, tph, K)
        idx = torch.cat([gc, gca], dim=2).reshape(B * G, 2 * tph)
        psw = torch.cat([s * (1.0 - v), s * v], dim=2).reshape(B * G, 2 * tph).to(W2.dtype)
        return F.embedding_bag(idx, W2, per_sample_weights=psw, mode="sum").reshape(B, G, d_out)

    # -- combine is not used (forward is overridden); kept to satisfy the ABC ---------------

    def _combine(self, y_hard, y_alt, u_abs_star):  # pragma: no cover - forward is overridden
        raise NotImplementedError

    def _fused_read(self, c, c_alt, s, v) -> torch.Tensor:
        """Score-weighted read via :func:`fused_scored_read` from the master table (``fused_read=True``)."""
        G, tph, K, d_out = self.weights.shape
        B = c.shape[0]
        W2 = self.weights.reshape(G * tph * K, d_out)
        if self.read_top_n == 1:
            idx = _global_cells(c, G, tph, K).reshape(B * G, tph)
            psw = s.reshape(B * G, tph)
        else:
            idx = torch.cat([_global_cells(c, G, tph, K), _global_cells(c_alt, G, tph, K)], dim=2).reshape(B * G, 2 * tph)
            psw = torch.cat([s * (1.0 - v), s * v], dim=2).reshape(B * G, 2 * tph)
        return fused_scored_read(idx, psw.to(W2.dtype), W2).reshape(B, G, d_out)

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        z, u, c, j_star, u_abs_star, c_alt = self._addresses(x)
        s = self._score(u)                                    # [B, G, tph]
        # Table dropout folds into the per-table score s (which gates each table's whole
        # contribution): no-op at eval / rate 0. Covers n=1 and n=2 (both read-outs use s).
        mask = self._table_dropout_mask(s.shape[0], s.device, s.dtype)
        if mask is not None:
            s = s * mask
        if self.fused_read:
            v = self._blend_v(u_abs_star) if self.read_top_n == 2 else None
            return self._route(self._fused_read(c, c_alt, s, v), x)
        if self.read_top_n == 1:
            if self.training:
                grp_out = self._scored_read(c, s)
            else:
                grp_out = (s.unsqueeze(-1) * self._read(c)).sum(dim=2)
        else:
            v = self._blend_v(u_abs_star)                     # [B, G, tph]
            if self.training:
                grp_out = self._scored_blend_read(c, c_alt, s, v)
            else:
                y_hard, y_alt = self._read_pair(c, c_alt)
                blend = y_hard + v.unsqueeze(-1) * (y_alt - y_hard)
                grp_out = (s.unsqueeze(-1) * blend).sum(dim=2)
        return self._route(grp_out, x)
