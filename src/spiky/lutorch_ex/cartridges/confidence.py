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

One read, for train and eval and both n: ``y = Σ_k w_k · W[r_k]`` over the flat table rows ``r`` and weights
``w`` from :meth:`_cells_and_weights` (``w = s`` at n = 1; ``s(1-v)`` and ``s v`` over ``c`` and ``c'`` at n = 2).
It runs in one of two forms of the same maths, chosen by whether the forward is being compiled:

* compiled (CUDA, the default): :func:`fused_scored_read` (``cartridges/_fused_read.py``), a gather × weight → sum
  with its own backward, which Inductor fuses into a few kernels. Fp32, h16 d48 tph64 nap8, 8,192 vectors: train
  step 6.55 → 3.48 ms (n = 1) and 11.83 → 5.35 ms (n = 2) against ``F.embedding_bag``;
* eager (CPU, or ``LUTORCH_EX_NO_COMPILE=1``): one ``F.embedding_bag`` with ``per_sample_weights``. Uncompiled, the
  fused read materialises its gathered rows (CPU n = 1 train step 97.6 → 350.7 ms at 1,024 vectors), so eager keeps
  the one-kernel embedding_bag.

The two agree to fp32 re-association (≤ 7e-7), not bitwise. Backward is autograd through either form: ``s_t`` / ``v``
(and their temperatures) are fully differentiable. FusedConfidenceLUT's ``cuda`` path never reads through here.
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
        **kw,
    ):
        super().__init__(spec, **kw)
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

    # -- the score-weighted read ---------------------------------------------------------------

    def _cells_and_weights(self, c, c_alt, s, v):
        """Flat table rows ``[B*G, n*tph]`` and their weights: ``s`` over ``c`` (n = 1), or ``s(1-v)`` over ``c``
        and ``s v`` over ``c'`` (n = 2)."""
        G, tph, K, _ = self.weights.shape
        B = c.shape[0]
        rows = _global_cells(c, G, tph, K)
        if self.read_top_n == 1:
            return rows.reshape(B * G, tph), s.reshape(B * G, tph)
        rows = torch.cat([rows, _global_cells(c_alt, G, tph, K)], dim=2)
        w = torch.cat([s * (1.0 - v), s * v], dim=2)
        return rows.reshape(B * G, 2 * tph), w.reshape(B * G, 2 * tph)

    def _scored_read(self, c, c_alt, s, v) -> torch.Tensor:
        """``Σ_k w_k · W[r_k]`` per (vector, group) -> ``[B, G, d_out]``; the two forms are in the module docstring."""
        G, tph, K, d_out = self.weights.shape
        table = self.weights.reshape(G * tph * K, d_out)
        rows, w = self._cells_and_weights(c, c_alt, s, v)
        w = w.to(table.dtype)
        if torch.compiler.is_compiling():
            out = fused_scored_read(rows, w, table)
        else:
            out = F.embedding_bag(rows, table, per_sample_weights=w, mode="sum")
        return out.reshape(c.shape[0], G, d_out)

    # -- combine is not used (forward is overridden); kept to satisfy the ABC ---------------

    def _combine(self, y_hard, y_alt, u_abs_star):  # pragma: no cover - forward is overridden
        raise NotImplementedError

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        z, u, c, j_star, u_abs_star, c_alt = self._addresses(x)
        s = self._score(u)                                    # [B, G, tph]
        # Table dropout folds into the per-table score s (which gates each table's whole
        # contribution): no-op at eval / rate 0. Covers n=1 and n=2 (both read-outs use s).
        mask = self._table_dropout_mask(s.shape[0], s.device, s.dtype)
        if mask is not None:
            s = s * mask
        v = self._blend_v(u_abs_star) if self.read_top_n == 2 else None
        return self._route(self._scored_read(c, c_alt, s, v), x)
