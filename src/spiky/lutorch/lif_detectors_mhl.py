"""LIFDetectorsMHL — a LIF-detector drop-in for the HyperplaneMultiHeadLUT index front-end.

Same multi-head-LUT skeleton as :class:`spiky.lutorch.hyperplane_multi_head_lut.HyperplaneMultiHeadLUT`
(``n_tables = n_heads * tables_per_head`` tables, each with ``n_anchor_pairs`` index bits selecting one of
``2**n_anchor_pairs`` rows of ``n_outputs`` values; rows summed within a head), but the per-table index
bits are produced by **combined LIF detectors over latency-coded inputs** instead of affine hyperplane
sign-tests, and addressing is **straight-through hard** so training and inference use the *same* discrete
lookup.

Per detector (arrivals ``a_i = t_i + d_i`` from latency-coded input ``t``):

    V_self = Σ_i  w_i · exp(-ReLU(r - a_i)/tau_s) · sigmoid((r - a_i)/eps)      # magnitude channel
    V_pair = Σ_{i≠j} P_ij · exp(-ReLU(a_j - a_i)/tau_p) · sigmoid((a_j - a_i)/eps)  # order/contrast channel
    V      = V_self + V_pair
    bit    = sigmoid((V - theta) / temp_bit)

``tau_s``, ``tau_p`` and ``temp_bit`` are softplus/exp-positive; ``P`` is off-diagonal (self-pairs masked);
the pair channel is initialised near zero so each detector starts as a pure value/range unit.

Addressing mirrors the teacher's ``forward_mode="hard"`` (hard forward = one cell per table via the packed
argmax address + ``embedding_bag``-style sum; soft backward) via an **OUTPUT-level straight-through**::

    prow_soft = Π_k [b_soft if code-bit=1 else 1-b_soft]   # b_soft = sigmoid((V-theta)/temp_bit); no detach
    prow_hard = one-hot at the packed argmax address       # non-differentiable
    y_soft = prow_soft @ table ;  y_hard = prow_hard @ table
    y = y_soft + (y_hard - y_soft).detach()                # forward = hard single cell; backward via y_soft

The forward VALUE equals the hard single-cell lookup, while the gradient flows entirely through ``y_soft``.
Because a product of independent per-bit Bernoullis over the ``2**nap`` outcomes IS the softmax over those
cells, this backward is the **exact full-K softmax over all 2**nap cells** — matching
HyperplaneMultiHeadLUT's hard-forward / soft-backward *exactly* (parity is exact, not an approximation).
Consequently the soft training objective and the hard/argmax inference objective coincide by construction
(no soft-blend "escape hatch"), and gradient reaches the bits of non-selected cells too.

Bit packing matches the teacher exactly: MSB-first, ``addr = Σ_k bit_k · 2**(nap-1-k)`` (== ``bits @
[2**(nap-1), …, 1]``; for nap=6 that is ``[32,16,8,4,2,1]``).
"""
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["LIFDetectorsMHL"]


class LIFDetectorsMHL(nn.Module):
    """LIF-detector replacement for the HyperplaneMultiHeadLUT index front-end.

    Args mirror ``HyperplaneMultiHeadLUT`` so it can be swapped in with minimal changes.

    Parameters
    ----------
    input_dim : int          number of (latency-coded) input features N.
    n_heads : int            number of output heads.
    n_outputs : int          values per table row (output dim per head).
    n_anchor_pairs : int     detectors (index bits) per table; rows = 2**n_anchor_pairs.
    tables_per_head : int    tables reduced (summed) per head. n_tables = n_heads * tables_per_head.
    t_window : float         latency rollout window T (latencies clamped to [0, T]).
    latency_c, latency_alpha : float   latency map t_i = latency_c - latency_alpha * x_i, clamped [0, T].
    pair_init : float        std of the near-zero ordered-pair weight init.
    temp_bit_init : float    initial bit temperature (trainable via a log-parameter).
    table_init : Tensor|None optional (n_tables, 2**nap, n_outputs) warm-start for the row values.
    """

    def __init__(self, input_dim: int, n_heads: int, n_outputs: int, n_anchor_pairs: int,
                 tables_per_head: int = 1, *, t_window: float = 32.0, latency_c: float = 16.0,
                 latency_alpha: float = 3.0, pair_init: float = 0.01, temp_bit_init: float = 1.0,
                 table_init: Optional[torch.Tensor] = None, device=None):
        super().__init__()
        if n_anchor_pairs < 1 or n_anchor_pairs > 20:
            raise ValueError(f"n_anchor_pairs must be in [1,20], got {n_anchor_pairs}")
        self.input_dim = int(input_dim)
        self.n_heads = int(n_heads)
        self.n_outputs = int(n_outputs)
        self.n_anchor_pairs = int(n_anchor_pairs)        # detectors per table
        self.tables_per_head = int(tables_per_head)
        self.n_tables = self.n_heads * self.tables_per_head
        self.n_rows = 1 << self.n_anchor_pairs
        self.t_window = float(t_window)
        self.latency_c = float(latency_c)
        self.latency_alpha = float(latency_alpha)

        N, T, NAP = self.input_dim, self.n_tables, self.n_anchor_pairs
        M = T * NAP                                       # total detectors
        self.n_detectors = M
        dev = device or torch.device("cpu")

        # --- detector params (flat over M = n_tables * n_anchor_pairs) ---
        self.d = nn.Parameter(torch.zeros(M, N, device=dev))
        self.w = nn.Parameter(0.2 * torch.randn(M, N, device=dev))
        self.r = nn.Parameter(torch.full((M,), 0.9 * self.t_window, device=dev))
        self.tau_s_raw = nn.Parameter(torch.ones(M, device=dev))
        self.P = nn.Parameter(pair_init * torch.randn(M, N, N, device=dev))
        self.tau_p_raw = nn.Parameter(torch.ones(M, device=dev))
        self.theta = nn.Parameter(torch.zeros(M, device=dev))
        self.log_temp_bit = nn.Parameter(torch.log(torch.tensor(float(temp_bit_init), device=dev)))

        # --- trainable table (row) values ---
        if table_init is not None:
            if tuple(table_init.shape) != (T, self.n_rows, self.n_outputs):
                raise ValueError(f"table_init shape {tuple(table_init.shape)} != "
                                 f"{(T, self.n_rows, self.n_outputs)}")
            self.table = nn.Parameter(table_init.clone().to(dev))
        else:
            self.table = nn.Parameter(0.1 * torch.randn(T, self.n_rows, self.n_outputs, device=dev))

        # --- buffers ---
        self.register_buffer("offdiag", 1.0 - torch.eye(N, device=dev))
        # MSB-first packing powers, general in NAP: [2**(nap-1), ..., 2, 1]
        self.register_buffer("pow2", (1 << torch.arange(NAP - 1, -1, -1, device=dev)).long())
        # bit-pattern matrix BM[c,k] = k-th (MSB-first) bit of code c
        codes = torch.arange(self.n_rows, device=dev).unsqueeze(1)
        self.register_buffer("bit_matrix",
                             ((codes >> torch.arange(NAP - 1, -1, -1, device=dev).unsqueeze(0)) & 1).float())

    # ---- positive-constrained params ----
    @property
    def tau_s(self):
        return F.softplus(self.tau_s_raw) + 1e-3

    @property
    def tau_p(self):
        return F.softplus(self.tau_p_raw) + 1e-3

    @property
    def temp_bit(self):
        return torch.exp(self.log_temp_bit)

    def latency(self, x: torch.Tensor) -> torch.Tensor:
        """Real input x -> spike latency t = clamp(c - alpha*x, 0, T) (earlier spike = larger value)."""
        return torch.clamp(self.latency_c - self.latency_alpha * x, 0.0, self.t_window)

    def detector_membrane(self, t: torch.Tensor, eps: float) -> torch.Tensor:
        """Combined LIF membrane V per detector. t:(B,N) latencies -> (B,M)."""
        a = t.unsqueeze(1) + self.d.unsqueeze(0)                       # (B,M,N)
        r = self.r.view(1, self.n_detectors, 1)
        dts = r - a
        Vself = (self.w.unsqueeze(0) * torch.exp(-F.relu(dts) / self.tau_s.view(1, -1, 1))
                 * torch.sigmoid(dts / eps)).sum(-1)                   # (B,M)
        D = a.unsqueeze(-2) - a.unsqueeze(-1)                          # D[...,i,j] = a_j - a_i
        g = torch.exp(-F.relu(D) / self.tau_p.view(1, -1, 1, 1)) * torch.sigmoid(D / eps)
        Vpair = ((self.P * self.offdiag).unsqueeze(0) * g).sum(dim=(-1, -2))
        return Vself + Vpair

    def _prow(self, bits: torch.Tensor) -> torch.Tensor:
        """Per-table cell distribution from index bits: (B,n_tables,nap) -> (B,n_tables,2**nap).

        prow[...,c] = Π_k [bits_k if code c's k-th bit == 1 else 1-bits_k]. For independent per-bit
        Bernoullis this product over the 2**nap outcomes is exactly the softmax over the cells; for hard
        0/1 bits it is a one-hot at the packed argmax address."""
        BM = self.bit_matrix.view(1, 1, self.n_rows, self.n_anchor_pairs)
        term = BM * bits.unsqueeze(2) + (1.0 - BM) * (1.0 - bits.unsqueeze(2))   # (B,n_tables,rows,nap)
        return term.prod(dim=-1)                                                 # (B,n_tables,rows)

    def _rows(self, prow: torch.Tensor) -> torch.Tensor:
        """(B,n_tables,rows) cell distribution -> (B,n_tables,n_outputs) per-table selected values."""
        return torch.einsum('btc,tco->bto', prow, self.table)

    def address(self, x: torch.Tensor, eps: float = 0.3) -> torch.Tensor:
        """Hard packed address per (sample, table): (B, n_tables) int64, MSB-first."""
        V = self.detector_membrane(self.latency(x), eps).view(x.shape[0], self.n_tables, self.n_anchor_pairs)
        bits = (V > self.theta.view(1, self.n_tables, self.n_anchor_pairs)).long()
        return (bits * self.pow2.view(1, 1, -1)).sum(-1)

    def forward(self, x: torch.Tensor, eps: float = 0.3, mode: str = "st") -> torch.Tensor:
        """x:(B, input_dim) -> actions (B, n_heads, n_outputs).

        mode: 'st' (OUTPUT-level straight-through, default; use for training) — forward value is the hard
        single cell per table, backward is the exact full-K softmax over all 2**nap cells; 'hard' (pure
        argmax inference, == 'st' forward value); 'soft' (differentiable 2**nap-cell blend, reference)."""
        B = x.shape[0]
        V = self.detector_membrane(self.latency(x), eps).view(B, self.n_tables, self.n_anchor_pairs)
        th = self.theta.view(1, self.n_tables, self.n_anchor_pairs)
        hard_bits = (V > th).float()                                   # non-differentiable argmax bits
        if mode == "hard":
            rows = self._rows(self._prow(hard_bits))                  # pure one-hot inference (no grad path)
        elif mode in ("soft", "st"):
            soft_bits = torch.sigmoid((V - th) / self.temp_bit).clamp(1e-6, 1 - 1e-6)
            y_soft = self._rows(self._prow(soft_bits))               # full 2**nap-cell softmax, differentiable
            if mode == "soft":
                rows = y_soft
            else:                                                     # 'st': output-level straight-through
                y_hard = self._rows(self._prow(hard_bits))          # one-hot forward value
                rows = y_soft + (y_hard - y_soft).detach()          # forward == hard; backward == full-K softmax
        else:
            raise ValueError(f"mode must be 'st'|'hard'|'soft', got {mode!r}")
        return rows.view(B, self.n_heads, self.tables_per_head, self.n_outputs).sum(dim=2)

    # ---- construction helpers ----
    @classmethod
    def from_hyperplane_config(cls, *, input_dim, n_heads, n_outputs, n_anchor_pairs,
                               tables_per_head=1, table_init=None, **kwargs):
        """Build a student matching a HyperplaneMultiHeadLUT config (optionally warm-starting the tables)."""
        return cls(input_dim, n_heads, n_outputs, n_anchor_pairs, tables_per_head,
                   table_init=table_init, **kwargs)
