"""Pure-PyTorch reference implementations of the two MultiHeadLUT families.

Companion code to ``mhlut_math.md``. Pure PyTorch, ``@torch.compile`` on the
hot forward path. Backward is plain autograd in both cases.

  * ``MultiHeadLUT``      — canonical (manifesto) formulation. Forward is a
                            row gather; ``smooth_mode=True`` blends the chosen
                            row with ``n_alternatives`` nearest rows via the
                            uncertainty kernel ``u(|d|) = 0.5 / (1 + |d|)``.

  * ``SoftMultiHeadLUT``  — softmax over all ``K = 2^NAP`` rows. ``p = d/(T+|d|)``,
                            ``ts = p @ B``, ``sel = softmax(ts/T_sel)``. ``hard=False``
                            uses the full soft mixture; ``hard=True`` applies the
                            standard STE wrapper so the forward output equals
                            the manifesto hard pick.

Both modules share the same discrete index ``chosen_t = sign-pack of (x[a] > x[b])``
and use the same balanced-anchor sampling at init.
"""

from __future__ import annotations
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------------
#  Shared init: balanced anchor pairs and the ±1 bit matrix.
# ----------------------------------------------------------------------------

def _sample_anchor_pairs(
    input_dim: int,
    n_tables: int,
    n_anchor_pairs: int,
    seed: Optional[int],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(anchor_a, anchor_b)`` of shape ``[n_tables, NAP]`` each.

    Per-table: a random permutation of ``[0, input_dim)`` is sliced into
    ``2*NAP`` distinct anchor positions; first ``NAP`` go to ``a``, last ``NAP``
    to ``b``. No anchor repeats within a table; no ``a == b`` pair within a
    table.
    """
    if 2 * n_anchor_pairs > input_dim:
        raise ValueError(
            f"need 2*NAP ({2*n_anchor_pairs}) <= input_dim ({input_dim})"
        )
    g = torch.Generator(device='cpu')
    if seed is not None:
        g.manual_seed(seed)
    a = torch.empty(n_tables, n_anchor_pairs, dtype=torch.long)
    b = torch.empty(n_tables, n_anchor_pairs, dtype=torch.long)
    for t in range(n_tables):
        perm = torch.randperm(input_dim, generator=g)[: 2 * n_anchor_pairs]
        a[t] = perm[:n_anchor_pairs]
        b[t] = perm[n_anchor_pairs:]
    return a.to(device), b.to(device)


def _bit_matrix(n_anchor_pairs: int, device: torch.device) -> torch.Tensor:
    """``B[i, k] = +1`` if bit ``(NAP-1-i)`` of integer ``k`` is 1, else ``-1``.

    Shape ``[NAP, 2^NAP]``. Column ``k`` is the ±1 expansion of ``k`` MSB first.
    """
    K = 1 << n_anchor_pairs
    idx = torch.arange(K, device=device).unsqueeze(0)
    shifts = torch.arange(n_anchor_pairs - 1, -1, -1, device=device).unsqueeze(1)
    bits = ((idx >> shifts) & 1).float()
    return bits * 2.0 - 1.0


# ============================================================================
#  §1 — Canonical MultiHeadLUT (manifesto + uncertainty-blend soft mode).
# ============================================================================

class MultiHeadLUT(nn.Module):
    """Canonical multi-head LUT.

    Forward (``smooth_mode=True``): blend chosen row with ``K' = n_alternatives``
    nearest rows, weighted by ``u(|d|) = 0.5 / (1 + |d|)``.

    Forward (``smooth_mode=False``): hard pick — the manifesto's deployed form.
    Backward x-gradient still flows through the uncertainty kernel via a
    standard STE wrapper; weight gradient is a hard ``index_add`` at the picked
    row. (This matches the canonical codebase behaviour: ``smooth_mode`` only
    controls the forward, not the backward x-chain.)

    Backward is autograd: smooth mode uses the soft forward directly; hard mode
    uses ``out = main + (soft − soft.detach())`` so the soft-chain gradient
    reaches ``x`` without changing the forward value or the weight gradient.

    Args:
        input_dim: dim of input vector.
        n_heads: number of heads.
        n_outputs: per-head output dim.
        n_anchor_pairs: NAP — per-table input bit-width (K = 2^NAP rows).
        tables_per_head: number of tables summed per head.
        n_alternatives: K' for the uncertainty blend (1 = manifesto soft default).
        smooth_mode: True for the soft blend, False for the hard pick (no gradient).
        init_std: uniform[-init_std, init_std] init for table weights.
        random_seed: anchor sampling + weight init seed.
        device: where to allocate buffers/parameters.
    """

    def __init__(
        self,
        input_dim: int,
        n_heads: int,
        n_outputs: int,
        n_anchor_pairs: int,
        tables_per_head: int = 1,
        n_alternatives: int = 1,
        smooth_mode: bool = True,
        init_std: float = 0.001,
        random_seed: Optional[int] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.n_heads = n_heads
        self.n_outputs = n_outputs
        self.n_anchor_pairs = n_anchor_pairs
        self.tables_per_head = tables_per_head
        self.n_tables = n_heads * tables_per_head
        self.K = 1 << n_anchor_pairs
        if not 1 <= n_alternatives <= n_anchor_pairs:
            raise ValueError(
                f"n_alternatives must be in [1, n_anchor_pairs]; got "
                f"{n_alternatives} (NAP={n_anchor_pairs}). The uncertainty kernel "
                f"needs at least one alt to produce x-gradient."
            )
        self.n_alternatives = n_alternatives
        self.smooth_mode = bool(smooth_mode)

        dev = device or torch.device('cpu')

        anchor_a, anchor_b = _sample_anchor_pairs(
            input_dim, self.n_tables, n_anchor_pairs, random_seed, dev
        )
        self.register_buffer('anchor_a', anchor_a)
        self.register_buffer('anchor_b', anchor_b)

        # Per-position bit weights for MSB-first bit-pack (powers[i] = 2^(NAP-1-i)).
        powers = (1 << torch.arange(n_anchor_pairs - 1, -1, -1, device=dev)).long()
        self.register_buffer('powers', powers)

        g = torch.Generator(device='cpu')
        if random_seed is not None:
            g.manual_seed(random_seed + 1)
        w = (torch.rand(self.n_tables, self.K, n_outputs, generator=g) - 0.5) * (2.0 * init_std)
        self.weights = nn.Parameter(w.to(dev))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, input_dim] → [B, n_heads, n_outputs]
        return _mhlut_forward(
            x, self.weights, self.anchor_a, self.anchor_b, self.powers,
            self.n_heads, self.tables_per_head,
            self.n_alternatives, self.smooth_mode,
        )


@torch.compile
def _mhlut_forward(
    x: torch.Tensor,
    weights: torch.Tensor,
    anchor_a: torch.Tensor,
    anchor_b: torch.Tensor,
    powers: torch.Tensor,
    n_heads: int,
    tables_per_head: int,
    n_alternatives: int,
    smooth_mode: bool,
) -> torch.Tensor:
    B = x.shape[0]
    n_tables, NAP = anchor_a.shape
    K = 1 << NAP
    n_outputs = weights.shape[-1]

    d = x[:, anchor_a] - x[:, anchor_b]                       # [B, n_tables, NAP]
    bits = (d > 0).long()
    chosen = (bits * powers.view(1, 1, -1)).sum(dim=-1)       # [B, n_tables]

    weights_flat = weights.view(n_tables * K, n_outputs)
    table_offset = torch.arange(n_tables, device=x.device).long() * K
    chosen_flat = (chosen + table_offset.view(1, -1)).reshape(-1)
    main_w = F.embedding(chosen_flat, weights_flat).view(B, n_tables, n_outputs)

    # Top-K' positions with smallest |d| → 1-bit-flip alts at those bits.
    abs_d = d.abs()
    _, top_pos = torch.topk(abs_d, n_alternatives, dim=-1, largest=False)   # [B, n_tables, K']
    flip = powers.view(1, 1, -1).expand(B, n_tables, -1).gather(-1, top_pos)
    alt_rows = chosen.unsqueeze(-1) ^ flip                                   # [B, n_tables, K']
    abs_d_alt = abs_d.gather(-1, top_pos)                                    # [B, n_tables, K']

    # u(|d_alt|) = 0.5 / (1 + |d_alt|), then split into α_main, α_alt.
    uncertainty = 0.5 / (1.0 + abs_d_alt)
    alpha_alt = uncertainty / n_alternatives                                 # [B, n_tables, K']
    alpha_main = 1.0 - alpha_alt.sum(dim=-1)                                 # [B, n_tables]

    # Gather alt-row weights.
    alt_flat = (alt_rows + table_offset.view(1, -1, 1)).reshape(-1)
    alt_w = F.embedding(alt_flat, weights_flat).view(
        B, n_tables, n_alternatives, n_outputs
    )

    if smooth_mode:
        # Soft blend — autograd handles forward + backward.
        out_t = alpha_main.unsqueeze(-1) * main_w
        out_t = out_t + (alpha_alt.unsqueeze(-1) * alt_w).sum(dim=-2)
    else:
        # Hard forward, uncertainty-kernel STE backward.
        # main_w carries the hard index_add weight gradient.
        # (soft - soft.detach()) routes the soft-chain gradient to x
        # without affecting the forward value or the weight gradient
        # (alt-row weight refs are detached, blocking grad flow there).
        soft = alpha_main.unsqueeze(-1) * main_w.detach()
        soft = soft + (alpha_alt.unsqueeze(-1) * alt_w.detach()).sum(dim=-2)
        out_t = main_w + soft - soft.detach()

    return out_t.view(B, n_heads, tables_per_head, n_outputs).sum(dim=2)


# ============================================================================
#  §2 — SoftMultiHeadLUT (softmax over all 2^NAP rows).
# ============================================================================

class SoftMultiHeadLUT(nn.Module):
    """Softmax-over-rows multi-head LUT.

    ``p[i] = d[i] / (T_soft + |d[i]|)`` are smooth signs.
    ``ts[k] = sum_i p[i] · B[i, k]`` is the per-row match score
    (``B`` is the ±1 bit matrix).
    ``sel[k] = softmax(ts / T_sel)`` is the per-row probability.

    Forward:
      ``hard=True``  — STE wrapper around the argmax pick. Numerically identical
                       to the manifesto hard pick; gradients flow through ``sel``.
      ``hard=False`` — full soft mixture ``out = Σ_k sel[k] · W[k, :]``.

    Backward is plain autograd.

    Args:
        input_dim: dim of input vector.
        n_heads: number of heads.
        n_outputs: per-head output dim.
        n_anchor_pairs: NAP — per-table input bit-width (K = 2^NAP rows).
        tables_per_head: number of tables summed per head.
        hard: forward = STE hard pick (True) or full soft mixture (False).
        soft_score_temp: initial T_soft (learnable).
        select_temp: initial T_sel (learnable).
        init_std: uniform[-init_std, init_std] init for table weights.
        random_seed: anchor sampling + weight init seed.
        device: where to allocate buffers/parameters.
    """

    def __init__(
        self,
        input_dim: int,
        n_heads: int,
        n_outputs: int,
        n_anchor_pairs: int,
        tables_per_head: int = 1,
        hard: bool = False,
        soft_score_temp: float = 0.5,
        select_temp: float = 0.5,
        init_std: float = 0.001,
        random_seed: Optional[int] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.n_heads = n_heads
        self.n_outputs = n_outputs
        self.n_anchor_pairs = n_anchor_pairs
        self.tables_per_head = tables_per_head
        self.n_tables = n_heads * tables_per_head
        self.K = 1 << n_anchor_pairs
        self.hard = bool(hard)

        dev = device or torch.device('cpu')

        anchor_a, anchor_b = _sample_anchor_pairs(
            input_dim, self.n_tables, n_anchor_pairs, random_seed, dev
        )
        self.register_buffer('anchor_a', anchor_a)
        self.register_buffer('anchor_b', anchor_b)

        self.register_buffer('bit_matrix', _bit_matrix(n_anchor_pairs, dev))

        # Log-parametrized temperatures (kept positive, optimizer sees scalars).
        self.log_T_soft = nn.Parameter(
            torch.tensor(math.log(float(soft_score_temp)), device=dev)
        )
        self.log_T_sel = nn.Parameter(
            torch.tensor(math.log(float(select_temp)), device=dev)
        )

        g = torch.Generator(device='cpu')
        if random_seed is not None:
            g.manual_seed(random_seed + 1)
        w = (torch.rand(self.n_tables, self.K, n_outputs, generator=g) - 0.5) * (2.0 * init_std)
        self.weights = nn.Parameter(w.to(dev))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T_soft = self.log_T_soft.exp()
        T_sel = self.log_T_sel.exp()
        return _soft_mhlut_forward(
            x, self.weights, self.anchor_a, self.anchor_b, self.bit_matrix,
            T_soft, T_sel, self.n_heads, self.tables_per_head, self.hard,
        )


@torch.compile
def _soft_mhlut_forward(
    x: torch.Tensor,
    weights: torch.Tensor,
    anchor_a: torch.Tensor,
    anchor_b: torch.Tensor,
    bit_matrix: torch.Tensor,
    T_soft: torch.Tensor,
    T_sel: torch.Tensor,
    n_heads: int,
    tables_per_head: int,
    hard: bool,
) -> torch.Tensor:
    B = x.shape[0]
    n_tables = anchor_a.shape[0]
    n_outputs = weights.shape[-1]

    d = x[:, anchor_a] - x[:, anchor_b]                  # [B, n_tables, NAP]
    p = d / (T_soft + d.abs())                            # [B, n_tables, NAP]

    # A more efficient hard-forward path is possible: at fp32, argmax(ts) is
    # provably equal to the sign-pack integer `Σ_i (d[i] > 0) · 2^(NAP-1-i)`,
    # so the [B, n_tables, K] `ts` tensor never needs to be materialised — a
    # bit-pack + embedding_bag gather suffices. But that breaks autograd:
    # PyTorch wouldn't see the soft `p → ts → sel_soft` chain, and the entire
    # backward (softmax bwd, einsum bwd through `bit_matrix`, soft-sign
    # Jacobian, x scatter to anchors, log_T_* grads) would have to be coded by
    # hand. Possible — that's what TinyMultiHeadLut(backward_mode='soft') does
    # — but much more verbose. Here we keep the einsum so autograd handles
    # everything for free; the price is `O(K · n_outputs)` activation memory
    # per LUT call per token.
    ts = torch.einsum('btp,pk->btk', p, bit_matrix)       # [B, n_tables, K]
    sel_soft = F.softmax(ts / T_sel, dim=-1)

    if hard:
        idx = sel_soft.argmax(dim=-1, keepdim=True)
        sel_hard = torch.zeros_like(sel_soft).scatter_(-1, idx, 1.0)
        sel = sel_hard - sel_soft.detach() + sel_soft     # STE
    else:
        sel = sel_soft

    out_t = torch.einsum('btk,tko->bto', sel, weights)    # [B, n_tables, n_outputs]
    return out_t.view(B, n_heads, tables_per_head, n_outputs).sum(dim=2)
