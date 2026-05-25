"""
Multi-head lookup table with LEARNABLE (trainable) anchoring.

Unlike ``MultiHeadLut`` / ``TinyMultiHeadLut``, which compare *fixed* pairs of
input coordinates ``(anchor_a, anchor_b)`` stored as integer buffers, this module
*learns* which two coordinates each (table, bit) compares, via two softmax
selectors over the input dimension. The selectors are hardened with a
straight-through estimator (STE): argmax (one-hot) on the forward, softmax on the
backward.

Pure PyTorch + autograd (no custom autograd.Function, no native CUDA kernels);
``torch.compile``-friendly. STE appears in two places and is expressed with the
additive trick ``hard + (soft - soft.detach())`` so plain autograd produces the
hard-forward / soft-backward gradient:

  1. anchor selection -- which input coords each bit compares  (softmax over D)
  2. soft signs       -- bit surrogate p = d / (T_sign + |d|)
  3. bitmatrix        -- match scores ts = einsum(p, bit_matrix)
  4. row selection    -- which of the 2**NAP rows to read       (softmax over K)

Forward (per batch element b, table t):
    a_{t,i} = argmax_d anchor_logits[t, i, 0, d]   # '+' coordinate for bit i
    b_{t,i} = argmax_d anchor_logits[t, i, 1, d]   # '-' coordinate for bit i
    d_i     = x[a_{t,i}] - x[b_{t,i}]
    row     = pack( sign(d_i) )                     # 0 .. 2**NAP - 1  (MSB-first)
    out     = sum over tables_per_head  weights[t, row, :]

Output shape: ``[B, n_heads, n_outputs]`` (matches MultiHeadLut).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def _bit_matrix_msb(nap: int, device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """``[NAP, K]`` bit-pattern matrix, MSB-first, with ``±1`` entries.

    ``bit_matrix[i, k] = +1`` iff MSB-first bit ``i`` of ``k`` is set, else ``-1``.
    Same convention as ``tiny_multi_head_lut._soft_bit_matrix_msb`` so that
    ``argmax_k einsum(sign(d), bit_matrix)`` equals the sign-bit-pack row index.
    """
    K = 1 << nap
    bits = ((torch.arange(K, device=device).unsqueeze(0)
             >> torch.arange(nap - 1, -1, -1, device=device).unsqueeze(1)) & 1)
    return ((bits.float() - 0.5) * 2.0).to(dtype)


def _ste(hard: torch.Tensor, soft: torch.Tensor) -> torch.Tensor:
    """Straight-through estimator: forward value is ``hard``, gradient flows
    through ``soft``."""
    return hard + (soft - soft.detach())


@torch.compile
def _forward_body(x, anchor_logits, weights, bit_matrix, log_temps,
                  n_heads, tph, n_outputs):
    """Hot path; see ``TrainableAnchorsMultiHeadLUT.forward`` for the contract."""
    T_anchor, T_sign, T_sel = log_temps.exp().unbind()
    B = x.shape[0]
    n_tables, nap, _, D = anchor_logits.shape

    # 1. trainable anchors: softmax over input dims, STE (argmax fwd / softmax bwd).
    #    anchor_logits is input-independent so `sel` is [T, NAP, 2, D], computed
    #    once per forward (never materializes a [B, ..., D] tensor).
    sel_soft = F.softmax(anchor_logits / T_anchor, dim=-1)
    sel_hard = F.one_hot(anchor_logits.argmax(dim=-1), D).to(sel_soft.dtype)
    sel = _ste(sel_hard, sel_soft)

    av = torch.einsum("tpd,bd->btp", sel[:, :, 0], x)            # picked '+' values
    bv = torch.einsum("tpd,bd->btp", sel[:, :, 1], x)            # picked '-' values
    d = av - bv                                                  # [B, T, NAP]

    # 2. soft signs -> 3. bitmatrix match scores
    p = d / (T_sign + d.abs())                                   # soft sign in (-1, 1)
    ts = torch.einsum("btp,pk->btk", p, bit_matrix.to(p.dtype))  # [B, T, K]

    # 4. row selection: softmax over K, STE. argmax(ts) == hard sign-pack row.
    row_soft = F.softmax(ts / T_sel, dim=-1)
    row_hard = F.one_hot(ts.argmax(dim=-1), ts.shape[-1]).to(row_soft.dtype)
    row = _ste(row_hard, row_soft)                              # [B, n_tables, K]

    # Gather selected rows and sum over tables_per_head. The head-sum is folded
    # into the contraction (over t and k jointly) so the [B, n_tables, n_outputs]
    # per-table tensor is never materialized.
    K = weights.shape[1]
    row_h = row.view(B, n_heads, tph, K)
    w_h = weights.view(n_heads, tph, K, n_outputs)
    return torch.einsum("bhtk,htko->bho", row_h, w_h)          # [B, n_heads, n_outputs]


class TrainableAnchorsMultiHeadLUT(nn.Module):
    """Multi-head LUT with learnable anchoring (pure-PyTorch, autograd, STE).

    Args:
        input_dim: Dimension of the input vector ``x`` (``[B, input_dim]``).
        n_heads: Number of heads.
        n_outputs: Output dims per head.
        n_anchor_pairs: NAP. Each table has ``K = 2**NAP`` rows and NAP comparison
            bits, i.e. NAP learnable coordinate *pairs*.
        tables_per_head: Lookup tables per head (default 1). Outputs are summed
            over the ``tables_per_head`` tables of a head.
        anchor_temp: Temperature of the anchor-selection softmax.
        sign_temp: ``T_soft`` for the soft sign ``d / (T_soft + |d|)``.
        select_temp: ``T_sel`` for the row-selection softmax.
        learnable_temps: If True, the three temperatures are learnable (stored in
            log space so they stay positive).
        anchor_init_std: Std of the anchor-logit init. Large values give nearly
            uniform/random initial anchors; the softmax sharpens as logits grow.
        weights_init_std: Std of the LUT-weight init.
        device: Device for the parameters/buffers.
    """

    def __init__(
        self,
        input_dim: int,
        n_heads: int,
        n_outputs: int,
        n_anchor_pairs: int,
        tables_per_head: int = 1,
        anchor_temp: float = 1.0,
        sign_temp: float = 0.5,
        select_temp: float = 0.5,
        learnable_temps: bool = True,
        anchor_init_std: float = 1.0,
        weights_init_std: float = 0.001,
        random_seed=None,
        device=None,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.n_heads = n_heads
        self.n_outputs = n_outputs
        self.nap = n_anchor_pairs
        self.tph = tables_per_head
        self.n_tables = n_heads * tables_per_head
        self.K = 1 << n_anchor_pairs

        dev = device or torch.device("cpu")
        gen = None
        if random_seed is not None:
            gen = torch.Generator(device=dev).manual_seed(int(random_seed))

        # Trainable anchors: selection logits [n_tables, NAP, 2, input_dim].
        # Dim -2: {0 -> '+' coordinate (a), 1 -> '-' coordinate (b)}.
        self.anchor_logits = nn.Parameter(
            torch.randn(self.n_tables, self.nap, 2, input_dim, device=dev, generator=gen)
            * anchor_init_std
        )
        # LUT table weights: [n_tables, K, n_outputs].
        self.weights = nn.Parameter(
            torch.randn(self.n_tables, self.K, n_outputs, device=dev, generator=gen)
            * weights_init_std
        )
        self.register_buffer("bit_matrix", _bit_matrix_msb(self.nap, device))

        log_t = torch.log(torch.tensor([anchor_temp, sign_temp, select_temp]))
        if learnable_temps:
            self.log_temps = nn.Parameter(log_t.to(device))
        else:
            self.register_buffer("log_temps", log_t.to(device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x``: ``[B, input_dim]`` -> ``[B, n_heads, n_outputs]``."""
        return _forward_body(
            x, self.anchor_logits, self.weights, self.bit_matrix, self.log_temps,
            self.n_heads, self.tph, self.n_outputs,
        )


# =============================================================================
# Soft-anchor PAIR variant (directed-gradient anchoring)
# =============================================================================
# Diagnosis of TrainableAnchorsMultiHeadLUT (exp533/534): hardening the anchor
# selection with argmax-STE gives an UNDIRECTED gradient — the softmax lives only
# in the backward (the forward operand is a hard single coordinate, the
# `- softmax.detach()` cancels it), so flips are a per-batch value-correlation
# random walk that pays a re-fit tax. bpb tracked flip-count upward, monotone.
#
# Fix (this class): move the softmax INTO the forward. Each anchor is a learned
# soft convex blend of coordinates, and we compare TWO of them:
#     a = softmax(alpha / tau) @ x        # soft anchor "+"
#     b = softmax(beta  / tau) @ x        # soft anchor "-"
#     bit = sign(a - b)                    # the PAIR comparison, explicit on fwd
# Now a, b move continuously with alpha, beta -> the boundary a=b rotates
# smoothly -> only near-boundary tokens flip -> the gradient is DIRECTED (no
# teleport, no re-fit tax). The difference also stays shift-invariant because
# softmax(alpha)-softmax(beta) is zero-sum (a genuine generalized difference).
#
# `anchor_tau` is a SCHEDULED buffer (a 0-dim tensor so torch.compile treats it
# as a runtime input, not a recompile trigger): the training loop anneals it from
# ~1.0 down to a very low value, so by the end each softmax collapses to one-hot
# and the model makes HARD pair decisions. Set `mod.hard = True` for an exact
# hard-argmax forward (cheap coordinate gather) to measure / deploy the hardened
# model.


@torch.compile
def _soft_anchor_forward(x, anchor_logits, weights, bit_matrix, log_temps,
                         anchor_tau, n_heads, tph, n_outputs):
    """Training/soft-eval forward: two soft-anchor blends compared as a pair."""
    T_sign, T_sel = log_temps.exp().unbind()
    B = x.shape[0]
    n_tables, nap, _, D = anchor_logits.shape

    # 1. soft anchors: softmax over input dims at the SCHEDULED temperature tau.
    #    input-independent -> [T, NAP, 2, D], computed once.
    sm = F.softmax(anchor_logits / anchor_tau, dim=-1)
    a = torch.einsum("tpd,bd->btp", sm[:, :, 0], x)            # soft anchor "+"
    b = torch.einsum("tpd,bd->btp", sm[:, :, 1], x)            # soft anchor "-"
    d = a - b                                                  # [B, T, NAP] (zero-sum diff)

    # 2. soft signs -> 3. bitmatrix match scores
    p = d / (T_sign + d.abs())
    ts = torch.einsum("btp,pk->btk", p, bit_matrix.to(p.dtype))  # [B, T, K]

    # 4. row selection: softmax over K, STE (this selection was always directed).
    row_soft = F.softmax(ts / T_sel, dim=-1)
    row_hard = F.one_hot(ts.argmax(dim=-1), ts.shape[-1]).to(row_soft.dtype)
    row = _ste(row_hard, row_soft)

    K = weights.shape[1]
    row_h = row.view(B, n_heads, tph, K)
    w_h = weights.view(n_heads, tph, K, n_outputs)
    return torch.einsum("bhtk,htko->bho", row_h, w_h)


def _hard_anchor_forward(x, anchor_logits, weights, n_heads, tph, n_outputs):
    """Exact hard forward: each anchor is its argmax coordinate, hard sign-pack
    row, gather. The deployable / hardened model (no soft blend, no STE)."""
    B = x.shape[0]
    n_tables, nap, _, D = anchor_logits.shape
    a_idx = anchor_logits[:, :, 0].argmax(dim=-1)             # [T, NAP]
    b_idx = anchor_logits[:, :, 1].argmax(dim=-1)
    a = x[:, a_idx]                                           # [B, T, NAP]
    b = x[:, b_idx]
    d = a - b
    powers = (1 << torch.arange(nap - 1, -1, -1, device=x.device, dtype=torch.long))
    row = ((d > 0).long() * powers.view(1, 1, -1)).sum(-1)    # [B, T]
    tix = torch.arange(n_tables, device=x.device).view(1, -1).expand(B, -1)
    out = weights[tix, row]                                   # [B, T, n_outputs]
    return out.view(B, n_heads, tph, n_outputs).sum(dim=2)


class SoftAnchorPairMHLUT(nn.Module):
    """Multi-head LUT whose anchor pairs are two LEARNED SOFT blends compared by a
    sign bit; the anchor-selection gradient is directed (softmax in the forward),
    and the softmax temperature is annealed to near-zero so the final model makes
    hard pair decisions. See the module-level note above for the why.

    Args mirror ``TrainableAnchorsMultiHeadLUT`` except ``anchor_temp`` is replaced
    by ``anchor_tau_init`` (the temperature is a scheduled buffer set externally).
    """

    def __init__(
        self,
        input_dim: int,
        n_heads: int,
        n_outputs: int,
        n_anchor_pairs: int,
        tables_per_head: int = 1,
        sign_temp: float = 0.5,
        select_temp: float = 0.5,
        learnable_temps: bool = True,
        anchor_init_std: float = 1.0,
        anchor_tau_init: float = 1.0,
        weights_init_std: float = 0.001,
        random_seed=None,
        device=None,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.n_heads = n_heads
        self.n_outputs = n_outputs
        self.nap = n_anchor_pairs
        self.tph = tables_per_head
        self.n_tables = n_heads * tables_per_head
        self.K = 1 << n_anchor_pairs
        self.hard = False  # set True to use the exact hard-argmax forward

        dev = device or torch.device("cpu")
        gen = None
        if random_seed is not None:
            gen = torch.Generator(device=dev).manual_seed(int(random_seed))

        # alpha/beta selection logits: [n_tables, NAP, 2, input_dim]; dim -2 = {0:+, 1:-}.
        self.anchor_logits = nn.Parameter(
            torch.randn(self.n_tables, self.nap, 2, input_dim, device=dev, generator=gen)
            * anchor_init_std
        )
        self.weights = nn.Parameter(
            torch.randn(self.n_tables, self.K, n_outputs, device=dev, generator=gen)
            * weights_init_std
        )
        self.register_buffer("bit_matrix", _bit_matrix_msb(self.nap, device))
        # Scheduled anchor temperature (0-dim tensor -> torch.compile runtime input).
        self.register_buffer("anchor_tau", torch.tensor(float(anchor_tau_init), device=dev))

        # Only T_sign, T_sel are learnable now (anchor temp is the scheduled tau).
        log_t = torch.log(torch.tensor([sign_temp, select_temp]))
        if learnable_temps:
            self.log_temps = nn.Parameter(log_t.to(dev))
        else:
            self.register_buffer("log_temps", log_t.to(dev))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x``: ``[B, input_dim]`` -> ``[B, n_heads, n_outputs]``."""
        if self.hard:
            return _hard_anchor_forward(
                x, self.anchor_logits, self.weights,
                self.n_heads, self.tph, self.n_outputs,
            )
        return _soft_anchor_forward(
            x, self.anchor_logits, self.weights, self.bit_matrix, self.log_temps,
            self.anchor_tau, self.n_heads, self.tph, self.n_outputs,
        )
