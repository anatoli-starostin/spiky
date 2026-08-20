"""SoftAnnealLut — soft full-K weighted-sum forward with annealed temperatures.

Experiment-local (exp_n_0054); does NOT touch the shared FastMultiHeadLut /
CompressionMultiHeadLUT. It is a drop-in replacement for a per-head FastMHL
(n_heads=1) inside a CompressionMHL: it copies that FastMHL's tables, anchor
pairs, and ±1 cluster-code matrix, but replaces the hard gather + surrogate
backward with the SAME soft math run ON THE FORWARD PASS, differentiated by
plain autograd (no straight-through estimator, no custom backward).

Forward (faithful to FastMHL's _soft_lut_bwd_body surrogate, unpinned to all K):
    d         = x[:, anchor_a] - x[:, anchor_b]          # signed distances, per anchor pair
    soft_sign = d / (T_soft + |d|)                       # soft ±1; -> sign(d) as T_soft -> 0
    score_k   = <bit_matrix[:,k], soft_sign>             # agreement of input signs with row k's code
    w         = softmax(score / T_sel, dim=K)            # -> one-hot (argmax row) as T_sel -> 0
    out_table = sum_k w_k * weights[table, k, :]
    out       = bag-sum over tables_per_head             # [N, n_heads, n_outputs]
As (T_soft, T_sel) -> 0 the soft weighted-sum concentrates on a single row, i.e.
the output approaches the hard FastMHL lookup. Temps are NOT learnable here —
they are driven by an external anneal schedule (set_temps each step).
"""
import torch
import torch.nn as nn


@torch.compile(dynamic=True)
def _soft_anneal_body(x, weights, anchor_a, anchor_b, bit_matrix,
                      T_soft, T_sel, n_heads: int, tph: int):
    # x: [N, input_dim]; anchor_a/b: [n_tables, nap]; weights: [n_tables, K, n_out]
    d = x[:, anchor_a] - x[:, anchor_b]                          # [N, n_tables, nap]
    soft_sign = d / (T_soft + d.abs())                          # [N, n_tables, nap], in (-1, 1)
    score = torch.einsum("btp,pk->btk", soft_sign, bit_matrix.to(soft_sign.dtype))  # [N, n_tables, K]
    w = torch.softmax(score / T_sel, dim=-1)                    # [N, n_tables, K]
    out = torch.einsum("btk,tko->bto", w, weights)              # [N, n_tables, n_out]
    N = x.shape[0]
    n_out = weights.shape[2]
    return out.view(N, n_heads, tph, n_out).sum(dim=2)          # [N, n_heads, n_out]


class SoftAnnealLut(nn.Module):
    """Soft-forward + annealed-temp drop-in for a per-head FastMultiHeadLut.

    Build from an already-constructed FastMultiHeadLut `src` (n_heads=1); it takes
    over `src`'s tables/anchors/bit_matrix (so init is bit-identical to the hard
    baseline) and discards `src`'s learnable temperatures. Call `set_temps(...)`
    once per training step with the annealed values before the forward pass.
    """

    def __init__(self, src, temp_init: float = 0.5):
        super().__init__()
        self.n_heads = int(src.n_heads)
        self.tph = int(src.tables_per_head)
        self.n_anchor_pairs = int(src.n_anchor_pairs)
        self.table_dim = int(src.table_dim)
        self.n_outputs = int(src.n_outputs)
        # Own copies of the LUT tables (trainable) + routing state (buffers).
        self.weights = nn.Parameter(src.weights.detach().clone())          # [n_tables, K, n_out]
        self.register_buffer("anchor_a", src.soft_anchor_a_long.clone())   # [n_tables, nap]
        self.register_buffer("anchor_b", src.soft_anchor_b_long.clone())
        self.register_buffer("bit_matrix", src.soft_bit_matrix.clone())    # [nap, K] in {-1,+1}
        # Annealed temperatures (non-persistent buffers; driven by the schedule).
        self.register_buffer("T_soft", torch.tensor(float(temp_init)), persistent=False)
        self.register_buffer("T_sel", torch.tensor(float(temp_init)), persistent=False)

    @torch.no_grad()
    def set_temps(self, t_soft: float, t_sel: float):
        self.T_soft.fill_(float(t_soft))
        self.T_sel.fill_(float(t_sel))

    def forward(self, x):
        return _soft_anneal_body(
            x, self.weights, self.anchor_a, self.anchor_b, self.bit_matrix,
            self.T_soft, self.T_sel, self.n_heads, self.tph,
        )


def anneal_temp(step: int, n_steps: int, start: float, floor: float) -> float:
    """Exponential decay start -> floor over n_steps (frac clamped to [0,1])."""
    frac = min(max(step, 0) / max(n_steps, 1), 1.0)
    return float(start) * (float(floor) / float(start)) ** frac
