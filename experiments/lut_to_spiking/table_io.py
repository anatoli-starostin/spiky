"""Load a real trained LUT table from the spiky repo and give it latency semantics.

Source: experiments/hyperplane_ffn/exp011_hyperplane_mhl_ffn_nap6_tph256_stack2_ln_resid
        (a trained HyperplaneMultiHeadLUT FFN, NAP=6 -> K=64 rows, tph=256, D_out=384).
The table body `weights[t] : [K, D_out]` has exactly FastMultiHeadLut semantics (the two
modules differ only in the index front-end), and the checkpoint also stores the
anchor-pair buffers, which is what a FastMHL front-end would use.

Latency semantics: a row W[k] in R^{D_out} is read as the first-spike latencies of the
output population.  Real values are mapped to integer ticks by a single affine map
(shared by the whole table, so cross-row comparisons stay meaningful) into [0, span].
"""
import torch

from paths import EXP011_CKPT as CKPT
PREFIX = "blocks.0.mlp.lut1"


def load_table(table_idx=0, d_out=None, span=120, ckpt=CKPT, prefix=PREFIX):
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    W = sd[f"{prefix}.weights"][table_idx].float()             # [K, D_out]
    a = sd[f"{prefix}.soft_anchor_a_long"][table_idx].long()   # [NAP]
    b = sd[f"{prefix}.soft_anchor_b_long"][table_idx].long()   # [NAP]
    if d_out is not None:
        W = W[:, :d_out]
    lo, hi = W.min().item(), W.max().item()
    lat = torch.round((W - lo) / (hi - lo) * span).long()      # [K, D_out] integer ticks
    return dict(W=W, lat=lat, anchor_a=a, anchor_b=b, span=span,
                K=W.shape[0], D=W.shape[1], NAP=a.numel(), lo=lo, hi=hi)


def bits_of(k, nap):
    """MSB-first bit pattern of row index k (same convention as FastMultiHeadLut)."""
    return [(k >> (nap - 1 - i)) & 1 for i in range(nap)]


def rank_of(x):
    """Ranks (0 = earliest) with ties broken by index, per row."""
    return torch.argsort(torch.argsort(x, dim=-1, stable=True), dim=-1, stable=True)
