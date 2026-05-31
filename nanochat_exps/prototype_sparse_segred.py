#!/usr/bin/env python3
"""Prototype: replace scatter_add with segment_reduce in sparse_scatter forward.

Compares correctness and speed of:
  A. Current path:  out.scatter_add_(2, idx, blended_pt.reshape(B, H, -1))
  B. New path:      gather using inverse map → segment_reduce by slot_offsets

Same numerical result; B avoids atomic contention.
"""
import sys, time
import torch
import torch.nn.functional as F

sys.path.insert(0, '/home/starost/spiky/src')
from spiky.lutorch.tiny_multi_head_lut import (
    TinyMultiHeadLut, _build_sparse_scatter_inverse_map,
)

DEV = torch.device('cuda:0')
B, T = 16, 512
BT = B * T
IN_DIM = 192
E = 384

m = TinyMultiHeadLut(
    input_dim=IN_DIM, n_heads=1, n_outputs=192, n_anchor_pairs=6,
    tables_per_head=512,
    weight_dtype=torch.float32, random_seed=0, device=DEV,
    backward_mode='hybrid_smooth',
    sparse_scatter_n_outputs=E, sparse_scatter_seed=11,
    use_bf16=False, learnable_temps=True,
)

x = torch.randn(BT, IN_DIM, device=DEV)

# Compute the per-table blended manually (replicates _hybrid_smooth_lut_fwd_body_scatter).
def manual_blended(x, m):
    weights = m.weights
    anchor_a = m.soft_anchor_a_long
    anchor_b = m.soft_anchor_b_long
    powers = m.soft_powers
    n_heads, tph, table_dim = m.n_heads, m.tables_per_head, m.table_dim
    n_tables = anchor_a.shape[0]
    n_per = weights.shape[2]   # per-table n_outputs
    T_soft = m.log_soft_score_temp.detach().exp()
    T_sel = m.log_select_temp.detach().exp()
    B_ = x.shape[0]

    d = x[:, anchor_a] - x[:, anchor_b]
    bits = (d > 0).to(torch.int64)
    main_index = (bits * powers.view(1, 1, -1)).sum(dim=-1)
    abs_d = d.abs()
    p_star = abs_d.argmin(dim=-1)
    flip_mask = powers.to(main_index.dtype)[p_star]
    alt_index = main_index ^ flip_mask
    d_min = abs_d.gather(-1, p_star.unsqueeze(-1)).squeeze(-1)
    delta_ts = 2.0 * d_min / (T_soft + d_min)
    u = torch.sigmoid(-delta_ts / T_sel)
    main_w = 1.0 - u

    table_offset = torch.arange(n_tables, device=weights.device, dtype=main_index.dtype) * table_dim
    w_flat = weights.view(n_tables * table_dim, n_per)
    mfi = (main_index + table_offset.view(1, -1)).reshape(-1)
    afi = (alt_index + table_offset.view(1, -1)).reshape(-1)
    main_rows = F.embedding(mfi, w_flat).view(B_, n_tables, n_per)
    alt_rows = F.embedding(afi, w_flat).view(B_, n_tables, n_per)
    blended = main_rows * main_w.unsqueeze(-1) + alt_rows * u.unsqueeze(-1)
    return blended, n_heads, tph, n_per


def path_A_scatter(blended, scatter_indices, n_heads, tph, n_per, sparse_n):
    """Current path: scatter_add."""
    B_ = blended.shape[0]
    blended_pt = blended.view(B_, n_heads, tph, n_per)
    out = blended_pt.new_zeros(B_, n_heads, sparse_n)
    idx = scatter_indices.unsqueeze(0).expand(B_, -1, -1, -1).reshape(B_, n_heads, tph * n_per)
    out.scatter_add_(2, idx, blended_pt.reshape(B_, n_heads, -1))
    return out


def path_B_segred(blended, slot_offsets, contrib_global_t, contrib_local_i,
                   n_heads, tph, n_per, sparse_n):
    """New path: gather via inverse map + segment_reduce."""
    B_ = blended.shape[0]
    n_tables = blended.shape[1]
    flat_idx_per_head = contrib_global_t * n_per + contrib_local_i   # [H, T*N]
    flat_idx_all = flat_idx_per_head.reshape(-1)
    blended_flat = blended.view(B_, n_tables * n_per)
    n_per_head = tph * n_per
    gathered = blended_flat.index_select(1, flat_idx_all).view(B_, n_heads, n_per_head)
    offsets_b = slot_offsets.unsqueeze(0).expand(B_, -1, -1).contiguous()
    out = torch.segment_reduce(gathered, 'sum', offsets=offsets_b, axis=2)
    return out


# Build inverse map.
slot_offsets, contrib_global_t, contrib_local_i = _build_sparse_scatter_inverse_map(
    m.scatter_indices, m.sparse_scatter_n_outputs,
)

# ---- correctness check ----
blended, n_heads, tph, n_per = manual_blended(x, m)
out_a = path_A_scatter(blended, m.scatter_indices, n_heads, tph, n_per, m.sparse_scatter_n_outputs)
out_b = path_B_segred(blended, slot_offsets, contrib_global_t, contrib_local_i,
                       n_heads, tph, n_per, m.sparse_scatter_n_outputs)
diff = (out_a - out_b).abs().max().item()
rel  = diff / out_a.abs().max().item()
print(f'correctness: |A - B|_max = {diff:.3e}  rel={rel:.3e}')
assert diff < 1e-3, f'mismatch! diff={diff}'

# ---- timing ----
def time_path(fn, name, n=100, warm=10):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(n):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / n

def fn_A():
    return path_A_scatter(blended, m.scatter_indices, n_heads, tph, n_per,
                           m.sparse_scatter_n_outputs)
def fn_B():
    return path_B_segred(blended, slot_offsets, contrib_global_t, contrib_local_i,
                          n_heads, tph, n_per, m.sparse_scatter_n_outputs)

a_ms = time_path(fn_A, 'A scatter')
b_ms = time_path(fn_B, 'B segred')
print(f'A (scatter_add): {a_ms:.3f} ms')
print(f'B (segred):      {b_ms:.3f} ms')
print(f'B/A = {b_ms/a_ms:.2f}x  (lower = faster)')
