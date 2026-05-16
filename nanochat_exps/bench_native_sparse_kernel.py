"""Validate + bench the new tiny_mhlut_sparse_scatter_forward kernel.

Builds the precomputed inverse map from a scatter_indices buffer, runs
the native kernel, compares to F.embedding + scatter_add_ reference, then
benches.
"""
import time, statistics, torch
import torch.nn.functional as F

import lutorch_cuda

DEV = torch.device('cuda:0')
torch.manual_seed(0)

# exp318 shapes.
B, H, TPH, NS, D = 4096, 1, 4096, 64, 384
NAP = 6
table_dim = 1 << NAP
n_lt = H * TPH

weights = torch.randn(n_lt, table_dim, NS, device=DEV, dtype=torch.bfloat16) * 0.01
lookup_indices = torch.randint(0, table_dim, (B, n_lt), device=DEV, dtype=torch.long)

# Build scatter_indices like TinyMHL would (iid sampling).
gen = torch.Generator().manual_seed(42)
scatter_indices = torch.empty(H, TPH, NS, dtype=torch.long)
for h in range(H):
    for t in range(TPH):
        scatter_indices[h, t] = torch.randperm(D, generator=gen)[:NS]
scatter_indices = scatter_indices.to(DEV)

# --- Build inverse-map buffers for the kernel ---
def build_inverse_map(scatter_indices):
    """scatter_indices: [H, TPH, NS] long. Returns slot_offsets [H, D+1],
    contrib_table [H, TPH*NS] (global table idx), contrib_local_i [H, TPH*NS]."""
    H_, T_, N_ = scatter_indices.shape
    flat_dest = scatter_indices.reshape(H_, T_ * N_)
    sorted_dest, perm = flat_dest.sort(dim=1, stable=True)
    flat_src = torch.arange(T_ * N_, device=scatter_indices.device, dtype=torch.long).unsqueeze(0).expand(H_, -1)
    sorted_src = flat_src.gather(1, perm)
    contrib_local_t = sorted_src // N_
    contrib_local_i = sorted_src % N_
    head_offset = (torch.arange(H_, device=scatter_indices.device, dtype=torch.long) * T_).unsqueeze(1)
    contrib_global_t = head_offset + contrib_local_t
    D_ = int(sorted_dest.max().item()) + 1
    counts = torch.zeros(H_, D_, device=scatter_indices.device, dtype=torch.long)
    counts.scatter_add_(1, sorted_dest, torch.ones_like(sorted_dest))
    slot_offsets = torch.zeros(H_, D_ + 1, device=scatter_indices.device, dtype=torch.long)
    slot_offsets[:, 1:] = counts.cumsum(dim=1)
    return slot_offsets, contrib_global_t, contrib_local_i

slot_offsets, contrib_table, contrib_local_i = build_inverse_map(scatter_indices)
assert slot_offsets.shape == (H, D + 1), slot_offsets.shape
assert contrib_table.shape == (H, TPH * NS), contrib_table.shape

# --- Reference (current path: F.embedding + scatter_add_) ---
def reference():
    weights_flat = weights.view(n_lt * table_dim, NS)
    table_offset = torch.arange(n_lt, device=DEV, dtype=torch.long) * table_dim
    flat_idx = (lookup_indices + table_offset.view(1, -1)).reshape(-1)
    per_table = F.embedding(flat_idx, weights_flat).view(B, H, TPH, NS)
    out = torch.zeros(B, H, D, device=DEV, dtype=weights.dtype)
    si = scatter_indices.unsqueeze(0).expand(B, -1, -1, -1).reshape(B, H, TPH * NS)
    out.scatter_add_(2, si, per_table.reshape(B, H, -1))
    return out

# --- Native kernel ---
mgr = lutorch_cuda.get_lutorch_manager()
def native():
    return mgr.tiny_mhlut_sparse_scatter_forward(
        weights, lookup_indices, slot_offsets, contrib_table, contrib_local_i,
        H, TPH, D,
    )

# Correctness check (small subset of B).
small_B = 8
lookup_small = lookup_indices[:small_B]
weights_flat = weights.view(n_lt * table_dim, NS)
table_offset = torch.arange(n_lt, device=DEV, dtype=torch.long) * table_dim
flat_idx = (lookup_small + table_offset.view(1, -1)).reshape(-1)
per_table = F.embedding(flat_idx, weights_flat).view(small_B, H, TPH, NS)
ref = torch.zeros(small_B, H, D, device=DEV, dtype=weights.dtype)
si = scatter_indices.unsqueeze(0).expand(small_B, -1, -1, -1).reshape(small_B, H, TPH * NS)
ref.scatter_add_(2, si, per_table.reshape(small_B, H, -1))

got = mgr.tiny_mhlut_sparse_scatter_forward(
    weights, lookup_small, slot_offsets, contrib_table, contrib_local_i,
    H, TPH, D,
)
diff = (ref - got).abs().max().item()
# float math reorder + bf16 accumulation differences. Native uses fp32 accumulator
# (more accurate). Reference uses bf16 atomicAdd (lossy at large bag sizes).
print(f"correctness: max |ref - native| = {diff:.5f}  (bf16 sum reorder; ~683 adds/slot)")

# Compare against an fp32-accumulated reference (computed in fp32, cast at end).
weights_fp32 = weights.float()
weights_flat_fp32 = weights_fp32.view(n_lt * table_dim, NS)
per_table_fp32 = F.embedding(flat_idx, weights_flat_fp32).view(small_B, H, TPH, NS)
ref_fp32 = torch.zeros(small_B, H, D, device=DEV, dtype=torch.float32)
ref_fp32.scatter_add_(2, si, per_table_fp32.reshape(small_B, H, -1))
diff_fp32 = (ref_fp32.to(weights.dtype) - got).abs().max().item()
print(f"correctness vs fp32-acc ref:   max diff = {diff_fp32:.5f}")
assert diff_fp32 < 0.05, f"fp32-acc correctness failed: {diff_fp32}"

# --- Bench ---
def bench(name, fn, n=30):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    t = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        t.append(time.perf_counter() - t0)
    print(f"{name:40s}  {statistics.median(t)*1000:7.3f} ms")

print(f"B={B}, H={H}, TPH={TPH}, NS={NS}, D={D}, table_dim={table_dim}")
bench("reference (F.embedding + scatter_add_)", reference)
bench("native CUDA kernel", native)
