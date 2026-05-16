"""Find which sub-op of the sparse residual_lut forward dominates."""
import time, statistics, torch
import torch.nn.functional as F
DEV = torch.device('cuda:0')
torch.manual_seed(0)

B_T, E, D, NAP, TPH, NS = 4096, 64, 384, 6, 4096, 64
n_tables = TPH
table_dim = 1 << NAP
n_outputs = NS

weights = torch.randn(n_tables, table_dim, n_outputs, device=DEV, dtype=torch.bfloat16)
scatter_indices = torch.randint(0, D, (1, TPH, NS), device=DEV)
anchor_a = torch.randint(0, E, (TPH, NAP), device=DEV, dtype=torch.long)
anchor_b = torch.randint(0, E, (TPH, NAP), device=DEV, dtype=torch.long)
powers = torch.tensor([1 << i for i in range(NAP-1, -1, -1)], device=DEV, dtype=torch.long)

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

x = torch.randn(B_T, E, device=DEV)

def step_d():
    d = x[:, anchor_a] - x[:, anchor_b]
bench("1. d gather (anchor pairs)", step_d)

d = x[:, anchor_a] - x[:, anchor_b]
def step_index():
    bits = (d > 0).to(torch.int64)
    idx = (bits * powers.view(1, 1, -1)).sum(dim=-1)
bench("2. index from bits", step_index)

bits = (d > 0).to(torch.int64)
index = (bits * powers.view(1, 1, -1)).sum(dim=-1)

weights_flat = weights.view(n_tables * table_dim, n_outputs)
table_offset = torch.arange(n_tables, device=DEV, dtype=torch.long) * table_dim
flat_indices = (index + table_offset.view(1, -1)).reshape(-1)

def step_embed():
    out = F.embedding(flat_indices, weights_flat)
bench("3. F.embedding (gather)", step_embed)

out_per_table = F.embedding(flat_indices, weights_flat).view(B_T, 1, TPH, n_outputs)
B = B_T

def step_scatter():
    out = out_per_table.new_zeros(B, 1, D)
    idx = scatter_indices.unsqueeze(0).expand(B, -1, -1, -1).reshape(B, 1, TPH * n_outputs)
    out.scatter_add_(2, idx, out_per_table.reshape(B, 1, -1))
bench("4. scatter_add_ (just the scatter)", step_scatter)

def step_embed_bag_dense():
    """Dense reference: embedding_bag with mode='sum' over tph bags."""
    n_bags = B * 1
    offsets = torch.arange(n_bags, device=DEV, dtype=torch.long) * TPH
    o = F.embedding_bag(flat_indices, weights_flat, offsets=offsets, mode='sum')
bench("5. F.embedding_bag(sum)  (dense ref)", step_embed_bag_dense)
