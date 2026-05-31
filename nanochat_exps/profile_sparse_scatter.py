#!/usr/bin/env python3
"""Micro-benchmark hybrid_smooth sparse_scatter vs dense out_proj.

Matches exp655 (sparse, tph=512, n_out=192 -> wide=384) vs
exp654 (dense, tph=320, n_out=384). Same hybrid_smooth backward,
same B*T, same H*d_v input. Reports per-op CUDA time so we can see
exactly where the sparse path spends its budget.
"""
import sys
import torch
from torch.profiler import profile, record_function, ProfilerActivity

sys.path.insert(0, '/home/starost/spiky/src')
from spiky.lutorch.tiny_multi_head_lut import TinyMultiHeadLut

DEV = torch.device('cuda:0')

# Shapes match exp654/exp655 out_proj at one forward pass:
#   input_dim = H * d_v = 6 * 32 = 192
#   B * T = 16 * 512 = 8192 (device_batch_size * seq_len)
#   wide_out = E = 384
B, T = 16, 512
BT = B * T
IN_DIM = 6 * 32   # H * d_v
E = 384
NAP = 6

COMMON = dict(
    input_dim=IN_DIM,
    n_heads=1,
    n_anchor_pairs=NAP,
    weight_dtype=torch.float32,
    random_seed=0,
    device=DEV,
    backward_mode='hybrid_smooth',
    use_bf16=True,
    learnable_temps=True,
)

# Dense exp654 shape.
m_dense = TinyMultiHeadLut(
    n_outputs=E, tables_per_head=320, **COMMON,
)
# Sparse exp655 shape.
m_sparse = TinyMultiHeadLut(
    n_outputs=192, tables_per_head=512,
    sparse_scatter_n_outputs=E, sparse_scatter_seed=11,
    **COMMON,
)


def make_input():
    return torch.randn(BT, IN_DIM, device=DEV, requires_grad=True)


def make_grad_out():
    return torch.randn(BT, 1, E, device=DEV)


def run_pass(mod, x, grad_out, label):
    out = mod(x)
    out.backward(grad_out)
    # Zero grads for next iter.
    x.grad = None
    mod.weights.grad = None
    if mod.log_soft_score_temp.grad is not None:
        mod.log_soft_score_temp.grad = None
    if mod.log_select_temp.grad is not None:
        mod.log_select_temp.grad = None


# ---- warmup ----
x = make_input()
go = make_grad_out()
for _ in range(3):
    run_pass(m_dense, x, go, 'warm_dense')
    run_pass(m_sparse, x, go, 'warm_sparse')
torch.cuda.synchronize()


# ---- timing (wall clock) ----
def time_module(mod, label, n=50):
    x = make_input()
    go = make_grad_out()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(n):
        run_pass(mod, x, go, label)
    end.record()
    end.synchronize()
    ms = start.elapsed_time(end) / n
    return ms

dense_ms = time_module(m_dense, 'dense')
sparse_ms = time_module(m_sparse, 'sparse')
print(f'\n=== wall-time (fwd+bwd, average over 50 iters) ===')
print(f'  dense  (tph=320, n_out=384):      {dense_ms:.3f} ms')
print(f'  sparse (tph=512, n_out=192, w=384): {sparse_ms:.3f} ms')
print(f'  sparse / dense = {sparse_ms/dense_ms:.2f}x')


# ---- detailed profile of sparse path ----
print(f'\n=== sparse profiler (5 iters) ===')
x = make_input()
go = make_grad_out()
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=False,
) as prof_sparse:
    for _ in range(5):
        run_pass(m_sparse, x, go, 'sparse')
        torch.cuda.synchronize()
print(prof_sparse.key_averages().table(
    sort_by='self_cuda_time_total', row_limit=20,
))


# ---- detailed profile of dense path ----
print(f'\n=== dense profiler (5 iters) ===')
x = make_input()
go = make_grad_out()
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=False,
) as prof_dense:
    for _ in range(5):
        run_pass(m_dense, x, go, 'dense')
        torch.cuda.synchronize()
print(prof_dense.key_averages().table(
    sort_by='self_cuda_time_total', row_limit=20,
))
