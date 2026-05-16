"""Benchmark exp303-shape (dense) vs exp318-shape (sparse) residual_lut.

Both run on cuda:0 in this process — if exp318 training is also on cuda:0
each module sees the same contention, so RELATIVE timing is what we report.

Run:
    .venv/bin/python -u nanochat_exps/bench_resid_lut_dense_vs_sparse.py
"""
import time
import statistics
import torch
import torch.nn.functional as F

from spiky.lutorch.tiny_multi_head_lut import TinyMultiHeadLut

DEVICE = torch.device('cuda:0')
torch.manual_seed(0)

B_T = 4096                          # device_batch_size * context_size = 8 * 512
E   = 64
D   = 384
NAP = 6
TPH_DENSE  = 2048                   # exp303
TPH_SPARSE = 4096                   # exp318
N_SPARSE   = 64                     # exp318 per-table n_outputs

_TINY_SOFT_KWARGS = dict(
    backward_mode='soft',
    soft_score_temp=0.5,
    select_temp=0.5,
    learnable_temps=True,
    use_bf16=True,
    argmax_noise_eps=0.002,
    weight_dtype=torch.bfloat16,
)

def make_dense():
    """exp303 shape: dense, tph=2048, n_out=D=384."""
    return TinyMultiHeadLut(
        input_dim=E, n_heads=1, n_outputs=D,
        n_anchor_pairs=NAP, tables_per_head=TPH_DENSE,
        random_seed=42, device=DEVICE,
        **_TINY_SOFT_KWARGS,
    ).to(DEVICE)

def make_sparse():
    """exp318 shape: tph=4096, n_out=64, sparse_scatter_n_outputs=D=384."""
    return TinyMultiHeadLut(
        input_dim=E, n_heads=1, n_outputs=N_SPARSE,
        n_anchor_pairs=NAP, tables_per_head=TPH_SPARSE,
        random_seed=42, device=DEVICE,
        sparse_scatter_n_outputs=D,
        sparse_scatter_seed=4242,
        **_TINY_SOFT_KWARGS,
    ).to(DEVICE)

def bench(name, model, n_warmup=10, n_iter=50):
    x = torch.randn(B_T, E, device=DEVICE, requires_grad=True)
    g_seed = torch.randn(B_T, 1, D, device=DEVICE)
    # Warmup (compiles, allocates, hits real steady state).
    for _ in range(n_warmup):
        y = model(x)
        loss = (y * g_seed).sum()
        loss.backward()
        x.grad = None
        for p in model.parameters():
            p.grad = None
    torch.cuda.synchronize()

    # Forward only
    fwd_times = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        y = model(x)
        torch.cuda.synchronize()
        fwd_times.append(time.perf_counter() - t0)

    # Forward + backward
    fb_times = []
    for _ in range(n_iter):
        x.grad = None
        for p in model.parameters():
            p.grad = None
        t0 = time.perf_counter()
        y = model(x)
        loss = (y * g_seed).sum()
        loss.backward()
        torch.cuda.synchronize()
        fb_times.append(time.perf_counter() - t0)

    fwd_med = statistics.median(fwd_times) * 1000
    fb_med  = statistics.median(fb_times)  * 1000
    bwd_med = fb_med - fwd_med
    params = sum(p.numel() for p in model.parameters())
    peak_mb = torch.cuda.max_memory_allocated() / (1024**2)
    print(f"{name:18s}  params={params/1e6:6.2f}M  fwd={fwd_med:7.3f}ms  bwd={bwd_med:7.3f}ms  fwd+bwd={fb_med:7.3f}ms  peak={peak_mb:7.1f} MB")
    return fwd_med, bwd_med, fb_med, params

print(f"B*T={B_T}, E={E}, D={D}, NAP={NAP}")
print(f"--- exp303 shape (dense, tph={TPH_DENSE}, n_out={D}) ---")
torch.cuda.reset_peak_memory_stats()
m_dense = make_dense()
fwd_d, bwd_d, fb_d, p_d = bench("exp303_dense", m_dense)
del m_dense
torch.cuda.empty_cache()

print(f"--- exp318 shape (sparse, tph={TPH_SPARSE}, n_out={N_SPARSE}, scatter -> {D}) ---")
torch.cuda.reset_peak_memory_stats()
m_sparse = make_sparse()
fwd_s, bwd_s, fb_s, p_s = bench("exp318_sparse", m_sparse)

print()
print(f"=== ratios (sparse / dense) ===")
print(f"  fwd:     {fwd_s/fwd_d:.2f}x  ({fwd_s:.3f}ms / {fwd_d:.3f}ms)")
print(f"  bwd:     {bwd_s/bwd_d:.2f}x  ({bwd_s:.3f}ms / {bwd_d:.3f}ms)")
print(f"  fwd+bwd: {fb_s/fb_d:.2f}x  ({fb_s:.3f}ms / {fb_d:.3f}ms)")
print(f"  params:  {p_s/p_d:.2f}x  ({p_s/1e6:.1f}M / {p_d/1e6:.1f}M)")
