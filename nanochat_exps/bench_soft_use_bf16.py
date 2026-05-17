"""Benchmark soft_use_bf16 on/off for the four TinyMHLut shapes used in exp407.

Per module: time forward + backward over many iters for use_bf16 in {True, False}.
Synthetic input matched to bs=16, ctx=512 (B*T = 8192).
"""
import time, torch
import torch.nn.functional as F

from spiky.lutorch.tiny_multi_head_lut import TinyMultiHeadLut
from spiky.lutorch.lut_helpers import AnchorSamplingPolicy

DEVICE = 'cuda'
torch.manual_seed(0)

# Mirror exp407 hyperparams
_COMMON = dict(
    weight_dtype=torch.float32,
    anchor_sampling_policy=AnchorSamplingPolicy.CANONICAL_FULL_COVERAGE,
    initial_weights_noise=0.001,
    backward_mode='soft',
    soft_score_temp=0.5,
    select_temp=0.5,
    learnable_temps=True,
    argmax_noise_eps=0.0,
)

MODULES = [
    # name,            input_dim, n_heads, n_outputs, NAP, tph, B
    ('qkv_lut',         48,        6,       2*64+16,   6,   16,  8192),
    ('v_lut',           48,        6,       16,        6,   128, 8192),
    ('out_proj',        6*16,      1,       48,        6,   512, 8192),
    ('residual_lut',    48,        1,       384,       6,   64,  8192),
]

WARMUP = 10
ITERS  = 50

def bench(name, input_dim, n_heads, n_outputs, nap, tph, B, use_bf16):
    mod = TinyMultiHeadLut(
        input_dim=input_dim,
        n_heads=n_heads,
        n_outputs=n_outputs,
        n_anchor_pairs=nap,
        tables_per_head=tph,
        random_seed=42,
        device=DEVICE,
        use_bf16=use_bf16,
        **_COMMON,
    ).to(DEVICE)
    x = torch.randn(B, input_dim, device=DEVICE, requires_grad=True)
    grad_target = torch.randn(B, n_heads, n_outputs, device=DEVICE)
    # warmup
    for _ in range(WARMUP):
        y = mod(x)
        y.backward(grad_target, retain_graph=False)
        x.grad = None
        for p in mod.parameters():
            p.grad = None
    torch.cuda.synchronize()
    # timed
    t0 = time.time()
    peak = 0
    torch.cuda.reset_peak_memory_stats()
    for _ in range(ITERS):
        y = mod(x)
        y.backward(grad_target, retain_graph=False)
        x.grad = None
        for p in mod.parameters():
            p.grad = None
    torch.cuda.synchronize()
    elapsed = (time.time() - t0) / ITERS * 1000  # ms / iter
    peak = torch.cuda.max_memory_allocated() / 1024**2  # MB
    return elapsed, peak

results = []
for spec in MODULES:
    name = spec[0]
    t_bf16, mem_bf16 = bench(*spec, use_bf16=True)
    t_fp32, mem_fp32 = bench(*spec, use_bf16=False)
    speedup = t_fp32 / t_bf16
    mem_red = mem_fp32 / mem_bf16
    results.append((name, t_bf16, t_fp32, speedup, mem_bf16, mem_fp32, mem_red))
    print(f'{name:14s}  bf16={t_bf16:6.2f} ms  fp32={t_fp32:6.2f} ms  speedup={speedup:.2f}x   peak bf16={mem_bf16:6.0f}MB  fp32={mem_fp32:6.0f}MB  ratio={mem_red:.2f}x')

print()
print('=== Summary ===')
print(f'{"module":14s} {"bf16(ms)":>10s} {"fp32(ms)":>10s} {"speedup":>10s} {"bf16(MB)":>10s} {"fp32(MB)":>10s}')
for r in results:
    print(f'{r[0]:14s} {r[1]:10.2f} {r[2]:10.2f} {r[3]:10.2f}x {r[4]:10.0f} {r[5]:10.0f}')
