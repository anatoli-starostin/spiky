"""Microbenchmark of SoftMultiHeadLUT in the exact shapes used by exp251.

Per layer in exp251 we run three SoftMultiHeadLUT calls; each dominates a
different cost center, so we benchmark all three plus an aggregated layer.

Outputs forward time, forward+backward time, and peak memory (allocator).
"""
import time
import torch
import torch.nn as nn

from spiky.lutorch.soft_multi_head_lut import SoftMultiHeadLUT
from spiky.lutorch.lut_helpers import AnchorSamplingPolicy

DEVICE = "cuda"
torch.manual_seed(0)
torch.backends.cuda.matmul.allow_tf32 = True

# Match exp251 dims
B_TOK = 8 * 512  # device_batch_size * context_size = 4096

CONFIGS = [
    dict(name="qk_joint",   input_dim=96, n_outputs=128, nap=6, tph=256, n_heads=6),
    dict(name="v_lut",      input_dim=96, n_outputs=32,  nap=8, tph=256, n_heads=6),
    dict(name="out_proj_L0",input_dim=192,n_outputs=96,  nap=6, tph=2048,n_heads=1),
    dict(name="out_proj_L2",input_dim=192,n_outputs=96,  nap=6, tph=1024,n_heads=1),
]

KW_BASE = dict(
    weight_dtype=torch.float32,
    anchor_sampling_policy=AnchorSamplingPolicy.CANONICAL_FULL_COVERAGE,
    initial_weights_noise=0.001,
    soft_score_temp=0.5,
    select_temp=0.5,
    gumbel=False,
    hard=True,
    learnable_temps=True,
    use_bf16=True,
    compile_forward=False,  # bench eager first
)


def bench(mod, x, label, n_warm=8, n_iter=40, do_backward=True):
    target = torch.randn(x.shape[0], mod.n_heads, mod.n_outputs, device=DEVICE)
    # warmup
    for _ in range(n_warm):
        out = mod(x)
        if do_backward:
            loss = (out - target).square().sum()
            loss.backward()
            x.grad = None
            for p in mod.parameters():
                p.grad = None
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    t0 = time.time()
    for _ in range(n_iter):
        out = mod(x)
        if do_backward:
            loss = (out - target).square().sum()
            loss.backward()
            x.grad = None
            for p in mod.parameters():
                p.grad = None
    torch.cuda.synchronize()
    dt = (time.time() - t0) / n_iter * 1000
    peak_mb = torch.cuda.max_memory_allocated() / 1e6
    suffix = " (fwd+bwd)" if do_backward else " (fwd only)"
    print(f"  {label:<28s} {dt:6.2f} ms{suffix:<12s}  peak={peak_mb:7.1f} MB")
    return dt


for cfg in CONFIGS:
    print(f"\n=== {cfg['name']}  in={cfg['input_dim']} out={cfg['n_outputs']} "
          f"nap={cfg['nap']} tph={cfg['tph']} H={cfg['n_heads']}  B={B_TOK} ===")
    mod = SoftMultiHeadLUT(
        input_dim=cfg["input_dim"],
        n_outputs=cfg["n_outputs"],
        n_anchor_pairs=cfg["nap"],
        tables_per_head=cfg["tph"],
        n_heads=cfg["n_heads"],
        device=DEVICE,
        **KW_BASE,
    ).to(DEVICE)
    x = torch.randn(B_TOK, cfg["input_dim"], device=DEVICE, requires_grad=True)
    bench(mod, x, "eager", do_backward=False)
    bench(mod, x, "eager", do_backward=True)

    mod.forward = torch.compile(mod.forward, dynamic=True)
    bench(mod, x, "compiled", do_backward=False)
    bench(mod, x, "compiled", do_backward=True)
