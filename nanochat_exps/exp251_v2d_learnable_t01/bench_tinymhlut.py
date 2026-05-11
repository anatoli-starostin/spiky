"""Bench TinyMultiHeadLut forward (no autograd) at the v_lut/qk/out_proj shapes
used by exp251 — just to know the absolute floor on what `forward` can be.
"""
import time
import torch

from spiky.lutorch.tiny_multi_head_lut import TinyMultiHeadLut
from spiky.lutorch.lut_helpers import AnchorSamplingPolicy

DEVICE = "cuda"

CONFIGS = [
    dict(name="qk_joint",   input_dim=96, n_outputs=128, nap=6, tph=256, n_heads=6),
    dict(name="v_lut",      input_dim=96, n_outputs=32,  nap=8, tph=256, n_heads=6),
    dict(name="out_proj_L0",input_dim=192,n_outputs=96,  nap=6, tph=2048,n_heads=1),
    dict(name="out_proj_L2",input_dim=192,n_outputs=96,  nap=6, tph=1024,n_heads=1),
]
B = 8 * 512
print(f"=== TinyMHLut forward, B={B} ===")
for cfg in CONFIGS:
    mod = TinyMultiHeadLut(
        input_dim=cfg["input_dim"],
        n_heads=cfg["n_heads"],
        n_outputs=cfg["n_outputs"],
        n_anchor_pairs=cfg["nap"],
        tables_per_head=cfg["tph"],
        weight_dtype=torch.bfloat16,
        anchor_sampling_policy=AnchorSamplingPolicy.CANONICAL_FULL_COVERAGE,
        random_seed=0, device=DEVICE,
    ).to(DEVICE)
    x = torch.randn(B, cfg["input_dim"], device=DEVICE, requires_grad=True)
    target = torch.randn(B, cfg["n_heads"], cfg["n_outputs"], device=DEVICE, dtype=torch.bfloat16)
    # Warmup
    for _ in range(8):
        out = mod(x); loss = (out - target).square().sum(); loss.backward()
        x.grad = None
        for p in mod.parameters(): p.grad = None
    n_iter = 40
    fwd_evts = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(n_iter)]
    bwd_evts = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(n_iter)]
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    for i in range(n_iter):
        fwd_evts[i][0].record(); out = mod(x); fwd_evts[i][1].record()
        loss = (out - target).square().sum()
        bwd_evts[i][0].record(); loss.backward(); bwd_evts[i][1].record()
        x.grad = None
        for p in mod.parameters(): p.grad = None
    torch.cuda.synchronize()
    fwd_ms = sum(s.elapsed_time(e) for s, e in fwd_evts) / n_iter
    bwd_ms = sum(s.elapsed_time(e) for s, e in bwd_evts) / n_iter
    peak = torch.cuda.max_memory_allocated() / 1e6
    print(f"  {cfg['name']:<14s}  fwd={fwd_ms:6.3f} ms  bwd(STE)={bwd_ms:6.3f} ms  total={fwd_ms+bwd_ms:6.3f} ms  peak={peak:7.1f} MB")
