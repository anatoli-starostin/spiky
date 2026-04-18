"""Profile BitPermutationLUT vs PermutationalLut (dominance, STE).

Measures forward (no_grad) and forward+backward (train) times, and reports
weight-memory footprint (bit vs float32).
"""
import time

import torch

from spiky.lutorch.bit_permutation_lut import BitPermutationLUT
from spiky.lutorch.lut_helpers import AnchorSamplingPolicy
from spiky.lutorch.permutational_lut import PermutationalLut


DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


def sync():
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()


def bench(name: str, fn, n: int = 50, warmup: int = 10) -> float:
    for _ in range(warmup):
        fn()
    sync()
    t0 = time.time()
    for _ in range(n):
        fn()
    sync()
    ms = (time.time() - t0) / n * 1000.0
    print(f"  {name:55s} {ms:8.4f} ms")
    return ms


def bench_config(n_inputs, n_outputs, n_heads, input_nap, output_nap, tph, batch):
    print(
        f"\n=== n_inputs={n_inputs}, n_outputs={n_outputs}, n_heads={n_heads}, "
        f"input_nap={input_nap}, output_nap={output_nap}, tph={tph}, B={batch} ==="
    )

    bit_lut = BitPermutationLUT(
        n_inputs=n_inputs, n_outputs=n_outputs,
        n_heads=n_heads, input_nap=input_nap, output_nap=output_nap, tph=tph,
        random_seed=42, device=DEVICE,
    )
    perm_lut = PermutationalLut(
        n_inputs=n_inputs, n_outputs=n_outputs,
        input_nap=input_nap, output_nap=output_nap,
        n_heads=n_heads, tph=tph,
        pair_mode='scrambled', soft_mode='ste', return_dominance=True,
        scrambled_policy=AnchorSamplingPolicy.CANONICAL_DISTINCT,
        input_anchor_policy=AnchorSamplingPolicy.CANONICAL_DISTINCT,
        random_seed=42, device=DEVICE,
    ).to(DEVICE)

    # ---- Forward only (eval / no_grad) ----
    bit_lut.eval(); perm_lut.eval()
    x = torch.randn(batch, n_inputs, device=DEVICE)
    with torch.no_grad():
        t_fwd_bit  = bench("BitPermLUT       fwd (no_grad)", lambda: bit_lut(x))
        t_fwd_perm = bench("PermLut(STE+dom) fwd (no_grad)", lambda: perm_lut(x))
    print(f"  fwd speedup (perm / bit): {t_fwd_perm / t_fwd_bit:.2f}x")

    # ---- Forward + Backward (train) ----
    bit_lut.train(); perm_lut.train()

    x_bit = torch.randn(batch, n_inputs, device=DEVICE, requires_grad=True)
    def step_bit():
        out = bit_lut(x_bit)
        out.sum().backward()
        x_bit.grad = None
    t_fb_bit = bench("BitPermLUT       fwd+bwd", step_bit)

    x_perm = torch.randn(batch, n_inputs, device=DEVICE, requires_grad=True)
    def step_perm():
        out = perm_lut(x_perm)
        out.sum().backward()
        x_perm.grad = None
        if perm_lut.inner.projection.weights.grad is not None:
            perm_lut.inner.projection.weights.grad = None
    t_fb_perm = bench("PermLut(STE+dom) fwd+bwd", step_perm)

    print(f"  fwd+bwd speedup (perm / bit): {t_fb_perm / t_fb_bit:.2f}x")
    print(
        f"  bwd-only:  bit ≈ {(t_fb_bit - t_fwd_bit):.3f} ms,  "
        f"perm ≈ {(t_fb_perm - t_fwd_perm):.3f} ms"
    )

    # ---- Weight-memory footprint ----
    bit_bytes = bit_lut.bit_weights.numel() * bit_lut.bit_weights.element_size()
    perm_bytes = (
        perm_lut.inner.projection.weights.numel()
        * perm_lut.inner.projection.weights.element_size()
    )
    print(f"  weight bytes: bit={bit_bytes:,}  perm={perm_bytes:,}  compression={perm_bytes / bit_bytes:.1f}x")


if __name__ == "__main__":
    print(f"Device: {DEVICE}")
    configs = [
        # (n_inputs, n_outputs, n_heads, input_nap, output_nap, tph, batch)
        # Representative transformer shapes (from recent PermLut experiments).
        (32,  32, 4, 6,  6, 128, 16384),   # Q/K inner @ bs=128, ctx=128
        (32,  32, 4, 6, 16, 128, 16384),
        (32, 128, 4, 6, 32, 128, 16384),   # out_proj inner
        (64,  64, 4, 8, 32, 128, 16384),
        (32,  64, 4, 6, 10, 128, 16384),
        (32,  64, 4, 6, 33, 128, 16384),   # > 32-bit block
        (32,  32, 4, 6,  6, 128,  1024),
        (16,  16, 4, 4,  6,  64,  4096),
    ]
    for cfg in configs:
        bench_config(*cfg)
