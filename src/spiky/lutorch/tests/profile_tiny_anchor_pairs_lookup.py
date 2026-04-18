"""Profile TinyAnchorPairsLookup: CUDA vs PyTorch, fwd + fwd+bwd, across shapes."""
import os
import time

import torch

from spiky.lutorch.tiny_anchor_pairs_lookup import (
    TinyAnchorPairsLookup,
    _can_use_native_tiny_apl,
    _tiny_forward_cuda,
    _tiny_forward_pytorch,
    _tiny_backward_cuda,
    _tiny_backward_pytorch,
)


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
    elapsed_ms = (time.time() - t0) / n * 1000.0
    print(f"  {name:45s} {elapsed_ms:8.4f} ms")
    return elapsed_ms


def bench_config(input_dim: int, n_tables: int, n_anchor_pairs: int, batch_size: int):
    print(f"\n=== input_dim={input_dim}, n_tables={n_tables}, nap={n_anchor_pairs}, B={batch_size} ===")
    lut = TinyAnchorPairsLookup(
        input_dim=input_dim,
        n_tables=n_tables,
        n_anchor_pairs=n_anchor_pairs,
        random_seed=42,
        device=DEVICE,
    )
    x_ng = torch.randn(batch_size, input_dim, device=DEVICE)
    x = x_ng.clone().requires_grad_(True)

    # --- Forward-only (inference / no-grad) ---
    print("Forward only (no_grad):")
    with torch.no_grad():
        t_ng_cuda = bench("CUDA    fwd", lambda: _tiny_forward_cuda(x_ng, lut.anchor_pairs_a, lut.anchor_pairs_b))
        t_ng_pt  = bench("PyTorch fwd", lambda: _tiny_forward_pytorch(x_ng, lut.anchor_pairs_a, lut.anchor_pairs_b, lut.powers))
    print(f"  speedup (CUDA vs PyTorch): {t_ng_pt / t_ng_cuda:.2f}x")

    # --- Full module forward (inference) ---
    print("Module forward (eval mode):")
    lut.eval()
    bench("CUDA   module(x) [eval]", lambda: lut(x_ng))

    # --- fwd + bwd (training) ---
    print("Forward+Backward (training):")
    lut.train()

    def step_with_grad():
        # Use distinct weights on main vs alt carriers so grad_diff is non-zero
        out = lut(x)
        loss = out[3].sum() + 2.0 * out[4].sum() + out[2].sum()  # + direct grad through alt_delta
        loss.backward()
        x.grad = None

    t_full = bench("CUDA    module fwd+bwd", step_with_grad)
    print(f"  (contributions: fwd={t_ng_cuda:.3f}ms, rest={(t_full - t_ng_cuda):.3f}ms)")

    # Switch off native
    os.environ["SPIKY_TINY_APL_NO_CUSTOM_CUDA"] = "1"
    import importlib
    import spiky.lutorch.tiny_anchor_pairs_lookup as tapl
    importlib.reload(tapl)
    lut2 = tapl.TinyAnchorPairsLookup(
        input_dim=input_dim, n_tables=n_tables, n_anchor_pairs=n_anchor_pairs,
        random_seed=42, device=DEVICE,
    )
    lut2.train()
    x2 = torch.randn(batch_size, input_dim, device=DEVICE, requires_grad=True)
    def step_pt():
        out = lut2(x2)
        loss = out[3].sum() + 2.0 * out[4].sum() + out[2].sum()
        loss.backward()
        x2.grad = None
    t_pt = bench("PyTorch module fwd+bwd (env NO_CUSTOM_CUDA=1)", step_pt)
    del os.environ["SPIKY_TINY_APL_NO_CUSTOM_CUDA"]
    importlib.reload(tapl)  # re-enable for next config

    print(f"  speedup (CUDA vs PyTorch, fwd+bwd): {t_pt / t_full:.2f}x")


if __name__ == "__main__":
    print(f"Device: {DEVICE}, can_use_native: {_can_use_native_tiny_apl(torch.zeros(1, 1, device=DEVICE))}")

    # Representative configs (covering typical PermLut shapes)
    configs = [
        # (input_dim, n_tables, n_anchor_pairs, batch_size)
        (32, 128, 6, 16384),   # Q/K inner at bs=128, ctx=128 (BT=16384)
        (32, 512, 6, 16384),   # out_proj inner
        (32, 128, 5, 16384),   # smaller nap
        (32, 128, 6, 1024),    # smaller batch
        (64, 128, 8, 16384),   # out_proj-style input_dim
        (32, 1024, 10, 16384), # larger tph
    ]
    for cfg in configs:
        bench_config(*cfg)
