"""Benchmark: hyperplane projection front-end vs anchor-pair front-end.

Measures the *front-end* cost only — the index-computation step that
HyperplaneMultiHeadLUT generalizes — plus the full module fwd+bwd, at a few
LUTGPT-representative shapes. The hyperplane projection is a dense
[B, input_dim] x [n_tables, NAP, input_dim] GEMM and is the new dominant FLOP;
the anchor-pair front-end is a cheap two-coordinate gather + subtract.

Run inside the sbox cage:
    sbox ~/projects/spiky/.venv/bin/python \
        experiments/hyperplane_ffn/bench_hyperplane_frontend.py
"""
import time

import torch

from spiky.lutorch.fast_multi_head_lut import FastMultiHeadLut
from spiky.lutorch.hyperplane_multi_head_lut import (
    HyperplaneMultiHeadLUT,
    _hyperplane_project,
)


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _time(fn, iters=50, warmup=10):
    for _ in range(warmup):
        fn()
    _sync()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    _sync()
    return (time.perf_counter() - t0) / iters * 1e3  # ms


def bench_frontend_only(dev, B, input_dim, n_tables, nap):
    """Isolated front-end: anchor gather+sub vs hyperplane GEMM (fwd only)."""
    x = torch.randn(B, input_dim, device=dev)
    a_idx = torch.randint(0, input_dim, (n_tables, nap), device=dev)
    b_idx = torch.randint(0, input_dim, (n_tables, nap), device=dev)
    w = torch.randn(n_tables, nap, input_dim, device=dev) * 0.02
    b = torch.zeros(n_tables, nap, device=dev)

    def anchor_fe():
        d = x[:, a_idx] - x[:, b_idx]
        (d > 0).to(torch.int64)

    def hyper_fe():
        aa = _hyperplane_project(x, w, b)
        (aa > 0).to(torch.int64)

    t_anchor = _time(anchor_fe)
    t_hyper = _time(hyper_fe)
    return t_anchor, t_hyper


def bench_full_module(dev, B, input_dim, n_heads, tph, nap, n_outputs,
                      forward_mode):
    """Full module fwd+bwd wall-clock: FastMultiHeadLut vs HyperplaneMultiHeadLUT."""
    common = dict(
        input_dim=input_dim, n_heads=n_heads, n_outputs=n_outputs,
        n_anchor_pairs=nap, tables_per_head=tph, forward_mode=forward_mode,
        weight_dtype=torch.float32, use_bf16=True, learnable_temps=True,
        random_seed=0, device=dev,
    )
    m_fast = FastMultiHeadLut(**common)
    m_hyp = HyperplaneMultiHeadLUT(hyperplane_init="anchor_pairs", **common)

    def step(m):
        x = torch.randn(B, input_dim, device=dev, requires_grad=True)
        y = m(x)
        y.float().sum().backward()

    t_fast = _time(lambda: step(m_fast), iters=30, warmup=8)
    t_hyp = _time(lambda: step(m_hyp), iters=30, warmup=8)
    return t_fast, t_hyp


def main():
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"device: {torch.cuda.get_device_name(0) if dev.type=='cuda' else 'cpu'}")
    print(f"torch: {torch.__version__}\n")

    print("== Front-end only (index computation), fwd, ms ==")
    print(f"{'B':>6} {'d_model':>8} {'n_tables':>9} {'NAP':>4} "
          f"{'anchor':>9} {'hyper':>9} {'x':>7}")
    fe_shapes = [
        # (B, input_dim, n_tables, nap)
        (4096, 384, 512, 6),
        (4096, 384, 3072, 6),   # n_heads=6 * tph=512
        (16384, 384, 512, 6),
        (4096, 384, 1536, 4),   # qk-like
    ]
    for B, d, nt, nap in fe_shapes:
        ta, th = bench_frontend_only(dev, B, d, nt, nap)
        print(f"{B:>6} {d:>8} {nt:>9} {nap:>4} {ta:>9.3f} {th:>9.3f} {th/ta:>6.1f}x")

    print("\n== Full module fwd+bwd (fp32 weights, bf16 autocast), ms ==")
    print(f"{'module':>14} {'B':>6} {'n_heads':>7} {'tph':>5} {'NAP':>4} "
          f"{'n_out':>6} {'mode':>13} {'fast':>9} {'hyper':>9} {'x':>7}")
    mod_shapes = [
        # (B, n_heads, tph, nap, n_outputs, forward_mode, name)
        (4096, 1, 512, 6, 384, "hard", "residual_lut"),
        (4096, 1, 512, 6, 384, "hybrid_smooth", "residual_lut"),
        (4096, 6, 512, 6, 128, "hard", "qk_lut"),
        (4096, 6, 512, 4, 16, "hard", "v_lut"),
    ]
    for B, nh, tph, nap, no, mode, name in mod_shapes:
        tf, th = bench_full_module(dev, B, 384, nh, tph, nap, no, mode)
        print(f"{name:>14} {B:>6} {nh:>7} {tph:>5} {nap:>4} {no:>6} "
              f"{mode:>13} {tf:>9.3f} {th:>9.3f} {th/tf:>6.1f}x")


if __name__ == "__main__":
    main()
