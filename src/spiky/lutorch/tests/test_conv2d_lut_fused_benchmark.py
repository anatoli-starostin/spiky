"""
Detailed benchmark: Conv2DLut vs Conv2DLutFused.

Measures per-component timing to identify bottlenecks.
Run with: SPIKY_LUTORCH_NO_COMPILE=1 python -m pytest test_conv2d_lut_fused_benchmark.py -v -s
"""
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytest

from spiky.lutorch.multi_head_lut import Conv2DLut, UnfoldConfiguration
from spiky.lutorch.conv2d_lut_fused import Conv2DLutFused

DEVICE = torch.device("cuda:0")
N_BATCHES = 50
WARMUP    = 10


def cuda_time(fn, warmup=WARMUP, n=N_BATCHES):
    """Return median ms over n runs using CUDA events."""
    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(n):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    times.sort()
    return times[len(times)//2]  # median


def make_layer(cls, H, k, s, ic, oc, nap=5, nh=4):
    tph = ic * k * k * 12 // (nap * 2)
    return cls(
        UnfoldConfiguration(H=H, W=H, kernel_size=k, stride=s, padding=1),
        in_channels=ic, out_channels=oc,
        n_anchor_pairs=nap, tables_per_head=tph,
        n_heads=nh, n_alternatives=3, smooth_mode=True,
        connected_anchors_mode=False, initial_weights_noise=0.001,
        device=DEVICE,
    ).to(DEVICE)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_benchmark_per_layer():
    """Time each layer individually, forward+backward, batch=64."""
    B = 64
    configs = [
        # (label, H, k, s, ic, oc)
        ("32x32 k=3 s=1 ic=32", 32, 3, 1, 32, 24),
        ("32x32 k=4 s=2 ic=24", 32, 4, 2, 24, 24),
        ("16x16 k=4 s=2 ic=24", 16, 4, 2, 24, 48),
        (" 8x8  k=3 s=1 ic=48", 8,  3, 1, 48, 24),
        (" 8x8  k=4 s=2 ic=24", 8,  4, 2, 24, 24),
    ]

    print("\n\n=== Per-layer benchmark (forward+backward, B=64, median of 50 runs) ===")
    print(f"{'Layer':<26} {'Ref ms':>8} {'Fused ms':>10} {'Speedup':>9}")
    print("-" * 58)

    for label, H, k, s, ic, oc in configs:
        x = torch.randn(B, ic, H, H, device=DEVICE, requires_grad=True)

        ref   = make_layer(Conv2DLut,      H, k, s, ic, oc)
        fused = make_layer(Conv2DLutFused, H, k, s, ic, oc)

        # copy weights
        fused.projection.weights.data.copy_(ref.lut.projection.weights.data)

        def run_ref():
            xi = x.detach().requires_grad_(True)
            out = ref(xi)
            out.sum().backward()

        def run_fused():
            xi = x.detach().requires_grad_(True)
            out = fused(xi)
            out.sum().backward()

        t_ref   = cuda_time(run_ref)
        t_fused = cuda_time(run_fused)
        speedup = t_ref / t_fused
        print(f"{label:<26} {t_ref:>8.2f} {t_fused:>10.2f} {speedup:>8.2f}x")

    print()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_benchmark_unfold_vs_kernel():
    """Isolate: how much time does F.unfold take vs the fused kernel?"""
    B = 64
    configs = [
        ("32x32 k=3 ic=32", 32, 3, 1, 32),
        ("32x32 k=4 ic=24", 32, 4, 2, 24),
        ("16x16 k=4 ic=24", 16, 4, 2, 24),
    ]

    print("\n\n=== F.unfold cost isolation (forward only, B=64) ===")
    print(f"{'Config':<22} {'unfold ms':>10} {'unfold+gather %':>16}")
    print("-" * 52)

    for label, H, k, s, ic in configs:
        x = torch.randn(B, ic, H, H, device=DEVICE)

        def do_unfold():
            patches = F.unfold(x, kernel_size=k, padding=1, stride=s)
            patches.transpose(1, 2).contiguous()

        t_unfold = cuda_time(do_unfold)
        print(f"{label:<22} {t_unfold:>10.3f}")

    print()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_benchmark_full_model():
    """Full exp149-style model: ref vs fused, different batch sizes."""
    CH_CONV, CH, NH, NAP = 32, 24, 4, 5

    def make_model(cls):
        def lut(H, k, s, ic, oc):
            tph = ic * k * k * 12 // (NAP * 2)
            return cls(
                UnfoldConfiguration(H=H, W=H, kernel_size=k, stride=s, padding=1),
                in_channels=ic, out_channels=oc,
                n_anchor_pairs=NAP, tables_per_head=tph,
                n_heads=NH, n_alternatives=3, smooth_mode=True,
                connected_anchors_mode=False, initial_weights_noise=0.001,
                device=DEVICE,
            )
        m = nn.Sequential(
            nn.Conv2d(3, CH_CONV, 3, padding=1),
            nn.BatchNorm2d(CH_CONV),
        )
        # store lut layers separately so we can time them
        return m, [
            lut(32, 3, 1, CH_CONV, CH),
            lut(32, 4, 2, CH,      CH),
            lut(16, 4, 2, CH,      CH*2),
            lut( 8, 3, 1, CH*2,    CH),
            lut( 8, 4, 2, CH,      CH),
        ]

    print("\n\n=== Full model benchmark (forward+backward, median of 50 runs) ===")
    print(f"{'Batch':>6} {'Ref ms':>9} {'Fused ms':>10} {'Speedup':>9} {'Memory saved':>14}")
    print("-" * 55)

    criterion = nn.CrossEntropyLoss()

    for B in [16, 32, 64, 128]:
        x   = torch.randn(B, 3, 32, 32, device=DEVICE)
        y   = torch.randint(0, 10, (B,), device=DEVICE)

        stem_ref,   luts_ref   = make_model(Conv2DLut)
        stem_fused, luts_fused = make_model(Conv2DLutFused)

        # Copy weights from ref to fused
        for lr, lf in zip(luts_ref, luts_fused):
            lf.projection.weights.data.copy_(lr.lut.projection.weights.data)

        classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(CH * 4 * 4, 1024),
            nn.BatchNorm1d(1024), nn.ReLU(inplace=True),
            nn.Dropout(0.5), nn.Linear(1024, 10),
        ).to(DEVICE)

        def run(stem, luts):
            h = stem(x)
            for l in luts:
                h = l(h)
            return criterion(classifier(h), y)

        torch.cuda.reset_peak_memory_stats()
        def run_ref():
            loss = run(stem_ref.to(DEVICE), [l.to(DEVICE) for l in luts_ref])
            loss.backward()

        def run_fused():
            loss = run(stem_fused.to(DEVICE), [l.to(DEVICE) for l in luts_fused])
            loss.backward()

        t_ref   = cuda_time(run_ref)
        t_fused = cuda_time(run_fused)

        # Measure peak memory for one forward pass
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            run(stem_ref.to(DEVICE), [l.to(DEVICE) for l in luts_ref])
        mem_ref = torch.cuda.max_memory_allocated() / 1e6

        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            run(stem_fused.to(DEVICE), [l.to(DEVICE) for l in luts_fused])
        mem_fused = torch.cuda.max_memory_allocated() / 1e6

        speedup = t_ref / t_fused
        mem_saved = mem_ref - mem_fused
        print(f"{B:>6} {t_ref:>9.1f} {t_fused:>10.1f} {speedup:>8.2f}x {mem_saved:>+12.1f} MB")

    print()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_verify_fused_path_taken():
    """Confirm Conv2DLutFused actually uses the fused kernel (not the fallback)."""
    layer = make_layer(Conv2DLutFused, 32, 3, 1, 32, 24)
    assert layer._can_fuse, "Fused path NOT available — check CUDA/n_alternatives/smooth_mode"
    x = torch.randn(4, 32, 32, 32, device=DEVICE)
    assert x.is_cuda
    out = layer(x)
    assert out.shape == (4, 24, 32, 32)
    print("\nFused path: ACTIVE")
