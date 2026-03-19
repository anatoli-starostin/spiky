"""
Tests for Conv2DLutFused: compare against reference Conv2DLut.
"""
import os
import time

os.environ["SPIKY_LUTORCH_NO_COMPILE"] = "1"

import pytest
import torch
import torch.nn as nn

from spiky.lutorch.multi_head_lut import Conv2DLut, UnfoldConfiguration
from spiky.lutorch.conv2d_lut_fused import Conv2DLutFused
from spiky.lutorch.lut_helpers import UncertaintyMode


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Realistic shared params
IN_CH   = 32
OUT_CH  = 24
N_HEADS = 4
NAP     = 5       # n_anchor_pairs
TPH     = 12      # tables_per_head
N_ALT   = 3
SEED    = 7


def _make_pair(unfold_cfg, device=DEVICE, seed=SEED):
    """Return (ref, fused) with identical weights on `device`."""
    common_kwargs = dict(
        n_alternatives=N_ALT,
        smooth_mode=True,
        cmp_eps=0.0,
        uncertainty_mode=UncertaintyMode.INVERSE_L1,
        random_seed=seed,
        initial_weights_noise=0.001,
    )
    ref = Conv2DLut(
        unfold_config=unfold_cfg,
        in_channels=IN_CH,
        out_channels=OUT_CH,
        n_anchor_pairs=NAP,
        n_heads=N_HEADS,
        tables_per_head=TPH,
        device=torch.device(device),
        **common_kwargs,
    ).to(device)

    fused = Conv2DLutFused(
        unfold_config=unfold_cfg,
        in_channels=IN_CH,
        out_channels=OUT_CH,
        n_anchor_pairs=NAP,
        n_heads=N_HEADS,
        tables_per_head=TPH,
        device=torch.device(device),
        **common_kwargs,
    ).to(device)

    # Synchronise anchor pairs and weights.
    with torch.no_grad():
        fused.anchor_pairs_a.copy_(ref.lut.lookup.anchor_pairs_a)
        fused.anchor_pairs_b.copy_(ref.lut.lookup.anchor_pairs_b)
        fused.projection.weights.copy_(ref.lut.projection.weights)

    return ref, fused


def _make_input(B, H, W, device=DEVICE, requires_grad=False):
    torch.manual_seed(42)
    return torch.randn(B, IN_CH, H, W, device=device, requires_grad=requires_grad)


# ---------------------------------------------------------------------------
# 1. Forward match
# ---------------------------------------------------------------------------

@pytest.mark.skipif(DEVICE == "cpu", reason="Fused kernels require CUDA")
@pytest.mark.parametrize("k,s,p,H,W", [
    (3, 1, 0, 16, 16),
    (4, 2, 0, 16, 16),
    (3, 1, 1, 16, 16),
    (4, 2, 0, 32, 32),
])
def test_conv2d_lut_fused_forward_matches_conv2dlut(k, s, p, H, W):
    """Fused forward must match reference Conv2DLut output within 1e-4."""
    cfg = UnfoldConfiguration(H=H, W=W, kernel_size=k, stride=s, padding=p)
    ref, fused = _make_pair(cfg)
    ref.eval()
    fused.eval()

    x = _make_input(4, H, W)
    with torch.no_grad():
        out_ref   = ref(x)
        out_fused = fused(x)

    assert out_ref.shape == out_fused.shape, (
        f"Shape mismatch: ref={out_ref.shape}, fused={out_fused.shape}"
    )
    max_diff = (out_ref - out_fused).abs().max().item()
    assert max_diff < 1e-4, (
        f"k={k},s={s},p={p},H={H},W={W}: max forward diff = {max_diff:.2e}"
    )


# ---------------------------------------------------------------------------
# 2. Backward match
# ---------------------------------------------------------------------------

@pytest.mark.skipif(DEVICE == "cpu", reason="Fused kernels require CUDA")
@pytest.mark.parametrize("k,s,p,H,W", [
    (3, 1, 0, 16, 16),
    (4, 2, 0, 16, 16),
    (3, 1, 1, 16, 16),
])
def test_conv2d_lut_fused_backward_matches_conv2dlut(k, s, p, H, W):
    """Fused grad_input and weight grads must match reference within 1e-4."""
    cfg = UnfoldConfiguration(H=H, W=W, kernel_size=k, stride=s, padding=p)

    B = 4
    common_kwargs = dict(
        n_alternatives=N_ALT,
        smooth_mode=True,
        cmp_eps=0.0,
        uncertainty_mode=UncertaintyMode.INVERSE_L1,
        random_seed=SEED,
        initial_weights_noise=0.001,
    )

    ref = Conv2DLut(
        unfold_config=cfg,
        in_channels=IN_CH,
        out_channels=OUT_CH,
        n_anchor_pairs=NAP,
        n_heads=N_HEADS,
        tables_per_head=TPH,
        device=torch.device(DEVICE),
        **common_kwargs,
    ).to(DEVICE).train()

    fused = Conv2DLutFused(
        unfold_config=cfg,
        in_channels=IN_CH,
        out_channels=OUT_CH,
        n_anchor_pairs=NAP,
        n_heads=N_HEADS,
        tables_per_head=TPH,
        device=torch.device(DEVICE),
        **common_kwargs,
    ).to(DEVICE).train()

    with torch.no_grad():
        fused.anchor_pairs_a.copy_(ref.lut.lookup.anchor_pairs_a)
        fused.anchor_pairs_b.copy_(ref.lut.lookup.anchor_pairs_b)
        fused.projection.weights.copy_(ref.lut.projection.weights)

    torch.manual_seed(42)
    x_ref   = torch.randn(B, IN_CH, H, W, device=DEVICE, requires_grad=True)
    x_fused = x_ref.detach().clone().requires_grad_(True)

    out_ref   = ref(x_ref)
    out_fused = fused(x_fused)

    # Same upstream gradient
    torch.manual_seed(99)
    grad_out = torch.randn_like(out_ref)
    out_ref.backward(grad_out)
    out_fused.backward(grad_out)

    # Input gradient
    max_diff_input = (x_ref.grad - x_fused.grad).abs().max().item()
    assert max_diff_input < 1e-4, (
        f"k={k},s={s},p={p}: max grad_input diff = {max_diff_input:.2e}"
    )

    # Weight gradient
    max_diff_w = (
        ref.lut.projection.weights.grad - fused.projection.weights.grad
    ).abs().max().item()
    assert max_diff_w < 1e-4, (
        f"k={k},s={s},p={p}: max weight_grad diff = {max_diff_w:.2e}"
    )


# ---------------------------------------------------------------------------
# 3. Training step (loss decreases)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(DEVICE == "cpu", reason="Fused kernels require CUDA")
def test_conv2d_lut_fused_training_step():
    """Loss should stay finite and decrease when fitting a fixed batch (overfit test)."""
    H, W = 16, 16
    cfg = UnfoldConfiguration(H=H, W=W, kernel_size=4, stride=2, padding=0)
    _, fused = _make_pair(cfg)
    fused.train()

    optimizer = torch.optim.SGD(fused.parameters(), lr=1.0)
    torch.manual_seed(1)
    # Fix input and target so the model can overfit.
    x_fixed      = torch.randn(4, IN_CH, H, W, device=DEVICE)
    target_fixed = torch.randn(4, OUT_CH, fused.H_p, fused.W_p, device=DEVICE)

    losses = []
    for _ in range(20):
        out  = fused(x_fixed)
        loss = ((out - target_fixed) ** 2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        assert torch.isfinite(torch.tensor(loss.item())), "Loss became non-finite"

    # The model must be able to decrease loss when fitting the same batch repeatedly.
    assert losses[-1] < losses[0], (
        f"Loss did not decrease over 20 steps on fixed batch: "
        f"first={losses[0]:.4f}, last={losses[-1]:.4f}"
    )


# ---------------------------------------------------------------------------
# 4. Benchmark (optional, prints timing)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(DEVICE == "cpu", reason="Fused kernels require CUDA")
def test_conv2d_lut_fused_benchmark():
    """Compare wall-clock time: Conv2DLut vs Conv2DLutFused."""
    H, W  = 32, 32
    B     = 16
    cfg   = UnfoldConfiguration(H=H, W=W, kernel_size=4, stride=2, padding=0)
    ref, fused = _make_pair(cfg)

    def _time_fn(mod, n_warmup=10, n_trials=30):
        mod.train()
        losses = []
        for i in range(n_warmup + n_trials):
            x = torch.randn(B, IN_CH, H, W, device=DEVICE)
            t0 = time.perf_counter()
            out = mod(x)
            loss = out.sum()
            loss.backward()
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            if i >= n_warmup:
                losses.append(t1 - t0)
        return sum(losses) / len(losses) * 1000  # ms

    t_ref   = _time_fn(ref)
    t_fused = _time_fn(fused)

    mem_ref   = torch.cuda.max_memory_allocated() // (1024 ** 2)
    print(
        f"\n[Benchmark] B={B} H={H} W={W}: "
        f"Conv2DLut={t_ref:.2f}ms  Conv2DLutFused={t_fused:.2f}ms  "
        f"Speedup={t_ref/t_fused:.2f}x  peak_mem={mem_ref}MB"
    )


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    pytest.main([__file__, "-v"] + sys.argv[1:])
