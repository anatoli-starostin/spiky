"""Diagnose drift between TinyMHLut(soft) and SoftMultiHeadLUT(hard=True).

Step 1: With identical weights + anchor pairs (same random_seed), under fp32
        and bf16 autocast, compare:
          - forward output
          - all 4 gradients (dL/dx, dL/dweights, dL/dlog_Ts, dL/dlog_Tx)
Step 2: Simulate N optimizer steps with both modules fed identical (x, grad_out)
        per step, copy the gradient from SoftMHLut into TinyMHLut after each
        step so they share weights — check that the per-step *delta* between
        their unmoved gradients stays bounded (no compounding).
"""
import math, contextlib
import torch
from spiky.lutorch.tiny_multi_head_lut import TinyMultiHeadLut
from spiky.lutorch.soft_multi_head_lut import SoftMultiHeadLUT
from spiky.lutorch.lut_helpers import AnchorSamplingPolicy


def build_pair(input_dim, n_heads, tph, nap, n_outputs, device, seed=42,
               weight_dtype=torch.float32, use_bf16=False, init_T=0.5):
    torch.manual_seed(seed)
    tiny = TinyMultiHeadLut(
        input_dim=input_dim, n_heads=n_heads, n_outputs=n_outputs,
        n_anchor_pairs=nap, tables_per_head=tph,
        weight_dtype=weight_dtype, random_seed=seed, device=device,
        anchor_sampling_policy=AnchorSamplingPolicy.CANONICAL_FULL_COVERAGE,
        backward_mode="soft", soft_score_temp=init_T, select_temp=init_T,
        learnable_temps=True, use_bf16=use_bf16,
    ).to(device)
    torch.manual_seed(seed)
    soft = SoftMultiHeadLUT(
        input_dim=input_dim, n_heads=n_heads, n_outputs=n_outputs,
        n_anchor_pairs=nap, tables_per_head=tph,
        soft_score_temp=init_T, select_temp=init_T,
        gumbel=False, hard=True, learnable_temps=True,
        anchor_sampling_policy=AnchorSamplingPolicy.CANONICAL_FULL_COVERAGE,
        weight_dtype=weight_dtype, random_seed=seed, device=device,
        use_bf16=use_bf16, compile_forward=False,
    ).to(device)
    # Verify they actually got the same anchor pairs and weight init.
    same_a = torch.equal(tiny.lookup.anchor_pairs_a.long(),
                         soft.anchor_pairs_a.long())
    same_b = torch.equal(tiny.lookup.anchor_pairs_b.long(),
                         soft.anchor_pairs_b.long())
    same_w = torch.equal(tiny.weights.detach(), soft.weights.detach())
    if not (same_a and same_b and same_w):
        raise RuntimeError(
            f"Init mismatch: anchor_a={same_a} anchor_b={same_b} weights={same_w}"
        )
    return tiny, soft


def compare_one(tiny, soft, x, target, label, max_print=10):
    """Run forward+backward on both with the SAME inputs, report per-grad diffs."""
    x1 = x.detach().clone().requires_grad_(True)
    x2 = x.detach().clone().requires_grad_(True)
    out1 = tiny(x1)
    out2 = soft(x2)
    loss1 = (out1 - target).pow(2).sum()
    loss2 = (out2 - target).pow(2).sum()
    loss1.backward()
    loss2.backward()
    out_diff = (out1 - out2).abs()
    out_max = out1.abs().max().item()
    print(f"\n=== {label} ===")
    print(f"  forward     abs|Δ|max = {out_diff.max().item():.3e}   ref|max| = {out_max:.3e}   "
          f"rel = {out_diff.max().item() / max(out_max, 1e-12):.3e}")
    for name, t1, t2 in [
        ("dL/dx",         x1.grad,                  x2.grad),
        ("dL/dweights",   tiny.weights.grad,        soft.weights.grad),
        ("dL/dlog_Ts",    tiny.log_soft_score_temp.grad, soft.log_soft_score_temp.grad),
        ("dL/dlog_Tx",    tiny.log_select_temp.grad,     soft.log_select_temp.grad),
    ]:
        if t1 is None or t2 is None:
            print(f"  {name:13s}  None grad (t1={t1 is None}, t2={t2 is None})")
            continue
        d = (t1 - t2).abs()
        r = t1.abs().max().item() if t1.dim() else abs(t1.item())
        print(f"  {name:13s}  abs|Δ|max = {d.max().item():.3e}   ref|max| = {r:.3e}   "
              f"rel = {d.max().item() / max(r, 1e-12):.3e}")
    return out1, out2


def step_drift(tiny, soft, n_steps, B, input_dim, device, lr=1e-3, use_bf16=False):
    """Simulate N optimizer steps. After each step, copy soft.weights into
    tiny.weights so the next step sees the same starting state — measures
    per-step grad divergence in isolation (no compounding from weight drift)."""
    n_outputs = tiny.n_outputs
    n_heads = tiny.n_heads
    print(f"\n=== Per-step drift (n_steps={n_steps}, bf16={use_bf16}) ===")
    print(f"  step  fwd_rel    gx_rel    gw_rel    gTs_diff   gTx_diff")
    rng = torch.Generator(device=device).manual_seed(0)
    for step in range(n_steps):
        x_raw = torch.randn(B, input_dim, generator=rng, device=device)
        target = torch.randn(B, n_heads, n_outputs, generator=rng, device=device)
        # Synchronize weights every step
        with torch.no_grad():
            tiny.weights.copy_(soft.weights)
            tiny.log_soft_score_temp.copy_(soft.log_soft_score_temp)
            tiny.log_select_temp.copy_(soft.log_select_temp)
        # Zero grads
        for m in (tiny, soft):
            for p in m.parameters():
                if p.grad is not None: p.grad.zero_()
        # Forward+backward
        x1 = x_raw.clone().requires_grad_(True)
        x2 = x_raw.clone().requires_grad_(True)
        out1 = tiny(x1); out2 = soft(x2)
        ((out1 - target).pow(2).sum()).backward()
        ((out2 - target).pow(2).sum()).backward()
        fwd_rel = (out1 - out2).abs().max() / out1.abs().max().clamp(min=1e-12)
        gx_rel  = (x1.grad - x2.grad).abs().max() / x1.grad.abs().max().clamp(min=1e-12)
        gw_rel  = (tiny.weights.grad - soft.weights.grad).abs().max() / \
                  tiny.weights.grad.abs().max().clamp(min=1e-12)
        gTs_d = (tiny.log_soft_score_temp.grad - soft.log_soft_score_temp.grad).abs().item()
        gTx_d = (tiny.log_select_temp.grad - soft.log_select_temp.grad).abs().item()
        # Apply soft's gradient to both (just to advance state)
        with torch.no_grad():
            soft.weights.add_(soft.weights.grad, alpha=-lr)
            soft.log_soft_score_temp.add_(soft.log_soft_score_temp.grad, alpha=-lr)
            soft.log_select_temp.add_(soft.log_select_temp.grad, alpha=-lr)
        print(f"  {step:4d}  {fwd_rel.item():.3e}  {gx_rel.item():.3e}  "
              f"{gw_rel.item():.3e}  {gTs_d:.3e}  {gTx_d:.3e}")


def main():
    device = torch.device("cuda")
    # Small case for clear fp32 comparison
    print("===" * 25)
    print("Small case: NAP=6 tph=8 n_heads=4 input_dim=64 n_outputs=12 B=16")
    print("===" * 25)
    tiny, soft = build_pair(64, 4, 8, 6, 12, device, seed=42, weight_dtype=torch.float32, use_bf16=False)
    torch.manual_seed(0)
    x = torch.randn(16, 64, device=device)
    target = torch.randn(16, 4, 12, device=device)
    compare_one(tiny, soft, x, target, "fp32 weights, no autocast")

    # Same shapes, bf16 autocast
    tiny, soft = build_pair(64, 4, 8, 6, 12, device, seed=42, weight_dtype=torch.float32, use_bf16=True)
    compare_one(tiny, soft, x, target, "fp32 weights, bf16 autocast")

    # v_lut shape (exp251 v_lut: nap=8, tph=256, n_heads=6, in=96, out=32)
    print("\n" + "===" * 25)
    print("v_lut shape: NAP=8 tph=256 n_heads=6 input_dim=96 n_outputs=32 B=4096")
    print("===" * 25)
    tiny, soft = build_pair(96, 6, 256, 8, 32, device, seed=42, weight_dtype=torch.float32, use_bf16=False)
    torch.manual_seed(0)
    x = torch.randn(4096, 96, device=device)
    target = torch.randn(4096, 6, 32, device=device)
    compare_one(tiny, soft, x, target, "fp32 weights, no autocast")

    tiny, soft = build_pair(96, 6, 256, 8, 32, device, seed=42, weight_dtype=torch.float32, use_bf16=True)
    compare_one(tiny, soft, x, target, "fp32 weights, bf16 autocast")

    # Step-by-step drift with bf16 autocast
    tiny, soft = build_pair(96, 6, 256, 8, 32, device, seed=42, weight_dtype=torch.float32, use_bf16=True)
    step_drift(tiny, soft, n_steps=12, B=512, input_dim=96, device=device, use_bf16=True)


if __name__ == "__main__":
    main()
