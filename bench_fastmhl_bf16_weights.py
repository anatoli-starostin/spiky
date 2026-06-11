"""Micro-benchmark: FastMultiHeadLUT fp32 vs bf16 weight_dtype.

LUTGPT exp732 shapes: B=12288 (= 24 dev_bs * 512 seq), E=384, H=6, d_qk=d_v=64.
Per-layer modules: qk_lut, v_lut, out_proj, residual_lut.

Measures forward, backward, and end-to-end (forward+backward) wall time per
iteration over the four module shapes used in exp732. Reports fp32 vs bf16
storage with use_bf16=True (compute autocast) for both.
"""

import time
import torch
import torch.nn.functional as F

from spiky.lutorch.fast_multi_head_lut import FastMultiHeadLUT
from spiky.lutorch.multi_head_lut import AnchorSamplingPolicy

DEVICE = torch.device("cuda:0")

# ---- LUTGPT exp732 shapes ----------------------------------------------------
B = 24 * 512               # = 12288 tokens per micro-batch (dev_bs=24, seq=512)
E = 384
H = 6
D_QK = 64
D_V = 64

MODULES = [
    # (name, input_dim, n_heads, n_outputs, n_anchor_pairs, tables_per_head)
    ("qk_lut",       E,       H, 2 * D_QK, 4, 256),  # n_out=128, K=16
    ("v_lut",        E,       H, D_V,      6, 256),  # n_out=64,  K=64
    ("out_proj",     H * D_V, 1, E,        7, 512),  # n_out=384, K=128
    ("residual_lut", E,       1, E,        6, 256),  # n_out=384, K=64
]

FAST_KWARGS_BASE = dict(
    forward_mode="hard",
    backward_mode="dense_K",
    anchor_sampling_policy=AnchorSamplingPolicy.CANONICAL_FULL_COVERAGE,
    initial_weights_noise=0.001,
    soft_score_temp=0.5,
    select_temp=0.5,
    learnable_temps=True,
    use_bf16=True,                # autocast compute = bf16 (same as exp732)
    random_seed=42,
)


def make_module(spec, weight_dtype):
    name, in_dim, n_heads, n_out, nap, tph = spec
    return FastMultiHeadLUT(
        input_dim=in_dim, n_heads=n_heads, n_outputs=n_out,
        n_anchor_pairs=nap, tables_per_head=tph,
        weight_dtype=weight_dtype, device=DEVICE, **FAST_KWARGS_BASE,
    )


def make_input(in_dim):
    # Match the autocast path: input is fp32, autocast handles the rest.
    return torch.randn(B, in_dim, device=DEVICE, dtype=torch.float32,
                       requires_grad=True)


def time_phase(module, x_fn, phase, n_warmup=3, n_iter=20):
    """phase in {'fwd', 'bwd', 'fwd_bwd'}."""
    for _ in range(n_warmup):
        x = x_fn()
        y = module(x)
        if phase != "fwd":
            grad_out = torch.randn_like(y)
            y.backward(grad_out)
            module.weights.grad = None
            if x.grad is not None:
                x.grad = None

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iter):
        x = x_fn()
        if phase == "fwd":
            y = module(x)
        elif phase == "bwd":
            # Forward outside timing window? No — bwd needs the saved tensors.
            # We time only the .backward() call.
            y = module(x)
            torch.cuda.synchronize()
            t_pre_bwd = time.perf_counter()
            grad_out = torch.randn_like(y)
            y.backward(grad_out)
            torch.cuda.synchronize()
            t_post_bwd = time.perf_counter()
            # Override sum manually
            time_phase._bwd_acc += t_post_bwd - t_pre_bwd
        else:  # fwd_bwd
            y = module(x)
            grad_out = torch.randn_like(y)
            y.backward(grad_out)
        module.weights.grad = None
        if x.grad is not None:
            x.grad = None
    torch.cuda.synchronize()
    t1 = time.perf_counter()

    if phase == "bwd":
        ms_per_iter = (time_phase._bwd_acc / n_iter) * 1000
        time_phase._bwd_acc = 0.0
    else:
        ms_per_iter = ((t1 - t0) / n_iter) * 1000
    return ms_per_iter


time_phase._bwd_acc = 0.0


def main():
    torch.manual_seed(0)
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Shapes: B={B} (=24*512), E={E}, H={H}, d_qk={D_QK}, d_v={D_V}")
    print(f"Modules: 4 per-layer FastMHL modules (exp732 config)\n")

    results = []

    for spec in MODULES:
        name, in_dim, n_heads, n_out, nap, tph = spec
        print(f"--- {name}: in={in_dim}, H={n_heads}, n_out={n_out}, NAP={nap}, tph={tph} ---")

        row = {"module": name, "n_out": n_out, "K": 2 ** nap}
        for dtype_name, dtype in [("fp32", torch.float32), ("bf16", torch.bfloat16)]:
            module = make_module(spec, weight_dtype=dtype)
            x_fn = lambda: make_input(in_dim)
            fwd_ms = time_phase(module, x_fn, "fwd")
            bwd_ms = time_phase(module, x_fn, "bwd")
            both_ms = time_phase(module, x_fn, "fwd_bwd")
            row[f"{dtype_name}_fwd_ms"] = fwd_ms
            row[f"{dtype_name}_bwd_ms"] = bwd_ms
            row[f"{dtype_name}_fwd_bwd_ms"] = both_ms
            print(f"  {dtype_name}: fwd={fwd_ms:7.3f} ms  bwd={bwd_ms:7.3f} ms  fwd+bwd={both_ms:7.3f} ms")
            del module
            torch.cuda.empty_cache()

        slow_fwd = row["bf16_fwd_ms"] / row["fp32_fwd_ms"]
        slow_bwd = row["bf16_bwd_ms"] / row["fp32_bwd_ms"]
        slow_both = row["bf16_fwd_bwd_ms"] / row["fp32_fwd_bwd_ms"]
        print(f"  ratio bf16/fp32:    fwd={slow_fwd:.3f}x   bwd={slow_bwd:.3f}x   fwd+bwd={slow_both:.3f}x")
        print()
        results.append(row)

    print("\n=== Per-step summary (sum over 4 modules x 6 layers = 24 module-instances/step, excluding qkv_proj 2-pass; ignores emb_resid run once/step) ===")
    fp32_total = sum(r["fp32_fwd_bwd_ms"] for r in results) * 6
    bf16_total = sum(r["bf16_fwd_bwd_ms"] for r in results) * 6
    print(f"  fp32 weights total fwd+bwd (6 layers, 4 modules):  {fp32_total:8.2f} ms")
    print(f"  bf16 weights total fwd+bwd (6 layers, 4 modules):  {bf16_total:8.2f} ms")
    print(f"  delta:                                              {bf16_total - fp32_total:+8.2f} ms ({(bf16_total/fp32_total - 1) * 100:+.1f} %)")


if __name__ == "__main__":
    main()
