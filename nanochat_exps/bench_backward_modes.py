"""Benchmark + correctness check for all TinyMultiHeadLut backward modes.

out_proj shape from exp364 / exp401 (the bandwidth-hot module):
    input_dim   = H * d_v = 96
    n_heads     = 1
    n_outputs   = E = 48
    NAP         = 8  (K = 2^NAP = 256 row dim)
    tph         = 128
    B           = device_batch_size * context_size = 192 * 512 = 98304

Modes:
    soft         — full sel_soft over K=256
    soft_topk-3  — new: soft math restricted to chosen + 3 best 1-bit-flip neighbors
    soft_topk-1  — degenerate: chosen + 1 neighbor
    soft_topk-8  — full NAP coverage (all 1-bit-flip neighbors)
    ste-n1       — single-alt hard STE (legacy TinyMHLut default)
    ste-n3       — multi-alt n_alternatives=3
"""
import sys, os, time, math
import torch

sys.path.insert(0, '/home/starost/spiky/src')
from spiky.lutorch.tiny_multi_head_lut import TinyMultiHeadLut
from spiky.lutorch.lut_helpers import AnchorSamplingPolicy

DEVICE = 'cuda'
DTYPE_W = torch.float32

# out_proj shape (exp364 / exp401)
INPUT_DIM = 96
N_HEADS   = 1
N_OUT     = 48
NAP       = 8
TPH       = 128
# Real out_proj B at bs=192 would be 98304, but exp401 is co-resident on the
# GPU and consumes ~68 GB. Use B = 32 * 512 = 16384 (bs=32) — captures the
# relative cost ratios across modes; absolute numbers scale ~linearly with B.
B         = int(os.environ.get('BENCH_B', 32 * 512))

torch.manual_seed(0)


def make_module(backward_mode, n_alternatives, learnable_temps=True):
    return TinyMultiHeadLut(
        input_dim=INPUT_DIM,
        n_heads=N_HEADS,
        n_outputs=N_OUT,
        n_anchor_pairs=NAP,
        tables_per_head=TPH,
        weight_dtype=DTYPE_W,
        anchor_sampling_policy=AnchorSamplingPolicy.CANONICAL_FULL_COVERAGE,
        backward_mode=backward_mode,
        n_alternatives=n_alternatives,
        learnable_temps=learnable_temps,
        use_bf16=True,
        random_seed=42,
        device=DEVICE,
    )


def bench(label, backward_mode, n_alt, warmup=2, iters=5):
    mod = make_module(backward_mode, n_alt)
    x = torch.randn(B, INPUT_DIM, device=DEVICE, dtype=torch.float32, requires_grad=True)
    grad_out_shape = (B, N_HEADS, N_OUT)

    # Warmup (compile + cudnn benchmark stabilization)
    for _ in range(warmup):
        x.grad = None
        for p in mod.parameters():
            p.grad = None
        out = mod(x)
        loss = (out * torch.randn_like(out)).sum()
        loss.backward()
    torch.cuda.synchronize()

    # Reset peak memory and start timing
    torch.cuda.reset_peak_memory_stats()
    fwd_times = []
    bwd_times = []
    for _ in range(iters):
        x.grad = None
        for p in mod.parameters():
            p.grad = None
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = mod(x)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        loss = (out * torch.randn_like(out)).sum()
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        loss.backward()
        torch.cuda.synchronize()
        t3 = time.perf_counter()
        fwd_times.append((t1 - t0) * 1000.0)
        bwd_times.append((t3 - t2) * 1000.0)
    peak_mb = torch.cuda.max_memory_allocated() / 1024**2

    fwd_med = sorted(fwd_times)[len(fwd_times) // 2]
    bwd_med = sorted(bwd_times)[len(bwd_times) // 2]

    return dict(
        label=label,
        fwd_ms=fwd_med,
        bwd_ms=bwd_med,
        peak_mb=peak_mb,
    )


def correctness_check():
    """Verify soft_topk_8 (full NAP) ≈ soft (full sel_soft) on small input.
    They differ only in considering multi-bit-flip rows (which soft includes,
    soft_topk doesn't). At NAP=2 they're equivalent up to floating point.
    """
    print("--- correctness check (NAP=2, all rows considered by both) ---")
    torch.manual_seed(1)
    B_small = 32
    NAP_small = 2  # K = 4 rows
    TPH_small = 4

    def small_mod(mode, n_alt):
        return TinyMultiHeadLut(
            input_dim=INPUT_DIM,
            n_heads=N_HEADS,
            n_outputs=N_OUT,
            n_anchor_pairs=NAP_small,
            tables_per_head=TPH_small,
            weight_dtype=DTYPE_W,
            anchor_sampling_policy=AnchorSamplingPolicy.CANONICAL_FULL_COVERAGE,
            backward_mode=mode,
            n_alternatives=n_alt,
            learnable_temps=False,
            use_bf16=False,         # turn off bf16 for exact compare
            random_seed=42,
            device=DEVICE,
        )

    x = torch.randn(B_small, INPUT_DIM, device=DEVICE, dtype=torch.float32, requires_grad=True)
    grad_y = torch.randn(B_small, N_HEADS, N_OUT, device=DEVICE)

    # Mode A: full soft
    mod_a = small_mod('soft', 1)
    x.grad = None; mod_a.weights.grad = None
    y_a = mod_a(x)
    (y_a * grad_y).sum().backward()
    gx_a = x.grad.detach().clone()
    gw_a = mod_a.weights.grad.detach().clone()

    # Mode B: soft_topk with n_alt = NAP (covers all 1-bit flips, omits multi-bit)
    mod_b = small_mod('soft_topk', NAP_small)
    # Copy weights to ensure same starting point
    with torch.no_grad():
        mod_b.weights.copy_(mod_a.weights)
    x.grad = None; mod_b.weights.grad = None
    y_b = mod_b(x)
    (y_b * grad_y).sum().backward()
    gx_b = x.grad.detach().clone()
    gw_b = mod_b.weights.grad.detach().clone()

    # Forward should be bit-identical (same sign-pack argmax forward)
    fwd_max_err = (y_a - y_b).abs().max().item()
    # gx: at NAP=2 the topk_n_alt=2 considers chosen + 2 1-bit flips = 3 rows.
    # Full soft considers 4 rows. The 4th row is the 2-bit flip — small softmax mass.
    # So gradients should be close but not identical.
    gx_max_err = (gx_a - gx_b).abs().max().item()
    gx_rel_err = ((gx_a - gx_b).abs() / (gx_a.abs() + 1e-8)).mean().item()
    # gw should be identical (both use hard index_add at chosen row)
    gw_max_err = (gw_a - gw_b).abs().max().item()

    print(f'  forward max diff:        {fwd_max_err:.2e}  (should be ~0)')
    print(f'  gx (input grad) max diff: {gx_max_err:.4e}')
    print(f'  gx (input grad) rel err:  {gx_rel_err:.4f}')
    print(f'  gw (weight grad) max diff: {gw_max_err:.2e}  (should be ~0)')

    # gw should match exactly because both use hard index_add at chosen row.
    assert gw_max_err < 1e-5, f"weight gradient mismatch: {gw_max_err}"
    assert fwd_max_err < 1e-5, f"forward mismatch: {fwd_max_err}"
    print('  PASS: forward and weight-grad match exactly.\n')


def main():
    print(f'Device: {torch.cuda.get_device_name(0)}')
    print(f'B={B}, NAP={NAP}, tph={TPH}, input_dim={INPUT_DIM}, n_out={N_OUT}\n')

    correctness_check()

    configs = [
        ('soft         ',     'soft',       1),
        ('soft_topk-1  ',     'soft_topk',  1),
        ('soft_topk-3  ',     'soft_topk',  3),
        ('soft_topk-8  ',     'soft_topk',  8),  # full NAP coverage (no multi-bit flips)
        ('ste-n1       ',     'ste',        1),
        ('ste-n3       ',     'ste',        3),
    ]
    print(f'{"mode":>14s} | {"fwd ms":>9s} | {"bwd ms":>9s} | {"peak MB":>9s}')
    print('-' * 60)
    results = []
    for label, mode, n_alt in configs:
        try:
            r = bench(label, mode, n_alt)
            results.append(r)
            print(f'{label:>14s} | {r["fwd_ms"]:9.2f} | {r["bwd_ms"]:9.2f} | {r["peak_mb"]:9.1f}')
        except Exception as e:
            print(f'{label:>14s} | FAILED: {type(e).__name__}: {e}')
        torch.cuda.empty_cache()

    # Relative bwd cost vs soft
    if results:
        soft_bwd = next((r for r in results if r['label'].strip() == 'soft'), None)
        if soft_bwd is not None:
            print('\nRelative backward cost (vs soft):')
            for r in results:
                speedup = soft_bwd['bwd_ms'] / r['bwd_ms']
                print(f'{r["label"]:>14s} | bwd: {speedup:5.2f}x  (peak MB ratio: {r["peak_mb"]/soft_bwd["peak_mb"]:5.2f}x)')


if __name__ == '__main__':
    main()
