"""
Profile LUT forward+backward with and without custom CUDA kernels.
Uses a single MultiHeadLut: input_dim=32, n_heads=1, n_outputs=32, nap=6.
"""
import sys, os, time
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

DEVICE = 'cuda:0'
E = 32
NAP = 6
BATCH = 128 * 32  # B*T tokens
SEED = 42
WARMUP = 5
ITERS = 50


def make_lut(tph, use_cuda):
    # Temporarily override env before import
    if not use_cuda:
        os.environ['SPIKY_LUTORCH_NO_CUSTOM_CUDA_KERNELS'] = '1'
    else:
        os.environ.pop('SPIKY_LUTORCH_NO_CUSTOM_CUDA_KERNELS', None)

    # Force reimport to pick up env change
    import importlib
    import spiky.lutorch.l_projection as lp_mod
    importlib.reload(lp_mod)
    import spiky.lutorch.multi_head_lut as mhl_mod
    importlib.reload(mhl_mod)

    from spiky.lutorch.multi_head_lut import MultiHeadLut
    from spiky.lutorch.lut_helpers import UncertaintyMode, AnchorSamplingPolicy

    return MultiHeadLut(
        input_dim=E, n_heads=1, n_outputs=E,
        n_anchor_pairs=NAP, tables_per_head=tph,
        smooth_mode=False, n_alternatives=1,
        normalize_weights=False, calibrate_output=False,
        anchor_sampling_policy=AnchorSamplingPolicy.FULL_COVERAGE,
        initial_weights_noise=0.001, uncertainty_mode=UncertaintyMode.INVERSE_L1,
        random_seed=SEED, device=DEVICE, recompute_in_backward=True,
    )


def bench(tph, use_cuda):
    lut = make_lut(tph, use_cuda).to(DEVICE)
    lut.train()
    x = torch.randn(BATCH, E, device=DEVICE)

    # Warmup
    for _ in range(WARMUP):
        xx = x.detach().requires_grad_(True)
        out = lut(xx)
        out.sum().backward()

    # Forward only
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(ITERS):
        out = lut(x.detach())
    torch.cuda.synchronize()
    t_fwd = (time.time() - t0) / ITERS * 1000

    # Forward + backward
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(ITERS):
        xx = x.detach().requires_grad_(True)
        out = lut(xx)
        out.sum().backward()
    torch.cuda.synchronize()
    t_total = (time.time() - t0) / ITERS * 1000

    t_bwd = t_total - t_fwd

    del lut
    torch.cuda.empty_cache()
    return t_fwd, t_bwd, t_total


print(f'Profiling: input_dim={E}, n_heads=1, n_outputs={E}, nap={NAP}, batch={BATCH}')
print(f'Warmup={WARMUP}, iters={ITERS}')
print()

header = f'{"tph":>6} | {"fwd_cuda":>9} | {"fwd_py":>9} | {"fwd_ratio":>9} | {"bwd_cuda":>9} | {"bwd_py":>9} | {"bwd_ratio":>9} | {"tot_cuda":>9} | {"tot_py":>9} | {"tot_ratio":>9}'
print(header)
print('-' * len(header))

for tph in [64, 128, 256, 512, 1024, 2048, 4096]:
    try:
        fwd_c, bwd_c, tot_c = bench(tph, use_cuda=True)
        fwd_p, bwd_p, tot_p = bench(tph, use_cuda=False)
        print(f'{tph:>6} | {fwd_c:>8.1f}ms | {fwd_p:>8.1f}ms | {fwd_p/fwd_c:>8.2f}x | '
              f'{bwd_c:>8.1f}ms | {bwd_p:>8.1f}ms | {bwd_p/bwd_c:>8.2f}x | '
              f'{tot_c:>8.1f}ms | {tot_p:>8.1f}ms | {tot_p/tot_c:>8.2f}x')
    except RuntimeError as e:
        print(f'{tph:>6} | OOM: {e}')
        break
