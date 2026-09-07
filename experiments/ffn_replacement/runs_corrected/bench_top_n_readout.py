"""Benchmark the top-n blended read-out: forward, backward and peak memory vs n=1.

Measures the FFN layer at the real anchor sizing (H4, d_in=d_out=48, nap8/K256, tph128 and
tph256) and the full 6-layer model step, so both the isolated cost and the cost that
actually shows up in a training run are visible.

GPU CONTENTION MATTERS FOR THESE NUMBERS and not for the correctness gates, so this refuses
to run while another CUDA process is on the device unless --allow-contention is passed.
bpb is deterministic; wall clock is not.

    python bench_top_n_readout.py
"""
import argparse
import subprocess
import sys
import time

import torch

sys.path.insert(0, '/home/astarostin/projects/spiky/experiments/ffn_replacement/tools')
from spiky.lutorch.compression_mhl import CompressionMultiHeadLUT   # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--allow-contention', action='store_true')
ap.add_argument('--iters', type=int, default=30)
ap.add_argument('--warmup', type=int, default=8)
a = ap.parse_args()

DEV = 'cuda'
N_TOK = 6144          # device_batch 12 x seq 512, one micro-batch of the real runs


def other_cuda_procs():
    out = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,process_name',
                          '--format=csv,noheader'], capture_output=True, text=True).stdout
    return [l for l in out.strip().splitlines()
            if l.strip() and 'python' in l.lower()]


busy = other_cuda_procs()
if busy and not a.allow_contention:
    print('*** another CUDA python process is running -- timings would be meaningless:')
    for b in busy:
        print('   ', b)
    print('*** re-run with --allow-contention to override, or wait.')
    sys.exit(2)
if busy:
    print(f'!! CONTENDED RUN: {len(busy)} other CUDA python process(es) present; '
          f'absolute numbers are NOT trustworthy, ratios only roughly so.\n')


def timeit(fn, iters, warmup):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / iters * 1e3          # ms


def bench_layer(tph, n, tau=0.1, impl='light', force_torch=False):
    """impl='fast' benchmarks FastMultiHeadLut at the same geometry -- the real
    alternative for a directional routing gradient, whose surrogate is a softmax over
    ALL 2^nap cells rather than the blend's n. force_torch disables the native fused-eval
    kernel so the n=1 forward is measured on the same code path the blend must use."""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    lay = CompressionMultiHeadLUT(
        input_dim=384, output_dim=384, inner_in_dim=48, inner_out_dim=48,
        nap=8, tph=tph, n_heads=4, lut_impl=impl, forward_confidence=True,
        confidence_form='margin', z_norm=(impl != 'bh4'), random_seed=1000,
        **({'read_top_n': n, 'read_tau': tau} if impl == 'light' else {}),
        device=DEV).to(DEV)
    if force_torch:
        from spiky.lutorch.light_multi_head_lut import LightMultiHeadLUT
        for m in lay.modules():
            if isinstance(m, LightMultiHeadLUT):
                m._native_msb_scored = None
                m._native_msb = None
    x = torch.randn(N_TOK, 384, device=DEV, requires_grad=True)

    def fwd():
        with torch.no_grad():
            lay(x)

    def fwdbwd():
        lay.zero_grad(set_to_none=True)
        if x.grad is not None:
            x.grad = None
        lay(x).pow(2).sum().backward()

    f = timeit(fwd, a.iters, a.warmup)
    fb = timeit(fwdbwd, a.iters, a.warmup)
    peak = torch.cuda.max_memory_allocated() / 2**20
    params = sum(p.numel() for p in lay.parameters())
    del lay, x
    torch.cuda.empty_cache()
    return f, fb, peak, params


print(f'device {torch.cuda.get_device_name(0)}   tokens/call {N_TOK:,} '
      f'(device_batch 12 x seq 512)   iters {a.iters}\n')
print(f'{"config":26} {"n":>2} {"fwd ms":>9} {"fwd+bwd ms":>12} {"bwd ms":>9} '
      f'{"peak MiB":>10} {"vs n=1 fwd":>11} {"vs n=1 f+b":>11}')
for tph in (128, 256):
    base = None
    rows = ([('light', 1, False), ('light', 1, True), ('light', 2, False),
             ('light', 3, False), ('fast', 1, False)])
    for impl, n, ft in rows:
        f, fb, peak, params = bench_layer(tph, n, impl=impl, force_torch=ft)
        if base is None:                       # n=1 on the torch path is the fair baseline
            base = (f, fb)
        if impl == 'light' and n == 1 and ft:
            base = (f, base[1])                # forward baseline := torch path
        lbl = ('Light n=1 (fused eval)' if impl == 'light' and n == 1 and not ft else
               'Light n=1 (torch path)' if impl == 'light' and n == 1 else
               f'Light n={n} blend' if impl == 'light' else
               'FAST (full-2^8 surrogate)')
        print(f'{lbl:26} {n:>2} {f:>9.3f} {fb:>12.3f} {fb - f:>9.3f} {peak:>10.1f} '
              f'{f / base[0]:>10.2f}x {fb / base[1]:>10.2f}x')
    print(f'   ^ nap8/tph{tph}, {params/1e6:.1f}M FFN params/layer\n')

print('NOTE ON THE FORWARD COLUMN: at n=1 the layer takes the native fused-eval kernel '
      'under no_grad,\nwhile n>1 cannot (it needs the margins), so "fwd" compares the '
      'fused kernel against the torch\npath and overstates the blend cost. The fwd+bwd '
      'column is the training-relevant one: both\nsides use the torch path there.')
