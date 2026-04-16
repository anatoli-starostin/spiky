"""
Profile PermutationalLut with exp265 configuration:
  pair_mode='scrambled', input_nap=6, output_nap=32, tph=2048,
  n_inputs=n_outputs=32, n_heads=1.
Measures: forward, backward, total per step, params, memory.
Compares against the prior exp263 config (in=10, out=16, tph=512) for reference.
"""
import sys, os, time, gc
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from spiky.lutorch.permutational_lut import PermutationalLut

DEVICE = 'cuda:0'
BATCH = 64 * 32  # B*T tokens (matches a single layer's per-step input)
WARMUP = 5
ITERS = 50


def make_lut(input_nap, output_nap, tph, soft_mode='rational'):
    return PermutationalLut(
        n_inputs=32, n_outputs=32,
        input_nap=input_nap, output_nap=output_nap,
        n_heads=1, tph=tph,
        pair_mode='scrambled',
        soft_mode=soft_mode,
        temperature=0.1,
        random_seed=42, device=DEVICE,
        recompute_in_backward=True,
        initial_weights_noise=0.001,
    )


def time_fn(fn, warmup=WARMUP, iters=ITERS):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t0) / iters * 1000


def bench(input_nap, output_nap, tph, soft_mode='rational'):
    lut = make_lut(input_nap, output_nap, tph, soft_mode).to(DEVICE)
    lut.train()

    n_params = sum(p.numel() for p in lut.parameters())
    inner_w = lut.inner.projection.weights.numel()

    x = torch.randn(BATCH, 32, device=DEVICE)

    # Forward only
    def fwd():
        with torch.no_grad():
            return lut(x)
    t_fwd = time_fn(fwd)

    # Forward + backward
    def fwd_bwd():
        xx = x.detach().requires_grad_(True)
        out = lut(xx)
        target = torch.randn_like(out)
        loss = ((out - target) ** 2).mean()
        loss.backward()

    t_total = time_fn(fwd_bwd)
    t_bwd = t_total - t_fwd

    # Peak memory during forward
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        _ = lut(x)
    torch.cuda.synchronize()
    peak_mb = torch.cuda.max_memory_allocated() / 1024**2

    del lut
    gc.collect()
    torch.cuda.empty_cache()
    return n_params, inner_w, t_fwd, t_bwd, t_total, peak_mb


def fmt(n):
    if n >= 1e6:
        return f"{n/1e6:.2f}M"
    if n >= 1e3:
        return f"{n/1e3:.1f}K"
    return f"{n}"


print(f'Profile PermutationalLut (1 layer, batch={BATCH} tokens)')
print(f'Warmup={WARMUP} iters={ITERS}')
print()
header = (f'{"in_nap":>6} {"out_nap":>7} {"tph":>5} {"soft":>9} | '
          f'{"params":>10} {"inner_w":>10} | '
          f'{"fwd ms":>8} {"bwd ms":>8} {"total":>8} | {"peak MB":>9}')
print(header)
print('-' * len(header))

configs = [
    # exp265 main (current run)
    (6, 32, 2048, 'rational'),
    # exp263 (prior best fast PermLut)
    (10, 16, 512, 'rational'),
    # exp259 (aligned baseline shape)
    (10, 10, 512, 'rational'),
    # ste variant of exp265
    (6, 32, 2048, 'ste'),
    # sigmoid variant of exp265
    (6, 32, 2048, 'sigmoid'),
]

for cfg in configs:
    in_nap, out_nap, tph, soft = cfg
    p, w, tf, tb, tt, mem = bench(*cfg)
    print(f'{in_nap:>6} {out_nap:>7} {tph:>5} {soft:>9} | '
          f'{fmt(p):>10} {fmt(w):>10} | '
          f'{tf:>7.2f} {tb:>7.2f} {tt:>7.2f} | {mem:>8.1f}')
