"""
Compare PermutationalLut speed with and without recompute_in_backward.
Uses exp271 config (in_nap=8, out_nap=32, tph=2048) which has the biggest
inner LUT and therefore the largest gap between the two modes.
"""
import sys, os, time, gc
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from spiky.lutorch.permutational_lut import PermutationalLut

DEVICE = 'cuda:0'
BATCH = 128 * 32  # bs=128 × ctx=32
WARMUP = 5
ITERS = 30


def make_lut(input_nap, output_nap, tph, soft_mode, recompute):
    return PermutationalLut(
        n_inputs=32, n_outputs=32,
        input_nap=input_nap, output_nap=output_nap,
        n_heads=1, tph=tph,
        pair_mode='scrambled',
        soft_mode=soft_mode,
        aggregation='matmul',
        temperature=0.1,
        random_seed=42, device=DEVICE,
        recompute_in_backward=recompute,
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


def bench(input_nap, output_nap, tph, soft_mode, recompute):
    lut = make_lut(input_nap, output_nap, tph, soft_mode, recompute).to(DEVICE)
    lut.train()
    x = torch.randn(BATCH, 32, device=DEVICE)

    def fwd():
        with torch.no_grad():
            return lut(x)
    t_fwd = time_fn(fwd)

    def fwd_bwd():
        xx = x.detach().requires_grad_(True)
        out = lut(xx)
        target = torch.randn_like(out)
        loss = ((out - target) ** 2).mean()
        loss.backward()
    t_total = time_fn(fwd_bwd)
    t_bwd = t_total - t_fwd

    # Peak memory during forward+backward
    torch.cuda.reset_peak_memory_stats()
    xx = x.detach().requires_grad_(True)
    out = lut(xx)
    loss = ((out - torch.randn_like(out)) ** 2).mean()
    loss.backward()
    torch.cuda.synchronize()
    peak_mb = torch.cuda.max_memory_allocated() / 1024**2

    del lut, xx, out, loss
    gc.collect()
    torch.cuda.empty_cache()
    return t_fwd, t_bwd, t_total, peak_mb


configs = [
    (6, 32, 2048),  # exp270 config
    (8, 32, 2048),  # exp271 config — bigger inner LUT
]

print(f'PermutationalLut: recompute_in_backward on/off comparison')
print(f'batch={BATCH} tokens, warmup={WARMUP}, iters={ITERS}')
print()

header = f'{"in_nap":>7} {"out_nap":>8} {"tph":>6} {"soft":>9} {"recompute":>10} | {"fwd":>7} {"bwd":>7} {"total":>7} {"peak MB":>9}'
print(header)
print('-' * len(header))

for cfg in configs:
    for soft in ['rational', 'ste']:
        for rec in [False, True]:
            t_fwd, t_bwd, t_total, peak = bench(*cfg, soft, rec)
            print(f'{cfg[0]:>7} {cfg[1]:>8} {cfg[2]:>6} {soft:>9} {str(rec):>10} | '
                  f'{t_fwd:>6.2f}ms {t_bwd:>6.2f}ms {t_total:>6.2f}ms {peak:>8.0f}')
