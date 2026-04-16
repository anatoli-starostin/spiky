"""
Profile matmul vs scatter aggregation for PermutationalLut at exp265 config.
"""
import sys, os, time
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from spiky.lutorch.permutational_lut import PermutationalLut

DEVICE = 'cuda:0'
BATCH = 64 * 32
WARMUP = 5
ITERS = 30


def make_lut(aggregation, soft_mode='rational'):
    return PermutationalLut(
        n_inputs=32, n_outputs=32,
        input_nap=6, output_nap=32,
        n_heads=1, tph=2048,
        pair_mode='scrambled',
        soft_mode=soft_mode,
        aggregation=aggregation,
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


def bench(aggregation, soft_mode):
    lut = make_lut(aggregation, soft_mode).to(DEVICE)
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

    del lut
    torch.cuda.empty_cache()
    return t_fwd, t_bwd, t_total


print(f'Profile: PermutationalLut aggregation comparison')
print(f'config: in=6 out=32 tph=2048 N=32, batch={BATCH}')
print()
header = f'{"aggregation":>12} {"soft":>9} | {"fwd":>7} {"bwd":>7} {"total":>7}'
print(header)
print('-' * len(header))

for soft in ['rational', 'sigmoid', 'ste']:
    for agg in ['scatter', 'matmul']:
        t_fwd, t_bwd, t_total = bench(agg, soft)
        print(f'{agg:>12} {soft:>9} | {t_fwd:>6.2f}ms {t_bwd:>6.2f}ms {t_total:>6.2f}ms')
