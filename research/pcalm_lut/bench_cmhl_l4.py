"""Microbenchmark for the L=4 CompressionMHL config. Trains nothing, writes no run directory.

The point is to check a reported wall clock honestly. The harness times each step with time.time()
around the optimiser step and NEVER calls torch.cuda.synchronize(), so on CUDA that measures kernel
LAUNCH time, not execution -- the queue drains later. Every number here is taken with an explicit
synchronize, so it is the real cost.
"""
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.expanduser('~/projects/spiky/src'))
from autoencoder import Autoencoder, branch_ratios, evaluate, lut_stats  # noqa: E402
from data import load  # noqa: E402

CFG = dict(width=128, depth_L=2, kind='lut', n_tables=128, device='cuda', seed=0, residual=False,
           block_norm='layernorm', lut_impl='compression', norm_position='pre', n_blocks=4,
           inner_out=-1, inner_in=128, final_norm=True)


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def timed(fn, n, warmup=5):
    for _ in range(warmup):
        fn()
    sync()
    t = time.time()
    for _ in range(n):
        fn()
    sync()
    return (time.time() - t) / n


def main():
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    xtr, _ = load('fashion', train=True, device=dev)
    xte, _ = load('fashion', train=False, device=dev)
    m = Autoencoder(784, **CFG)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    print(f'params {sum(p.numel() for p in m.parameters())/1e6:.2f}M on {dev}; '
          f'train tensor {tuple(xtr.shape)} on {xtr.device}, test {tuple(xte.shape)}')

    b = xtr[:128]

    def step():
        opt.zero_grad(set_to_none=True)
        loss = (m(b) - b).pow(2).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()

    per_step = timed(step, 50)
    print(f'\ntrain step (fwd+bwd+clip+Adam, batch 128, SYNCHRONISED): {per_step*1e3:.2f} ms')
    print(f'  -> 10000 steps of training alone = {per_step*10000:.1f} s')

    # what one probe costs: two full evaluate() calls plus the two diagnostics
    def probe():
        evaluate(m, xtr[:10000])
        evaluate(m, xte)
        lut_stats(m, xtr[:512])
        branch_ratios(m, xtr[:512])

    per_probe = timed(probe, 5, warmup=2)
    print(f'\none probe (20000 images evaluated + lut_stats + branch_ratios): {per_probe*1e3:.1f} ms')
    print(f'  -> 401 probes = {per_probe*401:.1f} s')

    total = per_step * 10000 + per_probe * 401
    print(f'\npredicted total for the 10000-step run: {total:.1f} s')
    print(f'the run reported: 39.7 s  ->  ratio {39.7/total:.2f}')

    # and the unsynchronised measurement the harness itself makes, for comparison
    for _ in range(5):
        step()
    t = time.time()
    for _ in range(50):
        step()
    unsync = (time.time() - t) / 50
    sync()
    print(f'\nthe same step timed WITHOUT synchronize (what the harness logs): {unsync*1e3:.2f} ms '
          f'-- {per_step/unsync:.1f}x optimistic')


if __name__ == '__main__':
    main()
