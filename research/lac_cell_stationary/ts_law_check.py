"""Confirm the G=32 / L=16 point against the 1/L law, instead of leaving it inferred.

THIS CONFIG SPILLS (255 registers, 1,264 B spill stores / 2,348 B spill loads), so it is
NOT a valid performance data point. It is measured here for one purpose only: to check
whether the law t ~= 4359/L predicts it. Reported as a law check, never as a config result.
"""
import os
import statistics
import sys

import torch

SPIKY = os.path.expanduser('~/projects/spiky')
sys.path.insert(0, os.path.join(SPIKY, 'src'))
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from spiky.lutorch import pow2_int8    # noqa: E402
import ts                              # noqa: E402
from bench_act3 import T, K, D, NAP, LO, HI, Q, BLOCK_N, real_cells  # noqa: E402

N, LAW_K = 24576, 4359.0
CHECKS = [(32, 16, 512), (16, 32, 512), (8, 32, 256)]


@torch.no_grad()
def main():
    ts.mod()
    g = torch.Generator(device='cuda').manual_seed(5)
    W = pow2_int8.stride_tables(
        torch.randint(-127, 128, (T * K, D), device='cuda', dtype=torch.int8, generator=g), D)
    cells, _, _, _ = real_cells(N)
    rows = ts.ptxas('/tmp/ts_build.log')
    print(f'{"config":<22}{"spill st":>9}{"measured ms":>13}{"law 4359/L":>12}'
          f'{"meas/law":>10}{"vs AS 1.42":>12}')
    for G, L, Bt in CHECKS:
        fn = lambda a=G, b=L, c=Bt: ts.run(W, cells, D, a, b, c, N=N, T=T)
        for _ in range(5):
            fn()
        torch.cuda.synchronize()
        t = []
        for _ in range(20):
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record()
            fn()
            e.record()
            torch.cuda.synchronize()
            t.append(s.elapsed_time(e))
        m = statistics.median(t)
        sp = rows.get(f'ts<G={G},L={L},Bt={Bt}>', {}).get('spill_st', -1)
        pred = LAW_K / L
        print(f'{f"G={G} L={L} Bt={Bt}":<22}{sp:>9}{m:>13.2f}{pred:>12.2f}'
              f'{m/pred:>10.3f}{1.4219/m:>12.4f}')
    print('\nAll three SPILL and are law checks only, not valid performance data points.')


if __name__ == '__main__':
    main()
