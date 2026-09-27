"""Attribute the cell-stationary kernels' runtime to their parts, by ablation.

MODE 0 = full kernel, 1 = everything but the global atomic flush, 2 = the row scan
alone (no shared atomics, no flush). Only MODE 0 is correct; 1 and 2 exist to be timed.
The differences say which term dominates -- which is the thing the owner asked to have
checked against his cost model rather than assumed.
"""
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import lac  # noqa: E402
from bench import make_shape_B  # noqa: E402

V1 = [(32, 32, 4), (16, 32, 4), (64, 32, 2)]
V0 = [(4, 48), (16, 96)]
BS = (1024, 24576)


def t(fn):
    r = lac.timeit(fn, iters=10, warmup=3)
    return r['median_ms']


def main():
    m = lac.mod()
    for B in BS:
        T, J, C = make_shape_B(B)
        NT, R, N = T.shape
        y = torch.zeros(B, N, device='cuda')
        print(f'\n===== B={B} =====')
        print(f'{"kernel":<28}{"full ms":>10}{"no flush":>10}{"scan only":>11}'
              f'{"flush":>9}{"sh.atomic":>11}{"scan %":>8}')
        for K, M, TB in V1:
            ts = []
            for mode in (0, 1, 2):
                fn = (lambda mo=mode: m.cs_v1_diag(T, J, C, y, K, M, TB, mo, 1))
                y.zero_()
                ts.append(t(fn))
            full, nofl, scan = ts
            print(f'{f"cs_v1 K={K} M={M} TB={TB}":<28}{full:>10.3f}{nofl:>10.3f}'
                  f'{scan:>11.3f}{full-nofl:>9.3f}{nofl-scan:>11.3f}'
                  f'{100*scan/full:>8.1f}')
        for K, M in V0:
            ts = []
            for mode in (0, 1):
                fn = (lambda mo=mode: m.cs_v0_diag(T, J, C, y, K, M, mo, 1))
                y.zero_()
                ts.append(t(fn))
            full, scan = ts
            print(f'{f"cs_v0 K={K} M={M}":<28}{full:>10.3f}{"-":>10}'
                  f'{scan:>11.3f}{full-scan:>9.3f}{"-":>11}{100*scan/full:>8.1f}')
        y.zero_()
        g = t(lambda: lac.run_gather(T, J, C, y=y))
        print(f'{"gather tsplit=1":<28}{g:>10.3f}')
        del T, J, C, y
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
