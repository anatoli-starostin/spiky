"""Locate the B at which load16=False overtakes load16=True.

The specified ladder jumps 1 -> 64 and the sign flips inside that gap, so this fills it
in. Same vehicle, same cells, same timing rules as bench_act3.py; hot only, since the
two arms' hot/cold ratios are indistinguishable above B=1.
"""
import os
import sys

import torch

SPIKY = os.path.expanduser('~/projects/spiky')
sys.path.insert(0, os.path.join(SPIKY, 'src'))
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from spiky.lutorch import pow2_int8        # noqa: E402
from bench_act3 import (H, T, NAP, K, D, LO, HI, Q, BLOCK_N, real_cells,  # noqa: E402
                        timeit, capture)

BS = [1, 2, 4, 8, 12, 16, 24, 32, 48, 64]


@torch.no_grad()
def main():
    g = torch.Generator(device='cuda').manual_seed(5)
    tab = pow2_int8.stride_tables(
        torch.randint(-127, 128, (H * T * K, D), device='cuda', dtype=torch.int8,
                      generator=g), D)
    cells_full, _, _, _ = real_cells(max(BS))
    print(f'{"B":>6}{"load16=T ms":>14}{"load16=F ms":>14}{"F/T":>8}  winner')
    for B in BS:
        c = cells_full[:B].contiguous()
        t = {}
        for l16 in (True, False):
            fn = (lambda l=l16, cc=c: pow2_int8.read_cells(
                tab, cc, NAP, D, LO, HI, Q, block_n=BLOCK_N, load16=l))
            gr = capture(fn)
            t[l16] = timeit(fn, graph=gr)
            del gr
        r = t[False] / t[True]
        print(f'{B:>6}{t[True]:>14.4f}{t[False]:>14.4f}{r:>8.3f}  '
              f'{"load16=False" if r < 1 else "load16=True"}')
        del c
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
