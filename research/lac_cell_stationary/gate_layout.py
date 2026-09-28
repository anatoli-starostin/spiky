"""Bit-exactness gate for the layout variants, before any timing.

Every valid config x {baseline, blocked tables, transposed cells, both} x {real, uniform}
x B in {512, 777 (ragged tile tail), 100 (smaller than every Bt)} must be bit-exact int32
against AS read_cells at both load16 settings.
"""
import os
import sys

import torch

SPIKY = os.path.expanduser('~/projects/spiky')
sys.path.insert(0, os.path.join(SPIKY, 'src'))
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from spiky.lutorch import pow2_int8    # noqa: E402
import ts                              # noqa: E402
from bench_act3 import (H, T, NAP, K, D, LO, HI, Q, BLOCK_N,  # noqa: E402
                        real_cells, uniform_cells)
from bench_layout import VALID          # noqa: E402

BATCHES = [512, 777, 100]
LAYOUTS = [(0, 0, 'baseline'), (1, 0, 'blocked tables'),
           (0, 1, 'transposed cells'), (1, 1, 'both')]


@torch.no_grad()
def main():
    ts.mod()
    g = torch.Generator(device='cuda').manual_seed(5)
    W = pow2_int8.stride_tables(
        torch.randint(-127, 128, (T * K, D), device='cuda', dtype=torch.int8, generator=g), D)
    full = {'real': real_cells(max(BATCHES))[0], 'uniform': uniform_cells(max(BATCHES))[0]}
    print(f'{"config":<18}{"layout":<18}{"dist":>9}{"B":>6}{"vs L16=T":>10}'
          f'{"vs L16=F":>10}  verdict')
    bad = n = 0
    for G, L, Bt in VALID:
        Wb = ts.blocked_tables(W, T, K, D, G, L)
        for tlay, clay, name in LAYOUTS:
            for dist in ('real', 'uniform'):
                for B in BATCHES:
                    cells = full[dist][:B].contiguous()
                    ct = ts.transposed_cells(cells)
                    rt = pow2_int8.read_cells(W, cells, NAP, D, LO, HI, Q,
                                              block_n=BLOCK_N, load16=True)
                    rf = pow2_int8.read_cells(W, cells, NAP, D, LO, HI, Q,
                                              block_n=BLOCK_N, load16=False)
                    got = ts.run(Wb if tlay else W, ct if clay else cells, D, G, L, Bt,
                                 tlay=tlay, clay=clay, N=B, T=T).float()
                    et = (got - rt).abs().max().item()
                    ef = (got - rf).abs().max().item()
                    ok = et == 0.0 and ef == 0.0
                    bad += not ok
                    n += 1
                    if not ok or (B == 777 and dist == 'real'):
                        print(f'{f"G={G} L={L} Bt={Bt}":<18}{name:<18}{dist:>9}{B:>6}'
                              f'{et:>10.1f}{ef:>10.1f}  {"OK" if ok else "MISMATCH"}')
                    if not ok:
                        print('  stopping on first mismatch')
                        return 1
        del Wb
        torch.cuda.empty_cache()
    print(f'\n{n} comparisons, {bad} mismatches '
          f'({len(VALID)} configs x {len(LAYOUTS)} layouts x 2 distributions x '
          f'{len(BATCHES)} batch sizes x 2 AS settings)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
