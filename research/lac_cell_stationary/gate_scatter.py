"""Bit-exactness gate for both scatter arms, before any timing. int32 exact equality."""
import os
import sys

import torch

SPIKY = os.path.expanduser('~/projects/spiky')
sys.path.insert(0, os.path.join(SPIKY, 'src'))
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from spiky.lutorch import pow2_int8    # noqa: E402
import scatter                          # noqa: E402
import ts                               # noqa: E402
from bench_act3 import (T, K, D, NAP, LO, HI, Q, BLOCK_N,  # noqa: E402
                        real_cells, uniform_cells, discard_stats)

BATCHES = [512, 777, 100]           # 777 is not a multiple of any Bt or tile


@torch.no_grad()
def main():
    print('building the scatter extension ...')
    scatter.mod(verbose=False, fresh=True)
    print('built\n')
    g = torch.Generator(device='cuda').manual_seed(5)
    W = pow2_int8.stride_tables(
        torch.randint(-127, 128, (T * K, D), device='cuda', dtype=torch.int8, generator=g), D)
    full = {'real': real_cells(max(BATCHES))[0], 'uniform': uniform_cells(max(BATCHES))[0]}

    n = bad = 0
    print('ARM A -- global atomics')
    print(f'{"L":>4}{"dist":>9}{"B":>6}{"vs AS L16=T":>13}{"vs AS L16=F":>13}  verdict')
    for L in scatter.LS:
        for dist in ('real', 'uniform'):
            for B in BATCHES:
                cells = full[dist][:B].contiguous()
                rt = pow2_int8.read_cells(W, cells, NAP, D, LO, HI, Q,
                                          block_n=BLOCK_N, load16=True)
                rf = pow2_int8.read_cells(W, cells, NAP, D, LO, HI, Q,
                                          block_n=BLOCK_N, load16=False)
                got = scatter.run_a(W, cells, D, L).float()
                et = (got - rt).abs().max().item()
                ef = (got - rf).abs().max().item()
                ok = et == 0.0 and ef == 0.0
                n += 1
                bad += not ok
                if not ok or (dist == 'real' and B == 777):
                    print(f'{L:>4}{dist:>9}{B:>6}{et:>13.1f}{ef:>13.1f}  '
                          f'{"OK" if ok else "MISMATCH"}')
                if not ok:
                    return 1

    print('\nARM B -- shared accumulate')
    print(f'{"Bt":>4}{"L":>4}{"pad":>5}{"dist":>9}{"B":>6}{"vs AS L16=T":>13}'
          f'{"vs AS L16=F":>13}  verdict')
    combos = [(bt, L, 0) for bt in scatter.BTS for L in scatter.LS]
    combos += [(16, 4, 1), (16, 4, 4), (8, 4, 1)]
    for Bt, L, pad in combos:
        b = scatter.budget_b(Bt, pad)
        if not b['fits']:
            try:
                scatter.run_b(W, full['real'][:512].contiguous(), D, L, Bt, pad)
                print(f'{Bt:>4}{L:>4}{pad:>5}  NO ERROR <-- unexpected')
            except Exception as ex:
                print(f'{Bt:>4}{L:>4}{pad:>5}  refused: {str(ex).splitlines()[0][:80]}')
            continue
        for dist in ('real', 'uniform'):
            for B in BATCHES:
                cells = full[dist][:B].contiguous()
                rt = pow2_int8.read_cells(W, cells, NAP, D, LO, HI, Q,
                                          block_n=BLOCK_N, load16=True)
                rf = pow2_int8.read_cells(W, cells, NAP, D, LO, HI, Q,
                                          block_n=BLOCK_N, load16=False)
                # POISON the caching allocator: arm B's output is now torch::empty, so an
                # element left unwritten would read back whatever was in that block. Filling
                # a same-shaped tensor with a sentinel and freeing it makes the allocator
                # hand the same memory back, so "unwritten" cannot hide behind a zero page.
                poison = torch.full((B, 1, D), -123456789, device='cuda', dtype=torch.int32)
                del poison
                got = scatter.run_b(W, cells, D, L, Bt, pad).float()
                et = (got - rt).abs().max().item()
                ef = (got - rf).abs().max().item()
                ok = et == 0.0 and ef == 0.0
                n += 1
                bad += not ok
                if not ok or (dist == 'real' and B == 777 and L == 4):
                    print(f'{Bt:>4}{L:>4}{pad:>5}{dist:>9}{B:>6}{et:>13.1f}{ef:>13.1f}  '
                          f'{"OK" if ok else "MISMATCH"}')
                if not ok:
                    return 1

    print(f'\n{n} comparisons, {bad} mismatches (int32 exact equality, no tolerance)')

    # the AS and TS paths must still pass their own gates
    print('\nregression: AS and TS unchanged?')
    cells = full['real'][:512].contiguous()
    rf = pow2_int8.read_cells(W, cells, NAP, D, LO, HI, Q, block_n=BLOCK_N, load16=False)
    tsv = ts.run(W, cells, D, 2, 64, 256, N=512, T=T).float()
    print(f'   TS G=2 L=64 vs AS: max|diff| {(tsv - rf).abs().max().item():.1f}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
