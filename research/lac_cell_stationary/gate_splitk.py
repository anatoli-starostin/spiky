"""Correctness gate for the split-K arm. No timings.

Every S x {real, uniform} cells x B in {1, 8, 100, 512, 777} against AS `read_cells` at
BOTH load16 settings, int32 exact equality. The output allocation is poisoned with a
sentinel before every launch so an unwritten element cannot pass silently.
"""
import os
import subprocess
import sys

import torch

SPIKY = os.path.expanduser('~/projects/spiky')
sys.path.insert(0, os.path.join(SPIKY, 'src'))
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from spiky.lutorch import pow2_int8   # noqa: E402
import splitk                          # noqa: E402
from bench_act3 import (T, K, D, NAP, LO, HI, Q, BLOCK_N,   # noqa: E402
                        real_cells, uniform_cells)

BATCHES = [1, 8, 100, 512, 777]
SENT = -123456789


@torch.no_grad()
def main():
    print('building the split-K extension ...')
    splitk.mod(verbose=False, fresh=True)
    print('built\n')
    g = torch.Generator(device='cuda').manual_seed(5)
    W = pow2_int8.stride_tables(
        torch.randint(-127, 128, (T * K, D), device='cuda', dtype=torch.int8, generator=g), D)
    full = {'real': real_cells(max(BATCHES))[0], 'uniform': uniform_cells(max(BATCHES))[0]}

    print(f'{"S":>5}{"dist":>9}{"B":>6}{"grid":>8}{"smem B":>9}'
          f'{"max|diff| T":>13}{"ndiff T":>9}{"max|diff| F":>13}{"ndiff F":>9}  verdict')
    n = bad = 0
    for S in splitk.SS:
        for dist in ('real', 'uniform'):
            for B in BATCHES:
                cells = full[dist][:B].contiguous()
                rt = pow2_int8.read_cells(W, cells, NAP, D, LO, HI, Q,
                                          block_n=BLOCK_N, load16=True)
                rf = pow2_int8.read_cells(W, cells, NAP, D, LO, HI, Q,
                                          block_n=BLOCK_N, load16=False)
                # poison: the output is zeroed inside the extension, so this proves the
                # zeroing itself happens rather than relying on a fresh page
                poison = torch.full((B, 1, D), SENT, device='cuda', dtype=torch.int32)
                del poison
                got = splitk.run(W, cells, D, S).float()
                dt, df = (got - rt).abs(), (got - rf).abs()
                et, ef = dt.max().item(), df.max().item()
                nt, nf = int((dt != 0).sum()), int((df != 0).sum())
                ok = et == 0.0 and ef == 0.0
                n += 1
                bad += not ok
                if not ok or (dist == 'real' and B == 777):
                    print(f'{S:>5}{dist:>9}{B:>6}{splitk.grid(B, S):>8}'
                          f'{splitk.smem(S):>9}{et:>13.1f}{nt:>9}{ef:>13.1f}{nf:>9}  '
                          f'{"OK" if ok else "MISMATCH"}')
                if not ok:
                    print('   stopping on first mismatch')
                    return 1
    print(f'\n{n} comparisons, {bad} mismatches — int32 exact equality, no tolerance.')
    print(f'({len(splitk.SS)} values of S x 2 cell distributions x {len(BATCHES)} batch sizes '
          f'x 2 AS load16 settings)')

    print('\nREFUSAL PATHS (no silent fallback)')
    c = full['real'][:64].contiguous()
    for desc, fn in [
        ('S=3 (does not divide T=256)', lambda: splitk.run(W, c, D, 3)),
        ('S=512 (> T)', lambda: splitk.run(W, c, D, 512)),
        ('S=0', lambda: splitk.run(W, c, D, 0)),
        ('D=512 (UPR would be 32, not the pinned AS shape)',
         lambda: splitk.run(W, c, 512, 4)),
        ('D=1000 (not a multiple of 16)', lambda: splitk.run(W, c, 1000, 4)),
    ]:
        try:
            fn()
            print(f'   {desc:<50} NO ERROR  <-- unexpected')
        except Exception as ex:
            print(f'   {desc:<50} {str(ex).splitlines()[0][:96]}')

    print('\nOCCUPANCY AND GRID (from the driver, on the real instantiation)')
    occ = splitk.mod().occupancy(splitk.smem(1))
    blk, regs, ssmem, local, maxthr = occ
    print(f'   cudaFuncGetAttributes: numRegs={regs}, static smem={ssmem} B, local={local} B, '
          f'maxThreadsPerBlock={maxthr}')
    print(f'   cudaOccupancyMaxActiveBlocksPerMultiprocessor (dyn smem {splitk.smem(1)} B) '
          f'= {blk} block(s)/SM')
    print(f'{"S":>5}{"tables/chunk":>14}{"dyn smem B":>12}{"blk/SM":>8}'
          f'{"grid at B=24576":>17}{"saturation B":>14}')
    for S in splitk.SS:
        b = splitk.mod().occupancy(splitk.smem(S))[0]
        print(f'{S:>5}{T//S:>14}{splitk.smem(S):>12}{b:>8}'
              f'{splitk.grid(24576, S):>17}{splitk.saturation_B(S, b):>14}')

    print('\nSASS COMBINE FORM')
    so = os.path.join(splitk.BUILD, 'lac_splitk.so')
    sass = subprocess.run(['/usr/local/cuda/bin/cuobjdump', '-sass', so],
                          capture_output=True, text=True).stdout
    import re
    for part in re.split(r'Function : ', sass)[1:]:
        if 'splitk_kernel' not in part.split('\n', 1)[0]:
            continue
        redg = len(re.findall(r'\bREDG[.A-Z0-9]*', part))
        atomg = len(re.findall(r'\bATOMG[.A-Z0-9]*', part))
        atoms = len(re.findall(r'\bATOMS[.A-Z0-9]*', part))
        print(f'   REDG (fire-and-forget) {redg}, ATOMG (returns a value) {atomg}, '
              f'ATOMS (shared) {atoms}')
        print(f'   => {"REDG, as intended" if redg and not atomg else "NOT the intended form"}; '
              f'{redg} per thread against nl=16 lanes in the combine loop')
        break
    return 0


if __name__ == '__main__':
    sys.exit(main())
