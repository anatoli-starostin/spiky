"""Correctness gate for the table-stationary kernel. No timing.

Every instantiated (G, L, Bt) must be bit-exact against the AS `read_cells` output at BOTH
load16 settings, on BOTH cells distributions, at three batch sizes: B=512 (round), B=777
(exercises the tile tail), B=100 (smaller than every Bt). Plus the unique-writer property
is asserted empirically on the real cells tensor.
"""
import json
import os
import sys

import torch

SPIKY = os.path.expanduser('~/projects/spiky')
sys.path.insert(0, os.path.join(SPIKY, 'src'))
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from spiky.lutorch import pow2_int8           # noqa: E402
import ts                                     # noqa: E402
from bench_act3 import (H, T, NAP, K, D, LO, HI, Q, BLOCK_N,   # noqa: E402
                        real_cells, uniform_cells, discard_stats)

BATCHES = [512, 777, 100]


def unique_writer(cells):
    """For every (token, table, slot) that is not discarded, exactly one row can contribute:
    the cell index is a single value in [0, K). Verified, not assumed."""
    c1, c2, sh = cells[..., 0].long(), cells[..., 1].long(), cells[..., 2].long()
    sh1, sh2 = sh & 15, sh >> 4
    live1, live2 = sh1 != 15, sh2 != 15
    bad1 = int(((c1 >= K) & live1).sum())
    bad2 = int(((c2 >= K) & live2).sum())
    print(f'   live c1 cells {int(live1.sum()):,} all in [0,{K}): {bad1 == 0}')
    print(f'   live c2 cells {int(live2.sum()):,} all in [0,{K}): {bad2 == 0}')
    print(f'   => exactly one owning row per (token, table, slot): '
          f'{"HOLDS" if bad1 == 0 and bad2 == 0 else "VIOLATED"}')
    return bad1 == 0 and bad2 == 0


@torch.no_grad()
def main():
    print(f'pow2_int8: {pow2_int8.available()[1]}')
    print('building the TS extension (10 instantiations, a minute) ...')
    ts.mod(verbose=False)
    print('built\n')

    g = torch.Generator(device='cuda').manual_seed(5)
    packed = torch.randint(-127, 128, (H * T * K, D), device='cuda', dtype=torch.int8,
                           generator=g)
    tab = pow2_int8.stride_tables(packed, D)

    cells_by_dist = {}
    for dist in ('real', 'uniform'):
        c, _, _, _ = (real_cells(max(BATCHES)) if dist == 'real'
                      else uniform_cells(max(BATCHES)))
        cells_by_dist[dist] = c
    print('unique-writer property, real cells:')
    assert unique_writer(cells_by_dist['real'])
    print('unique-writer property, uniform cells:')
    assert unique_writer(cells_by_dist['uniform'])

    fits = [c for c in ts.CONFIGS if ts.budget(*c)['fits_smem']]
    over = [c for c in ts.CONFIGS if not ts.budget(*c)['fits_smem']]

    print('\nconfigs that exceed shared memory -- checking they REFUSE rather than substitute:')
    for G, L, Bt in over:
        b = ts.budget(G, L, Bt)
        try:
            ts.run(tab, cells_by_dist['real'][:512].contiguous(), D, G, L, Bt)
            print(f'   G={G:<3} L={L:<3} Bt={Bt:<5} NO ERROR  <-- unexpected')
        except Exception as ex:
            msg = str(ex).splitlines()[0]
            m = 'over by' in msg and str(b['smem_bytes'] - ts.SMEM_LIMIT) in msg
            print(f'   G={G:<3} L={L:<3} Bt={Bt:<5} refused, arithmetic '
                  f'{"matches" if m else "DIFFERS"}: needs {b["smem_bytes"]:,} B, '
                  f'over by {b["smem_bytes"] - ts.SMEM_LIMIT:,}')

    print(f'\nbit-exactness of the {len(fits)} fitting configs vs AS read_cells:')
    print(f'{"G":>4}{"L":>4}{"Bt":>6}{"dist":>9}{"B":>6}{"vs load16=T":>13}'
          f'{"vs load16=F":>13}{"|acc|max":>11}  verdict')
    bad = 0
    for G, L, Bt in fits:
        for dist in ('real', 'uniform'):
            for B in BATCHES:
                cells = cells_by_dist[dist][:B].contiguous()
                ref_t = pow2_int8.read_cells(tab, cells, NAP, D, LO, HI, Q,
                                             block_n=BLOCK_N, load16=True)
                ref_f = pow2_int8.read_cells(tab, cells, NAP, D, LO, HI, Q,
                                             block_n=BLOCK_N, load16=False)
                got = ts.run(tab, cells, D, G, L, Bt).float()
                e_t = (got - ref_t).abs().max().item()
                e_f = (got - ref_f).abs().max().item()
                amax = ref_t.abs().max().item()
                ok = (e_t == 0.0 and e_f == 0.0)
                bad += not ok
                print(f'{G:>4}{L:>4}{Bt:>6}{dist:>9}{B:>6}{e_t:>13.1f}{e_f:>13.1f}'
                      f'{amax:>11.0f}  {"OK" if ok else "MISMATCH"}')
                if not ok:
                    print('   stopping on first mismatch')
                    return 1
    print(f'\n{len(fits) * 2 * len(BATCHES)} comparisons, {bad} mismatches')
    print(f'largest |accumulator| seen: {amax:.0f}  '
          f'(float32 is exact on integers to 2^24 = 16,777,216; '
          f'{"below it, so the int32->float conversion is lossless here" if amax < 2**24 else "ABOVE it"})')

    rows = ts.ptxas(os.path.join(ts.BUILD, 'ptxas.log')) if os.path.exists(
        os.path.join(ts.BUILD, 'ptxas.log')) else {}
    json.dump({'fits': fits, 'over': over,
               'budgets': {f'{g}_{l}_{b}': ts.budget(g, l, b) for g, l, b in ts.CONFIGS},
               'ptxas': rows},
              open(os.path.join(HERE, 'artifacts', 'ts_gate.json'), 'w'), indent=1)
    return 0


if __name__ == '__main__':
    sys.exit(main())
