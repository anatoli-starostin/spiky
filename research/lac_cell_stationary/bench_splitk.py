"""Split-K S x B sweep against the AS baseline.

TIMING DISCIPLINE, decided up front:
  (a) AS bare          -- read_cells as it is, torch::empty inside the timed region
  (b) AS + zeroed buf  -- AS plus an equivalently-sized torch::zeros in the timed region,
                          so split-K's mandatory memset is matched rather than ignored
  (c) split-K          -- as it is, torch::zeros inside the timed region (atomics need it)
  (d) memset alone     -- at every B, so (c) can be corrected cleanly if preferred
L2 is HOT: a tight repeat loop with no flush between iterations, identical to how the
step-1 AS ladder was measured. NO CUDA graphs anywhere in this sweep — plain stream
launches, so the ~5 us launch floor is inside every number, as it was in step 1.
"""
import json
import os
import statistics
import sys

import torch

SPIKY = os.path.expanduser('~/projects/spiky')
sys.path.insert(0, os.path.join(SPIKY, 'src'))
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from spiky.lutorch import pow2_int8   # noqa: E402
import splitk                          # noqa: E402
import scatter                         # noqa: E402
from bench_act3 import (T, K, D, NAP, LO, HI, Q, BLOCK_N, real_cells)  # noqa: E402

BS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 24576]
SS = splitk.SS
FETCH = 1.3272
SMS = 170


def stats(fn, budget_s=1.2):
    fn()
    torch.cuda.synchronize()
    import time
    t0 = time.perf_counter()
    fn()
    torch.cuda.synchronize()
    one = time.perf_counter() - t0
    iters = max(12, min(80, int(budget_s / max(one, 1e-5))))
    for _ in range(max(5, iters // 4)):
        fn()
    torch.cuda.synchronize()
    t = []
    for _ in range(iters):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        t.append(s.elapsed_time(e))
    t.sort()
    return {'median': statistics.median(t), 'stdev': statistics.stdev(t), 'iters': iters}


@torch.no_grad()
def main():
    pow2_int8.ensure_registered()
    splitk.mod()
    scatter.mod()
    g = torch.Generator(device='cuda').manual_seed(5)
    W = pow2_int8.stride_tables(
        torch.randint(-127, 128, (T * K, D), device='cuda', dtype=torch.int8, generator=g), D)
    full, _, _, _ = real_cells(max(BS))

    r = stats(lambda: scatter.mod().empty_launch(170, 1024))
    launch_floor = r['median']
    print(f'launch floor (empty kernel, 170x1024, plain stream): {launch_floor:.4f} ms')
    print('L2 HOT throughout; no CUDA graphs; real cells.\n')

    base, base_z, memset = {}, {}, {}
    print(f'{"B":>7}{"AS bare":>10}{"stdev":>8}{"AS+zero":>10}{"memset":>9}'
          f'{"AS blocks":>11}{"SM occ":>8}')
    for B in BS:
        c = full[:B].contiguous()
        rb = stats(lambda cc=c: pow2_int8.read_cells(W, cc, NAP, D, LO, HI, Q,
                                                     block_n=BLOCK_N, load16=False))
        def asz(cc=c, b=B):
            y = torch.zeros(b, 1, D, device='cuda', dtype=torch.int32)
            y.add_(pow2_int8.read_cells(W, cc, NAP, D, LO, HI, Q,
                                        block_n=BLOCK_N, load16=False).to(torch.int32))
        rz = stats(asz)
        rm = stats(lambda b=B: torch.zeros(b, 1, D, device='cuda', dtype=torch.int32))
        base[B], base_z[B], memset[B] = rb['median'], rz['median'], rm['median']
        nb = (B + BLOCK_N - 1) // BLOCK_N
        print(f'{B:>7}{rb["median"]:>10.4f}{rb["stdev"]:>8.4f}{rz["median"]:>10.4f}'
              f'{rm["median"]:>9.4f}{nb:>11}{min(1.0, nb/SMS):>8.3f}')
        del c
        torch.cuda.empty_cache()

    print(f'\nSPLIT-K SWEEP (median ms, torch::zeros inside the timed region)')
    print(f'{"B":>7}' + ''.join(f'{"S="+str(s):>10}' for s in SS))
    grid = {}
    for B in BS:
        c = full[:B].contiguous()
        row = {}
        line = f'{B:>7}'
        for S in SS:
            rr = stats(lambda cc=c, s=S: splitk.run(W, cc, D, s))
            row[S] = rr['median']
            line += f'{rr["median"]:>10.4f}'
        grid[B] = row
        print(line)
        del c
        torch.cuda.empty_cache()

    print(f'\n{"B":>7}{"AS bare":>10}{"best S":>8}{"best ms":>10}{"vs AS":>8}'
          f'{"vs AS+zero":>12}{"pred sat B":>12}{"rate /s":>11}{"combine GB":>12}')
    out = []
    for B in BS:
        bs = min(grid[B], key=lambda s: grid[B][s])
        bm = grid[B][bs]
        lu = B * T * FETCH * D
        comb = bs * B * D * 4 / 1e9
        sat = ((SMS + bs - 1) // bs) * BLOCK_N
        print(f'{B:>7}{base[B]:>10.4f}{bs:>8}{bm:>10.4f}{base[B]/bm:>8.3f}'
              f'{base_z[B]/bm:>12.3f}{sat:>12}{lu/(bm*1e-3):>11.3e}{comb:>12.3f}')
        out.append({'B': B, 'as_bare': base[B], 'as_zero': base_z[B], 'memset': memset[B],
                    'best_S': bs, 'best_ms': bm, 'vs_as': base[B] / bm,
                    'vs_as_zero': base_z[B] / bm, 'pred_sat_B': sat,
                    'rate': lu / (bm * 1e-3), 'combine_gb': comb,
                    'row': grid[B]})
    json.dump({'launch_floor_ms': launch_floor, 'rows': out},
              open(os.path.join(HERE, 'artifacts', 'splitk_sweep.json'), 'w'), indent=1)
    print('\nwrote artifacts/splitk_sweep.json')

    # --- gate re-check after all the timing work ---
    print('\nPOST-SWEEP GATE RE-CHECK (nothing drifted?)')
    for S, B in ((1, 512), (16, 777), (256, 100)):
        c = full[:B].contiguous()
        rf = pow2_int8.read_cells(W, c, NAP, D, LO, HI, Q, block_n=BLOCK_N, load16=False)
        poison = torch.full((B, 1, D), -123456789, device='cuda', dtype=torch.int32)
        del poison
        got = splitk.run(W, c, D, S).float()
        print(f'   S={S:<4} B={B:<5} max|diff| {(got - rf).abs().max().item():.1f}')


if __name__ == '__main__':
    main()
