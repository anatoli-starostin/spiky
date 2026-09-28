"""Step 1: is AS's small-batch cost grid underfill, or something else?

Measurement only, no new compute kernel. Two additive helpers were needed and are
read-only: `pow2_int8.occupancy(...)` (cudaOccupancyMaxActiveBlocksPerMultiprocessor +
cudaFuncGetAttributes on the exact instantiation, launches nothing) and an `empty_kernel`
in the research scatter extension for the launch-overhead floor.

AS grid geometry: read_cells launches ceil(N / block_n) x H blocks of block_n * UPR
threads. At D=1024 the lane slice is UPR = 64 units, so block_n * 64 <= 1024 forces
block_n = 16 -- and that is already the FINEST gridding the kernel can do.
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
import scatter                         # noqa: E402
from bench_act3 import (T, K, D, NAP, LO, HI, Q, real_cells, discard_stats)  # noqa: E402

BS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 24576]
SMS = 170
BLOCK_N, UPR = 16, D // 16
FETCH = 1.3272


def stats(fn, iters=60, warmup=15):
    for _ in range(warmup):
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
    return {'median': statistics.median(t), 'min': t[0], 'stdev': statistics.stdev(t)}


@torch.no_grad()
def main():
    pow2_int8.ensure_registered()
    ext = pow2_int8.load()
    scatter.mod()
    g = torch.Generator(device='cuda').manual_seed(5)
    W = pow2_int8.stride_tables(
        torch.randint(-127, 128, (T * K, D), device='cuda', dtype=torch.int8, generator=g), D)
    cells_full, _, _, _ = real_cells(max(BS))

    smem = T * BLOCK_N * 3 + (4 - (T * BLOCK_N * 3) % 4) % 4
    occ = ext.occupancy(BLOCK_N, UPR, False, False, False, smem)
    blocks_sm, regs, st_smem, local, maxthr = occ
    thr = BLOCK_N * UPR
    print(f'AS instantiation: block_n={BLOCK_N}, UPR={UPR}, threads/block={thr}, '
          f'load16=False, prologue compiled out')
    print(f'  cudaFuncGetAttributes: numRegs={regs}, static smem={st_smem} B, '
          f'local={local} B, maxThreadsPerBlock={maxthr}')
    print(f'  dynamic smem requested = {smem} B (T*block_n*3, no Z tile on this path)')
    print(f'  cudaOccupancyMaxActiveBlocksPerMultiprocessor = {blocks_sm} block(s)/SM')
    print(f'  => the GPU saturates at {SMS * blocks_sm} blocks, i.e. '
          f'B = {SMS * blocks_sm * BLOCK_N:,} tokens\n')

    # --- launch-overhead floor ---
    print('LAUNCH-OVERHEAD FLOOR (empty kernel)')
    for nb, nt in ((1, 32), (170, 1024), (1536, 1024)):
        f = lambda b=nb, t=nt: scatter.mod().empty_launch(b, t)
        r = stats(f)
        try:
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(3):
                    f()
            torch.cuda.current_stream().wait_stream(s)
            torch.cuda.synchronize()
            gr = torch.cuda.CUDAGraph()
            with torch.cuda.graph(gr):
                f()
            rg = stats(lambda: gr.replay())
            gtxt = f'{rg["median"]:.4f}'
        except Exception as ex:
            gtxt = f'capture failed ({type(ex).__name__})'
        print(f'   {nb:>5} blocks x {nt:>4} threads:  plain {r["median"]:.4f} ms, '
              f'graph {gtxt} ms')
    print()

    # --- the ladder ---
    print(f'{"B":>7}{"blocks":>8}{"blk/SM":>8}{"SM occ":>8}{"median ms":>11}{"stdev":>8}'
          f'{"lane upd":>11}{"rate /s":>11}{"rate/occ":>11}{"ms/B":>10}')
    rows = []
    sat_occ = sat_rate = None
    prev_rate = None
    for B in BS:
        cells = cells_full[:B].contiguous()
        nblocks = (B + BLOCK_N - 1) // BLOCK_N
        occupancy = min(1.0, nblocks / (SMS * blocks_sm))
        r = stats(lambda c=cells: pow2_int8.read_cells(W, c, NAP, D, LO, HI, Q,
                                                       block_n=BLOCK_N, load16=False))
        lu = B * T * FETCH * D
        rate = lu / (r['median'] * 1e-3)
        print(f'{B:>7}{nblocks:>8}{blocks_sm:>8}{occupancy:>8.3f}{r["median"]:>11.4f}'
              f'{r["stdev"]:>8.4f}{lu:>11.3e}{rate:>11.3e}{rate/occupancy:>11.3e}'
              f'{r["median"]/B*1e3:>10.4f}')
        rows.append({'B': B, 'blocks': nblocks, 'occ': occupancy, 'ms': r['median'],
                     'lane_updates': lu, 'rate': rate, 'rate_norm': rate / occupancy})
        if sat_occ is None and occupancy >= 1.0:
            sat_occ = B
        if prev_rate is not None and rate < prev_rate * 1.05 and sat_rate is None and B > 16:
            sat_rate = B
        prev_rate = max(prev_rate or 0, rate)
        del cells
        torch.cuda.empty_cache()

    print(f'\noccupancy first reaches 1.0 at B = {sat_occ}')
    print(f'measured rate first plateaus (next step < +5%) at B = {sat_rate}')
    print('ms/B is the per-token cost; a flat rate/occ column means underfill explains it.')

    # --- block_n cross-check ---
    print('\nBLOCK_N CROSS-CHECK at D=1024')
    for bn in (16, 32, 64, 128):
        try:
            pow2_int8.read_cells(W, cells_full[:64].contiguous(), NAP, D, LO, HI, Q,
                                 block_n=bn, load16=False)
            print(f'   block_n={bn:<4} OK ({bn*UPR} threads/block)')
        except Exception as ex:
            print(f'   block_n={bn:<4} refused: {str(ex).splitlines()[0][:96]}')
    print(f'   smem on the prologue-free path is only T*block_n*3, so block_n=64 would need '
          f'{T*64*3:,} B -- well inside the limit. The binding constraint is THREADS: '
          f'block_n * UPR = block_n * {UPR} must be <= 1024.')

    json.dump({'occ': {'blocks_per_sm': blocks_sm, 'regs': regs, 'smem': smem,
                       'threads': thr}, 'rows': rows,
               'saturation_occ_B': sat_occ, 'saturation_rate_B': sat_rate},
              open(os.path.join(HERE, 'artifacts', 'underfill.json'), 'w'), indent=1)
    print('\nwrote artifacts/underfill.json')


if __name__ == '__main__':
    main()
