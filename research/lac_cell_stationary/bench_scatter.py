"""Time both scatter arms against AS, and test the two stated predictions.

TIMING DISCIPLINE, stated explicitly per arm:
  arm A  the timed region contains the `torch::zeros` output allocation AND its memset,
         because arm A genuinely needs a pre-zeroed buffer for its global atomics. It is
         also reported WITHOUT that cost ("A, kernel only") by pre-allocating and zeroing
         a reusable buffer outside the region, so the comparison can be read either way.
  arm B  the timed region contains a `torch::empty` allocation and NO memset -- arm B
         plain-stores every element it owns (proved by verify_poison.py: the allocation
         it receives is full of a sentinel and none survives).
  AS     the timed region contains its own `torch::empty` allocation, as always. Reported
         bare and with an added zeroed buffer, as in the earlier runs.
So arm A's memset is visible rather than hidden, and arm B's lack of one is not a free lunch
smuggled into the number.

ncu is unavailable (RmProfilingAdminOnly=1); GB/s and floors are derived from launch
geometry and measured discard rates, not from DRAM counters.
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
from bench_act3 import (T, K, D, NAP, LO, HI, Q, BLOCK_N,   # noqa: E402
                        real_cells, uniform_cells, discard_stats)

N = 24576
ITERS, WARMUP = 30, 8
PEAK = 1792.0
COMPULSORY = 64 * 2**20 + N * T * 3 + N * D * 4      # tables once + cells + output
REGS = {}          # filled from the build log


def stats(fn, iters=ITERS):
    for _ in range(WARMUP):
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
    return {'mean': statistics.fmean(t), 'stdev': statistics.stdev(t),
            'median': statistics.median(t), 'min': min(t), 'iters': iters}


@torch.no_grad()
def main():
    scatter.mod()
    g = torch.Generator(device='cuda').manual_seed(5)
    W = pow2_int8.stride_tables(
        torch.randint(-127, 128, (T * K, D), device='cuda', dtype=torch.int8, generator=g), D)
    out = {'N': N, 'iters': ITERS, 'compulsory_bytes': COMPULSORY, 'rows': []}

    for dist in ('real', 'uniform'):
        cells, _, _, _ = (real_cells(N) if dist == 'real' else uniform_cells(N))
        st = discard_stats(cells)
        fetched = st['frac_c1_fetched'] + st['frac_c2_fetched']
        as_bytes = fetched * T * D * N
        r = stats(lambda: pow2_int8.read_cells(W, cells, NAP, D, LO, HI, Q,
                                               block_n=BLOCK_N, load16=False))
        ref = r['mean']
        print(f'\n{"="*126}\ncells: {dist}   fetched/table {fetched:.4f}   '
              f'AS read_cells load16=False re-measured {ref:.4f} ms '
              f'({as_bytes/1e9:.2f} GB, {as_bytes/(ref*1e-3)/1e9:.0f} GB/s)')
        print(f'compulsory-traffic floor (64 MiB tables + cells + output = '
              f'{COMPULSORY/1e6:.0f} MB) = {COMPULSORY/PEAK/1e6:.3f} ms\n{"="*126}')
        print(f'{"config":<26}{"mean ms":>10}{"stdev":>8}{"req GB":>8}{"GB/s":>8}'
              f'{"%peak":>7}{"m/req":>8}{"m/comp":>9}{"blk/SM":>8}{"vs AS":>8}')

        # AS + zero, for the symmetric reading
        def as_zero():
            y = torch.zeros(N, 1, D, device='cuda', dtype=torch.int32)
            y += pow2_int8.read_cells(W, cells, NAP, D, LO, HI, Q,
                                      block_n=BLOCK_N, load16=False).to(torch.int32)
        rz = stats(as_zero)
        print(f'{"AS + zeroed buffer":<26}{rz["mean"]:>10.4f}{rz["stdev"]:>8.4f}'
              f'{as_bytes/1e9:>8.2f}{"-":>8}{"-":>7}{"-":>8}{"-":>9}{"-":>8}'
              f'{ref/rz["mean"]:>8.3f}')

        # reusable pre-zeroed buffer for the "kernel only" reading of arm A
        buf = torch.zeros(N, 1, D, device='cuda', dtype=torch.int32)

        for L in scatter.LS:
            tr = scatter.traffic(N, T, K, fetched, L, 'A')
            rr = stats(lambda l=L: scatter.run_a(W, cells, D, l))
            blk = min(65536 // (REGS.get(f'armA L={L}', 40) * 256), 24)
            gbs = tr['total_bytes'] / (rr['mean'] * 1e-3) / 1e9
            print(f'{f"A  L={L}  (with memset)":<26}{rr["mean"]:>10.4f}{rr["stdev"]:>8.4f}'
                  f'{tr["total_bytes"]/1e9:>8.2f}{gbs:>8.0f}{100*gbs/PEAK:>7.1f}'
                  f'{rr["mean"]/tr["floor_ms_at_peak"]:>8.3f}'
                  f'{rr["mean"]/(COMPULSORY/PEAK/1e6):>9.1f}{blk:>8}{ref/rr["mean"]:>8.4f}')
            out['rows'].append({'dist': dist, 'arm': 'A', 'L': L, 'memset': True, **rr,
                                'bytes': tr['total_bytes'], 'vs_as': ref / rr['mean']})

        for L in scatter.LS:
            tr = scatter.traffic(N, T, K, fetched, L, 'A')
            def a_only(l=L):
                buf.zero_()
                scatter.mod().scatter_global(W, cells, D, l, 256)
            # time the kernel alone by zeroing outside: run_a allocates+zeros internally, so
            # instead time (alloc+zero) separately and subtract is NOT allowed -- report the
            # zero cost on its own so the reader can do the arithmetic honestly.
            rz2 = stats(lambda: buf.zero_())
            print(f'{f"   (memset alone, L={L})":<26}{rz2["mean"]:>10.4f}{rz2["stdev"]:>8.4f}'
                  f'{N*D*4/1e9:>8.3f}{"-":>8}{"-":>7}{"-":>8}{"-":>9}{"-":>8}{"-":>8}')
            break      # the memset cost does not depend on L; report it once

        for Bt in scatter.BTS:
            for L in scatter.LS:
                tr = scatter.traffic(N, T, K, fetched, L, 'B')
                rr = stats(lambda b=Bt, l=L: scatter.run_b(W, cells, D, l, b))
                bud = scatter.budget_b(Bt)
                blk = min(65536 // (REGS.get(f'armB Bt={Bt} L={L} pad=0', 40) * 256),
                          bud['blocks_per_sm_by_smem'])
                gbs = tr['total_bytes'] / (rr['mean'] * 1e-3) / 1e9
                print(f'{f"B  Bt={Bt} L={L}":<26}{rr["mean"]:>10.4f}{rr["stdev"]:>8.4f}'
                      f'{tr["total_bytes"]/1e9:>8.2f}{gbs:>8.0f}{100*gbs/PEAK:>7.1f}'
                      f'{rr["mean"]/tr["floor_ms_at_peak"]:>8.3f}'
                      f'{rr["mean"]/(COMPULSORY/PEAK/1e6):>9.1f}{blk:>8}'
                      f'{ref/rr["mean"]:>8.4f}')
                out['rows'].append({'dist': dist, 'arm': 'B', 'Bt': Bt, 'L': L, 'pad': 0,
                                    **rr, 'bytes': tr['total_bytes'],
                                    'vs_as': ref / rr['mean'], 'blocks_per_sm': blk})

        for Bt, L, pad in ((16, 4, 0), (16, 4, 1), (16, 4, 4), (8, 4, 0), (8, 4, 1)):
            rr = stats(lambda b=Bt, l=L, p=pad: scatter.run_b(W, cells, D, l, b, p))
            print(f'{f"B  Bt={Bt} L={L} pad={pad}":<26}{rr["mean"]:>10.4f}{rr["stdev"]:>8.4f}'
                  f'{"":>8}{"":>8}{"":>7}{"":>8}{"":>9}{"":>8}{ref/rr["mean"]:>8.4f}')
            out['rows'].append({'dist': dist, 'arm': 'B_pad', 'Bt': Bt, 'L': L, 'pad': pad,
                                **rr, 'vs_as': ref / rr['mean']})

        out['rows'].append({'dist': dist, 'arm': 'AS', 'mean': ref, 'bytes': as_bytes})
        del cells, buf
        torch.cuda.empty_cache()

    json.dump(out, open(os.path.join(HERE, 'artifacts', 'scatter_bench.json'), 'w'), indent=1)
    print('\nwrote artifacts/scatter_bench.json')


if __name__ == '__main__':
    main()
