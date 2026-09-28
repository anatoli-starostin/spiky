"""Does memory layout move TS at all? A falsification test of the scan-bound attribution.

Four arms per config: baseline layouts, blocked tables, transposed cells, both. The 1/L law
predicts NO material change; any real speedup falsifies the scan-bound reading.

BOTH LAYOUT CONVERSIONS ARE HOST-SIDE ONE-OFFS, done once before timing and NEVER inside
the timed region. Zeroing stays symmetric: TS allocates its zeroed int32 output inside
ts_read (inside the timed region); the AS baseline is reported bare and with an added
zeroed buffer, as before.

ncu counters are still unavailable (RmProfilingAdminOnly=1), so the amplification figures
are sector accounting from the access pattern, not measured DRAM bytes, and are labelled
as such.
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

from spiky.lutorch import pow2_int8          # noqa: E402
import ts                                    # noqa: E402
from bench_act3 import (H, T, NAP, K, D, LO, HI, Q, BLOCK_N,  # noqa: E402
                        real_cells, uniform_cells, discard_stats)

N = 24576
ITERS, WARMUP = 50, 10
PEAK = 1792.0
SECTOR = 32
VALID = [(2, 64, 256), (4, 32, 256), (4, 32, 512), (8, 16, 256), (8, 16, 1024),
         (16, 8, 512), (32, 4, 512)]
LAW_K = 4359.0          # fitted t*L from the step-1 ladder, ms-lanes


def stats(fn):
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    t = []
    for _ in range(ITERS):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        t.append(s.elapsed_time(e))
    t.sort()
    return {'median': statistics.median(t), 'min': t[0], 'stdev': statistics.stdev(t)}


def amp_table(L, blocked):
    """Sector amplification on the one-off table load, from the access pattern."""
    if blocked:
        return 1.0                      # thread r reads G*L contiguous bytes; warp covers 32*G*L
    useful = min(L, SECTOR)             # L contiguous bytes out of a 32 B sector, stride = D
    return SECTOR / useful


def amp_cells(G, trans):
    if trans:
        return 1.0                      # 3*Bt contiguous per table per tile
    useful = min(3 * G, SECTOR)         # 3G contiguous of every 3T, stride 3T = 768
    return SECTOR / useful


def traffic(G, L, Bt, tlay, clay, n):
    blocks = (T // G) * (D // L)
    table = blocks * 256 * G * L * amp_table(L, tlay)
    cells = blocks * ((n + Bt - 1) // Bt) * Bt * G * 3 * amp_cells(G, clay)
    atomic = blocks * n * L * 8
    return table, cells, atomic, table + cells + atomic


@torch.no_grad()
def main():
    ts.mod()
    g = torch.Generator(device='cuda').manual_seed(5)
    W = pow2_int8.stride_tables(
        torch.randint(-127, 128, (T * K, D), device='cuda', dtype=torch.int8, generator=g), D)
    out = {'N': N, 'rows': []}

    for dist in ('real', 'uniform'):
        cells, _, _, _ = (real_cells(N) if dist == 'real' else uniform_cells(N))
        st = discard_stats(cells)
        fetched = st['frac_c1_fetched'] + st['frac_c2_fetched']
        cells_t = ts.transposed_cells(cells)                 # host-side one-off, untimed
        as_bytes = fetched * T * D * N

        f = lambda: pow2_int8.read_cells(W, cells, NAP, D, LO, HI, Q,
                                         block_n=BLOCK_N, load16=False)
        base_as = stats(f)
        print(f'\n{"="*118}\ncells: {dist}   AS load16=False re-measured '
              f'{base_as["median"]:.4f} ms   ({as_bytes/1e9:.2f} GB, '
              f'{as_bytes/(base_as["median"]*1e-3)/1e9:.0f} GB/s)\n{"="*118}')
        print(f'{"config":<20}{"layout":<20}{"ms":>10}{"stdev":>8}{"GB":>8}{"GB/s":>8}'
              f'{"%peak":>7}{"vs base":>9}{"vs law":>8}{"vs AS":>8}')

        for G, L, Bt in VALID:
            Wb = ts.blocked_tables(W, T, K, D, G, L)         # host-side one-off, untimed
            base_ms = None
            for tlay, clay, name in ((0, 0, 'baseline'), (1, 0, 'blocked tables'),
                                     (0, 1, 'transposed cells'), (1, 1, 'both')):
                Wx, Cx = (Wb if tlay else W), (cells_t if clay else cells)
                fn = (lambda a=G, b=L, c=Bt, t=tlay, u=clay, w=Wx, cc=Cx:
                      ts.run(w, cc, D, a, b, c, tlay=t, clay=u, N=N, T=T))
                r = stats(fn)
                tb, ce, at, tot = traffic(G, L, Bt, tlay, clay, N)
                if base_ms is None:
                    base_ms = r['median']
                gbs = tot / (r['median'] * 1e-3) / 1e9
                print(f'{f"G={G} L={L} Bt={Bt}":<20}{name:<20}{r["median"]:>10.2f}'
                      f'{r["stdev"]:>8.2f}{tot/1e9:>8.2f}{gbs:>8.0f}{100*gbs/PEAK:>7.1f}'
                      f'{base_ms/r["median"]:>9.3f}{r["median"]/(LAW_K/L):>8.2f}'
                      f'{base_as["median"]/r["median"]:>8.4f}')
                out['rows'].append({'dist': dist, 'G': G, 'L': L, 'Bt': Bt, 'layout': name,
                                    'tlay': tlay, 'clay': clay, **r,
                                    'table_bytes': tb, 'cells_bytes': ce,
                                    'atomic_bytes': at, 'bytes': tot,
                                    'vs_base': base_ms / r['median'],
                                    'vs_as': base_as['median'] / r['median']})
            del Wb
            torch.cuda.empty_cache()
        out['rows'].append({'dist': dist, 'layout': 'AS_load16_False', **base_as,
                            'bytes': as_bytes})
        del cells, cells_t
        torch.cuda.empty_cache()

    json.dump(out, open(os.path.join(HERE, 'artifacts', 'ts_layout.json'), 'w'), indent=1)
    print('\nwrote artifacts/ts_layout.json')


if __name__ == '__main__':
    main()
