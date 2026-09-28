"""Time the spill-free TS configs against the AS baseline, and test the traffic model.

SYMMETRIC ZEROING (failure mode 5). Both arms allocate their output inside the timed
region, so neither gets a free buffer:
  TS  `ts_read` calls torch::zeros({N,1,D}, int32) itself -- allocation + memset are
      inside the kernel launch wrapper and therefore inside the timed region.
  AS  `read_cells` calls torch::empty({N,H,D}, float) and writes every element, so it
      allocates inside the timed region too but does NOT memset -- it cannot, it has no
      atomics to combine into. To keep the comparison honest the AS arm is ALSO timed
      with an explicit `torch.zeros_like`-equivalent added in front, reported as
      "AS + zero", alongside the bare AS number. The bare number is the fair one for
      "what the AS kernel costs"; the +zero number is the fair one for "what it costs to
      produce a summed output the way TS has to".
Both are reported so the choice is visible rather than baked in.

Timing: CUDA events with the sync outside the window, >=50 iterations, warmup discarded,
median / min / p90 reported. The AS baseline is RE-MEASURED here, not quoted.
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
HBM_PEAK_GBS = 1792.0


def stats(fn, iters=ITERS, warmup=WARMUP):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t = []
    for _ in range(iters):
        s, e = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        t.append(s.elapsed_time(e))
    t.sort()
    return {'median': statistics.median(t), 'min': t[0],
            'p90': t[int(0.9 * (len(t) - 1))], 'stdev': statistics.stdev(t)}


def ts_traffic(G, L, Bt, n):
    blocks = (T // G) * (D // L)
    table = blocks * 256 * G * L
    cells = blocks * ((n + Bt - 1) // Bt) * Bt * G * 3
    atomic = blocks * n * L * 8                      # int32 RMW: read + write
    return table, cells, atomic, table + cells + atomic, blocks


@torch.no_grad()
def main():
    print(f'pow2_int8: {pow2_int8.available()[1]}')
    ts.mod()
    g = torch.Generator(device='cuda').manual_seed(5)
    tab = pow2_int8.stride_tables(
        torch.randint(-127, 128, (H * T * K, D), device='cuda', dtype=torch.int8,
                      generator=g), D)

    out = {'N': N, 'iters': ITERS, 'rows': []}
    for dist in ('real', 'uniform'):
        cells, _, _, _ = (real_cells(N) if dist == 'real' else uniform_cells(N))
        st = discard_stats(cells)
        fetched = st['frac_c1_fetched'] + st['frac_c2_fetched']
        as_bytes_f = fetched * T * D * N
        as_bytes_t = 2.0 * T * D * N
        print(f'\n{"="*112}\ncells: {dist}   cells fetched per table '
              f'{fetched:.4f} of 2   (discarded tables {100*st["frac_table_discarded"]:.2f}%, '
              f'second cell dropped {100*st["frac_second_cell_dropped"]:.2f}%)')
        print(f'{"="*112}')

        # ---- AS baseline, re-measured, both load16 settings, bare and +zero ----
        print(f'{"arm":<30}{"median ms":>11}{"min":>9}{"p90":>9}{"stdev":>9}'
              f'{"GB":>9}{"GB/s":>9}{"%peak":>7}{"vs AS":>8}')
        base = {}
        for l16 in (True, False):
            f = (lambda l=l16: pow2_int8.read_cells(tab, cells, NAP, D, LO, HI, Q,
                                                    block_n=BLOCK_N, load16=l))
            r = stats(f)
            b = as_bytes_t if l16 else as_bytes_f
            base[l16] = r['median']
            print(f'{f"AS read_cells load16={l16}":<30}{r["median"]:>11.4f}{r["min"]:>9.4f}'
                  f'{r["p90"]:>9.4f}{r["stdev"]:>9.4f}{b/1e9:>9.2f}'
                  f'{b/(r["median"]*1e-3)/1e9:>9.0f}'
                  f'{100*b/(r["median"]*1e-3)/1e9/HBM_PEAK_GBS:>7.0f}{1.0:>8.2f}')
            out['rows'].append({'dist': dist, 'arm': f'AS_load16_{l16}', **r,
                                'bytes': b})
        ref = base[False]

        def as_plus_zero():
            y = torch.zeros(N, H, D, device='cuda', dtype=torch.float32)
            y += pow2_int8.read_cells(tab, cells, NAP, D, LO, HI, Q,
                                      block_n=BLOCK_N, load16=False)
        r = stats(as_plus_zero)
        print(f'{"AS load16=False + zero buf":<30}{r["median"]:>11.4f}{r["min"]:>9.4f}'
              f'{r["p90"]:>9.4f}{r["stdev"]:>9.4f}{as_bytes_f/1e9:>9.2f}'
              f'{"-":>9}{"-":>7}{ref/r["median"]:>8.2f}')
        out['rows'].append({'dist': dist, 'arm': 'AS_plus_zero', **r})

        # ---- TS, the spill-free configs ----
        # the spill-free set; (2, 64, 256) added in step 2 as the widest lane slice that is
        # simultaneously spill-free, inside the shared-memory limit and at Bt >= 256.
        valid = [(8, 16, 256), (8, 16, 1024), (4, 32, 256), (4, 32, 512),
                 (16, 8, 512), (32, 4, 512), (2, 64, 256)]
        print()
        for G, L, Bt in valid:
            f = (lambda a=G, b=L, c=Bt: ts.run(tab, cells, D, a, b, c))
            r = stats(f)
            tb, ce, at, tot, blocks = ts_traffic(G, L, Bt, N)
            gbs = tot / (r['median'] * 1e-3) / 1e9
            print(f'{f"TS G={G} L={L} Bt={Bt}":<30}{r["median"]:>11.4f}{r["min"]:>9.4f}'
                  f'{r["p90"]:>9.4f}{r["stdev"]:>9.4f}{tot/1e9:>9.2f}{gbs:>9.0f}'
                  f'{100*gbs/HBM_PEAK_GBS:>7.0f}{ref/r["median"]:>8.2f}')
            out['rows'].append({'dist': dist, 'arm': f'TS_{G}_{L}_{Bt}', **r,
                                'blocks': blocks, 'table_bytes': tb, 'cells_bytes': ce,
                                'atomic_bytes': at, 'bytes': tot,
                                'vs_as': ref / r['median']})

        # ---- is the traffic model predictive? ----
        rows = [x for x in out['rows'] if x['dist'] == dist and x['arm'].startswith('TS_')]
        by_t = sorted(rows, key=lambda x: x['bytes'])
        by_m = sorted(rows, key=lambda x: x['median'])
        print(f'\n  traffic order (least first): {[x["arm"] for x in by_t]}')
        print(f'  measured order (fastest first): {[x["arm"] for x in by_m]}')
        agree = [x['arm'] for x in by_t] == [x['arm'] for x in by_m]
        print(f'  ORDERINGS {"AGREE" if agree else "DISAGREE"}')
        if len(rows) > 1:
            import math
            xs = [math.log(x['bytes']) for x in rows]
            ys = [math.log(x['median']) for x in rows]
            mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
            num = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
            den = (sum((a - mx) ** 2 for a in xs) * sum((b - my) ** 2 for b in ys)) ** 0.5
            print(f'  log-log correlation of traffic vs time across the six configs: '
                  f'{num/den if den else float("nan"):+.3f}')
        del cells
        torch.cuda.empty_cache()

    json.dump(out, open(os.path.join(HERE, 'artifacts', 'ts_bench.json'), 'w'), indent=1)
    print('\nwrote artifacts/ts_bench.json')


if __name__ == '__main__':
    main()
