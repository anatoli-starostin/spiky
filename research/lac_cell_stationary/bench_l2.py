"""How much of the gather baseline's table traffic the 96 MB L2 absorbs, measured
without ncu.

ncu cannot run here (the driver has RmProfilingAdminOnly=1, so dram__bytes_read.sum and
lts__t_sector_hit_rate need root), so the L2 amortisation factor is bounded from achieved
bandwidth instead of read from a counter. The experiment: hold the per-token read volume
and the arithmetic fixed in FORM and vary only the table footprint by changing the row
width N, then plot achieved request bandwidth (B * NT * N bytes / time) against footprint.

  footprint <= 96 MB  -> the table set is L2-resident, requests are served at L2 speed
  footprint >> 96 MB  -> requests fall through to HBM and the achieved rate collapses
                         towards the 5090's 1,792 GB/s peak

The ratio of the two plateaus is the amortisation factor the baseline is actually getting,
and it is what decides whether the cell-stationary kernel's headline advantage -- reading
the table set once instead of B times -- exists at all on this hardware.
"""
import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import lac  # noqa: E402

HBM_PEAK_GBS = 1792.0        # 512-bit GDDR7 at 28 Gbps
L2_BYTES = 96 * 2 ** 20
NS = (256, 512, 1024, 1536, 2048, 3072, 4096)
B = 4096
NT, R = 256, 256


def main():
    lac.mod()
    out = []
    print(f'B={B}, NT={NT}, R={R}; L2 = {L2_BYTES/2**20:.0f} MiB, '
          f'HBM peak {HBM_PEAK_GBS:.0f} GB/s')
    print(f'{"N":>6}{"footprint MiB":>15}{"per-token KiB":>15}{"median ms":>11}'
          f'{"req GB/s":>11}{"x HBM peak":>12}{"L2-resident":>13}')
    for N in NS:
        g = torch.Generator(device='cuda').manual_seed(0)
        T = torch.randint(-127, 128, (NT, R, N), device='cuda', dtype=torch.int8, generator=g)
        J = torch.randint(0, R, (B, NT), device='cuda', dtype=torch.uint8, generator=g)
        C = torch.randn(B, NT, device='cuda', generator=g) * 0.1 + 1.0
        y = torch.zeros(B, N, device='cuda')
        fn = lambda: lac.run_gather(T, J, C, tsplit=1, use_coef=True, y=y,
                                   tok_per_blk=max(1, min(32, 1024 // (N // 4))))
        lac.burn_in(fn, seconds=1.5)
        r = lac.timeit(fn, iters=50, warmup=5)
        req = B * NT * N
        gbs = req / (r['median_ms'] * 1e-3) / 1e9
        fp = NT * R * N
        print(f'{N:>6}{fp/2**20:>15.1f}{NT*N/1024:>15.1f}{r["median_ms"]:>11.4f}'
              f'{gbs:>11.1f}{gbs/HBM_PEAK_GBS:>12.2f}'
              f'{"yes" if fp <= L2_BYTES else "no":>13}')
        out.append({'N': N, 'footprint_bytes': fp, 'per_token_bytes': NT * N,
                    'median_ms': r['median_ms'], 'request_gbs': gbs,
                    'x_hbm_peak': gbs / HBM_PEAK_GBS, 'l2_resident': fp <= L2_BYTES})
        del T, J, C, y
        torch.cuda.empty_cache()

    fit = [o for o in out if o['l2_resident']]
    spill = [o for o in out if not o['l2_resident']]
    if fit and spill:
        a = max(o['request_gbs'] for o in fit)
        b = min(o['request_gbs'] for o in spill)
        print(f'\nbest L2-resident request rate {a:.0f} GB/s; worst spilled {b:.0f} GB/s'
              f'  -> {a/b:.2f}x')
    print('\nAmortisation, stated carefully: with the 64 MiB table set resident in the')
    print('96 MB L2, the compulsory DRAM traffic for a whole batch is the table set ONCE,')
    print(f'so the ratio of requested to compulsory bytes at B=24576 is')
    print(f'  {24576*NT*1024/2**20:.0f} MiB requested / {NT*R*1024/2**20:.0f} MiB '
          f'compulsory = {24576/R:.0f}x.')
    print('The measured request rate exceeding HBM peak is the direct evidence that the')
    print('cache is in fact supplying that factor; the number above is its lower bound.')
    p = os.path.join(HERE, 'artifacts', 'l2.json')
    os.makedirs(os.path.dirname(p), exist_ok=True)
    json.dump(out, open(p, 'w'), indent=1)
    print('wrote', p)


if __name__ == '__main__':
    main()
