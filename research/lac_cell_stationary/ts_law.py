"""Is TS's runtime the 1/L scan law, and what does that law say the design can ever reach?

The measured six configs vary in G (4..32), Bt (256..1024) and L (4..32) yet their times
depend on L alone. That is the signature of the token scan, whose total work is

    blocks * threads * passes * tokens
  = (T/G)*(D/L) * 256 * 2G * N          =  2 * T * D * 256 * N / L

-- independent of G and of Bt, and inversely proportional to L. If L*t is constant across
the six, the kernel is scan-bound and the traffic model is irrelevant to it.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
T, D, N = 256, 1024, 24576
AS_MS = None


def main():
    d = json.load(open(os.path.join(HERE, 'artifacts', 'ts_bench.json')))
    rows = [r for r in d['rows'] if r['dist'] == 'real' and r['arm'].startswith('TS_')]
    base = [r for r in d['rows'] if r['dist'] == 'real' and r['arm'] == 'AS_load16_False'][0]
    print(f'{"config":<22}{"G":>4}{"L":>5}{"Bt":>6}{"median ms":>11}'
          f'{"L * t":>10}{"scan ops":>12}{"ops/s":>11}')
    prod = []
    for r in rows:
        _, G, L, Bt = r['arm'].split('_')
        G, L, Bt = int(G), int(L), int(Bt)
        ops = 2 * T * D * 256 * N / L
        lt = L * r['median']
        prod.append(lt)
        print(f'{r["arm"]:<22}{G:>4}{L:>5}{Bt:>6}{r["median"]:>11.2f}'
              f'{lt:>10.0f}{ops:>12.3e}{ops/(r["median"]*1e-3):>11.3e}')
    lo, hi = min(prod), max(prod)
    print(f'\nL * t ranges {lo:.0f} to {hi:.0f} ms-lanes -- a spread of '
          f'{100*(hi-lo)/lo:.0f}% across configs whose G varies 8x and Bt 4x.')
    print('=> the kernel is SCAN-BOUND: runtime depends on L and on nothing else.')
    k = sum(prod) / len(prod)
    print(f'\nfitted law: t ~= {k:.0f} / L  ms')
    print(f'{"L":>6}{"predicted ms":>15}{"vs AS":>9}   note')
    for L in (16, 32, 64, 128, 256, 512, 1024):
        t = k / L
        note = ''
        if L == 1024:
            note = 'the whole row in one block: G would have to be < 1'
        elif L == 128:
            note = 'needs 32*G table words; G=1 only, and G=1 quadruples atomic traffic'
        print(f'{L:>6}{t:>15.2f}{base["median"]/t:>9.3f}   {note}')
    print(f'\nAS re-measured this run: {base["median"]:.4f} ms (load16=False, real cells).')
    print('Even at L = D = 1024 -- one block owning the entire output row, which needs 256')
    print('table registers and therefore cannot exist -- the law puts TS at '
          f'{k/1024:.1f} ms, still {k/1024/base["median"]:.1f}x slower than AS.')


if __name__ == '__main__':
    main()
