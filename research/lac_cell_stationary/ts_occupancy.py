"""Where the registers go, and what they cost in occupancy.

If the non-table register count were driven by the g-unrolled body it would grow with G;
if by the q-unrolled lane loop it would grow with L. Tabulating both against the measured
register count settles it, and the blocks-per-SM column is the consequence that matters.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ts  # noqa: E402

REGS_PER_SM, SMEM_PER_SM, THREADS = 65536, 101376, 256


def main(log):
    rows = ts.ptxas(log)
    print(f'{"config":<24}{"2G passes":>10}{"q iters":>8}{"tbl reg":>8}{"regs":>6}'
          f'{"non-table":>11}{"blk/SM":>8}{"warps/SM":>10}{"spill":>7}')
    for G, L, Bt in ts.CONFIGS:
        n = f'ts<G={G},L={L},Bt={Bt}>'
        r = rows.get(n)
        if not r or not r['regs']:
            continue
        b = ts.budget(G, L, Bt)
        by_reg = REGS_PER_SM // (r['regs'] * THREADS) if r['regs'] * THREADS <= REGS_PER_SM else 0
        by_smem = SMEM_PER_SM // b['smem_bytes'] if b['fits_smem'] else 0
        blocks = min(by_reg, by_smem) if by_smem else by_reg
        print(f'{n:<24}{2*G:>10}{L//4:>8}{b["table_regs"]:>8}{r["regs"]:>6}'
              f'{r["regs"]-b["table_regs"]:>11}{blocks:>8}{blocks*THREADS//32:>10}'
              f'{r["spill_st"]:>7}')
    print(f'\nA block is {THREADS} threads = 8 warps by construction (one thread per row).')
    print('The SM can hold 48+ warps; these kernels hold 8, because the register-resident')
    print('table pushes regs/thread high enough that a second block cannot fit.')
    print('That is the design\'s own cost: table-stationary IN REGISTERS => high regs/thread')
    print('=> one block per SM => 8 warps => nothing to hide the dependent shared-memory')
    print('scan latency behind. It is why the scan runs at ~1.5% of instruction issue rate.')


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else '/tmp/ts_build.log')
