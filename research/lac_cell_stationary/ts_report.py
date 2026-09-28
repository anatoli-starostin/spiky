"""Registers / spills / shared memory per TS instantiation, plus the traffic accounting."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ts  # noqa: E402

T, K, D = 256, 256, 1024
N = 24576
# measured on the cells tensors by bench_act3 / gate_ts
FETCH = {'real': 1.3272, 'uniform': 1.4379}


def main(log):
    rows = ts.ptxas(log)
    print(f'{"config":<26}{"regs":>6}{"spill st":>10}{"spill ld":>10}{"stack":>7}'
          f'{"tbl regs":>10}{"dyn smem":>10}  valid')
    for G, L, Bt in ts.CONFIGS:
        n = f'ts<G={G},L={L},Bt={Bt}>'
        r = rows.get(n)
        b = ts.budget(G, L, Bt)
        if r is None:
            print(f'{n:<26}{"(not in log)":>43}')
            continue
        spills = r['spill_st'] or r['spill_ld']
        valid = b['fits_smem'] and not spills
        print(f'{n:<26}{r["regs"]:>6}{r["spill_st"]:>10}{r["spill_ld"]:>10}{r["stack"]:>7}'
              f'{b["table_regs"]:>10}{b["smem_bytes"]:>10,}  '
              f'{"yes" if valid else ("SMEM" if not b["fits_smem"] else "SPILLS")}')

    print(f'\ntraffic per launch at N={N}, T={T}, K={K}, D={D} '
          f'(blocks = T/G x D/L, each streaming the whole batch)')
    print(f'{"config":<22}{"blocks":>8}{"table MiB":>11}{"cells MB":>10}'
          f'{"atomic GB":>11}{"total GB":>10}{"vs AS 8.56 GB":>15}')
    for G, L, Bt in ts.CONFIGS:
        if not ts.budget(G, L, Bt)['fits_smem']:
            continue
        blocks = (T // G) * (D // L)
        table = blocks * 256 * G * L                    # = T*K*D, compulsory, once
        cells = blocks * ((N + Bt - 1) // Bt) * Bt * G * 3
        atomics = blocks * N * L                        # one int32 RMW per (token, lane) per block
        atom_b = atomics * 8                            # read + write
        tot = table + cells + atom_b
        print(f'{f"G={G} L={L} Bt={Bt}":<22}{blocks:>8}{table/2**20:>11.1f}'
              f'{cells/1e6:>10.1f}{atom_b/1e9:>11.2f}{tot/1e9:>10.2f}'
              f'{tot/8.56e9:>14.2f}x')
    print('\nAS reference at the same shape: 8.56 GB of table reads (load16=False, real cells),')
    print('plus 18.9 MB of cells and 100 MB of output stores.')
    print('TS reads the table set exactly once (64.0 MiB) by construction; what it pays instead')
    print('is the cells tile re-read by every lane-slice block, and the global atomic combine')
    print('across the T/G table-group blocks.')


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else '/tmp/ts_gate.log')
