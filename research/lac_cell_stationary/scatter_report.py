"""Instantiation table, SASS RED/ATOM check, refusal demo and traffic accounting. No timings."""
import os
import re
import subprocess
import sys

import torch

SPIKY = os.path.expanduser('~/projects/spiky')
sys.path.insert(0, os.path.join(SPIKY, 'src'))
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import scatter  # noqa: E402

N, T, K, D = 24576, 256, 256, 1024
FETCH_REAL, FETCH_UNIF = 1.3272, 1.4379
REGS_PER_SM, SMEM_PER_SM = 65536, 101376


def ptxas(log):
    rows, cur = {}, None
    for line in open(log, errors='replace'):
        m = re.search(r"Compiling entry function '([^']+)'", line)
        if m:
            nm = m.group(1)
            nums = [int(x) for x in re.findall(r'Li(\d+)E', nm)]
            if 'scatter_global' in nm:
                key = f'armA L={nums[0]}'
            elif 'scatter_shared' in nm:
                key = f'armB Bt={nums[0]} L={nums[1]} pad={nums[2]}'
            else:
                key = nm
            cur = {'regs': 0, 'spill_st': 0, 'spill_ld': 0, 'smem': 0, 'stack': 0}
            rows[key] = cur
            continue
        if cur is None:
            continue
        m = re.search(r'(\d+) bytes stack frame, (\d+) bytes spill stores, '
                      r'(\d+) bytes spill loads', line)
        if m:
            cur['stack'], cur['spill_st'], cur['spill_ld'] = (int(x) for x in m.groups())
        m = re.search(r'Used (\d+) registers', line)
        if m:
            cur['regs'] = int(m.group(1))
        m = re.search(r'(\d+) bytes smem', line)
        if m:
            cur['smem'] = int(m.group(1))
    return rows


def main():
    log = sys.argv[1] if len(sys.argv) > 1 else '/tmp/gs.log'
    rows = ptxas(log)
    print('INSTANTIATION TABLE (ptxas -v; threads/block = 256 for both arms)')
    print(f'{"kernel":<28}{"regs":>6}{"spill st":>10}{"spill ld":>10}{"stack":>7}'
          f'{"dyn smem":>10}{"blk/SM":>8}')
    for k in sorted(rows, key=lambda s: (s.split()[0], s)):
        r = rows[k]
        if not r['regs']:
            continue
        by_reg = REGS_PER_SM // (r['regs'] * 256)
        if k.startswith('armB'):
            bt = int(re.search(r'Bt=(\d+)', k).group(1))
            pad = int(re.search(r'pad=(\d+)', k).group(1))
            sm = scatter.budget_b(bt, pad)['smem']
            blk = min(by_reg, SMEM_PER_SM // sm)
        else:
            sm, blk = 0, min(by_reg, 24)
        print(f'{k:<28}{r["regs"]:>6}{r["spill_st"]:>10}{r["spill_ld"]:>10}{r["stack"]:>7}'
              f'{sm:>10,}{blk:>8}')

    print('\nARM B SHARED-MEMORY BUDGET (acc = Bt * (D + pad) * 4)')
    print(f'{"Bt":>4}{"pad":>5}{"smem B":>10}{"fits":>7}{"blk/SM by smem":>16}'
          f'{"2 blocks?":>11}')
    for bt in scatter.BTS:
        b = scatter.budget_b(bt)
        print(f'{bt:>4}{0:>5}{b["smem"]:>10,}{"yes" if b["fits"] else "NO":>7}'
              f'{b["blocks_per_sm_by_smem"]:>16}'
              f'{"yes" if b["smem"] <= scatter.SMEM_2BLOCK else "no":>11}')

    print('\nPREDICTED SHARED-MEMORY BANK-CONFLICT FACTOR (arm B)')
    print('  acc is int32, so 32 consecutive ints span the 32 banks exactly. A thread writes L')
    print('  consecutive lanes; 32 adjacent threads at stride L cover lanes [0, 32L), so every')
    print('  bank is addressed L times per warp-wide access.')
    for L in scatter.LS:
        print(f'   L={L:<3} predicted conflict factor {L}-way'
              f'{"  (conflict-free)" if L == 1 else ""}')
    print('  Padding the ROW stride (D + pad) shifts row-to-row alignment only; it cannot change')
    print('  the within-row stride, so it does NOT reduce this factor. Padded variants are built')
    print('  (pad = 1 and 4 at Bt=16, L=4) so the prediction can be tested rather than asserted.')

    print('\nTRAFFIC / WORK ACCOUNTING at N=24,576, real cells (1.3272 cells fetched per table)')
    print(f'{"arm":<12}{"L":>4}{"lane updates":>14}{"tbl amp":>9}{"table GB":>10}'
          f'{"cells MB":>10}{"atomic GB":>11}{"out MB":>9}{"zero MB":>9}'
          f'{"total GB":>10}{"floor ms":>10}')
    for arm in ('A', 'B'):
        for L in scatter.LS:
            t = scatter.traffic(N, T, K, FETCH_REAL, L, arm)
            print(f'{"scatter " + arm:<12}{L:>4}{t["lane_updates"]:>14.3e}'
                  f'{t["table_amp"]:>9.0f}{t["table"]/1e9:>10.2f}{t["cells"]/1e6:>10.1f}'
                  f'{t["atomic_bytes"]/1e9:>11.2f}{t["out_bytes"]/1e6:>9.1f}'
                  f'{t["zero_bytes"]/1e6:>9.1f}{t["total_bytes"]/1e9:>10.2f}'
                  f'{t["floor_ms_at_peak"]:>10.2f}')
    print('\n  lane updates = N * T * 1.3272 * D = 8.55e9, matching the stated expectation.')
    print('  arm B additionally issues that many SHARED-atomic ops (no global atomics at all).')
    print('  floor ms = total bytes / 1792 GB/s, the HBM peak; it is a lower bound that ignores')
    print('  atomic throughput, occupancy and the L2 (which will serve much of the table).')
    print('  Derived from launch geometry + measured discard rates; ncu is unavailable here')
    print('  (RmProfilingAdminOnly=1), so these are NOT measured DRAM counters.')


if __name__ == '__main__':
    main()
