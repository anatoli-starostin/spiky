"""Turn an nvcc -Xptxas -v build log into the registers/spills/smem table.

Usage: ptxas_table.py <build.log>   (the log the smoke/bench build wrote)
Demangles the template parameters out of the mangled entry names, so a row reads
`cs_v1<K=32,M=1,TB=4,CLU=1>` rather than `_ZN54_GLOBAL__N__...ILi32ELi1E...`.
"""
import re
import sys

NAMES = {'cs_v0_diag_kernel': ('cs_v0_diag', ['K', 'M', 'MODE']),
         'cs_v1_diag_kernel': ('cs_v1_diag', ['K', 'M', 'TB', 'MODE']),
         'cs_v0_kernel': ('cs_v0', ['K', 'M']),
         'cs_v1_kernel': ('cs_v1', ['K', 'M', 'TB', 'CLU']),
         'cs_v2_kernel': ('cs_v2', ['K', 'M', 'TB', 'TRANS']),
         'cs_v3_kernel': ('cs_v3', ['K', 'M', 'G', 'TRANS']),
         'gather_kernel': ('gather', [])}


def pretty(mangled):
    for key, (short, params) in NAMES.items():
        if key in mangled:
            nums = [int(x) for x in re.findall(r'Li(\d+)E', mangled)]
            if not params:
                return short
            return f'{short}<' + ','.join(f'{p}={v}' for p, v in zip(params, nums)) + '>'
    return mangled


def parse(path):
    rows, cur = [], None
    for line in open(path, errors='replace'):
        m = re.search(r"Compiling entry function '([^']+)'", line)
        if m:
            cur = {'name': pretty(m.group(1)), 'regs': None, 'spill_st': 0, 'spill_ld': 0,
                   'smem': 0, 'stack': 0}
            rows.append(cur)
            continue
        if cur is None:
            continue
        m = re.search(r'(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads',
                      line)
        if m:
            cur['stack'], cur['spill_st'], cur['spill_ld'] = (int(g) for g in m.groups())
        m = re.search(r'Used (\d+) registers', line)
        if m:
            cur['regs'] = int(m.group(1))
        m = re.search(r'(\d+) bytes smem', line)
        if m:
            cur['smem'] = int(m.group(1))
    return [r for r in rows if r['regs'] is not None]


def main():
    rows = parse(sys.argv[1])
    seen = {}
    for r in rows:
        seen[r['name']] = r
    print(f'{"kernel":<34}{"regs":>6}{"spill st":>10}{"spill ld":>10}{"smem B":>9}{"stack":>7}')
    order = sorted(seen, key=lambda n: (n.split('<')[0], n))
    for n in order:
        r = seen[n]
        print(f'{n:<34}{r["regs"]:>6}{r["spill_st"]:>10}{r["spill_ld"]:>10}'
              f'{r["smem"]:>9}{r["stack"]:>7}')
    print(f'\n{len(seen)} entry functions; '
          f'{sum(1 for r in seen.values() if r["spill_st"] or r["spill_ld"])} with spills')
    # occupancy implication: 65536 registers per SM on the 5090
    print('\nblock-residency implied by the register file (65,536 registers per SM):')
    for n in order:
        r = seen[n]
        if n.startswith('cs_v1'):
            tb = int(re.search(r'TB=(\d+)', n).group(1))
            thr = 256 * tb
        elif n.startswith('cs_v0'):
            thr = 256
        else:
            continue
        per_sm = 65536 // (r['regs'] * thr) if r['regs'] * thr <= 65536 else 0
        print(f'  {n:<32} {thr:>5} threads x {r["regs"]:>3} regs = {thr*r["regs"]:>6}'
              f'  -> {per_sm} block(s)/SM')


if __name__ == '__main__':
    main()
