"""Count the AS inner-loop instructions per lane update, from SASS, so the scalar-ALU
anchor rests on a measured op count rather than a guessed one."""
import re
import sys
from collections import Counter

PATH = sys.argv[1] if len(sys.argv) > 1 else '/tmp/as_sass.txt'
T, D = 256, 1024


def main():
    txt = open(PATH, errors='replace').read()
    found = 0
    for part in re.split(r'Function : ', txt)[1:]:
        nm = part.split('\n', 1)[0]
        if 'p2_int8_kernel' not in nm:
            continue
        # bool template args mangle as Lb0E / Lb1E, not Li..E
        nums = [int(x) for x in re.findall(r'L[ib](\d+)E', nm)]
        if len(nums) < 5:
            continue
        BN, UPR, PRO, L16, B16 = nums[:5]
        if PRO != 0 or B16 != 0:
            continue                       # read_cells, fp32 output
        ops = re.findall(r'^\s+/\*[0-9a-f]+\*/\s+(?:@!?P\d+\s+)?([A-Z][A-Z0-9_]*)',
                         part, re.M)
        c = Counter(ops)
        tot = sum(c.values())
        # the accumulate arithmetic, per the source: byte extract, mask, shift, add
        arith = sum(c[k] for k in ('SHF', 'LOP3', 'IADD3', 'PRMT', 'IADD', 'IMAD', 'I2I'))
        lanes_per_thread = 16              # acc[16], one 16-lane unit per thread
        cells = 2                          # both slots per table per iteration
        print(f'p2_int8_kernel BN={BN} UPR={UPR} load16={bool(L16)}: '
              f'{tot} instructions total')
        print(f'   {"mnemonic":<10}{"count":>7}')
        for k, v in c.most_common(10):
            print(f'   {k:<10}{v:>7}')
        print(f'   integer-arith subtotal (SHF/LOP3/IADD3/PRMT/IMAD/I2I): {arith}')
        print(f'   the table loop body covers {cells} cells x {lanes_per_thread} lanes = '
              f'{cells*lanes_per_thread} lane updates per iteration')
        print(f'   => roughly {arith/(cells*lanes_per_thread):.2f} integer ops per lane '
              f'update (whole-kernel count / per-iteration lane updates; an upper bound, '
              f'since prologue and flush are included)\n')
        found += 1
        if found >= 2:
            break
    if not found:
        print('no matching read_cells instantiation found in the SASS dump')


if __name__ == '__main__':
    main()
