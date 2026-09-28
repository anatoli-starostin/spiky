"""Shared-memory and register budgets for the table-stationary (G, L, Bt) points.

The limit is cudaDevAttrMaxSharedMemoryPerBlockOptin on sm_120 = 101,376 B, which the
kernel opts into with cudaFuncSetAttribute. Table registers are G*L/4 32-bit words,
against the 255-register cap, before any working set.
"""
LIMIT = 101376
REG_CAP = 255

SPEC = [(8, 96, 256, 'specified (a)'), (32, 24, 1024, 'specified (b)'),
        (16, 48, 512, 'specified midpoint')]
ALT = [(8, 64, 256, 'fitting neighbour of (a)'), (8, 32, 256, 'wide-lane low-shared'),
       (16, 32, 512, 'fitting neighbour of midpoint'),
       (32, 16, 512, 'fitting neighbour of (b)'), (32, 16, 256, 'narrow, small tile'),
       (16, 64, 256, 'wide lanes, more registers')]


def row(G, L, Bt, tag):
    acc, ce = Bt * L * 4, 3 * Bt * G
    tot, regs = acc + ce, G * L // 4
    ok = tot <= LIMIT and regs <= 200          # 200 leaves ~55 for the working set
    print(f'{G:>4}{L:>5}{Bt:>6}{acc:>10,}{ce:>9,}{tot:>10,}{tot - LIMIT:>+11,}'
          f'{regs:>7}{"  yes" if ok else "  NO ":>6}   {tag}')
    return ok


def main():
    print(f'{"G":>4}{"L":>5}{"Bt":>6}{"acc B":>10}{"cells B":>9}{"total B":>10}'
          f'{"vs limit":>11}{"tblreg":>7}  fits   note')
    print('-- the three points named in the task --')
    for g, l, b, t in SPEC:
        row(g, l, b, t)
    print('-- fitting alternatives --')
    for g, l, b, t in ALT:
        row(g, l, b, t)
    print(f'\nlimit {LIMIT:,} B; register cap {REG_CAP}, budgeted at 200 for table data')
    print('common cause of the three failures: all have Bt*L = 24,576, i.e. 98,304 B of')
    print('accumulator alone = 97.0% of the limit, before a single byte of cells tile.')


if __name__ == '__main__':
    main()
