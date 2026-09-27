"""Per-family best configuration and time from a (possibly still running) bench log."""
import re
import sys


def main(path):
    best, cur = {}, None
    for line in open(path):
        m = re.match(r'-- sweep at B=(\d+)', line)
        if m:
            cur = int(m.group(1))
            continue
        m = re.match(r'\s+(gather|cs_v\d) ([^ ].*?)\s+([\d.]+) ms\s+rel=([\de.+-]+)(.*)$', line)
        if m and cur:
            fam, lab = m.group(1), (m.group(1) + ' ' + m.group(2)).strip()
            ms, rel, cap = float(m.group(3)), float(m.group(4)), '1 launch' in m.group(5)
            k = (cur, fam)
            if k not in best or ms < best[k][1]:
                best[k] = (lab, ms, rel, cap)
    for B, fam in sorted(best):
        lab, ms, rel, cap = best[(B, fam)]
        tag = '(1 launch)' if cap else ''
        print(f'B={B:>6} {fam:<6} {ms:>9.4f} ms  rel={rel:.1e} {tag:<11} {lab}')
    txt = open(path).read()
    rels = [float(x) for x in re.findall(r'rel=([\de.+-]+)', txt)]
    print(f'\n{len(rels)} correctness checks so far; max rel err {max(rels):.2e}')
    for B in sorted({b for b, _ in best}):
        g = best.get((B, 'gather'))
        cs = [best[(B, f)] for f in ('cs_v0', 'cs_v1', 'cs_v2', 'cs_v3') if (B, f) in best]
        if g and cs:
            bcs = min(cs, key=lambda x: x[1])
            print(f'B={B:>6}: best cell-stationary {bcs[1]:.4f} ms vs gather {g[1]:.4f} ms '
                  f'-> {bcs[1]/g[1]:.1f}x slower   ({bcs[0]})')


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else '/tmp/bench.log')
