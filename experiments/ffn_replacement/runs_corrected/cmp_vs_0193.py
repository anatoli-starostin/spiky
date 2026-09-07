"""Matched-step delta of a running experiment against exp_g_0193's eval curve.

Both runs use seed 1 and identical schedules, so eval points align exactly. This module
NEVER silently aligns mismatched steps: a step with no exact counterpart in 0193's curve is
reported as "NO EXACT COUNTERPART", and interpolation happens only via the CLI's --interp
flag and is labelled as a fallback.

`delta_table()` is importable so the Slack progress watcher renders the SAME comparison this
CLI does. That matters: the reason the delta was missing from progress messages is that the
watcher had no such call at all -- it reported step/bpb/ETA only.

    python cmp_vs_0193.py <run_dir> [--last N] [--interp]
"""
import argparse
import csv
import os

HERE = os.path.dirname(os.path.abspath(__file__))
REF = os.path.join(HERE, 'exp_g_0193_B16k_light_margin_tph128_noznorm_seed1')
SD = 0.00335          # vanilla@16K seed spread, the noise unit


def curve(d):
    p = os.path.join(d, 'metrics.csv')
    if not os.path.exists(p):
        return {}
    return {int(r['step']): float(r['val_bpb'])
            for r in csv.DictReader(open(p)) if r.get('val_bpb')}


def delta_table(run_dir, last=6):
    """-> (trend_line, [row strings]). Exact step matching only."""
    new, ref = curve(run_dir), curve(REF)
    if not new:
        return 'no evals yet', []
    rows, lines = [], []
    for s in sorted(new)[-last:]:
        if s in ref:
            d = new[s] - ref[s]
            rows.append((s, d))
            lines.append(f'{s:>6} {new[s]:>9.6f} {ref[s]:>9.6f} {d:>+10.6f} {d / SD:>+6.2f}s')
        else:
            lines.append(f'{s:>6} {new[s]:>9.6f}   NO EXACT COUNTERPART in exp_g_0193')
    trend = ''
    if len(rows) >= 2:
        f, l = rows[0][1], rows[-1][1]
        word = 'opening' if l - f > 1e-6 else 'closing' if l - f < -1e-6 else 'flat'
        signs = ''.join('+' if d > 0 else '-' for _, d in rows)
        trend = (f'trend {f:+.6f} -> {l:+.6f} ({l - f:+.6f}, {word}); signs {signs} '
                 f'("-" = 0194 ahead); |d|={abs(l):.6f} = {abs(l) / SD:.2f}x seed spread'
                 + ('  NOT YET MEANINGFUL' if abs(l) < SD else ''))
    return trend, lines


def render(run_dir, step=None, total=16000, last=6):
    """The exact text the Slack progress watcher posts."""
    trend, lines = delta_table(run_dir, last=last)
    head = f'step {step:,}/{total:,}' if step else 'progress'
    body = ('  step   this run      0193      delta  sigma\n  ' + '\n  '.join(lines)
            if lines else '  (no evals yet)')
    return f'{head} · vs exp_g_0193 (matched step, seed spread {SD}):\n{body}\n  {trend}'


def _cli():
    ap = argparse.ArgumentParser()
    ap.add_argument('run')
    ap.add_argument('--last', type=int, default=8)
    ap.add_argument('--interp', action='store_true')
    a = ap.parse_args()
    run_dir = a.run if os.path.isabs(a.run) else os.path.join(HERE, a.run)
    new, ref = curve(run_dir), curve(REF)
    if not new:
        raise SystemExit('no evals yet in ' + os.path.basename(run_dir))
    print(f'{os.path.basename(run_dir)}  vs  exp_g_0193   (seed spread {SD})')
    print(f'{"step":>7} {"this run":>10} {"0193":>10} {"delta":>11} {"sigma":>7}  match')
    rows = []
    for s in sorted(new)[-a.last:]:
        if s in ref:
            d = new[s] - ref[s]
            rows.append((s, d))
            print(f'{s:>7} {new[s]:>10.6f} {ref[s]:>10.6f} {d:>+11.6f} {d / SD:>+7.2f}  exact')
        elif a.interp:
            lo = max([k for k in ref if k < s], default=None)
            hi = min([k for k in ref if k > s], default=None)
            if lo is None or hi is None:
                print(f'{s:>7} {new[s]:>10.6f} {"-":>10} {"-":>11} {"-":>7}  '
                      f'NO COUNTERPART (outside 0193 range)')
                continue
            w = (s - lo) / (hi - lo)
            v = ref[lo] * (1 - w) + ref[hi] * w
            d = new[s] - v
            rows.append((s, d))
            print(f'{s:>7} {new[s]:>10.6f} {v:>10.6f} {d:>+11.6f} {d / SD:>+7.2f}  '
                  f'*** INTERPOLATED between {lo} and {hi} -- fallback')
        else:
            print(f'{s:>7} {new[s]:>10.6f} {"-":>10} {"-":>11} {"-":>7}  '
                  f'NO EXACT COUNTERPART (pass --interp to estimate)')
    trend, _ = delta_table(run_dir, last=a.last)
    print('\n' + trend)


if __name__ == '__main__':
    _cli()
