"""Read the 500-step L=4 readout-debug runs and show where the readout's magnitude goes.

Columns are the quantities the local update is built from: the readout's output scale against the target,
the confidence score that multiplies every table read, the margins that produce that score, and the
weight-gradient RMS split into readout vs interior, each beside backprop's on the same weights.

Usage: python3 analyze_readout.py [--plot]
"""
import argparse
import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = [('dbg-bp', 'BP'), ('dbg-pcA', 'PC-A'), ('dbg-pcalmB', 'PC-ALM-B'), ('dbg-hybrid', 'hybrid')]


def load(n):
    p = os.path.join(HERE, 'runs_debug', n, 'run.json')
    return json.load(open(p)) if os.path.exists(p) else None


def at(h, k):
    return {r['step']: r[k] for r in h
            if k in r and isinstance(r[k], (int, float)) and not math.isnan(r[k])}


def col(d, k, s):
    return d.get(k, {}).get(s, float('nan'))


def fmt(v, w=10, p=4):
    if isinstance(v, float) and v != v:
        return f'{"-":>{w}}'
    return (f'{v:>{w}.{p-1}e}' if isinstance(v, float) and v != 0 and abs(v) < 1e-3
            else f'{v:>{w}.{p}f}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--plot', action='store_true')
    a = ap.parse_args()
    runs = {n: load(n) for n, _ in RUNS}

    print('== 500 steps, L=4, N=64, 32 tables, T=8, seed 0, table dropout p=0.25, pinned clamping')
    print('   hybrid = readout trained by BP, interior by PC-A.\n')
    print(f'   {"arm":10s} {"test acc":>9s} {"data loss":>10s} {"readout RMS":>12s} {"target RMS":>11s} '
          f'{"out/target":>11s} {"m_min":>8s}')
    for n, label in RUNS:
        d = runs[n]
        if not d:
            print(f'   {label:10s}   (missing)')
            continue
        h = d['hist']
        s = max(at(h, 'eval/test_acc'))
        print(f'   {label:10s} {col({"a": at(h, "eval/test_acc")}, "a", s):>9.4f} '
              f'{col({"a": at(h, "train/loss_at_h") or at(h, "train/loss")}, "a", s):>10.4f} '
              f'{fmt(col({"a": at(h, "out/readout_rms")}, "a", s), 12)} '
              f'{col({"a": at(h, "out/target_rms")}, "a", s):>11.4f} '
              f'{col({"a": at(h, "out/readout_over_target")}, "a", s):>11.4f} '
              f'{col({"a": at(h, "margin/m_min_p50_mean")}, "a", s):>8.4f}')

    print('\n== gradient RMS by bucket, at matched steps (arm on the same weights as BP)')
    for key in ('readout_tables', 'interior_tables'):
        print(f'\n   {key}')
        print(f'   {"arm":10s} ' + ' '.join(f'{s:>11d}' for s in (1, 100, 250, 500)))
        for n, label in RUNS:
            d = runs[n]
            if not d:
                continue
            h = d['hist']
            arm_rms = at(h, f'grad/{key}_rms')
            print(f'   {label:10s} ' + ' '.join(fmt(arm_rms.get(s, float('nan')), 11) for s in (1, 100, 250, 500)))
        d = runs['dbg-pcA']
        if d:
            bp = at(d['hist'], f'grad/{key}_rms_bp')
            ratio = at(d['hist'], f'grad/{key}_ratio')
            print(f'   {"BP (probe)":10s} ' + ' '.join(fmt(bp.get(s, float('nan')), 11) for s in (1, 100, 250, 500)))
            print(f'   {"A/BP":10s} ' + ' '.join(fmt(ratio.get(s, float('nan')), 11) for s in (1, 100, 250, 500)))

    print('\n== the readout output scale over training (RMS output / RMS target)')
    steps = sorted(at(runs['dbg-bp']['hist'], 'out/readout_over_target')) if runs['dbg-bp'] else []
    show = [s for s in steps if s <= 100 or s % 100 == 0]
    print(f'   {"step":>5s} ' + ' '.join(f'{l:>11s}' for _, l in RUNS))
    for s in show:
        print(f'   {s:>5d} ' + ' '.join(
            fmt(col({'a': at(runs[n]['hist'], 'out/readout_over_target')}, 'a', s), 11) if runs[n] else f'{"-":>11}'
            for n, _ in RUNS))

    if a.plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
        col_ = {'dbg-bp': '#444444', 'dbg-pcA': '#1f77b4', 'dbg-pcalmB': '#d62728', 'dbg-hybrid': '#2ca02c'}
        for n, label in RUNS:
            d = runs[n]
            if not d:
                continue
            h = d['hist']
            for j, k in enumerate(('eval/test_acc', 'out/readout_over_target', 'margin/m_min_p50_mean')):
                s = at(h, k)
                ax[j].plot(sorted(s), [s[q] for q in sorted(s)], color=col_[n], alpha=.85, label=label)
        for j, t in enumerate(('test accuracy', 'readout output / target scale', 'median smallest margin')):
            ax[j].set(xlabel='step', title=t)
            ax[j].legend(fontsize=8)
            ax[j].grid(alpha=.25)
        ax[1].set_yscale('log')
        fig.tight_layout()
        out = os.path.join(HERE, 'runs_debug', 'readout_scale.png')
        fig.savefig(out, dpi=140)
        print(f'\nwrote {out}')


if __name__ == '__main__':
    main()
