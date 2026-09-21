"""Head-to-head early training: is PC actually AHEAD of backprop before it destroys itself?

The L=4 pinned smoke has a peak-then-decay shape for every PC arm, so the interesting comparison is not
the final number but BP's accuracy at the SAME step as each PC arm's peak. This reads the per-checkpoint
curves out of runs_pinned/ (probe cadence 25 steps) and reports them beside the margin, tau and table
norm, so the accuracy peak can be placed against the onset of margin collapse.

Usage: python3 analyze_early.py [--until 300] [--thin 100] [--plot]
"""
import argparse
import json
import math
import os
import statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
ARMS = [('pin-bp-L4', 'BP'), ('pin-pcA-L4', 'PC-A'), ('pin-pcalmB-L4', 'PC-ALM-B'),
        ('nopin-pcA-L4', 'PC-A no-pin'), ('nopin-pcalmB-L4', 'PC-ALM-B no-pin')]


def load(n):
    p = os.path.join(HERE, 'runs_pinned', n, 'run.json')
    return json.load(open(p)) if os.path.exists(p) else None


def at(h, key):
    """step -> value for one metric."""
    return {r['step']: r[key] for r in h
            if key in r and isinstance(r[key], (int, float)) and not math.isnan(r[key])}


def tau_at(h):
    out = {}
    for r in h:
        v = [r[k] for k in r if k.startswith('tau/f_L')]
        if v:
            out[r['step']] = st.mean(v)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--until', type=int, default=300, help='print every checkpoint up to this step')
    ap.add_argument('--thin', type=int, default=100, help='after --until, print every N steps')
    ap.add_argument('--plot', action='store_true')
    a = ap.parse_args()

    runs = {n: load(n) for n, _ in ARMS}
    missing = [n for n, d in runs.items() if d is None]
    if missing:
        print(f'missing runs: {missing}')
    acc = {n: at(d['hist'], 'eval/test_acc') for n, d in runs.items() if d}
    loss = {n: at(d['hist'], 'train/loss_at_h') or at(d['hist'], 'train/loss')
            for n, d in runs.items() if d}
    marg = {n: at(d['hist'], 'margin/m_min_p50_mean') for n, d in runs.items() if d}
    tau = {n: tau_at(d['hist']) for n, d in runs.items() if d}
    tnorm = {n: at(d['hist'], 'norm/f_tables_mean') for n, d in runs.items() if d}

    steps = sorted(acc['pin-bp-L4'])
    show = [s for s in steps if s <= a.until or s % a.thin == 0]

    print('== L=4, N=64, 32 tables, T=8, seed 0, table dropout p=0.25 (mask pinned), pinned clamping')
    print('   test accuracy at every logged checkpoint; data loss is 1/2||yhat-y||^2 per sample')
    print(f'   (every checkpoint to step {a.until}, then every {a.thin} steps)')
    print(f'\n   {"step":>5s} {"BP acc":>7s} {"PC-A":>7s} {"PC-ALM-B":>9s} | {"BP loss":>8s} '
          f'{"PC-A loss":>10s} {"PC-B loss":>10s} | {"A margin":>9s} {"A tau":>7s} {"A tables":>9s} | '
          f'{"B margin":>9s} {"B tau":>7s} {"B tables":>9s}')
    for s in show:
        def g(d, n):
            return d.get(n, {}).get(s, float('nan'))
        print(f'   {s:>5d} {g(acc, "pin-bp-L4"):>7.4f} {g(acc, "pin-pcA-L4"):>7.4f} '
              f'{g(acc, "pin-pcalmB-L4"):>9.4f} | {g(loss, "pin-bp-L4"):>8.4f} '
              f'{g(loss, "pin-pcA-L4"):>10.4f} {g(loss, "pin-pcalmB-L4"):>10.4f} | '
              f'{g(marg, "pin-pcA-L4"):>9.4f} {g(tau, "pin-pcA-L4"):>7.4f} '
              f'{g(tnorm, "pin-pcA-L4"):>9.4f} | {g(marg, "pin-pcalmB-L4"):>9.4f} '
              f'{g(tau, "pin-pcalmB-L4"):>7.4f} {g(tnorm, "pin-pcalmB-L4"):>9.4f}')

    print('\n== THE HEAD-TO-HEAD: each arm at its own peak, against BP at that SAME step')
    print(f'   {"arm":18s} {"peak step":>10s} {"peak acc":>9s} {"BP @ that step":>15s} {"PC - BP":>9s} '
          f'{"BP best ever":>13s} {"margin @ peak":>14s} {"margin start":>13s}')
    bp = acc['pin-bp-L4']
    for n, label in ARMS:
        if n == 'pin-bp-L4' or n not in acc:
            continue
        pstep = max(acc[n], key=lambda s: acc[n][s])
        pacc = acc[n][pstep]
        bp_here = bp.get(pstep, float('nan'))
        m0 = marg[n][min(marg[n])]
        print(f'   {label:18s} {pstep:>10d} {pacc:>9.4f} {bp_here:>15.4f} {pacc - bp_here:>+9.4f} '
              f'{max(bp.values()):>13.4f} {marg[n].get(pstep, float("nan")):>14.4f} {m0:>13.4f}')

    print('\n== does the accuracy peak PRECEDE the margin collapse, or coincide with it?')
    print(f'   {"arm":18s} {"peak step":>10s} {"margin peak step":>17s} {"margin at":>10s} '
          f'{"half-margin step":>17s} {"acc at half-margin":>19s}')
    for n, label in ARMS:
        if n == 'pin-bp-L4' or n not in acc:
            continue
        pstep = max(acc[n], key=lambda s: acc[n][s])
        mstep = max(marg[n], key=lambda s: marg[n][s])
        m0 = marg[n][mstep]
        half = next((s for s in sorted(marg[n]) if s > mstep and marg[n][s] < 0.5 * m0), None)
        print(f'   {label:18s} {pstep:>10d} {mstep:>17d} {m0:>10.4f} '
              f'{(half if half is not None else -1):>17d} '
              f'{acc[n].get(half, float("nan")) if half else float("nan"):>19.4f}')

    if a.plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
        col = {'pin-bp-L4': '#444444', 'pin-pcA-L4': '#1f77b4', 'pin-pcalmB-L4': '#d62728',
               'nopin-pcA-L4': '#1f77b4', 'nopin-pcalmB-L4': '#d62728'}
        for n, label in ARMS:
            if n not in acc:
                continue
            ls = '--' if n.startswith('nopin') else '-'
            s = sorted(acc[n])
            for j, lim in enumerate((300, max(s))):
                sel = [q for q in s if q <= lim]
                ax[j].plot(sel, [acc[n][q] for q in sel], color=col[n], ls=ls, alpha=.85, label=label)
        ax[0].set(xlabel='step', ylabel='test accuracy', title='first 300 steps (head-to-head)')
        ax[1].set(xlabel='step', ylabel='test accuracy', title='all 2000 steps')
        for a_ in ax:
            a_.legend(fontsize=8)
            a_.grid(alpha=.25)
        fig.tight_layout()
        out = os.path.join(HERE, 'runs_pinned', 'early_headtohead.png')
        fig.savefig(out, dpi=140)
        print(f'\nwrote {out}')


if __name__ == '__main__':
    main()
