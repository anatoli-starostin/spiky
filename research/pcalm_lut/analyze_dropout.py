"""Read runs_dropout/*/run.json and answer task 5fd246bb's question: does depth start paying once the
deep stack is regularised, and does table dropout slow the margin growth that suppresses the routing block.

Usage: python3 analyze_dropout.py [--plot]
"""
import argparse
import json
import math
import os
import statistics as st

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'runs_dropout')
CELLS = [('none', 0.0)] + [(v, p) for v in ('table', 'resid') for p in (0.1, 0.25)]
LABEL = {'none': 'no dropout', 'table': 'table dropout', 'resid': 'residual dropout'}


SEEDS = (0, 1, 2)


def name(variant, p, L, seed=0):
    sfx = '' if seed == 0 else f'-s{seed}'
    return (f'drop-none-L{L}{sfx}' if variant == 'none' else f'drop-{variant}-p{p}-L{L}{sfx}')


def load(n):
    p = os.path.join(R, n, 'run.json')
    return json.load(open(p)) if os.path.exists(p) else None


def loads(variant, p, L):
    """Every seed of one cell."""
    return [d for d in (load(name(variant, p, L, s)) for s in SEEDS) if d]


def agg(ds, fn):
    v = [fn(d) for d in ds]
    v = [x for x in v if not (isinstance(x, float) and math.isnan(x))]
    if not v:
        return float('nan'), float('nan'), 0
    return st.mean(v), (st.stdev(v) if len(v) > 1 else 0.0), len(v)


def series(h, k):
    return [(r['step'], r[k]) for r in h if k in r and not math.isnan(r[k])]


def tail(h, k, n=3):
    """Mean of the last n probes -- a single final probe on 2000 test rows is +-1% noise."""
    s = [v for _, v in series(h, k)]
    return st.mean(s[-n:]) if s else float('nan')


def per_layer_mean(row, prefix, nl):
    v = [row[f'{prefix}{i}'] for i in range(nl) if f'{prefix}{i}' in row]
    return st.mean(v) if v else float('nan')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--plot', action='store_true')
    a = ap.parse_args()
    runs = {(v, p, L): loads(v, p, L) for v, p in CELLS for L in (4, 16)}

    acc = lambda d: tail(d['hist'], 'eval/test_acc')                                        # noqa: E731
    tau_last = lambda d: per_layer_mean([r for r in d['hist'] if 'tau/f_L0' in r][-1],       # noqa: E731
                                        'tau/f_L', d['cfg']['depth'])

    print('== BP dropout probe: N=64, 32 tables, 2000 steps, Fashion-MNIST')
    print('   Each cell is 3 seeds; acc/gap are the mean of the last 3 probes of each run (a single')
    print('   2,000-row accuracy estimate has sd ~0.0075, so single-seed differences mean nothing).')
    print(f'\n{"variant":18s} {"p":>5s} {"L":>3s} {"n":>2s} {"train loss":>11s} {"train acc":>10s} '
          f'{"test acc":>9s} {"+-sd":>6s} {"gap":>7s} {"m_min last":>11s} {"tau last":>9s} {"s/step":>8s}')
    for v, p in CELLS:
        for L in (4, 16):
            ds = runs[(v, p, L)]
            if not ds:
                print(f'{LABEL[v]:18s} {p:>5} {L:>3d}   (missing)')
                continue
            a, sd, n = agg(ds, acc)
            print(f'{LABEL[v]:18s} {p:>5} {L:>3d} {n:>2d} '
                  f'{agg(ds, lambda d: tail(d["hist"], "eval/train_loss_full"))[0]:>11.4f} '
                  f'{agg(ds, lambda d: tail(d["hist"], "eval/train_acc"))[0]:>10.4f} '
                  f'{a:>9.4f} {sd:>6.4f} '
                  f'{agg(ds, lambda d: tail(d["hist"], "eval/gap"))[0]:>7.4f} '
                  f'{agg(ds, lambda d: tail(d["hist"], "margin/m_min_p50_mean"))[0]:>11.4f} '
                  f'{agg(ds, tau_last)[0]:>9.4f} '
                  f'{agg(ds, lambda d: st.median([x for _, x in series(d["hist"], "train/s_per_step")]))[0]:>8.4f}')

    print('\n== THE QUESTION: does depth pay under dropout?  (L=16 test acc) - (L=4 test acc)')
    print('   sd of the difference is sqrt(sd4^2/n + sd16^2/n): a difference smaller than ~2 sd is noise.')
    print(f'   {"variant":18s} {"p":>5s} {"L=4":>8s} {"L=16":>8s} {"L16 - L4":>10s} {"+-sd":>7s}')
    base = None
    for v, p in CELLS:
        ds4, ds16 = runs[(v, p, 4)], runs[(v, p, 16)]
        if not (ds4 and ds16):
            continue
        a4, s4, n4 = agg(ds4, acc)
        a16, s16, n16 = agg(ds16, acc)
        dsd = math.sqrt(s4 ** 2 / max(n4, 1) + s16 ** 2 / max(n16, 1))
        if v == 'none':
            base = a16 - a4
        print(f'   {LABEL[v]:18s} {p:>5} {a4:>8.4f} {a16:>8.4f} {a16 - a4:>+10.4f} {dsd:>7.4f}'
              + ('   <- baseline' if v == 'none' else
                 f'   ({a16 - a4 - base:+.4f} vs baseline)' if base is not None else ''))

    print('\n== MARGIN GROWTH: does dropping tables slow the sharpening?')
    print('   median smallest margin m_j*, first probe -> last, and the growth factor (3-seed means).')
    print(f'   {"variant":18s} {"p":>5s} {"L":>3s} {"m first":>8s} {"m last":>8s} {"growth":>7s} '
          f'{"tau first":>10s} {"tau last":>9s}')
    for v, p in CELLS:
        for L in (4, 16):
            ds = runs[(v, p, L)]
            if not ds:
                continue
            m0 = agg(ds, lambda d: series(d['hist'], 'margin/m_min_p50_mean')[0][1])[0]
            m1 = agg(ds, lambda d: series(d['hist'], 'margin/m_min_p50_mean')[-1][1])[0]
            t0 = agg(ds, lambda d: per_layer_mean([r for r in d['hist'] if 'tau/f_L0' in r][0],
                                                  'tau/f_L', d['cfg']['depth']))[0]
            print(f'   {LABEL[v]:18s} {p:>5} {L:>3d} {m0:>8.4f} {m1:>8.4f} {m1 / m0:>7.2f} '
                  f'{t0:>10.4f} {agg(ds, tau_last)[0]:>9.4f}')

    if a.plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
        style = {('none', 0.0): ('#444444', '-'), ('table', 0.1): ('#1f77b4', '-'),
                 ('table', 0.25): ('#1f77b4', '--'), ('resid', 0.1): ('#d62728', '-'),
                 ('resid', 0.25): ('#d62728', '--')}
        for j, L in enumerate((4, 16)):
            for v, p in CELLS:
                ds = runs[(v, p, L)]
                if not ds:
                    continue
                d = ds[0]                      # seed 0; the seed spread is in the tables
                c, ls = style[(v, p)]
                s = series(d['hist'], 'eval/test_acc')
                ax[j].plot([q for q, _ in s], [q for _, q in s], color=c, ls=ls, alpha=.85,
                           label=f'{LABEL[v]}' + (f' p={p}' if v != 'none' else ''))
                m = series(d['hist'], 'margin/m_min_p50_mean')
                if L == 16:
                    ax[2].plot([q for q, _ in m], [q for _, q in m], color=c, ls=ls, alpha=.85,
                               label=f'{LABEL[v]}' + (f' p={p}' if v != 'none' else ''))
            ax[j].set(xlabel='step', ylabel='test acc', title=f'L={L}', ylim=(0.6, 0.92))
        ax[2].set(xlabel='step', ylabel='median smallest margin', title='margin growth, L=16')
        for a_ in ax:
            a_.legend(fontsize=7)
            a_.grid(alpha=.25)
        fig.tight_layout()
        out = os.path.join(R, 'dropout_probe.png')
        fig.savefig(out, dpi=140)
        print(f'\nwrote {out}')


if __name__ == '__main__':
    main()
