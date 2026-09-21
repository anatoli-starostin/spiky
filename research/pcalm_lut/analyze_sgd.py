"""Read the SGD-vs-Adam lr probe (runs_sgd/) and answer the five questions of task 27f350f3.

Usage: python3 analyze_sgd.py [--plot]
"""
import argparse
import json
import math
import os
import statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
LRS = ['3e-4', '1e-3', '3e-3', '1e-2', '3e-2']
ARMS = [('bp', 'BP'), ('pcA', 'PC-A'), ('pcalmB', 'PC-ALM-B')]


def load(n):
    p = os.path.join(HERE, 'runs_sgd', n, 'run.json')
    return json.load(open(p)) if os.path.exists(p) else None


def ser(h, k):
    return [(r['step'], r[k]) for r in h
            if k in r and isinstance(r[k], (int, float)) and not math.isnan(r[k])]


def last(h, k, d=float('nan')):
    s = ser(h, k)
    return s[-1][1] if s else d


def taus(h, which='f'):
    out = []
    for r in h:
        v = [r[k] for k in r if k.startswith(f'tau/{which}_L')]
        if v:
            out.append((r['step'], st.mean(v)))
    return out


def shape(h):
    """Peak-then-decay or monotone: where the peak is and how much is given back after it."""
    s = ser(h, 'eval/test_acc')
    if not s:
        return float('nan'), float('nan'), float('nan')
    pk = max(s, key=lambda t: t[1])
    return pk[0], pk[1], s[-1][1] - pk[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--plot', action='store_true')
    a = ap.parse_args()

    print('== 500 steps, L=4, N=64, 32 tables, T=8, seed 0, table dropout p=0.25, pinned, Fashion-MNIST')
    print('   SGD = plain, momentum 0, no weight decay. "decay" = final acc minus peak acc (0 = monotone).')
    print(f'\n   {"arm":10s} {"opt":5s} {"lr":>6s} {"train acc":>10s} {"test acc":>9s} {"data loss":>10s} '
          f'{"peak acc":>9s} {"peak@":>6s} {"decay":>8s} {"m_min":>7s} {"tau_f":>7s} {"ms/step":>8s}')
    rows = {}
    for arm, label in ARMS:
        for opt, lrs in (('sgd', LRS), ('adam', ['1e-3'])):
            for lr in lrs:
                n = f'{opt}-{arm}-lr{lr}'
                d = load(n)
                if not d:
                    print(f'   {label:10s} {opt:5s} {lr:>6s}   (missing)')
                    continue
                h = d['hist']
                rows[(arm, opt, lr)] = d
                pkat, pk, dec = shape(h)
                dl = last(h, 'train/loss_at_h', last(h, 'train/loss'))
                print(f'   {label:10s} {opt:5s} {lr:>6s} {last(h, "eval/train_acc"):>10.4f} '
                      f'{last(h, "eval/test_acc"):>9.4f} {dl:>10.4f} {pk:>9.4f} {pkat:>6d} '
                      f'{dec:>+8.4f} {last(h, "margin/m_min_p50_mean"):>7.4f} '
                      f'{(taus(h)[-1][1] if taus(h) else float("nan")):>7.4f} '
                      f'{1e3 * st.median([v for _, v in ser(h, "train/s_per_step")]):>8.1f}')

    print('\n== (b) does any PC arm leave data loss ~0.49?  (trivial value for one-hot targets is 0.5)')
    best = None
    for (arm, opt, lr), d in rows.items():
        if arm == 'bp':
            continue
        dl = last(d['hist'], 'train/loss_at_h', last(d['hist'], 'train/loss'))
        if best is None or dl < best[0]:
            best = (dl, arm, opt, lr)
    print(f'   best PC data loss anywhere in the grid: {best[0]:.4f}  ({best[1]}, {best[2]}, lr {best[3]})'
          if best else '   no PC runs')

    print('\n== (c) tau trajectory, SGD vs Adam (start -> end, mean over forward LUTs)')
    print(f'   {"arm":10s} {"opt":5s} {"lr":>6s} {"tau start":>10s} {"tau end":>9s} {"ratio":>7s} '
          f'{"tau_g end":>10s}')
    for (arm, opt, lr), d in sorted(rows.items()):
        t = taus(d['hist'])
        tg = taus(d['hist'], 'g')
        if not t:
            continue
        print(f'   {dict(ARMS)[arm]:10s} {opt:5s} {lr:>6s} {t[0][1]:>10.4f} {t[-1][1]:>9.4f} '
              f'{t[-1][1] / t[0][1]:>7.3f} {(tg[-1][1] if tg else float("nan")):>10.4f}')

    print('\n== (d) margin trajectory (median smallest margin, start -> end)')
    print(f'   {"arm":10s} {"opt":5s} {"lr":>6s} {"m start":>9s} {"m end":>8s} {"ratio":>7s}')
    for (arm, opt, lr), d in sorted(rows.items()):
        m = ser(d['hist'], 'margin/m_min_p50_mean')
        if not m:
            continue
        print(f'   {dict(ARMS)[arm]:10s} {opt:5s} {lr:>6s} {m[0][1]:>9.4f} {m[-1][1]:>8.4f} '
              f'{m[-1][1] / m[0][1]:>7.3f}')

    print('\n== applied-update RMS vs gradient RMS (the point of the optimiser comparison)')
    print(f'   {"arm":10s} {"opt":5s} {"lr":>6s} {"upd readout":>12s} {"upd interior":>13s} '
          f'{"upd backward":>13s} {"grad readout":>13s} {"grad interior":>14s}')
    for (arm, opt, lr), d in sorted(rows.items()):
        h = d['hist']
        print(f'   {dict(ARMS)[arm]:10s} {opt:5s} {lr:>6s} '
              f'{last(h, "update/readout_tables_rms"):>12.3e} '
              f'{last(h, "update/interior_tables_rms"):>13.3e} '
              f'{last(h, "update/backward_tables_rms"):>13.3e} '
              f'{last(h, "grad/readout_tables_rms"):>13.3e} '
              f'{last(h, "grad/interior_tables_rms"):>14.3e}')

    if a.plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
        cm = plt.get_cmap('viridis')
        for j, (arm, label) in enumerate(ARMS):
            for i, lr in enumerate(LRS):
                d = rows.get((arm, 'sgd', lr))
                if not d:
                    continue
                s = ser(d['hist'], 'eval/test_acc')
                ax[j].plot([q for q, _ in s], [v for _, v in s], color=cm(i / max(len(LRS) - 1, 1)),
                           label=f'SGD {lr}')
            d = rows.get((arm, 'adam', '1e-3'))
            if d:
                s = ser(d['hist'], 'eval/test_acc')
                ax[j].plot([q for q, _ in s], [v for _, v in s], color='#d62728', ls='--',
                           label='Adam 1e-3')
            ax[j].set(xlabel='step', ylabel='test acc', title=label, ylim=(0.1, 0.9))
            ax[j].legend(fontsize=7)
            ax[j].grid(alpha=.25)
        fig.tight_layout()
        out = os.path.join(HERE, 'runs_sgd', 'sgd_lr_probe.png')
        fig.savefig(out, dpi=140)
        print(f'\nwrote {out}')


if __name__ == '__main__':
    main()
