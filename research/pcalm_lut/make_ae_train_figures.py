"""Train-set (and train-vs-test) curves for every autoencoder run on disk.

Reads runs_autoencoder/*/run.json only -- no training is run. Every curve is labelled with the run
directory name exactly as it exists on disk, so each line traces to one artifact.

Palette: Okabe-Ito, a published colourblind-safe set, assigned per entity and never cycled. The dataviz
skill's own validator (scripts/validate_palette.js) is NOT installed on this machine, so the palette
check it asks for could not be run; a pre-validated published palette is the substitute.

Usage: python3 make_ae_train_figures.py
"""
import json
import math
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, 'runs_autoencoder')
OUT = os.path.join(R, 'plots')
INK = '#222222'
OKABE = ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#E69F00', '#56B4E9', '#F0E442', '#000000']
plt.rcParams.update({'axes.edgecolor': '#BBBBBB', 'axes.labelcolor': INK, 'text.color': INK,
                     'xtick.color': INK, 'ytick.color': INK, 'font.size': 8.5, 'axes.grid': True,
                     'grid.color': '#E4E4E4', 'figure.facecolor': 'white', 'axes.facecolor': 'white'})

# the sweep families, exactly as they were run
FAMILIES = [
    ('a. depth sweep, tph=64, 500 steps',
     ['lut-L2-adam', 'lut-L4-adam', 'lut-L8-adam', 'mlp-L2-adam', 'mlp-L4-adam', 'mlp-L8-adam',
      'linear-adam']),
    ('b. table sweep, tph in {64,128,256}, 500 steps',
     ['lut-L2-adam', 'lut-L2-tph128', 'lut-L2-tph256', 'lut-L4-adam', 'lut-L4-tph128', 'lut-L4-tph256',
      'lut-L8-adam', 'lut-L8-tph128', 'lut-L8-tph256']),
    ('c. no-residual arms and stabilisers, 2000 steps',
     ['lut-L2-tph64-nores-s2000', 'lut-L2-tph128-nores-s2000', 'lut-L2-tph128-nores-lr1e-4',
      'lut-L2-tph64-noresgn-s2000', 'lut-L2-tph128-noresgn-s2000', 'lut-L2-tph256-noresgn-s2000',
      'lut-L2-tph64-nores-ln-s2000', 'lut-L2-tph128-nores-ln-s2000', 'lut-L2-tph256-nores-ln-s2000',
      'mlp-L2-adam-nores-s2000', 'mlp-L2-adam-noresgn-s2000', 'mlp-L2-nores-ln-s2000',
      'mlpwide-L2-h8192-nores-s2000', 'mlpwide-L2-h8192-noresgn-s2000',
      'mlpwide-L2-h8192-nores-ln-s2000', 'linear-adam-s2000']),
    ('d. residual 2000-step runs (L=2)',
     ['lut-L2-tph64-s2000', 'lut-L2-tph128-s2000', 'mlp-L2-adam-s2000', 'mlpwide-L2-h8192-s2000',
      'linear-adam-s2000']),
]
DIVERGED_CUT = 10.0          # a run ending above this is called out in the legend


def load(n):
    p = os.path.join(R, n, 'run.json')
    return json.load(open(p)) if os.path.exists(p) else None


def ser(h, k):
    return [(r['step'], r[k]) for r in h
            if k in r and isinstance(r[k], (int, float)) and not math.isnan(r[k])]


def tidy(ax, t, ylab='train MSE (standardised)'):
    ax.set(xlabel='step', ylabel=ylab, yscale='log')
    ax.set_title(t, loc='left', fontsize=9.5)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.grid(alpha=0.3)
    # a 16-entry legend will sit on top of the curves in a 2x2 grid, so crowded panels get two
    # columns parked in the empty mid-band that the diverged runs leave behind
    n = len(ax.get_lines())
    if n > 9:
        ax.legend(fontsize=5.8, frameon=False, ncol=2, loc='center right')
    else:
        ax.legend(fontsize=6.5, frameon=False, ncol=1)


def main():
    os.makedirs(OUT, exist_ok=True)
    present, missing = {}, []
    for _, names in FAMILIES:
        for n in names:
            if n in present:
                continue
            d = load(n)
            if d:
                present[n] = d
            elif n not in missing:
                missing.append(n)

    # ---- figure 1: train MSE per family
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    for ax, (title, names) in zip(axes.ravel(), FAMILIES):
        avail = [n for n in names if n in present]
        for i, n in enumerate(avail):
            s = ser(present[n]['hist'], 'eval/train_mse')
            if not s:
                continue
            fin = s[-1][1]
            lbl = n + ('  [DIVERGED]' if fin > DIVERGED_CUT else '')
            ax.plot([q for q, _ in s], [v for _, v in s], color=OKABE[i % len(OKABE)],
                    ls='--' if fin > DIVERGED_CUT else '-', lw=1.6, label=lbl)
        mb = next((present[n]['summary']['mean_baseline_train'] for n in avail
                   if 'mean_baseline_train' in present[n]['summary']), None)
        if mb:
            ax.axhline(mb, color='#888888', ls=(0, (2, 2)), lw=1.0)
            ax.annotate(f'per-pixel train mean {mb:.3f}', xy=(1, mb), xytext=(3, 3),
                        textcoords='offset points', fontsize=7, color='#666666')
        tidy(ax, title)
    fig.suptitle('Autoencoder TRAIN MSE (eval/train_mse, xtr[:10000]) — log y so diverged runs do not '
                 'flatten the rest', fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    p1 = os.path.join(OUT, 'ae_train_by_family.png')
    fig.savefig(p1, dpi=170)

    # ---- figure 2: train (solid) vs test (dashed), same axes
    fig2, axes2 = plt.subplots(2, 2, figsize=(14, 9))
    for ax, (title, names) in zip(axes2.ravel(), FAMILIES):
        avail = [n for n in names if n in present]
        for i, n in enumerate(avail):
            c = OKABE[i % len(OKABE)]
            tr, te = ser(present[n]['hist'], 'eval/train_mse'), ser(present[n]['hist'], 'eval/test_mse')
            if tr:
                ax.plot([q for q, _ in tr], [v for _, v in tr], color=c, ls='-', lw=1.6, label=n)
            if te:
                ax.plot([q for q, _ in te], [v for _, v in te], color=c, ls='--', lw=1.2, alpha=0.9)
        lin = present.get('linear-adam-s2000') or present.get('linear-adam')
        if lin:
            v = lin['summary']['test_mse']
            ax.axhline(v, color='#555555', ls=(0, (4, 3)), lw=1.1)
            ax.annotate(f'linear baseline (test) {v:.4f}', xy=(1, v), xytext=(3, -10),
                        textcoords='offset points', fontsize=7, color='#555555')
        tidy(ax, title + '   — solid = train, dashed = test', 'MSE (standardised)')
    fig2.suptitle('Autoencoder train (solid) vs held-out test (dashed), same colour per run', fontsize=11)
    fig2.tight_layout(rect=(0, 0, 1, 0.97))
    p2 = os.path.join(OUT, 'ae_train_vs_test.png')
    fig2.savefig(p2, dpi=170)

    # ---- the table
    print(f'\n{"run":34s} {"steps":>6s} {"train MSE":>12s} {"test MSE":>12s} {"test - train":>13s}')
    rows = []
    for n in sorted(present):
        s = present[n]['summary']
        tr, te = s['train_mse'], s['test_mse']
        rows.append((n, s.get('steps_done', present[n]['hist'][-1]['step']), tr, te, te - tr))
    for n, st, tr, te, gap in rows:
        f = (lambda v: f'{v:>12.4f}' if abs(v) < 1e4 else f'{v:>12.3e}')
        print(f'{n:34s} {st:>6d} {f(tr)} {f(te)} {f(gap)}')
    if missing:
        print('\nnot on disk: ' + ', '.join(missing))
    print(f'\nwrote {p1}\nwrote {p2}')


if __name__ == '__main__':
    main()
