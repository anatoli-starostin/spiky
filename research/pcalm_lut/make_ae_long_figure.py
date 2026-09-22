"""Charts for the 2000-step autoencoder runs, from runs_autoencoder/*-s2000/run.json.

Three questions, three panels: has anything converged, have the residual blocks left the identity path,
and does the LUT routing stay healthy over four times the training.

Palette: Okabe-Ito (published colourblind-safe); hues assigned per entity and never cycled. Points are
markers because the runs probe every 100 steps.
"""
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, 'runs_autoencoder')
OUT = os.path.join(R, 'plots')
RUNS = [('linear-adam-s2000', 'linear 784-64-784 (0.10M)', '#555555'),
        ('lut-L2-tph64-s2000', 'LUT tph=64 (4.30M)', '#0072B2'),
        ('lut-L2-tph128-s2000', 'LUT tph=128 (8.49M)', '#56B4E9'),
        ('mlpwide-L2-h8192-s2000', 'MLP wide h=8192 (4.30M)', '#D55E00'),
        ('mlp-L2-adam-s2000', 'MLP narrow (0.12M)', '#E69F00')]
INK = '#222222'
plt.rcParams.update({'axes.edgecolor': '#BBBBBB', 'axes.labelcolor': INK, 'text.color': INK,
                     'xtick.color': INK, 'ytick.color': INK, 'font.size': 9, 'axes.grid': True,
                     'grid.color': '#E4E4E4', 'figure.facecolor': 'white', 'axes.facecolor': 'white'})


def series(h, k):
    return [(r['step'], r[k]) for r in h if k in r]


def tidy(ax, xl, yl, t):
    ax.set(xlabel=xl, ylabel=yl)
    ax.set_title(t, loc='left')
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(alpha=0.3)


def main():
    os.makedirs(OUT, exist_ok=True)
    data = {}
    for n, lbl, c in RUNS:
        p = os.path.join(R, n, 'run.json')
        if os.path.exists(p):
            data[n] = (json.load(open(p)), lbl, c)

    # (a) MSE vs step, with the linear baseline marked
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    lin = data.get('linear-adam-s2000')
    for n, (d, lbl, c) in data.items():
        s = series(d['hist'], 'eval/test_mse')
        ax.plot([q for q, _ in s], [v for _, v in s], 'o-', ms=3.5, lw=1.7, color=c, label=lbl,
                markeredgecolor='white', markeredgewidth=0.6)
    if lin:
        fin = lin[0]['summary']['test_mse']
        ax.axhline(fin, color='#555555', ls=(0, (4, 3)), lw=1.2)
        ax.annotate(f'linear baseline at 2000 steps: {fin:.4f}', xy=(2000, fin), xytext=(-4, 6),
                    textcoords='offset points', ha='right', fontsize=8.5, color='#555555')
    tidy(ax, 'step', 'held-out MSE (standardised units)',
         'Reconstruction through a 64-dim bottleneck, 2000 steps (Adam 1e-3)')
    ax.set_yscale('log')
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    p1 = os.path.join(OUT, 'ae_long_mse.png')
    fig.savefig(p1, dpi=170)

    # (b) how far the residual blocks have left the identity path
    fig2, ax2 = plt.subplots(figsize=(8.0, 4.4))
    for n, (d, lbl, c) in data.items():
        s = series(d['hist'], 'branch/ratio_mean')
        if not s or max(v for _, v in s) == 0:
            continue
        ax2.plot([q for q, _ in s], [v for _, v in s], 'o-', ms=3.5, lw=1.7, color=c, label=lbl,
                 markeredgecolor='white', markeredgewidth=0.6)
    tidy(ax2, 'step', 'mean ||a_i block(h)|| / ||h||',
         'How far the residual blocks have departed from the identity path')
    ax2.legend(fontsize=8, frameon=False)
    fig2.tight_layout()
    p2 = os.path.join(OUT, 'ae_long_branch.png')
    fig2.savefig(p2, dpi=170)

    # (c) LUT routing health
    fig3, ax3 = plt.subplots(1, 3, figsize=(13.5, 4.0))
    for n, (d, lbl, c) in data.items():
        if not n.startswith('lut'):
            continue
        for j, (k, t) in enumerate((('lut/m_min_mean', 'median smallest margin'),
                                    ('lut/tau_mean', 'learned tau'),
                                    ('flips/mean', 'address flips since the previous probe'))):
            s = [(q, v) for q, v in series(d['hist'], k) if v == v]
            ax3[j].plot([q for q, _ in s], [v for _, v in s], 'o-', ms=3.5, lw=1.7, color=c, label=lbl,
                        markeredgecolor='white', markeredgewidth=0.6)
            tidy(ax3[j], 'step', '', t)
    for a_ in ax3:
        a_.legend(fontsize=8, frameon=False)
    fig3.tight_layout()
    p3 = os.path.join(OUT, 'ae_long_routing.png')
    fig3.savefig(p3, dpi=170)
    for p in (p1, p2, p3):
        print('wrote', os.path.relpath(p, HERE))


if __name__ == '__main__':
    main()
