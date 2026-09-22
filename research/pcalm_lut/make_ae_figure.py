"""Reconstruction grid and loss curves for the autoencoder sweep, from the saved checkpoints.

Images are shown in [0,1] pixel space (the loader's standardisation undone), because a standardised
image is not something you can look at. Every panel is labelled with the run that produced it and its
held-out test MSE, so no number in the figure is detached from its run.
"""
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.expanduser('~/projects/spiky/src'))
from autoencoder import PIX_STD, Autoencoder  # noqa: E402
from data import load  # noqa: E402

R = os.path.join(HERE, 'runs_autoencoder')
OUT = os.path.join(HERE, 'runs_autoencoder', 'plots')
MEAN, STD = 0.1307, PIX_STD
ROWS = ['lut-L2-adam', 'lut-L4-adam', 'lut-L8-adam', 'mlp-L4-adam', 'linear-adam']
N_IMG = 8


def unnorm(x):
    return (x * STD + MEAN).clamp(0, 1)


def load_run(name, dev):
    p = os.path.join(R, name, 'run.json')
    if not os.path.exists(p):
        return None, None
    d = json.load(open(p))
    c = d['cfg']
    m = Autoencoder(784, c['width'], c['depth_L'], c['kind'], c['tables'], dev, c['seed'],
                    hidden=c.get('hidden', 0))
    m.load_state_dict(torch.load(os.path.join(R, name, 'model.pt'), map_location=dev))
    m.eval()
    return m, d


def main():
    os.makedirs(OUT, exist_ok=True)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    xte, _ = load('fashion', train=False, device=dev)
    x = xte[:N_IMG]
    rows = [(n, *load_run(n, dev)) for n in ROWS]
    rows = [(n, m, d) for n, m, d in rows if m is not None]

    fig, ax = plt.subplots(len(rows) + 1, N_IMG, figsize=(1.15 * N_IMG, 1.25 * (len(rows) + 1)))
    for j in range(N_IMG):
        ax[0][j].imshow(unnorm(x[j]).reshape(28, 28).cpu(), cmap='gray', vmin=0, vmax=1)
        ax[0][j].axis('off')
    ax[0][0].set_ylabel('original')
    ax[0][0].axis('on')
    ax[0][0].set_xticks([])
    ax[0][0].set_yticks([])
    for i, (name, m, d) in enumerate(rows, start=1):
        with torch.no_grad():
            rec = unnorm(m(x))
        for j in range(N_IMG):
            ax[i][j].imshow(rec[j].reshape(28, 28).cpu(), cmap='gray', vmin=0, vmax=1)
            ax[i][j].axis('off')
        ax[i][0].axis('on')
        ax[i][0].set_xticks([])
        ax[i][0].set_yticks([])
        mse = d['summary']['test_mse']
        ax[i][0].set_ylabel(f'{name}\ntest MSE {mse:.3f}', fontsize=7)
    ax[0][0].set_ylabel('original', fontsize=7)
    fig.suptitle('Fashion-MNIST through a 64-dim bottleneck, 500 steps (MSE in standardised units)',
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    p1 = os.path.join(OUT, 'ae_reconstructions.png')
    fig.savefig(p1, dpi=170)

    # loss curves: is anything converged at 500 steps?
    fig2, ax2 = plt.subplots(figsize=(7.6, 4.4))
    col = {'lut': '#0072B2', 'mlp': '#D55E00', 'linear': '#555555'}
    style = {2: '-', 4: '--', 8: ':'}
    for name in sorted(os.listdir(R)):
        p = os.path.join(R, name, 'run.json')
        if not os.path.exists(p) or 'sgd' in name:
            continue
        d = json.load(open(p))
        s = [(r['step'], r['eval/test_mse']) for r in d['hist']]
        k, L = d['cfg']['kind'], d['cfg']['depth_L']
        ax2.plot([q for q, _ in s], [v for _, v in s], color=col[k], ls=style.get(L, '-'),
                 lw=1.8, label=name)
    mb = json.load(open(os.path.join(R, ROWS[0], 'run.json')))['summary']['mean_baseline_test']
    ax2.axhline(mb, color='#888888', ls=(0, (4, 3)), lw=1.2)
    ax2.annotate(f'predicting the per-pixel mean: {mb:.3f}', xy=(500, mb), xytext=(-4, 5),
                 textcoords='offset points', ha='right', fontsize=8.5, color='#555555')
    ax2.set(xlabel='step', ylabel='held-out MSE (standardised units)',
            title='Reconstruction through a 64-dim bottleneck (Adam 1e-3)')
    ax2.set_yscale('log')
    for side in ('top', 'right'):
        ax2.spines[side].set_visible(False)
    ax2.grid(alpha=0.25)
    ax2.legend(fontsize=8, frameon=False)
    fig2.tight_layout()
    p2 = os.path.join(OUT, 'ae_curves.png')
    fig2.savefig(p2, dpi=170)
    print('wrote', p1)
    print('wrote', p2)


if __name__ == '__main__':
    main()
