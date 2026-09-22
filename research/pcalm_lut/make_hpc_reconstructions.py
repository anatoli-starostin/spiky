"""Reconstruction grids for the 60K HPC run, including the per-level decomposition.

Same ten test items and conventions as every other figure in this work: first occurrence of each label
in the test split (so sample 3 is the pullover with the printed graphic the owner has been tracking),
un-standardised to pixel space for display, nearest-neighbour upscaling, no sharpening or contrast
adjustment of any kind.

The decomposition figure shows each level's CONTRIBUTION, not each level's running sum: level 0's
prediction, then levels 1, 2 and 3's own outputs, then the running sums. A level that contributes
nothing shows as a blank panel, which is the point of drawing it.

Level contributions are signed and mostly near zero, so they are drawn on a SYMMETRIC diverging scale
around 0 with the range printed under each panel -- putting them on the [0,1] image scale would render
every one of them as flat grey and hide exactly what the figure exists to show.
"""
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
from torchvision import datasets  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.expanduser('~/projects/spiky/src'))
from autoencoder import Autoencoder  # noqa: E402
from data import DATA_ROOT, load  # noqa: E402

R = os.path.join(HERE, 'runs_ae')
RUN = 'cmhl-L4-hpc-diag-s60000'
OUT = os.path.join(R, 'plots_hpc60k')
MEAN, STD = 0.1307, 0.3081
CLASSES = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
           'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
DECOMP = [2, 0, 5, 8]          # pullover (the printed one), t-shirt, sandal, bag


def build(weights='model.pt'):
    d = json.load(open(os.path.join(R, RUN, 'run.json')))
    c = d['cfg']
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    m = Autoencoder(784, c['width'], c['depth_L'], c['kind'], c['tables'], dev, c['seed'],
                    residual=not c['no_residual'], block_norm=c['block_norm'],
                    lut_impl=c['lut_impl'], norm_position=c['norm_position'],
                    n_blocks=c['n_blocks'], inner_out=c['inner_out'], inner_in=c['inner_in'],
                    final_norm=c['final_norm'], deep_supervision=c['deep_supervision'])
    m.load_state_dict(torch.load(os.path.join(R, RUN, weights), map_location=dev))
    m.eval()
    return m, d, dev


def img(t):
    return (t * STD + MEAN).clamp(0, 1).reshape(28, 28).cpu()


def bare(a):
    a.set_xticks([])
    a.set_yticks([])
    for s in a.spines.values():
        s.set_color('#CCCCCC')


@torch.no_grad()
def main():
    os.makedirs(OUT, exist_ok=True)
    m, d, dev = build()
    xte, _ = load('fashion', train=False, device=dev)
    labels = datasets.FashionMNIST(DATA_ROOT, train=False, download=False).targets.to(dev)
    idx = [int((labels == c).nonzero()[0]) for c in range(10)]
    x = xte[idx]
    preds = m.levels(x)
    recon = sum(preds)
    print(f'selection {idx}; sample 3 is test index {idx[2]} (pullover)')

    # --- figure A: originals over reconstructions -------------------------------------------------
    per = (recon - x).pow(2).mean(-1)
    fig, ax = plt.subplots(2, 10, figsize=(13.5, 3.6))
    for jx in range(10):
        ax[0][jx].set_title(CLASSES[jx], fontsize=8.5, pad=4)
        ax[0][jx].imshow(img(x[jx]), cmap='gray', vmin=0, vmax=1, interpolation='nearest')
        ax[1][jx].imshow(img(recon[jx]), cmap='gray', vmin=0, vmax=1, interpolation='nearest')
        ax[1][jx].set_xlabel(f'{float(per[jx]):.3f}', fontsize=7.5, labelpad=2, color='#333333')
        for r in (0, 1):
            bare(ax[r][jx])
    ax[0][0].set_ylabel('original', fontsize=8.5, rotation=0, ha='right', va='center', labelpad=8)
    ax[1][0].set_ylabel(f'HPC 60K\ntest {d["summary"]["test_mse"]:.5f}', fontsize=8.5, rotation=0,
                        ha='right', va='center', labelpad=8)
    fig.suptitle('HPC at 60K: summed reconstruction, per-image MSE below (standardised units)',
                 fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    p = os.path.join(OUT, '7_reconstructions.png')
    fig.savefig(p, dpi=190)
    plt.close(fig)
    print('wrote', p)

    # --- figure B: per-level decomposition --------------------------------------------------------
    nl = len(preds)
    cols = 1 + nl + nl          # original, then nl contributions, then nl running sums
    fig, ax = plt.subplots(len(DECOMP), cols, figsize=(1.35 * cols, 1.5 * len(DECOMP) + 1.0))
    for r, s_i in enumerate(DECOMP):
        run = torch.zeros_like(x[s_i])
        c = 0
        ax[r][c].imshow(img(x[s_i]), cmap='gray', vmin=0, vmax=1, interpolation='nearest')
        bare(ax[r][c])
        if r == 0:
            ax[r][c].set_title('original', fontsize=7.5, pad=4)
        ax[r][c].set_ylabel(CLASSES[s_i], fontsize=8, rotation=0, ha='right', va='center', labelpad=6)
        c += 1
        for li in range(nl):
            v = preds[li][s_i]
            run = run + v
            lim = float(v.abs().max())
            ax[r][c].imshow(v.reshape(28, 28).cpu(), cmap='RdBu_r', vmin=-lim, vmax=lim,
                            interpolation='nearest')
            bare(ax[r][c])
            ax[r][c].set_xlabel(f'+-{lim:.2f}', fontsize=6.5, labelpad=1.5, color='#333333')
            if r == 0:
                ax[r][c].set_title(f'level {li}\ncontribution', fontsize=7.5, pad=4)
            c += 1
        run = torch.zeros_like(x[s_i])
        for li in range(nl):
            run = run + preds[li][s_i]
            ax[r][c].imshow(img(run), cmap='gray', vmin=0, vmax=1, interpolation='nearest')
            bare(ax[r][c])
            ax[r][c].set_xlabel(f'{float((run - x[s_i]).pow(2).mean()):.3f}', fontsize=6.5,
                                labelpad=1.5, color='#333333')
            if r == 0:
                ax[r][c].set_title(f'sum of\nlevels 0-{li}', fontsize=7.5, pad=4)
            c += 1
    fig.suptitle('Per-level decomposition. Contributions are signed, each on its own symmetric scale '
                 'with its range printed;\nrunning sums are on the image scale with MSE below. A level '
                 'that contributes nothing shows as a flat panel.', fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    p = os.path.join(OUT, '8_level_decomposition.png')
    fig.savefig(p, dpi=190)
    plt.close(fig)
    print('wrote', p)

    print('\nper-level contribution magnitude (mean |contribution| over the 10 items):')
    for li, pr in enumerate(preds):
        print(f'  level {li}: mean |v| {float(pr.abs().mean()):.5f}   '
              f'RMS {float(pr.pow(2).mean().sqrt()):.5f}')
    print(f'  original    mean |x| {float(x.abs().mean()):.5f}')


if __name__ == '__main__':
    main()
