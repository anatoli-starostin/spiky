"""Reconstructions along one ViT training run: original, linear baseline, then each saved checkpoint.

Nothing is trained here. The ViT rows all come from ONE run (vit-p2-k8-e4d4-nowarm-full-s60000) via the
per-step checkpoints --ckpt-every wrote, so the rows are successive points on a single trajectory rather
than six independent runs of different lengths. The linear row is the existing linear-full-s10000 result,
which is flat well before 10K and was not retrained.

Same ten test items and the same conventions as make_vit_recon_figure.py: first occurrence of each label
in the test split, un-standardised to pixel space for display, per-image MSE kept in standardised units
so it is commensurate with the test MSEs in the row labels.
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
from data import DATA_ROOT, load  # noqa: E402
from vit_autoencoder import LinearAE, ViTAutoencoder, make_eval  # noqa: E402

R = os.path.join(HERE, 'runs_vit')
OUT = os.path.join(R, 'plots')
MEAN, STD = 0.1307, 0.3081
CLASSES = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
           'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
LINEAR, VIT = 'linear-full-s10000', 'vit-p2-k8-e4d4-nowarm-full-s60000'


def build(name, dev, weights='model.pt'):
    """Rebuild a model from the cfg recorded in its own run.json and load one of its checkpoints."""
    d = json.load(open(os.path.join(R, name, 'run.json')))
    c, s = d['cfg'], d['summary']
    if c['arch'] == 'linear':
        m = LinearAE(s['n_pixels'], c['latent'], dev, c['seed'])
    else:
        m = ViTAutoencoder(s['n_tokens'], s['patch_dim'], c['d_model'], c['n_heads'], c['enc_layers'],
                           c['dec_layers'], c['latent'], c['latent_tokens'], c['ffn_mult'], c['ffn'],
                           c['tables'], dev, c['seed'])
    m.load_state_dict(torch.load(os.path.join(R, name, weights), map_location=dev))
    m.eval()
    return make_eval(m, c['arch'], 28, c['patch']), d


@torch.no_grad()
def main():
    os.makedirs(OUT, exist_ok=True)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    xte, _ = load('fashion', train=False, device=dev)          # 28x28, no downsampling
    labels = datasets.FashionMNIST(DATA_ROOT, train=False, download=False).targets.to(dev)
    idx = [int((labels == c).nonzero()[0]) for c in range(10)]  # first of each class: reproducible
    x = xte[idx]

    lin_f, lin_d = build(LINEAR, dev)
    rows = [('original', None)]
    rows.append((f'linear 784-64-784\n10K steps  test {lin_d["summary"]["test_mse"]:.5f}', lin_f(x)))

    vit_d = json.load(open(os.path.join(R, VIT, 'run.json')))
    for ck in vit_d['summary']['checkpoints']:
        f, _ = build(VIT, dev, ck['file'])
        rows.append((f'ViT  {ck["step"]//1000}K steps\ntest {ck["test_mse"]:.5f}', f(x)))

    def img(t):
        return (t * STD + MEAN).clamp(0, 1).reshape(28, 28).cpu()

    n = len(rows)
    fig, ax = plt.subplots(n, 10, figsize=(13.5, 1.42 * n + 0.9))
    for j in range(10):
        ax[0][j].set_title(CLASSES[j], fontsize=8.5, pad=4)
    for i, (label, r) in enumerate(rows):
        p = None if r is None else (r - x).pow(2).mean(-1)
        for j in range(10):
            a = ax[i][j]
            a.imshow(img(x[j] if r is None else r[j]), cmap='gray', vmin=0, vmax=1)
            a.set_xticks([])
            a.set_yticks([])
            for s in a.spines.values():
                s.set_color('#CCCCCC')
            if p is not None:
                a.set_xlabel(f'{float(p[j]):.3f}', fontsize=7, labelpad=1.5, color='#333333')
        ax[i][0].set_ylabel(label, fontsize=8.5, rotation=0, ha='right', va='center', labelpad=8)
        if p is not None:
            print(f'{label.splitlines()[0]:<22} mean {float(p.mean()):.5f}  ' +
                  ' '.join(f'{float(v):.3f}' for v in p))
    fig.suptitle('Fashion-MNIST at 28x28, 64-dim bottleneck: one ViT run seen at six points in training '
                 '— per-image MSE under each reconstruction (standardised units)', fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.36 / (1.42 * n + 0.9)))
    path = os.path.join(OUT, 'recon_28_vit_progression.png')
    fig.savefig(path, dpi=160)
    print('wrote', path)


if __name__ == '__main__':
    main()
