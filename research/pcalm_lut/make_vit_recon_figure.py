"""Reconstruction grid at 28x28: original / linear / ViT, from the saved checkpoints.

No training happens here. Both models are rebuilt from the cfg in their own run.json and loaded from the
model.pt written beside it.

The ten columns are the FIRST occurrence of each Fashion-MNIST class in the test split, so the selection
is reproducible without a seed. Images are un-standardised back to [0,1] with the loader's own constants
((x/255 - 0.1307)/0.3081, data.py:17-18) before display, because a standardised image is not something
you can look at. Per-image MSE is printed under each reconstruction in the same standardised units the
training loss used, so the numbers match the reported test MSEs.
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
LINEAR, VIT = 'linear-full-s10000', 'vit-p2-k8-e4d4-nowarm-full-s10000'


def build(name, dev):
    d = json.load(open(os.path.join(R, name, 'run.json')))
    c = d['cfg']
    n_pix = d['summary']['n_pixels']
    if c['arch'] == 'linear':
        m = LinearAE(n_pix, c['latent'], dev, c['seed'])
    else:
        m = ViTAutoencoder(d['summary']['n_tokens'], d['summary']['patch_dim'], c['d_model'],
                           c['n_heads'], c['enc_layers'], c['dec_layers'], c['latent'],
                           c['latent_tokens'], c['ffn_mult'], c['ffn'], c['tables'], dev, c['seed'])
    m.load_state_dict(torch.load(os.path.join(R, name, 'model.pt'), map_location=dev))
    m.eval()
    return m, make_eval(m, c['arch'], 28, c['patch']), d['summary']['test_mse']


@torch.no_grad()
def main():
    os.makedirs(OUT, exist_ok=True)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    xte, _ = load('fashion', train=False, device=dev)          # 28x28, no downsampling
    labels = datasets.FashionMNIST(DATA_ROOT, train=False, download=False).targets.to(dev)
    # first occurrence of each class, in test-set order: reproducible without a seed
    idx = [int((labels == c).nonzero()[0]) for c in range(10)]
    x = xte[idx]

    lin_m, lin_f, lin_mse = build(LINEAR, dev)
    vit_m, vit_f, vit_mse = build(VIT, dev)
    rec = {'linear': lin_f(x), 'vit': vit_f(x)}
    per = {k: (v - x).pow(2).mean(-1) for k, v in rec.items()}

    def img(t):
        return (t * STD + MEAN).clamp(0, 1).reshape(28, 28).cpu()

    rows = [('original', None, None),
            (f'linear 784-64-784\ntest MSE {lin_mse:.5f}', rec['linear'], per['linear']),
            (f'ViT p2-k8-e4d4\ntest MSE {vit_mse:.5f}', rec['vit'], per['vit'])]
    fig, ax = plt.subplots(3, 10, figsize=(13.5, 4.9))
    for j in range(10):
        ax[0][j].set_title(CLASSES[j], fontsize=8.5, pad=4)
    for i, (label, r, p) in enumerate(rows):
        for j in range(10):
            a = ax[i][j]
            a.imshow(img(x[j] if r is None else r[j]), cmap='gray', vmin=0, vmax=1)
            a.set_xticks([])
            a.set_yticks([])
            for s in a.spines.values():
                s.set_color('#CCCCCC')
            if p is not None:
                a.set_xlabel(f'{float(p[j]):.3f}', fontsize=7.5, labelpad=2, color='#333333')
        ax[i][0].set_ylabel(label, fontsize=8.5, rotation=0, ha='right', va='center', labelpad=8)
    fig.suptitle('Fashion-MNIST at 28x28 through a 64-dim bottleneck — per-image MSE under each '
                 'reconstruction (standardised units)', fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    p1 = os.path.join(OUT, 'recon_28_linear_vs_vit.png')
    fig.savefig(p1, dpi=170)
    print(f'class indices in the test split: {idx}')
    print(f'per-image MSE linear: ' + ' '.join(f'{float(v):.3f}' for v in per['linear']))
    print(f'per-image MSE vit   : ' + ' '.join(f'{float(v):.3f}' for v in per['vit']))
    print(f'mean over these 10 — linear {float(per["linear"].mean()):.5f}, '
          f'vit {float(per["vit"].mean()):.5f}')
    print('wrote', p1)


if __name__ == '__main__':
    main()
