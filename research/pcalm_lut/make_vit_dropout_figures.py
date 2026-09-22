"""Two figures for the dropout arm: its own training progression, and a head-to-head against no dropout.

Nothing is trained here; every row is a saved checkpoint loaded in eval mode, so dropout is off in all
reconstructions regardless of what the model was trained with.

Same ten test items and the same conventions as the earlier figures: first occurrence of each label in
the test split, un-standardised to pixel space for display, per-image MSE kept in standardised units.
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
LINEAR = 'linear-full-s10000'
PLAIN = 'vit-p2-k8-e4d4-nowarm-full-s60000'
DROP = 'vit-p2-k8-e4d4-nowarm-full-drop20-s60000'


def build(name, dev, weights='model.pt'):
    d = json.load(open(os.path.join(R, name, 'run.json')))
    c, s = d['cfg'], d['summary']
    if c['arch'] == 'linear':
        m = LinearAE(s['n_pixels'], c['latent'], dev, c['seed'])
    else:
        m = ViTAutoencoder(s['n_tokens'], s['patch_dim'], c['d_model'], c['n_heads'], c['enc_layers'],
                           c['dec_layers'], c['latent'], c['latent_tokens'], c['ffn_mult'], c['ffn'],
                           c['tables'], dev, c['seed'], c.get('dropout', 0.0))
    m.load_state_dict(torch.load(os.path.join(R, name, weights), map_location=dev))
    m.eval()   # dropout off for every reconstruction, whatever the run was trained with
    return make_eval(m, c['arch'], 28, c['patch']), d


def grid(rows, x, title, path):
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
            print(f'{label.splitlines()[0]:<26} mean {float(p.mean()):.5f}  ' +
                  ' '.join(f'{float(v):.3f}' for v in p))
    fig.suptitle(title, fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.36 / (1.42 * n + 0.9)))
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print('wrote', path, '\n')


@torch.no_grad()
def main():
    os.makedirs(OUT, exist_ok=True)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    xte, _ = load('fashion', train=False, device=dev)
    labels = datasets.FashionMNIST(DATA_ROOT, train=False, download=False).targets.to(dev)
    idx = [int((labels == c).nonzero()[0]) for c in range(10)]
    x = xte[idx]

    lin_f, lin_d = build(LINEAR, dev)
    lin_row = (f'linear 784-64-784\n10K steps  test {lin_d["summary"]["test_mse"]:.5f}', lin_f(x))

    # figure 1: the dropout run's own progression
    drop_d = json.load(open(os.path.join(R, DROP, 'run.json')))
    rows = [('original', None), lin_row]
    for ck in drop_d['summary']['checkpoints']:
        f, _ = build(DROP, dev, ck['file'])
        rows.append((f'ViT drop 0.2\n{ck["step"]//1000}K  test {ck["test_mse"]:.5f}', f(x)))
    grid(rows, x, 'Fashion-MNIST 28x28, 64-dim bottleneck, ViT with dropout 0.2 at six points in '
                  'training — per-image MSE under each reconstruction (standardised units)',
         os.path.join(OUT, 'recon_28_vit_drop20_progression.png'))

    # figure 2: the two 60K models head to head
    plain_f, plain_d = build(PLAIN, dev)
    drop_f, _ = build(DROP, dev)
    grid([('original', None), lin_row,
          (f'ViT 60K no dropout\ntest {plain_d["summary"]["test_mse"]:.5f}', plain_f(x)),
          (f'ViT 60K dropout 0.2\ntest {drop_d["summary"]["test_mse"]:.5f}', drop_f(x))],
         x, 'The two 60K ViTs head to head, same ten test items — per-image MSE under each '
            'reconstruction (standardised units)',
         os.path.join(OUT, 'recon_28_vit_dropout_vs_plain.png'))


if __name__ == '__main__':
    main()
