"""Reconstruction grids for the d128 / latent-128 / augmented arm.

Nothing is trained here; every row is a saved checkpoint loaded in eval mode, so neither dropout nor
augmentation is active in any reconstruction. Same ten test items and conventions as the earlier
figures: first occurrence of each label, un-standardised to pixel space for display, per-image MSE in
standardised units.

TWO linear baselines appear in both figures on purpose. 784-64-784 (0.11026) is the floor at 12.25x
compression, which is what the d64 ViT's 0.05193 was measured against; 784-128-784 is the floor at the
6.125x this arm actually runs at. The second is the honest reference for the new number.
"""
import json
import os
import sys

import torch
from torchvision import datasets

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.expanduser('~/projects/spiky/src'))
from data import DATA_ROOT, load  # noqa: E402
from make_vit_dropout_figures import LINEAR, OUT, R, build, grid  # noqa: E402

LINEAR128 = 'linear-full-lat128-s10000'
D64 = 'vit-p2-k8-e4d4-nowarm-full-s60000'
D128 = 'vit-p2-k8-e4d4-d128h4-lat128-nowarm-full-aug-s60000'


@torch.no_grad()
def main():
    os.makedirs(OUT, exist_ok=True)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    xte, _ = load('fashion', train=False, device=dev)
    labels = datasets.FashionMNIST(DATA_ROOT, train=False, download=False).targets.to(dev)
    idx = [int((labels == c).nonzero()[0]) for c in range(10)]
    x = xte[idx]

    l64_f, l64_d = build(LINEAR, dev)
    l128_f, l128_d = build(LINEAR128, dev)
    base = [('original', None),
            (f'linear 784-64-784\n12.25x  test {l64_d["summary"]["test_mse"]:.5f}', l64_f(x)),
            (f'linear 784-128-784\n6.125x  test {l128_d["summary"]["test_mse"]:.5f}', l128_f(x))]

    d = json.load(open(os.path.join(R, D128, 'run.json')))
    rows = list(base)
    for ck in d['summary']['checkpoints']:
        f, _ = build(D128, dev, ck['file'])
        rows.append((f'ViT d128 h4 lat128 + aug\n{ck["step"]//1000}K  test {ck["test_mse"]:.5f}', f(x)))
    grid(rows, x, 'Fashion-MNIST 28x28, d_model 128 / 4 heads / 128-dim latent (6.125x) with train-time '
                  'augmentation, at six points in training — per-image MSE (standardised units)',
         os.path.join(OUT, 'recon_28_vit_d128h4_lat128_aug_progression.png'))

    d64_f, d64_d = build(D64, dev)
    d128_f, _ = build(D128, dev)
    grid(base + [(f'ViT d64 lat64, no aug\n12.25x  test {d64_d["summary"]["test_mse"]:.5f}', d64_f(x)),
                 (f'ViT d128 lat128 + aug\n6.125x  test {d["summary"]["test_mse"]:.5f}', d128_f(x))],
         x, 'The two ViTs at 60K, each beside the linear floor for ITS OWN compression ratio — the two '
            'ViT rows are not like-for-like (12.25x vs 6.125x)',
         os.path.join(OUT, 'recon_28_vit_d128h4_lat128_aug_vs_d64.png'))


if __name__ == '__main__':
    main()
