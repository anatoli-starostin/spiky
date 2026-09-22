"""Put our reconstruction error on the scales the VAE literature uses. No training; saved checkpoints.

Our MSE is in STANDARDISED units, (x/255 - 0.1307)/0.3081. Nothing in the literature is. Converting is
a multiply by 0.3081^2 for [0,1] pixel units and a further 255^2 for 0-255 units -- but PSNR is the one
that needs care, because a decoder can emit values outside [0,1] and PSNR is defined against a peak of
1. This computes BOTH the unclamped number (what the training loss actually minimises) and the clamped
one (what a viewer sees, and what an image-quality paper would report), so neither can be quietly
substituted for the other.
"""
import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.expanduser('~/projects/spiky/src'))
from data import load  # noqa: E402
from make_vit_dropout_figures import R, build  # noqa: E402

MEAN, STD = 0.1307, 0.3081
RUNS = [('ViT d128 h4 lat128 +aug, 6.125x', 'vit-p2-k8-e4d4-d128h4-lat128-nowarm-full-aug-s60000'),
        ('ViT d64 lat64 plain, 12.25x', 'vit-p2-k8-e4d4-nowarm-full-s60000'),
        ('linear 784-128-784, 6.125x', 'linear-full-lat128-s10000'),
        ('linear 784-64-784, 12.25x', 'linear-full-s10000')]


@torch.no_grad()
def main():
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    xte, _ = load('fashion', train=False, device=dev)
    print(f'{"model":<34}{"MSE std":>9}{"MSE [0,1]":>11}{"MSE 0-255":>11}{"PSNR dB":>9}'
          f'{"PSNR clamp":>11}')
    rows = {}
    for label, run in RUNS:
        f, d = build(run, dev)
        tot = tot_c = n = 0.0
        for i in range(0, xte.shape[0], 2048):
            b = xte[i:i + 2048]
            r = f(b)
            p, q = r * STD + MEAN, b * STD + MEAN          # back to [0,1] pixel units
            tot += float((p - q).pow(2).sum())
            tot_c += float((p.clamp(0, 1) - q.clamp(0, 1)).pow(2).sum())
            n += b.numel()
        m01, m01c = tot / n, tot_c / n
        mstd, m255 = m01 / STD ** 2, m01 * 255 ** 2
        psnr, psnr_c = 10 * torch.log10(torch.tensor(1.0 / m01)), 10 * torch.log10(torch.tensor(1.0 / m01c))
        print(f'{label:<34}{mstd:>9.5f}{m01:>11.6f}{m255:>11.1f}{float(psnr):>9.2f}{float(psnr_c):>11.2f}')
        rows[label] = dict(mse_std=mstd, mse_01=m01, mse_01_clamped=m01c, mse_255=m255,
                           psnr_db=float(psnr), psnr_db_clamped=float(psnr_c),
                           recorded_test_mse=d['summary']['test_mse'])
    print(f'\nscale factors: MSE[0,1] = MSE_std * {STD**2:.8f};  MSE[0,255] = MSE[0,1] * 65025')
    json.dump(rows, open(os.path.join(R, 'mse_scale_conversions.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
