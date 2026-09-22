"""Compact ledger for the CompressionMHL autoencoder arms, on every scale the project uses.

Standardised MSE is what the loss minimises; [0,1] pixel MSE and PSNR are what the comparison set and
the image-quality literature use. PSNR here is unclamped, matching the training objective -- see
convert_mse_scales.py for why the clamped and unclamped numbers differ for a linear decoder.
"""
import json
import math
import os

S = 0.3081
HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, 'runs_ae')
ARMS = [('cmhl-L2-w128-tph128-din128-dout-1-prenorm-noresid', 'din=128 dout=-1  no resid  SPEC'),
        ('cmhl-L2-w128-tph128-din128-dout-1-prenorm-resid', 'din=128 dout=-1  residual'),
        ('cmhl-L2-w128-tph128-din128-dout128-prenorm-noresid', 'din=128 dout=128 no resid  extra'),
        ('cmhl-L2-w128-tph128-din128-dout128-prenorm-resid', 'din=128 dout=128 residual  extra'),
        ('cmhl-L2-w128-tph128-dout-1-prenorm-noresid', 'din=-1  dout=-1  no resid  (prior)'),
        ('cmhl-L2-w128-tph128-dout-1-prenorm-resid', 'din=-1  dout=-1  residual   (prior)'),
        ('cmhl-L2-w128-tph128-dout128-prenorm-noresid', 'din=-1  dout=128 no resid  (prior)'),
        ('cmhl-L2-w128-tph128-dout128-prenorm-resid', 'din=-1  dout=128 residual   (prior)')]
FLOOR = ('linear 784-128-784 (6.125x floor)', 0.06877)


def line(label, test, train=None, gap=None, branch=None, impr=None, params=None):
    m01 = test * S * S
    s = (f'{label:<34}{test:>9.5f}{m01:>10.6f}{10 * math.log10(1 / m01):>9.2f}')
    if train is not None:
        s += f'{train:>9.5f}{gap:>6.1f}%{branch:>8.3f}{impr:>8.2f}{params / 1e6:>8.2f}M'
    return s


def main():
    print(f'{"arm":<34}{"test std":>9}{"MSE[0,1]":>10}{"PSNR dB":>9}{"train":>9}{"gap":>7}'
          f'{"branch":>8}{"%/75":>8}{"params":>9}')
    for d, lab in ARMS:
        j = json.load(open(os.path.join(R, d, 'run.json')))
        s, h = j['summary'], j['hist'][-1]
        print(line(lab, s['test_mse'], s['train_mse'],
                   100 * (s['test_mse'] - s['train_mse']) / s['train_mse'],
                   s['branch_last'], s['still_improving_pct'], s['params']))
        nb = s['n_blocks']
        per = lambda k, f='.1f': '/'.join(  # noqa: E731
            format(h[f'lut/{k}_b{b}'], f) for b in range(nb) if f'lut/{k}_b{b}' in h)
        sc = per('score_sum')
        extra = (f', score sum {sc}, |in| {per("norm_in")} -> |out| {per("norm_out")}' if sc
                 else ', score sum / per-block norms not logged (probe added after this run)')
        print(f'{"":<34}gain {s["branch_first"]:.3f} -> {s["branch_last"]:.3f}, tau {s["tau_last"]:.3f}, '
              f'm_min {s["m_min_first"]:.3f} -> {s["m_min_last"]:.3f}, '
              f'flips {s["flips_last"]:.3f}{extra}')
    print(line(*FLOOR))


if __name__ == '__main__':
    main()
