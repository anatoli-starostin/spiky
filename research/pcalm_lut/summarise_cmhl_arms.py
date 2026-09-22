"""The CompressionMHL autoencoder ledger: every arm, on every scale the project uses.

Standardised MSE is what the loss minimises; [0,1] pixel MSE and PSNR are what the comparison set uses.
PSNR is unclamped, matching the objective. Per-block diagnostics come from the 25-step probe and are
shown at the LAST probe: gain = ||block out|| / ||block in||, the summed confidence score across tables
(the quantity that grows with ||h|| under confidence_form='margin'), the median smallest margin, and tau.
Margins and scores are measured on the tensor the LUT actually addresses -- after the pre-norm and after
the compress projection -- not on the block input.
"""
import json
import math
import os

S = 0.3081
HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, 'runs_ae')
#        directory                                                 label            L  din  dout res fn
ARMS = [('cmhl-L4-w128-tph128-din128-dout-1-prenorm-finalnorm-noresid', 'L4 SPEC no resid', 4, 128, -1, 'no', 'yes'),
        ('cmhl-L4-w128-tph128-din128-dout-1-prenorm-finalnorm-resid', 'L4 residual', 4, 128, -1, 'yes', 'yes'),
        ('cmhl-L2-w128-tph128-din128-dout-1-prenorm-noresid', 'L2 no resid', 2, 128, -1, 'no', 'no'),
        ('cmhl-L2-w128-tph128-din128-dout-1-prenorm-resid', 'L2 residual', 2, 128, -1, 'yes', 'no'),
        ('cmhl-L2-w128-tph128-din128-dout128-prenorm-noresid', 'L2 no resid', 2, 128, 128, 'no', 'no'),
        ('cmhl-L2-w128-tph128-din128-dout128-prenorm-resid', 'L2 residual', 2, 128, 128, 'yes', 'no'),
        ('cmhl-L2-w128-tph128-dout-1-prenorm-noresid', 'L2 no resid', 2, -1, -1, 'no', 'no'),
        ('cmhl-L2-w128-tph128-dout-1-prenorm-resid', 'L2 residual', 2, -1, -1, 'yes', 'no'),
        ('cmhl-L2-w128-tph128-dout128-prenorm-noresid', 'L2 no resid', 2, -1, 128, 'no', 'no'),
        ('cmhl-L2-w128-tph128-dout128-prenorm-resid', 'L2 residual', 2, -1, 128, 'yes', 'no')]
FLOOR = 0.06877

HDR = (f'{"arm":<18}{"L":>2}{"din":>5}{"dout":>5}{"res":>5}{"fnorm":>6}{"test std":>10}'
       f'{"MSE[0,1]":>10}{"PSNR":>7}{"gap":>7}{"gain":>7}{"%/75":>7}{"params":>9}')


def main():
    print(HDR)
    print('-' * len(HDR))
    for d, lab, L, din, dout, res, fn in ARMS:
        p = os.path.join(R, d, 'run.json')
        if not os.path.exists(p):
            print(f'{lab:<18}{L:>2}{din:>5}{dout:>5}{res:>5}{fn:>6}   (not run)')
            continue
        j = json.load(open(p))
        s, h = j['summary'], j['hist'][-1]
        m01 = s['test_mse'] * S * S
        print(f'{lab:<18}{L:>2}{din:>5}{dout:>5}{res:>5}{fn:>6}{s["test_mse"]:>10.5f}{m01:>10.6f}'
              f'{10 * math.log10(1 / m01):>7.2f}'
              f'{100 * (s["test_mse"] - s["train_mse"]) / s["train_mse"]:>6.1f}%'
              f'{s["branch_last"]:>7.3f}{s["still_improving_pct"]:>7.2f}{s["params"] / 1e6:>8.2f}M')
        nb = s['n_blocks']

        def per(k, f='7.3f'):
            ks = [f'lut/{k}_b{b}' for b in range(nb)]
            return ' '.join(format(h[x], f) for x in ks) if all(x in h for x in ks) else None

        g = ' '.join(f'{h[f"branch/ratio_b{b}"]:7.3f}' for b in range(nb))
        print(f'{"":<18}per-block gain  {g}')
        for key, fmt, name in [('norm_in', '7.2f', '|in|  '), ('norm_out', '7.2f', '|out| '),
                               ('score_sum', '7.1f', 'score '), ('m_min', '7.3f', 'm_min '),
                               ('tau', '7.3f', 'tau   ')]:
            v = per(key, fmt)
            if v:
                print(f'{"":<18}per-block {name}{v}')
    m01 = FLOOR * S * S
    print('-' * len(HDR))
    print(f'{"linear 784-128-784":<18}{"":>2}{"":>5}{"":>5}{"":>5}{"":>6}{FLOOR:>10.5f}{m01:>10.6f}'
          f'{10 * math.log10(1 / m01):>7.2f}{"":>7}{"":>7}{"":>7}{0.202:>8.2f}M')


if __name__ == '__main__':
    main()
