"""Checkpoint tables for the 60K hierarchical-predictive-coding run."""
import json
import math
import os

S = 0.3081
HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, 'runs_ae')
RUN = 'cmhl-L4-hpc-diag-s60000'
MARKS = (10000, 20000, 30000, 40000, 50000, 60000)
FLOOR, HPC10K, BASE10K = 0.06877, 0.09102, 0.14576


def main():
    j = json.load(open(os.path.join(R, RUN, 'run.json')))
    H = {r['step']: r for r in j['hist']}
    nb = j['summary']['n_blocks']
    nl = sum(1 for k in j['hist'][-1] if k.startswith('hpc/level'))
    marks = [m for m in MARKS if m in H]

    print('HEADLINE -- test MSE of the SUMMED reconstruction')
    print(f'{"ckpt":>7}{"test std":>11}{"MSE[0,1]":>11}{"PSNR":>8}{"train":>10}{"gap":>8}'
          f'{"vs floor":>10}')
    for m in marks:
        r = H[m]
        t, tr = r['eval/test_mse'], r['eval/train_mse']
        m01 = t * S * S
        print(f'{m:>7}{t:>11.5f}{m01:>11.6f}{10 * math.log10(1 / m01):>8.2f}{tr:>10.5f}'
              f'{100 * (t - tr) / tr:>7.1f}%{100 * (t - FLOOR) / FLOOR:>+9.1f}%')
    print(f'{"":>7}{"reference points:":>11}  HPC at 10K {HPC10K:.5f} | baseline at 10K '
          f'{BASE10K:.5f} | linear floor {FLOOR:.5f}')

    print('\nPER-LEVEL train MSE (level i is scored against the residual levels < i left)')
    print(f'{"step":>7}' + ''.join(f'{"level " + str(i):>11}' for i in range(nl)))
    for m in marks:
        print(f'{m:>7}' + ''.join(f'{H[m][f"hpc/level{i}"]:>11.5f}' for i in range(nl)))

    for key, name, fmt in [('util/pr_mean', 'participation ratio, of 256', '11.2f'),
                           ('util/entropy_mean', 'usage entropy, nats (max 5.545)', '11.3f'),
                           ('util/dead_frac', 'dead-row fraction (512-sample batch)', '11.3f'),
                           ('util/top1_share', 'share of the most-used row', '11.3f'),
                           ('util/row_norm_mean', 'mean table row norm', '11.3f'),
                           ('util/common_ratio', '||mean row|| / mean ||row||', '11.3f'),
                           ('norm/t5_block_out_mean', 'readout norm after the block', '11.1f'),
                           ('branch/ratio', 'block gain (out/in, pre-norm input)', '11.2f')]:
        print(f'\n{name}')
        print(f'{"step":>7}' + ''.join(f'{"b" + str(b):>11}' for b in range(nb)))
        for m in marks:
            print(f'{m:>7}' + ''.join(format(H[m][f'{key}_b{b}'], fmt) for b in range(nb)))

    ks = [k for k in ('grad/enc_weight', 'grad/b0_compress', 'grad/dec_weight', 'grad/tables_b0')
          if k in H[marks[-1]]]
    print('\ngradient norms')
    print(f'{"step":>7}' + ''.join(f'{k.split("/")[1]:>18}' for k in ks))
    for m in marks:
        print(f'{m:>7}' + ''.join(f'{H[m][k]:>18.5f}' for k in ks))

    s = j['summary']
    print(f'\nwall {s["wall_s"]:.0f} s, {s["params"]/1e6:.2f}M params, '
          f'still improving {s["still_improving_pct"]:.2f}%/75 at the stop')


if __name__ == '__main__':
    main()
