"""Tables for the hierarchical-predictive-coding run against the matched baseline."""
import json
import math
import os

S = 0.3081
HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, 'runs_ae')
MARKS = (500, 1000, 2000, 3000, 5000, 7000, 9000, 10000)
ARMS = [('cmhl-L4-hpc-diag-s10000', 'HPC deep supervision'),
        ('cmhl-L4-base-diag-s10000', 'baseline (same seed)')]
FLOOR = 0.06877


def load(n):
    return json.load(open(os.path.join(R, n, 'run.json')))


def main():
    print(f'{"arm":<24}{"test std":>10}{"MSE[0,1]":>10}{"PSNR":>8}{"train":>9}{"gap":>7}'
          f'{"vs floor":>10}{"vs base":>9}')
    base = None
    for d, lab in ARMS:
        s = load(d)['summary']
        m01 = s['test_mse'] * S * S
        base = base or s['test_mse']
        print(f'{lab:<24}{s["test_mse"]:>10.5f}{m01:>10.6f}{10*math.log10(1/m01):>8.2f}'
              f'{s["train_mse"]:>9.5f}'
              f'{100*(s["test_mse"]-s["train_mse"])/s["train_mse"]:>6.1f}%'
              f'{100*(s["test_mse"]-FLOOR)/FLOOR:>+9.1f}%'
              f'{100*(s["test_mse"]-load(ARMS[1][0])["summary"]["test_mse"])/load(ARMS[1][0])["summary"]["test_mse"]:>+8.1f}%')
    m01 = FLOOR * S * S
    print(f'{"linear 784-128-784":<24}{FLOOR:>10.5f}{m01:>10.6f}{10*math.log10(1/m01):>8.2f}')

    h = load(ARMS[0][0])['hist']
    by = {r['step']: r for r in h}
    steps = [m for m in MARKS if m in by]
    nl = sum(1 for k in by[steps[0]] if k.startswith('hpc/level'))
    if nl:
        print('\nPER-LEVEL training MSE (each against the residual its predecessors left)')
        print(f'{"level":<8}' + ''.join(f'{m:>10}' for m in steps))
        for i in range(nl):
            print(f'{i:<8}' + ''.join(f'{by[m][f"hpc/level{i}"]:>10.5f}' for m in steps))

    for d, lab in ARMS:
        j = load(d)
        by = {r['step']: r for r in j['hist']}
        nb = j['summary']['n_blocks']
        steps = [m for m in MARKS if m in by]
        print(f'\n### {lab}')
        for key, name, fmt in [('util/pr_mean', 'participation ratio (max 256)', '10.1f'),
                               ('util/entropy_mean', 'usage entropy, nats (max 5.545)', '10.3f'),
                               ('util/dead_frac', 'dead-row fraction (batch 512)', '10.3f'),
                               ('util/top1_share', 'share of the most-used row', '10.3f'),
                               ('util/row_norm_mean', 'mean table row norm', '10.3f'),
                               ('util/common_ratio', '||mean row|| / mean ||row||', '10.3f')]:
            print(f'{name:<34}' + ''.join(f'{"b" + str(b):>10}' for b in range(nb)))
            print(f'{"  at step " + str(steps[-1]):<34}'
                  + ''.join(format(by[steps[-1]][f'{key}_b{b}'], fmt) for b in range(nb)))
        print(f'{"participation ratio over training":<34}'
              + ''.join(f'{m:>10}' for m in steps))
        for b in range(nb):
            print(f'{"  block " + str(b):<34}'
                  + ''.join(f'{by[m][f"util/pr_mean_b{b}"]:>10.1f}' for m in steps))
        print(f'{"gradient norms":<34}' + ''.join(f'{m:>10}' for m in steps))
        for g, n in (('grad/enc_weight', '  encoder'), ('grad/b0_compress', '  block-0 compress'),
                     ('grad/dec_weight', '  decoder'), ('grad/tables_b0', '  tables b0')):
            if g in by[steps[0]]:
                print(f'{n:<34}' + ''.join(f'{by[m][g]:>10.4f}' for m in steps))


if __name__ == '__main__':
    main()
