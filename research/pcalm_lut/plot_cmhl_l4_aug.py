"""Plots for the L=4 CompressionMHL arm: augmented vs un-augmented, 10000 steps each.

Colours are Okabe-Ito (colourblind-safe). Solid = augmented, dashed = un-augmented, throughout, so the
line style alone carries the run identity and colour is free to carry the block index. No dual axes.

A NOTE THE GAIN PLOT CARRIES IN ITS AXIS LABEL, because it would otherwise be read as a bound violation:
the ratio is ||block out|| / ||block in||, where "in" is the inter-block tensor BEFORE the pre-norm.
That is not the tensor the LUT reads -- the LUT reads the normalised version -- so a ratio above 1 is
not a norm being violated, it is the stack's stream growing between blocks.
"""
import json
import math
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, 'runs_ae')
OUT = os.path.join(R, 'plots_cmhl_l4_aug')
AUG = 'cmhl-L4-w128-tph128-din128-dout-1-prenorm-finalnorm-noresid-aug-s10000'
PLAIN = 'cmhl-L4-w128-tph128-din128-dout-1-prenorm-finalnorm-noresid-s10000'
FLOOR = 0.06877
# Okabe-Ito, minus the yellow (too low contrast on white)
BLOCK = ['#0072B2', '#D55E00', '#009E73', '#CC79A7']
TRAIN, TEST = '#0072B2', '#D55E00'
GRID = dict(color='#DDDDDD', linewidth=0.6)


def load(name):
    return json.load(open(os.path.join(R, name, 'run.json')))


def style(ax, title, xlab='training step', ylab=''):
    ax.set_title(title, fontsize=10.5, pad=8)
    ax.set_xlabel(xlab, fontsize=9)
    ax.set_ylabel(ylab, fontsize=9)
    ax.grid(True, **GRID)
    ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color('#999999')
    ax.tick_params(labelsize=8, color='#999999')


def series(h, key):
    return [r['step'] for r in h if key in r], [r[key] for r in h if key in r]


def save(fig, name):
    p = os.path.join(OUT, name)
    fig.savefig(p, dpi=170, bbox_inches='tight')
    plt.close(fig)
    print('wrote', p)


def main():
    os.makedirs(OUT, exist_ok=True)
    a, p = load(AUG), load(PLAIN)
    ha, hp = a['hist'], p['hist']
    nb = a['summary']['n_blocks']

    # 1 -- loss curves
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    for h, ls, tag in ((ha, '-', 'augmented'), (hp, '--', 'no augmentation')):
        for key, c, lab in (('eval/train_mse', TRAIN, 'train'), ('eval/test_mse', TEST, 'test')):
            x, y = series(h, key)
            ax.plot(x, y, ls, color=c, linewidth=1.6 if ls == '-' else 1.2, label=f'{lab}, {tag}')
    ax.axhline(FLOOR, color='#000000', linewidth=1.4, linestyle=':',
               label=f'linear 784-128-784 floor ({FLOOR:.5f})')
    ax.set_yscale('log')
    ax.set_xlim(0, 10000)
    style(ax, 'L=4 CompressionMHL: neither run approaches the linear floor\n'
              'solid = with augmentation, dashed = without', ylab='MSE, standardised (log scale)')
    ax.legend(fontsize=8, frameon=False, loc='upper right')
    # The late drift is real but far too small to see on a log axis spanning a decade, so the claim
    # gets its own linear-scale inset rather than being asserted over a plot that cannot show it.
    ins = ax.inset_axes((0.50, 0.12, 0.47, 0.33))
    for h, ls, tag, c in ((ha, '-', 'augmented', TEST), (hp, '--', 'no aug', '#7F7F7F')):
        x, y = series(h, 'eval/test_mse')
        w = [(xx, yy) for xx, yy in zip(x, y) if xx >= 6000]
        ins.plot([q[0] for q in w], [q[1] for q in w], ls, color=c, linewidth=1.3, label=tag)
    ins.set_title('test MSE, last 4000 steps, linear scale', fontsize=7.5, pad=3)
    ins.tick_params(labelsize=6.5, color='#999999')
    ins.grid(True, **GRID)
    ins.set_axisbelow(True)
    for s in ('top', 'right'):
        ins.spines[s].set_visible(False)
    ins.legend(fontsize=6.5, frameon=False, loc='upper right')
    save(fig, '1_loss_curves.png')

    # 2 -- train/test gap
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    for h, ls, tag, c in ((ha, '-', 'augmented', '#009E73'), (hp, '--', 'no augmentation', '#D55E00')):
        x = [r['step'] for r in h]
        y = [100 * (r['eval/test_mse'] - r['eval/train_mse']) / r['eval/train_mse'] for r in h]
        ax.plot(x, y, ls, color=c, linewidth=1.6, label=tag)
    ax.axhline(0, color='#999999', linewidth=0.8)
    ax.set_xlim(0, 10000)
    style(ax, 'Train/test gap: augmentation closes it almost completely (22.2% -> 1.3%)',
          ylab='100 x (test - train) / train, %')
    ax.legend(fontsize=9, frameon=False)
    save(fig, '2_train_test_gap.png')

    # 3 -- per-block gain
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    for b in range(nb):
        x, y = series(ha, f'branch/ratio_b{b}')
        ax.plot(x, y, '-', color=BLOCK[b], linewidth=1.6, label=f'block {b}, augmented')
    x, y = series(hp, 'branch/ratio_b0')
    ax.plot(x, y, '--', color=BLOCK[0], linewidth=1.4, label='block 0, NO augmentation')
    ax.axhline(1.0, color='#000000', linewidth=1.0, linestyle=':', label='gain = 1')
    ax.set_xlim(0, 10000)
    style(ax, 'Per-block gain: augmented block 0 plateaus near 2.4 from ~7000;\n'
              'un-augmented block 0 is still climbing at 4.16 when the run stops',
          ylab='out/in ratio, measured against the PRE-NORM inter-block tensor\n'
               '(not what the block reads; >1 is stream growth, not a violated bound)')
    ax.legend(fontsize=8, frameon=False, loc='upper left')
    save(fig, '3_per_block_gain.png')

    # 4 -- per-block stream norm
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    for h, ls, tag in ((ha, '-', 'augmented'), (hp, '--', 'no aug')):
        for b in range(nb):
            x, y = series(h, f'lut/norm_in_b{b}')
            ax.plot(x, y, ls, color=BLOCK[b], linewidth=1.5 if ls == '-' else 1.1,
                    label=f'block {b}, {tag}')
    ax.set_xlim(0, 10000)
    style(ax, 'Stream norm entering each block: it grows through the stack in both runs,\n'
              'and augmentation lowers the level without removing the growth',
          ylab='mean ||h|| entering the block')
    ax.legend(fontsize=7.5, frameon=False, loc='upper left', ncol=2)
    save(fig, '4_stream_norm.png')

    # 5 -- summed score and smallest margin, two panels sharing the x axis
    fig, axes = plt.subplots(2, 1, figsize=(8.2, 7.6), sharex=True)
    for h, ls, tag in ((ha, '-', 'augmented'), (hp, '--', 'no aug')):
        for b in range(nb):
            x, y = series(h, f'lut/score_sum_b{b}')
            axes[0].plot(x, y, ls, color=BLOCK[b], linewidth=1.5 if ls == '-' else 1.1,
                         label=f'block {b}, {tag}')
            x, y = series(h, f'lut/m_min_b{b}')
            axes[1].plot(x, y, ls, color=BLOCK[b], linewidth=1.5 if ls == '-' else 1.1,
                         label=f'block {b}, {tag}')
    axes[0].set_xlim(0, 10000)
    style(axes[0], 'Summed confidence score across tables (top) and smallest margin (bottom)\n'
                   'the score is proportional to ||h||, so it should track the stream norm', xlab='',
          ylab='sum of s_t over tables')
    style(axes[1], '', ylab='median smallest margin |m|')
    axes[0].legend(fontsize=7.5, frameon=False, loc='upper left', ncol=2)
    save(fig, '5_score_and_margin.png')

    print(f'\nfinal: augmented test {a["summary"]["test_mse"]:.5f}, '
          f'un-augmented {p["summary"]["test_mse"]:.5f}, floor {FLOOR:.5f}')
    print(f'PSNR: {10*math.log10(1/(a["summary"]["test_mse"]*0.3081**2)):.2f} dB vs '
          f'{10*math.log10(1/(p["summary"]["test_mse"]*0.3081**2)):.2f} dB')


if __name__ == '__main__':
    main()
