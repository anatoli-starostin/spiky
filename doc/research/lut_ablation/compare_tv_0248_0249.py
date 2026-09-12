"""exp_g_0248 (learned_margin, g frozen) vs exp_g_0249 (the same + Hamming-1 cell TV, weight 10). Read-only.

    python compare_tv_0248_0249.py [--png out.png]

1. Per-layer beta / gamma trajectories of both runs (metrics.csv): selected steps, final difference, late drift.
2. The TV penalty over training in exp_g_0249 (metrics.csv lut_tv, lut_tv_L*), and exp_g_0248's own value at its
   saved checkpoints (it did not log it), so "did TV come down" has a no-TV reference.
3. Cell-to-cell differences on the trained tables of both runs, per layer: mean ||v_c - v_c'||^2 over Hamming-1
   pairs, mean ||v_c||^2, and their ratio (2.0 = neighbouring cells as different as independent random cells).
"""
import csv
import os
import sys

import torch

RC = os.path.expanduser('~/projects/spiky/experiments/ffn_replacement/runs_corrected')
R248 = os.path.join(RC, 'exp_g_0248_B16k_light_learnedmargin_frozeng_tph128_seed1')
R249 = os.path.join(RC, 'exp_g_0249_B16k_light_learnedmargin_frozeng_tv10_tph128_seed1')
L = 6
SHOW = (1000, 2000, 4000, 8000, 12000, 14000, 16000)


def rows(rd):
    return {int(r['step']): r for r in csv.DictReader(open(os.path.join(rd, 'metrics.csv')))}


a, b = rows(R248), rows(R249)
steps = sorted(set(a) & set(b))
for k in ('gamma', 'beta'):
    print('=' * 110)
    print(f'{k} per layer: 0248 | 0249')
    for s in SHOW:
        if s in a and s in b:
            print(f'   {s:>6} ' + '  '.join(f'L{i} {float(a[s][f"lm_{k}_L{i}"]):.4f}|{float(b[s][f"lm_{k}_L{i}"]):.4f}'
                                         for i in range(L)))
    f = steps[-1]
    print('   final 0249-0248 ' + ' '.join(f'{float(b[f][f"lm_{k}_L{i}"]) - float(a[f][f"lm_{k}_L{i}"]):+.4f}'
                                        for i in range(L)))
    if 14000 in b and 16000 in b:
        print('   drift 14K->16K 0249 ' + ' '.join(f'{float(b[16000][f"lm_{k}_L{i}"]) - float(b[14000][f"lm_{k}_L{i}"]):+.4f}'
                                              for i in range(L)))
        print('   drift 14K->16K 0248 ' + ' '.join(f'{float(a[16000][f"lm_{k}_L{i}"]) - float(a[14000][f"lm_{k}_L{i}"]):+.4f}'
                                              for i in range(L)))

print('=' * 110)
print('TV penalty in exp_g_0249 (mean over layers, and per layer) at eval steps')
for s in sorted(b):
    if s in (500, 1000, 2000, 4000, 8000, 12000, 16000):
        print(f'   {s:>6} lut_tv {float(b[s]["lut_tv"]):.4e} | ' + ' '.join(f'{float(b[s][f"lut_tv_L{i}"]):.3e}' for i in range(L)))


def cells(path):
    sd = torch.load(path, map_location='cpu')
    out = []
    for i in range(L):
        t = sd[f'blocks.{i}.ffn.lut_light.tables'].double()             # [n_tables, 2^nap, D]
        nt, C, Dd = t.shape
        nap = C.bit_length() - 1
        h = t.view(nt, *([2] * nap), Dd)
        diff2 = sum((h.diff(dim=ax) ** 2).sum() for ax in range(1, nap + 1)) / (nt * nap * (1 << (nap - 1)))
        cell2 = (t ** 2).sum(-1).mean()
        out.append((diff2.item(), cell2.item()))
    return out


print('=' * 110)
print("0248's TV penalty at its checkpoints (no-TV reference; per layer mean ||dv||^2, the quantity 0249 penalises)")
for step, fn in ((4000, 'checkpoint_step4000.pt'), (8000, 'checkpoint_step8000.pt'), (12000, 'checkpoint_step12000.pt'),
                 (16000, 'checkpoint.pt')):
    c = cells(os.path.join(R248, fn))
    print(f'   0248 @ {step:>5}: lut_tv {sum(x[0] for x in c) / L:.4e} | ' + ' '.join(f'{x[0]:.3e}' for x in c))
print('=' * 110)
print('cell-to-cell differences on the TRAINED tables: mean ||v_c - v_c\'||^2 (Hamming-1) | mean ||v_c||^2 | ratio')
c48, c49 = cells(os.path.join(R248, 'checkpoint.pt')), cells(os.path.join(R249, 'checkpoint.pt'))
for i in range(L):
    print(f'   L{i}: 0248 {c48[i][0]:.4e} | {c48[i][1]:.4e} | {c48[i][0] / c48[i][1]:.3f}    '
          f'0249 {c49[i][0]:.4e} | {c49[i][1]:.4e} | {c49[i][0] / c49[i][1]:.3f}    '
          f'diff2 ratio 0249/0248 {c49[i][0] / c48[i][0]:.3f}')

if '--png' in sys.argv:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out = sys.argv[sys.argv.index('--png') + 1]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for ax, k in zip(axes[:2], ('gamma', 'beta')):
        for i in range(L):
            ax.plot(steps, [float(a[s][f'lm_{k}_L{i}']) for s in steps], color=colors[i], lw=1.5, label=f'L{i} 0248')
            ax.plot(steps, [float(b[s][f'lm_{k}_L{i}']) for s in steps], color=colors[i], lw=1.5, ls='--', label=f'L{i} 0249')
        ax.set(xlabel='step', title=f'learned {k}: solid 0248, dashed 0249 (TV 10)')
        ax.grid(True, alpha=.3)
    axes[0].legend(fontsize=6, ncol=2)
    bs = sorted(b)
    for i in range(L):
        axes[2].plot(bs, [float(b[s][f'lut_tv_L{i}']) for s in bs], color=colors[i], lw=1.5, label=f'L{i}')
    axes[2].set(xlabel='step', ylabel='mean ||v_c - v_c\'||^2', title='TV penalty per layer (exp_g_0249)', yscale='log')
    axes[2].grid(True, alpha=.3)
    axes[2].legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    print('wrote', out)
