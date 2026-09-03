#!/usr/bin/env python3
"""Learning-curve comparison of the three super-long 144k runs (val_bpb vs step):
vanilla dense (0157), CompressionMHL nap7/tph128 (0158), CompressionMHL nap8/tph64 (0159).
Writes two PNGs next to this script. Portable: paths derived from __file__."""
import csv
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.dirname(HERE)   # experiments/hyperplane_ffn
RUNS = [('exp_n_0157_long144k_vanilla', 'vanilla dense (0157)', 'tab:blue'),
        ('exp_n_0158_long144k_from_0127', '0158  nap7/tph128', 'tab:orange'),
        ('exp_n_0159_long144k_from_0128', '0159  nap8/tph64', 'tab:green')]


def load(d):
    steps, bpb = [], []
    with open(os.path.join(R, d, 'metrics.csv')) as f:
        for row in csv.DictReader(f):
            steps.append(int(row['step']))
            bpb.append(float(row['val_bpb']))
    return steps, bpb


def final(d, bpb):
    try:
        return json.load(open(os.path.join(R, d, 'summary.json')))['final_val_bpb']
    except Exception:
        return bpb[-1]


data = [(lab, col, *load(d), final(d, load(d)[1])) for d, lab, col in RUNS]

# ---- Figure A: standard scale (full) + last-20k zoom panel ----
figA, (a1, a2) = plt.subplots(1, 2, figsize=(15, 6))
for lab, col, s, b, fv in data:
    a1.plot(s, b, color=col, lw=1.7, label=f'{lab}  final {fv:.5f}')
    a1.axhline(fv, color=col, ls=':', lw=0.9, alpha=0.5)
a1.set(xlabel='training step', ylabel='validation bpb',
       title='Val BPB vs step (0 -> 144k), standard scale', xlim=(0, 144000))
a1.grid(True, alpha=0.3)
a1.legend(fontsize=9, loc='upper right')
for lab, col, s, b, fv in data:
    m = [i for i, st in enumerate(s) if st >= 124000]
    a2.plot([s[i] for i in m], [b[i] for i in m], color=col, lw=1.7, label=lab)
    a2.axhline(fv, color=col, ls=':', lw=0.9, alpha=0.6)
a2.set(xlabel='training step', ylabel='validation bpb',
       title='Zoom: last 20k steps (124k -> 144k)', xlim=(124000, 144000))
a2.grid(True, alpha=0.3)
a2.legend(fontsize=8, loc='upper right')
figA.suptitle('Three 144k runs — val BPB (finals: vanilla 1.147985 · 0158 1.148013 · 0159 1.148886)',
              fontsize=11)
figA.tight_layout(rect=[0, 0, 1, 0.96])
outA = os.path.join(HERE, 'val_bpb_144k_three_standard.png')
figA.savefig(outA, dpi=135)

# ---- Figure B: log-log, tail zoomed so ~1.146-1.16 is readable ----
def smooth(y, w=9):
    """Light CENTERED moving average (no lag; shrinking window at the edges)."""
    h = w // 2
    return [sum(y[max(0, i - h):min(len(y), i + h + 1)]) /
            len(y[max(0, i - h):min(len(y), i + h + 1)]) for i in range(len(y))]


# Last-HALF zoom (step 72k..144k), LINEAR x + LINEAR y (over a 2x span log-x adds nothing;
# linear reads cleaner). y-limits FIT tightly to the smoothed data in this window so the
# final ~0.001 separation occupies a real fraction of the height.
XLO_W, XHI_W = 72000, 144000
figB, ax = plt.subplots(figsize=(14, 6))
series, allvals = [], []
for lab, col, s, b, fv in data:
    idx = [i for i, st in enumerate(s) if XLO_W <= st <= XHI_W]
    sx = [s[i] for i in idx]
    by = [b[i] for i in idx]
    bs = smooth(by, 9)
    series.append((lab, col, sx, by, bs, fv))
    allvals.extend(bs)
ymin, ymax = min(allvals), max(allvals)
pad = 0.03 * (ymax - ymin)
YLO, YHI = ymin - pad, ymax + pad
for lab, col, sx, by, bs, fv in series:
    ax.plot(sx, by, color=col, lw=0.7, alpha=0.20)                        # raw, faint
    ax.plot(sx, bs, color=col, lw=2.2, label=f'{lab}  final {fv:.5f}')    # smoothed, main
    ax.axhline(fv, color=col, ls=':', lw=0.9, alpha=0.55)
ax.set_xlim(XLO_W, XHI_W)
ax.set_ylim(YLO, YHI)
ax.set_xlabel('training step (72k -> 144k, linear)')
ax.set_ylabel('validation bpb (linear)')
ax.set_title('Val BPB, last half of training (step 72k->144k), lightly smoothed (raw faint behind)\n'
             'three 144k runs — 0159 nap8/tph64 (green) runs highest; vanilla (blue) & 0158 nap7/tph128 (orange) track lower')
ax.grid(True, which='both', alpha=0.3)
ax.legend(fontsize=9, loc='upper right')
figB.tight_layout()
outB = os.path.join(HERE, 'val_bpb_144k_three_loglog.png')
figB.savefig(outB, dpi=135)
print(f'loglog(semilogx) y-limits: [{YLO:.5f}, {YHI:.5f}] (data min {ymin:.5f}, max {ymax:.5f})')

print(f'wrote {outA}')
print(f'wrote {outB}')
for lab, col, s, b, fv in data:
    print(f'{lab}: {len(s)} pts, final {fv:.5f}, last_step {s[-1]}')
