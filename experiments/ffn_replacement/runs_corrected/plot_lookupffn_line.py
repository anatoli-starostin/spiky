"""Summary figure for the LookupFFN line: the arms, the collapse, and the selectivity."""
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.path.expanduser('~/projects/spiky/src'))
from spiky.lutorch.fast_multi_head_lut import _confidence_score   # noqa: E402

RC = os.path.expanduser('~/projects/spiky/experiments/ffn_replacement/runs_corrected')
BASE = {500: 1.852331, 1000: 1.693385, 1500: 1.607591, 2000: 1.548721,
        2500: 1.501781, 3000: 1.467895}
BASE_FINAL, SD, VANILLA = 1.434572, 0.0096, 1.474749
ARM_A = {500: 1.987700, 1000: 1.868186, 1500: 1.804922, 2000: 1.753607,
         2500: 1.718596, 3000: 1.692834}

C_RED, C_BLUE, C_GREEN, C_ORANGE = '#a4443a', '#3b5b8c', '#2f6f4f', '#b5793a'


def evals_of(run):
    p = os.path.join(RC, run, 'train.log')
    if not os.path.exists(p):
        return {}
    out = {}
    for line in open(p):
        if line.startswith('[VAL] step '):
            s, b = line.split('step ')[1].split(': bpb=')
            out[int(s)] = float(b)
    return out


def final_of(run):
    p = os.path.join(RC, run, 'corrected_score.json')
    if not os.path.exists(p):
        return None
    j = json.load(open(p))
    for k in ('proxy_val_bpb', 'corrected_val_bpb', 'bpb_fixed'):
        if j.get(k) is not None:
            return j[k]
    return None


ARMS = [("arm A  bounded", 'exp_n_0178_A_fwdconf_seed1', C_RED),
        ("arm A' bounded_norm", 'exp_n_0180_Aprime_fwdconf_norm_seed1', C_BLUE),
        ("arm B  Light+bnorm", 'exp_n_0181_B_light_bnorm_seed1', C_ORANGE),
        ("arm C  bounded x12.61", 'exp_n_0182_C_bounded_gain_seed1', '#7a4f9c'),
        ("arm D  margin x2.99", 'exp_n_0183_D_margin_gain_seed1', '#3f8f8f')]

fig = plt.figure(figsize=(21.5, 9.6))
gs = fig.add_gridspec(2, 4, hspace=0.36, wspace=0.30)

# ---- 1. gap to baseline over training, full range and zoomed --------------------------
steps = sorted(BASE)
for panel, (ax, ylim, title) in enumerate([
        (fig.add_subplot(gs[0, 0]), None,
         'Arm A diverges monotonically\n(same score shape as arm C — only the scale differs)'),
        (fig.add_subplot(gs[0, 1]), (-0.012, 0.055),
         'Zoomed: at matched SCALE every gate\nsits on the baseline; only Light lags')]):
    ax.axhline(0, color='#333', lw=1.4)
    ax.axhspan(-SD, SD, color='#2f6f4f', alpha=.13,
               label=f'±{SD} seed sd' + ('' if panel else ' (nothing inside is a result)'))
    for label, run, col in ARMS:
        ev = evals_of(run)
        xs = [s for s in steps if s in ev]
        if xs:
            ax.plot(xs, [ev[s] - BASE[s] for s in xs], 'o-', lw=2, ms=5.5,
                    color=col, label=label)
    if ylim:
        ax.set_ylim(*ylim)
    ax.set_xlabel('step'), ax.set_ylabel('bpb gap to baseline S5')
    ax.set_title(title, fontsize=10.5)
    ax.grid(alpha=.25), ax.legend(fontsize=8, loc='upper left')
a = None

# ---- 2. final scores, as deltas (a bar from an arbitrary origin would exaggerate) -------
b = fig.add_subplot(gs[0, 2:])
names, vals, cols = ['baseline S5'], [BASE_FINAL], [C_GREEN]
for label, run, col in ARMS:
    f = final_of(run)
    if f is not None:
        names.append(label)
        vals.append(f)
        cols.append(col)
names.append('vanilla dense S0'), vals.append(VANILLA), cols.append('#888')
y = np.arange(len(names))
b.axvline(0, color='#333', ls='--', lw=1.3)
b.axvspan(-SD, SD, color=C_GREEN, alpha=.15, label=f'±{SD} seed sd')
for i, (v, col) in enumerate(zip(vals, cols)):
    b.plot(v - BASE_FINAL, i, 'o', color=col, ms=11)
    b.annotate(f'{v:.4f}   ({v - BASE_FINAL:+.4f})', (v - BASE_FINAL, i),
               xytext=(11, 0), textcoords='offset points', va='center', fontsize=8)
b.set_yticks(y), b.set_yticklabels(names, fontsize=8.5)
b.set_ylim(len(names) - .4, -.6)
b.set_xlim(-0.013, (VANILLA - BASE_FINAL) * 2.1)
b.set_xlabel('final proxy val bpb  −  baseline S5')
b.set_title('Final score, as a delta\n(a bar from an arbitrary origin would exaggerate these)',
            fontsize=10.5)
b.grid(axis='x', alpha=.25), b.legend(fontsize=8, loc='lower right')

# ---- 3. the nap collapse ---------------------------------------------------------------
c = fig.add_subplot(gs[1, 0])
m = torch.load('/tmp/margins_anchor.pt')['d'].abs().reshape(-1)
samp = m[torch.randint(0, m.numel(), (120000, 16))]
naps = [1, 2, 4, 6, 8, 12, 16]
for form, col, lab in (('bounded', C_RED, 'bounded (product)'),
                       ('bounded_norm', C_BLUE, 'bounded_norm (geo mean)'),
                       ('margin', '#888', 'margin')):
    c.plot(naps, [_confidence_score(samp[:, :n], form).mean().item() for n in naps],
           'o-', color=col, lw=2, ms=5, label=lab)
c.axvline(8, color='#333', ls=':', lw=1.2)
c.annotate('our nap', (8, 0.9), fontsize=8, rotation=90, va='top', ha='right')
c.set_yscale('log'), c.set_xlabel('nap (anchor pairs per table)')
c.set_ylabel('mean score'), c.grid(alpha=.25), c.legend(fontsize=8)
c.set_title('Why bounded fails: a product over nap\nfactors collapses; the geometric mean '
            'does not', fontsize=11)

# ---- 4. selectivity --------------------------------------------------------------------
d = fig.add_subplot(gs[1, 1])
dd = torch.load('/tmp/margins_anchor.pt')['d']
sets = [('bounded\n(arm A)', _confidence_score(dd, 'bounded'), C_RED),
        ("bounded_norm\n(A', B)  CV 0.07", _confidence_score(dd, 'bounded_norm'), C_BLUE),
        ('bounded x12.61\n(C)  CV 0.58', _confidence_score(dd, 'bounded') * 12.61, '#7a4f9c'),
        ('margin x2.99\n(D)  CV 0.95', _confidence_score(dd, 'margin') * 2.99, '#3f8f8f')]
for i, (name, s, col) in enumerate(sets):
    q = torch.quantile(s[torch.randperm(s.numel())[:200000]],
                       torch.tensor([0.05, .25, .5, .75, 0.95]))
    d.plot([i, i], [q[0], q[4]], color=col, lw=1.3)
    d.plot([i, i], [q[1], q[3]], color=col, lw=13, solid_capstyle='butt')
    d.plot(i, q[2], 'o', color='white', ms=6, zorder=3)
    d.annotate(f'p75/p25\n{(q[3]/q[1]).item():.2f}x', (i, q[3]), xytext=(13, 0),
               textcoords='offset points', fontsize=8, va='center')
d.axhline(1.0, color='#333', ls='--', lw=1.2)
d.set_yscale('log'), d.set_xticks(range(len(sets)))
d.set_xticklabels([s[0] for s in sets], fontsize=8)
d.set_ylabel('confidence score (log)')
d.set_title("Three arms, one scale (0.6838),\nthree spreads — the selectivity axis",
            fontsize=11)
d.grid(axis='y', alpha=.25)

# ---- 5. margin growth ------------------------------------------------------------------
e = fig.add_subplot(gs[1, 2])
tr = torch.load('/tmp/margins_anchor_trained.pt')['d']
rows = [('at init', dd, '#bbb'), ('after 4,000 steps', tr, C_GREEN)]
w = 0.35
for i, (name, t, col) in enumerate(rows):
    for j, form in enumerate(('bounded', 'bounded_norm', 'margin')):
        v = _confidence_score(t, form).mean().item()
        e.bar(j + (i - .5) * w, v, w, color=col,
              label=name if j == 0 else None)
        e.annotate(f'{v:.3f}', (j + (i - .5) * w, v), xytext=(0, 3),
                   textcoords='offset points', ha='center', fontsize=7.5)
e.axhline(1.0, color='#333', ls='--', lw=1.2)
e.set_xticks(range(3)), e.set_xticklabels(['bounded', 'bounded_norm', 'margin'], fontsize=8.5)
e.set_ylabel('mean score'), e.legend(fontsize=8), e.grid(axis='y', alpha=.25)
e.set_title('Margins widen 1.69x with training, but\nbounded only reaches 0.130 — never heals',
            fontsize=11)

# ---- 6. the selectivity trend, as a small overlay axes -------------------------------
ax6 = fig.add_subplot(gs[1, 3])
cvs = [0.067, 0.584, 0.946]
gaps = [f - BASE_FINAL for f in
        (final_of('exp_n_0180_Aprime_fwdconf_norm_seed1'),
         final_of('exp_n_0182_C_bounded_gain_seed1'),
         final_of('exp_n_0183_D_margin_gain_seed1'))]
ax6.axhspan(-SD, SD, color=C_GREEN, alpha=.18)
ax6.axhline(0, color='#333', lw=1)
ax6.plot(cvs, gaps, 'o-', color='#7a4f9c', lw=2, ms=7)
for cv, g, n in zip(cvs, gaps, ["A'", 'C', 'D']):
    ax6.annotate(n, (cv, g), xytext=(0, 7), textcoords='offset points',
                 ha='center', fontsize=8)
ax6.set_xlabel('within-token CV (how much the gate discriminates)', fontsize=8.5)
ax6.set_ylabel('final bpb gap to baseline S5', fontsize=8.5)
ax6.set_title('More selective = better, monotonically —\nbut the whole spread (0.0087) fits '
              'inside\none seed sd, so this is a lead, not a result', fontsize=10)
ax6.grid(alpha=.25)

fig.suptitle('The LookupFFN confidence gate on our LUT FFN (#112) — anchor sizing, '
             '4k proxy budget, seed 1', fontsize=13.5, y=0.985)
plt.savefig('/tmp/lookupffn_line.png', dpi=125, bbox_inches='tight')
print('wrote /tmp/lookupffn_line.png')
