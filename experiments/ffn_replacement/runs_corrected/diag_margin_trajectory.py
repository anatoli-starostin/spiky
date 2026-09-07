"""Did exp_g_0194 fight a mis-set frozen tau all through training?

THE HYPOTHESIS. tau was frozen at Delta_m measured on a FULLY TRAINED exp_g_0193. Early in
training the anchor margins are smaller, so 2m/tau starts below 1 and w1 = sigmoid(-2m/tau)
starts above the intended sigmoid(-1) = 0.2689 -- over-blended, near-even mixing of winner
and runner-up -- and only settles as margins grow into tau. If so, the deficit vs 0193 was
an artefact of the init and should close IN LOCKSTEP with 2m/tau climbing toward 1.

THE COMPETING, MUNDANE EXPLANATION. The blend imposes a cost early and the model simply
adapts to it -- which closes the gap regardless of what tau was set to, with no particular
relationship to 2m/tau.

WHY exp_g_0193 IS MEASURED TOO, and it is the point of the design. Both runs saved
checkpoints at the SAME steps (4k/8k/12k/final) and share seed 1, so 0193 gives the margin
trajectory of an otherwise identical model with NO blend at all. If 0194's margins track
0193's, then the blend is not reshaping the routing geometry and 2m/tau is simply reading
off a margin growth that would have happened anyway -- which would make any correlation with
the deficit uninformative.

Same tokens for every checkpoint: one fixed slab drawn through the corrected eval protocol
(bs48, leading 12 rows skipped), so nothing varies between measurements except the weights.

    python diag_margin_trajectory.py [--out FIG] [--json OUT]
"""
import argparse
import csv
import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', 'tools')))
NANOCHAT_ROOT = os.environ.get('NANOCHAT_ROOT', os.path.expanduser('~/projects/nanochat'))
if NANOCHAT_ROOT not in sys.path:
    sys.path.insert(0, NANOCHAT_ROOT)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                                    # noqa: E402
from nanochat.common import get_base_dir                           # noqa: E402
from nanochat.tokenizer import RustBPETokenizer                    # noqa: E402
from model_build import build_model                                # noqa: E402
from fixed_eval import EVAL_BATCH_SIZE, EVAL_SKIP_ROWS             # noqa: E402
from spiky.lutorch.light_multi_head_lut import LightMultiHeadLUT   # noqa: E402

R94 = os.path.join(HERE, 'exp_g_0194_B16k_light_margin_blend_n2_tau_auto_seed1')
R93 = os.path.join(HERE, 'exp_g_0193_B16k_light_margin_tph128_noznorm_seed1')
TAU = [0.03309, 0.07241, 0.07801, 0.08158, 0.09190, 0.10786]   # 0194's frozen per-layer tau
STEPS = [4000, 8000, 12000, 16000]
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'

ap = argparse.ArgumentParser()
ap.add_argument('--out', default=os.path.join(HERE, '..', 'figs',
                                              'margin_trajectory_0194.png'))
ap.add_argument('--json', default=None)
ap.add_argument('--rows', type=int, default=16)      # 16 x 512 = 8,192 tokens
a = ap.parse_args()

tok = RustBPETokenizer.from_directory(os.path.join(get_base_dir(), 'tokenizer'))
cfg94 = json.load(open(os.path.join(R94, 'config.json')))

# ---- ONE fixed token slab, drawn exactly as the corrected eval protocol draws ------------
from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit  # noqa: E402
it = iter(tokenizing_distributed_data_loader_bos_bestfit(
    tok, EVAL_BATCH_SIZE, cfg94['seq_len'], split='val', device=DEV))
x0, _ = next(it)
IDS = x0[EVAL_SKIP_ROWS:EVAL_SKIP_ROWS + a.rows].clone()    # skip the 12 anomalous rows
print(f'fixed slab: {tuple(IDS.shape)} = {IDS.numel():,} tokens, rows '
      f'[{EVAL_SKIP_ROWS}, {EVAL_SKIP_ROWS + a.rows}) of the corrected eval window, '
      f'IDENTICAL for every checkpoint')


def margins(run_dir, step):
    ck = (os.path.join(run_dir, 'checkpoint.pt') if step == 16000
          else os.path.join(run_dir, f'checkpoint_step{step}.pt'))
    if not os.path.exists(ck):
        return None
    cfg = json.load(open(os.path.join(run_dir, 'config.json')))
    model = build_model(cfg, tok.get_vocab_size(), device=DEV)
    model.load_state_dict(torch.load(ck, map_location=DEV), strict=False)
    model.eval()
    got = []

    def hook(mod, inp, out, _s=got):
        x = inp[0]
        H, T, K = mod.n_heads, mod.tables_per_head, mod.n_anchor_pairs
        B = x.shape[0]
        ia = mod.anchor_a.reshape(1, H, T * K).expand(B, H, T * K)
        ib = mod.anchor_b.reshape(1, H, T * K).expand(B, H, T * K)
        d = (torch.gather(x, 2, ia) - torch.gather(x, 2, ib)).view(B, H, T, K).abs()
        _s.append(d.detach().float().reshape(-1, K).min(dim=-1).values.cpu())

    hs = [m.register_forward_hook(hook) for m in model.modules()
          if isinstance(m, LightMultiHeadLUT)]
    with torch.no_grad():
        model(IDS)
    for h in hs:
        h.remove()
    del model
    torch.cuda.empty_cache()
    return got                                       # list of [N] m_(1) per layer


res = {'0194': {}, '0193': {}}
for tag, rd in (('0194', R94), ('0193', R93)):
    for s in STEPS:
        m = margins(rd, s)
        if m is None:
            print(f'{tag} step {s}: no checkpoint, skipped')
            continue
        res[tag][s] = [{'median': float(v.median()), 'mean': float(v.mean()),
                        'p25': float(v.quantile(.25)), 'p75': float(v.quantile(.75))}
                       for v in m]
        print(f'{tag} step {s:>5}: median m_(1) by layer = '
              + ' '.join(f'{d["median"]:.5f}' for d in res[tag][s]))

# ---- the deficit, from the matched-step table -------------------------------------------
def curve(d):
    return {int(r['step']): float(r['val_bpb'])
            for r in csv.DictReader(open(os.path.join(d, 'metrics.csv')))
            if r.get('val_bpb')}


c94, c93 = curve(R94), curve(R93)
deficit = {s: c94[s] - c93[s] for s in sorted(c94) if s in c93}

print(f'\n{"step":>6} {"deficit":>10} | ' +
      ' '.join(f'{"L" + str(i):>16}' for i in range(6)))
print(f'{"":>6} {"":>10} | ' + ' '.join(f'{"2m/tau  w1":>16}' for _ in range(6)))
rows = []
for s in STEPS:
    if s not in res['0194']:
        continue
    line = f'{s:>6} {deficit.get(s, float("nan")):>+10.6f} | '
    r2, rw = [], []
    for i in range(6):
        m = res['0194'][s][i]['median']
        ratio = 2 * m / TAU[i]
        w1 = 1.0 / (1.0 + torch.tensor(ratio).exp().item())
        r2.append(ratio); rw.append(w1)
        line += f'{ratio:>7.3f} {w1:>8.4f} '
    rows.append((s, deficit.get(s), r2, rw))
    print(line)

print(f'\ntarget: 2m/tau = 1.000, w1 = 0.2689')

# ---- the discriminating test -------------------------------------------------------------
print('\n=== DISCRIMINATING TEST ===')
pk = max(deficit, key=lambda s: deficit[s])
print(f'deficit PEAKS at step {pk} ({deficit[pk]:+.6f})')
print(f'earliest checkpoint available: {min(res["0194"])}')
if pk < min(res['0194']):
    print(f'*** The peak is BEFORE the earliest checkpoint, so 2m/tau AT THE PEAK cannot be')
    print(f'    measured. The timing test is UNANSWERABLE with the checkpoints that exist.')

xs = [r[0] for r in rows]
import statistics
for i in range(6):
    rr = [r[2][i] for r in rows]
    dd = [r[1] for r in rows]
    if len(rr) > 2:
        mr, md = statistics.mean(rr), statistics.mean(dd)
        num = sum((x - mr) * (y - md) for x, y in zip(rr, dd))
        den = (sum((x - mr) ** 2 for x in rr) * sum((y - md) ** 2 for y in dd)) ** .5
        print(f'  L{i}: corr(2m/tau, deficit) over steps {xs} = {num/den:+.4f}')
print('  NOTE: both series are monotone in training step, so this correlation is close to')
print('  tautological -- ANY monotone quantity would score near -1. It is reported because')
print('  it was asked for, not because it discriminates.')

print('\n=== CONTROL: does the blend reshape the margins at all? ===')
print(f'{"step":>6} | ' + ' '.join(f'{"L" + str(i):>15}' for i in range(6)))
print(f'{"":>6} | ' + ' '.join(f'{"0194 / 0193":>15}' for _ in range(6)))
for s in STEPS:
    if s in res['0194'] and s in res['0193']:
        line = f'{s:>6} | '
        for i in range(6):
            a94 = res['0194'][s][i]['median']; a93 = res['0193'][s][i]['median']
            line += f'{a94:.4f}/{a93:.4f} '
        print(line)

# ---- figure -------------------------------------------------------------------------------
os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
cols = plt.cm.viridis([i / 5 for i in range(6)])
for i in range(6):
    axes[0].plot(xs, [res['0194'][s][i]['median'] for s in xs], 'o-', color=cols[i],
                 label=f'L{i}')
    axes[0].plot(xs, [res['0193'][s][i]['median'] for s in xs if s in res['0193']], 's--',
                 color=cols[i], alpha=.45)
    axes[1].plot(xs, [r[2][i] for r in rows], 'o-', color=cols[i], label=f'L{i}')
    axes[2].plot(xs, [r[3][i] for r in rows], 'o-', color=cols[i], label=f'L{i}')
axes[0].set(title='median m$_{(1)}$  (solid 0194, dashed 0193)', xlabel='step', ylabel='m')
axes[1].axhline(1.0, color='k', ls=':', lw=1.5)
axes[1].set(title=r'2m$_{(1)}$/$\tau$   (target 1.0, dotted)', xlabel='step')
axes[2].axhline(0.2689, color='k', ls=':', lw=1.5)
axes[2].set(title=r'w$_1$ = $\sigma$(-2m/$\tau$)   (target 0.2689, dotted)', xlabel='step')
ax2 = axes[1].twinx()
ax2.plot(xs, [r[1] for r in rows], 'k^-', lw=2, label='deficit vs 0193')
ax2.set_ylabel('deficit (bpb)')
for ax in axes:
    ax.grid(alpha=.3); ax.legend(fontsize=7, ncol=2)
fig.suptitle('exp_g_0194: did the frozen tau stay mis-set? '
             f'(fixed {IDS.numel():,}-token slab, corrected eval window)')
fig.tight_layout()
fig.savefig(a.out, dpi=130)
print(f'\nwrote {a.out}')
if a.json:
    json.dump({'margins': res, 'deficit': deficit, 'tau': TAU}, open(a.json, 'w'), indent=1)
    print(f'wrote {a.json}')
