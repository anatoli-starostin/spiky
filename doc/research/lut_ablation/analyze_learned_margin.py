"""Post-run analysis of exp_g_0247 (learned_margin). Read-only; CPU.

    python analyze_learned_margin.py [<run_dir name, default exp_g_0247>] [--png out.png]

1. TRAJECTORIES of g, beta, gamma per layer from metrics.csv (logged at every eval): table at selected
   steps, final values, and drift over the last 2,000 steps (still moving at 16K?).
2. BOUNDARY SURVIVAL at the learned values per layer: 0.5^(8*gamma) (all-margins-zero corner of the
   probability factor) and 0.5^gamma (one boundary crossing, relative factor).
3. (beta, gamma) DEGENERACY on the run's OWN trained margins (4 val rows, skip 12), per layer:
   * sensitivities of log s per (token, head, table): d log s / d g = 1,
     d log s / d log beta = gamma * sum_j (beta m_j) sigmoid(-beta m_j),
     d log s / d log gamma = gamma * log P.
     A constant sensitivity is exactly absorbed by g (and g itself by downstream weights), so what matters
     is the CENTERED pair: their correlation and the condition number of their 2x2 covariance. |corr| -> 1
     means moving log beta and log gamma in a fixed ratio changes log s only by a constant, i.e. those
     directions are near-equivalent up to a gain.
   * a valley scan: for gamma' on a grid, the beta' (with the best gain offset g') whose log-score is
     closest to the learned layer's, and the RMS residual in log s. A flat valley (small residual over a
     wide gamma' range) means the individual learned (beta, gamma) values are not identified. The
     residual of margin's (2, 1) and of sharp_margin's (2, 1.75) against the learned score are given for
     scale.
"""
import csv
import json
import math
import os
import sys

import torch
import torch.nn.functional as F

FR = os.path.expanduser('~/projects/spiky/experiments/ffn_replacement')
_ARGS = [a for i, a in enumerate(sys.argv[1:], 1) if not a.startswith('--') and sys.argv[i - 1] != '--png']
RD = os.path.join(FR, 'runs_corrected', _ARGS[0] if _ARGS else 'exp_g_0247_B16k_light_learnedmargin_tph128_seed1')
RUN = os.path.basename(RD)[:10]
print(f'run: {os.path.basename(RD)}')
sys.path.insert(0, os.path.join(FR, 'tools'))
sys.path.insert(0, os.path.join(FR, 'distill'))
N_LAYERS = 6
SUB = 200_000

rows = list(csv.DictReader(open(os.path.join(RD, 'metrics.csv'))))
steps = [int(r['step']) for r in rows]
traj = {k: [[float(r[f'lm_{k}_L{i}']) for r in rows] for i in range(N_LAYERS)] for k in ('g', 'beta', 'gamma')}

print('1. TRAJECTORIES (value at eval step; init g=0, beta=2, gamma=1)')
show = [s for s in (500, 1000, 2000, 4000, 8000, 12000, 14000, 16000) if s in steps]
for k in ('g', 'beta', 'gamma'):
    print(f'   {k}:')
    print('      ' + f'{"step":>6} ' + ' '.join(f'{"L" + str(i):>9}' for i in range(N_LAYERS)))
    for s in show:
        j = steps.index(s)
        print('      ' + f'{s:>6} ' + ' '.join(f'{traj[k][i][j]:>9.4f}' for i in range(N_LAYERS)))
    if 14000 in steps and 16000 in steps:
        a, b = steps.index(14000), steps.index(16000)
        c, d = steps.index(8000), steps.index(10000)
        print('      drift 14K->16K ' + ' '.join(f'{traj[k][i][b] - traj[k][i][a]:>+9.4f}' for i in range(N_LAYERS)))
        print('      drift  8K->10K ' + ' '.join(f'{traj[k][i][d] - traj[k][i][c]:>+9.4f}' for i in range(N_LAYERS)))

final = {k: [traj[k][i][-1] for i in range(N_LAYERS)] for k in traj}
print('\n2. BOUNDARY SURVIVAL at the learned gamma')
for i in range(N_LAYERS):
    gm = final['gamma'][i]
    print(f'   L{i}: gamma {gm:.4f}  beta {final["beta"][i]:.4f}  g {final["g"][i]:+.4f} (exp g {math.exp(final["g"][i]):.4f})'
          f'  | 0.5^(8 gamma) = {0.5 ** (8 * gm):.3e}  0.5^gamma = {0.5 ** gm:.4f}')

import distill_ffn as D                                                  # noqa: E402
from model_build import build_model                                      # noqa: E402

cfg = json.load(open(os.path.join(RD, 'config.json')))
tok = D.RustBPETokenizer.from_directory(os.path.join(D.get_base_dir(), 'tokenizer'))
model = build_model(cfg, tok.get_vocab_size(), device='cpu')
miss, unexp = model.load_state_dict(torch.load(os.path.join(RD, 'checkpoint.pt'), map_location='cpu'), strict=False)
model.eval()
idx = D.take_rows(D.tokenizing_distributed_data_loader_bos_bestfit(tok, 16, cfg['seq_len'], split='val', device='cpu'),
                  4, skip_rows=12)
got = []
hooks = [b.ffn.lut_light.register_forward_hook(lambda mod, inp, out: got.append((mod, inp[0].detach())))
         for b in model.blocks]
with torch.no_grad():
    x = model.tok_emb(idx)
    for b in model.blocks:
        x = b(x, model.rope.cos, model.rope.sin)
for h in hooks:
    h.remove()
print(f'\n3. (beta, gamma) DEGENERACY on own trained margins (checkpoint missing={len(miss)} unexpected={len(unexp)})')
gen = torch.Generator().manual_seed(0)
GAMMAS = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0]
BETAS = torch.logspace(math.log10(0.25), math.log10(32.0), 60).tolist()


def logs(m, beta, gamma):
    return torch.log(m.sum(-1)) + gamma * F.logsigmoid(beta * m).sum(-1)


def resid(target, cand):
    dlt = cand - target
    return (dlt - dlt.mean()).pow(2).mean().sqrt().item()          # best gain offset g' removed


for li, (lut, z) in enumerate(got):
    H, T, NAP = lut.n_heads, lut.tables_per_head, lut.n_anchor_pairs
    a = lut.anchor_a.view(1, H, T * NAP).expand(z.shape[0], H, T * NAP)
    b = lut.anchor_b.view(1, H, T * NAP).expand(z.shape[0], H, T * NAP)
    m = (torch.gather(z, 2, a) - torch.gather(z, 2, b)).view(-1, NAP).abs().double()
    m = m[torch.randperm(m.shape[0], generator=gen)[:SUB]]
    m = m[m.sum(-1) > 0]
    v = lut.learned_confidence_values()
    beta, gamma = v['beta'], v['gamma']
    bm = beta * m
    s_lb = gamma * (bm * torch.sigmoid(-bm)).sum(-1)
    s_lg = gamma * F.logsigmoid(bm).sum(-1)
    C = torch.cov(torch.stack([s_lb, s_lg]))
    corr = (C[0, 1] / (C[0, 0] * C[1, 1]).sqrt()).item()
    ev = torch.linalg.eigvalsh(C)
    target = logs(m, beta, gamma)
    valley = []
    for gp in GAMMAS:
        best = min((resid(target, logs(m, bp, gp)), bp) for bp in BETAS)
        valley.append((gp, best[1], best[0]))
    print(f'   L{li}: learned beta {beta:.4f} gamma {gamma:.4f} | centered sensitivities: sd(dlogs/dlogbeta) '
          f'{C[0, 0].sqrt().item():.4f} sd(dlogs/dloggamma) {C[1, 1].sqrt().item():.4f} corr {corr:+.4f} '
          f'cond {(ev[-1] / ev[0]).item():.1f}')
    print(f'      residual RMS in log s (best gain removed): margin(2,1) {resid(target, logs(m, 2.0, 1.0)):.4f}  '
          f'sharp(2,1.75) {resid(target, logs(m, 2.0, 1.75)):.4f}  | log-score sd {target.std().item():.4f}')
    print('      valley  gamma\' -> best beta\' (resid): ' +
          '  '.join(f'{gp:g}->{bp:.2f} ({r:.3f})' for gp, bp, r in valley))
    # Does beta's departure from 2 matter for the score SHAPE? Hold beta' = 2, refit gamma' finely.
    fine = [0.5 + 0.01 * i for i in range(201)]
    r2, g2 = min((resid(target, logs(m, 2.0, gp)), gp) for gp in fine)
    rl, bl = min((resid(target, logs(m, bp, gamma)), bp) for bp in [0.5 + 0.01 * i for i in range(351)])
    print(f'      beta\' fixed at 2: best gamma\' {g2:.2f}, residual {r2:.4f}  |  gamma\' fixed at learned '
          f'{gamma:.3f}: best beta\' {bl:.2f}, residual {rl:.4f}')

if '--png' in sys.argv:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out = sys.argv[sys.argv.index('--png') + 1]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, k, init in zip(axes, ('g', 'beta', 'gamma'), (0.0, 2.0, 1.0)):
        for i in range(N_LAYERS):
            ax.plot(steps, traj[k][i], lw=1.6, label=f'L{i}')
        ax.axhline(init, color='0.5', ls='--', lw=1, label=f'init {init:g}')
        if k == 'gamma':
            ax.axhline(1.75, color='tab:red', ls=':', lw=1, label='sharp_margin 1.75')
        ax.set(xlabel='step', title=f'learned {k} per layer ({RUN})')
        ax.grid(True, alpha=.3)
        ax.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    print('wrote', out)
