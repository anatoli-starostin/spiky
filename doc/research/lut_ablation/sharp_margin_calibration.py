"""Calibrate a selectivity-matched but DISCONTINUOUS control for the LightMHL continuity ablation.

Candidates (both stay discontinuous: sigmoid(0) = 0.5 > 0, so the score never vanishes at a boundary):
    gamma form   s = (sum_j m_j) * (prod_j sigmoid(2 m_j)) ** gamma      boundary factor 0.5^(n*gamma)
    beta form    s = (sum_j m_j) * prod_j sigmoid(beta * m_j)            boundary factor 0.5^n
gamma = 1 and beta = 2 are exactly "margin".

Data: exp_g_0193's own trained margins (4 real val rows, all 6 layers; CPU), grouped per (token, head)
across TPH = 128 tables for the within-token spread -- the same grouping selectivity_trained.py uses.

For every value: raw mean score, gain that scale-matches the mean to margin's mean on the same margins,
and on the GAIN-MATCHED scores: within-token CV, p75/p25, fraction below 1e-3 (the near-zero fraction is
only meaningful at matched scale), overall and per layer. Also the median boundary/before score ratio
(one margin set to 0) -- margin 0.56, a continuous form 0 -- as the discontinuity check.

TARGET (tanh_margin on its own trained margins, exp_g_0244): within-token CV 1.872 overall,
per layer 6.458 / 2.040 / 1.846 / 1.695 / 1.580 / 1.352; p75/p25 59.96; frac<1e-3 0.1055
(per layer 0.4025 / 0.0829 / 0.0595 / 0.0415 / 0.0312 / 0.0151). A log-space distance to that
profile ranks the candidates.
"""
import math
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tanh_margin_scale import light_blocks            # noqa: E402  (exp_g_0193 margins, per layer, CPU)

GAMMAS = (1.0, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.5, 4.0, 6.0, 8.0)
BETAS = (2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0)
TARGET = dict(wcv=1.872, ratio=59.96, small=0.1055,
              wcv_layers=[6.458, 2.040, 1.846, 1.695, 1.580, 1.352],
              small_layers=[0.4025, 0.0829, 0.0595, 0.0415, 0.0312, 0.0151])
# overall selectivity of the two continuous arms on their OWN trained margins (selectivity_trained.py)
ENVELOPE = dict(wcv=(1.872, 2.000), ratio=(12.33, 59.96), small=(0.0085, 0.1055))
CACHES = [('init (sweep_s05, TPH=256)', '/tmp/margins_anchor.pt'),
          ('after 4K steps (sweep_s05, TPH=256)', '/tmp/margins_anchor_trained.pt')]


def score_gamma(m, g):
    return m.sum(dim=-1) * torch.exp(g * F.logsigmoid(2.0 * m).sum(dim=-1))


def score_beta(m, b):
    return m.sum(dim=-1) * torch.exp(F.logsigmoid(b * m).sum(dim=-1))


def quant(t, ps=(0.25, 0.75)):
    g = torch.Generator().manual_seed(0)
    t = t[torch.randperm(t.numel(), generator=g)[:400000]] if t.numel() > 400000 else t
    return torch.quantile(t.double(), torch.tensor(ps, dtype=torch.float64)).tolist()


def wcv(s, group):
    usable = (s.numel() // group) * group
    grp = s[:usable].view(-1, group)
    return ((grp - grp.mean(dim=1, keepdim=True)).std() / grp.mean()).item()


def profile(fn, blocks, tph, ref_mean):
    per = [fn(b) for b in blocks]
    allS = torch.cat(per)
    gain = ref_mean / allS.mean().item()
    out = dict(mean=allS.mean().item(), gain=gain, layers=[])
    s = allS * gain
    p25, p75 = quant(s)
    out.update(wcv=wcv(s, tph), ratio=p75 / max(p25, 1e-30), small=(s < 1e-3).float().mean().item())
    for x in per:
        xs = x * gain
        q25, q75 = quant(xs)
        out['layers'].append(dict(wcv=wcv(xs, tph), ratio=q75 / max(q25, 1e-30),
                                  small=(xs < 1e-3).float().mean().item()))
    # discontinuity: set one (random) margin to 0 and compare the score before/after
    g = torch.Generator().manual_seed(1)
    sub = torch.cat(blocks)[torch.randint(0, allS.numel(), (200000,), generator=g)].clone()
    j = torch.randint(0, sub.shape[1], (sub.shape[0],), generator=g)
    before = fn(sub)
    sub[torch.arange(sub.shape[0]), j] = 0.0
    out['boundary_ratio'] = (fn(sub) / before).median().item()
    return out


def rms(xs):
    return math.sqrt(sum(x * x for x in xs) / len(xs))


def distance(p):
    """Separate log-space distances to the target; a single combined number is dominated by the
    near-zero fractions (log of ~0), so the axes are reported apart."""
    cv = rms([math.log(p['wcv'] / TARGET['wcv'])] +
             [math.log(l['wcv'] / t) for l, t in zip(p['layers'], TARGET['wcv_layers'])])
    ratio = abs(math.log(p['ratio'] / TARGET['ratio']))
    small = rms([math.log(max(p['small'], 1e-3) / TARGET['small'])] +
                [math.log(max(l['small'], 1e-3) / t) for l, t in zip(p['layers'], TARGET['small_layers'])])
    return cv, ratio, small


def envelope(p):
    return ' '.join(f'{k}:{"in" if lo <= p[k] <= hi else ("LOW" if p[k] < lo else "HIGH")}'
                    for k, (lo, hi) in ENVELOPE.items())


def show(name, p, extra=''):
    lw = ' '.join(f'{l["wcv"]:5.2f}' for l in p['layers'])
    lr = ' '.join(f'{l["ratio"]:6.1f}' for l in p['layers'])
    ls = ' '.join(f'{l["small"]:.3f}' for l in p['layers'])
    dcv, dr, ds = distance(p)
    print(f'{name:<14} mean {p["mean"]:.5f} gain {p["gain"]:9.3f} | wCV {p["wcv"]:5.3f} p75/p25 {p["ratio"]:8.2f} '
          f'frac<1e-3 {p["small"]:.4f} | boundary/before {p["boundary_ratio"]:.4g} {extra}')
    print(f'{"":<14}   per-layer wCV [{lw}]  p75/p25 [{lr}]  frac<1e-3 [{ls}]')
    print(f'{"":<14}   log-dist to target: CV {dcv:.2f}  p75/p25 {dr:.2f}  frac {ds:.2f} | envelope of the two '
          f'continuous arms: {envelope(p)}')


if __name__ == '__main__':
    blocks, tph = light_blocks()
    blocks = [b.abs().double() for b in blocks]
    NAP = blocks[0].shape[1]
    ref = torch.cat([score_gamma(b, 1.0) for b in blocks]).mean().item()
    print(f'exp_g_0193 trained margins: {sum(b.shape[0] for b in blocks):,} vectors, nap={NAP}, TPH={tph}; '
          f'margin mean score = {ref:.4f}')
    print('TARGET (tanh_margin on its own margins, exp_g_0244): wCV 1.872 [6.46 2.04 1.85 1.70 1.58 1.35], '
          'p75/p25 59.96, frac<1e-3 0.1055 [0.403 0.083 0.060 0.042 0.031 0.015]')
    print('ENVELOPE of the two continuous arms on their own margins: wCV 1.872-2.000, p75/p25 12.33-59.96, '
          'frac<1e-3 0.0085-0.1055 (min_margin own-margin per layer: wCV [2.92 1.70 1.71 1.62 1.62 1.65], '
          'frac [0.031 0.007 0.005 0.004 0.003 0.002])')
    print('like-for-like references (the continuous forms scored on the SAME exp_g_0193 margins):')
    show('tanh a=2', profile(lambda m: m.sum(dim=-1) * torch.tanh(2.0 * m).prod(dim=-1), blocks, tph, ref))
    show('min_margin', profile(lambda m: m.min(dim=-1).values * torch.exp(F.logsigmoid(2.0 * m).sum(dim=-1)),
                               blocks, tph, ref))
    print('-' * 150)
    best = []
    for gm in GAMMAS:
        p = profile(lambda m, gm=gm: score_gamma(m, gm), blocks, tph, ref)
        show(f'gamma={gm:g}', p, f'| 0.5^(n*g)={0.5 ** (NAP * gm):.3e} 0.5^g={0.5 ** gm:.3f} ')
        best.append((distance(p), f'gamma={gm:g}'))
    print('-' * 150)
    for bt in BETAS:
        p = profile(lambda m, bt=bt: score_beta(m, bt), blocks, tph, ref)
        show(f'beta={bt:g}', p, f'| 0.5^n={0.5 ** NAP:.3e} ')
        best.append((distance(p), f'beta={bt:g}'))
    print('=' * 150)
    for i, axis in enumerate(('CV (overall + per layer)', 'p75/p25', 'frac<1e-3 (overall + per layer)')):
        ranked = sorted(best, key=lambda x: x[0][i])[:3]
        print(f'closest on {axis}: ' + ', '.join(f'{n} ({d[i]:.2f})' for d, n in ranked))
    print('=' * 150)
    print('scale match at other training stages: margin:gamma-form mean-score ratio')
    for tag, path in CACHES:
        if not os.path.exists(path):
            print(f'   {tag}: {path} missing')
            continue
        allm = torch.cat([b.abs().double() for b in torch.load(path)['per_block']])
        mref = score_gamma(allm, 1.0).mean().item()
        print(f'   {tag:<38} ' + '  '.join(f'g={gm:g}:{mref / score_gamma(allm, gm).mean().item():.2f}'
                                          for gm in GAMMAS[1:9]))
