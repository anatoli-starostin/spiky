"""Triage the confidence-form design space on the CACHED real margins, before any GPU.

Reads /tmp/margins_anchor.pt (dump_margins.py) -- 6,291,456 real [NAP=8] margin vectors
from the anchor sizing -- and answers, cheaply:

  A. What does each form's score distribution look like, and how SELECTIVE is it?
     A gate that multiplies every row by roughly the same number is not a gate; it is a
     constant, and a constant is absorbed by the (linear, zero-initialised) decompress.
     So the quantity that matters is not the score's mean but its SPREAD -- and,
     specifically, the spread ACROSS TABLES WITHIN ONE TOKEN, because that is the only
     part that reweights the ensemble sum. The across-token part merely rescales that
     token's whole FFN output.
  B. nap-invariance: does the geometric mean really hold its scale as nap varies, where
     the product collapses? (Uses independent draws from the pooled empirical |d| to
     build vectors at any nap; the measured within-table anchor correlation is reported
     so that assumption is visible rather than hidden.)
  C. A sharpened variant: score_beta = exp(mean_j logsigmoid(2*beta*|d_j|)). beta>1 buys
     back selectivity that the geometric mean flattens away, without reintroducing the
     nap-compounding. Sweep beta and report where it matches "bounded"'s selectivity.

    python diag_confidence_forms.py
"""
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.expanduser('~/projects/spiky/src'))
from spiky.lutorch.fast_multi_head_lut import _confidence_score        # noqa: E402

CACHE = '/tmp/margins_anchor.pt'
TPH = 256          # tables per head at the anchor sizing -- the ensemble that gets summed


def score_beta(m, beta):
    """The geometric-mean form with a sharpness knob: beta=1 IS bounded_norm."""
    return torch.exp(F.logsigmoid(2.0 * beta * m).mean(dim=-1))


def q(t, ps=(0.25, 0.5, 0.75)):
    t = t[torch.randperm(t.numel())[:400000]] if t.numel() > 400000 else t
    return torch.quantile(t, torch.tensor(ps, dtype=t.dtype)).tolist()


def describe(name, s):
    p25, med, p75 = q(s)
    cv = (s.std() / s.mean()).item()
    print(f'   {name:<22} mean {s.mean():8.4f}  p25 {p25:7.4f}  med {med:7.4f}  '
          f'p75 {p75:7.4f}  p75/p25 {p75 / max(p25, 1e-12):6.2f}  CV {cv:6.3f}')
    return dict(mean=s.mean().item(), p25=p25, p75=p75, cv=cv)


def main():
    blob = torch.load(CACHE)
    d = blob['d']                                   # [N, NAP] real margins
    m = d.abs()
    N, NAP = m.shape
    print(f'cached margins: {N:,} vectors x NAP={NAP}   '
          f'(|d| median {m.median():.4f}, mean {m.mean():.4f})')

    print('\n' + '=' * 96)
    print('A. SCORE DISTRIBUTION AND SELECTIVITY  (mean is cosmetic; spread is the gate)')
    print('=' * 96)
    forms = {'bounded': _confidence_score(d, 'bounded'),
             'margin': _confidence_score(d, 'margin'),
             'bounded_norm': _confidence_score(d, 'bounded_norm')}
    stats = {k: describe(k, v) for k, v in forms.items()}

    print('\n   The point: p75/p25 is how many times more a confident row is weighted than')
    print('   an unconfident one. A ratio near 1 means the gate is a constant multiplier,')
    print('   and a constant multiplier is EXACTLY absorbable into the linear decompress')
    print('   that follows -- i.e. it changes nothing the model cannot already express.')

    print('\n' + '=' * 96)
    print('A2. WHERE THE VARIANCE LIVES: across tokens (rescales) vs within a token')
    print('    across tables (actually reweights the ensemble sum -- the only useful part)')
    print('=' * 96)
    usable = (N // TPH) * TPH
    for k, s in forms.items():
        g = s[:usable].view(-1, TPH)                # [tokens*heads, tables]
        per_tok = g.mean(dim=1)
        within = g - per_tok.unsqueeze(1)
        tot = g.var().item()
        print(f'   {k:<22} total var {tot:11.6g}   across-token {per_tok.var().item() / tot:6.1%}'
              f'   within-token {within.var().item() / tot:6.1%}'
              f'   within-token CV {(within.std() / g.mean()).item():6.3f}')
    print('\n   Within-token CV is the honest "how much gating is happening" number.')

    print('\n' + '=' * 96)
    print('B. nap-INVARIANCE  (does the geometric mean hold its scale where the product')
    print('   collapses?)  Vectors built from independent draws of the pooled empirical |d|.')
    print('=' * 96)
    flat = m.reshape(-1)
    corr = torch.corrcoef(m[:200000, :2].T.double())[0, 1].item()
    print(f'   within-table correlation between two anchors of the same table: {corr:+.4f}'
          f'   ({"independent enough" if abs(corr) < 0.05 else "NOT independent -- caveat"})')
    idx = torch.randint(0, flat.numel(), (200000, 16))
    samp = flat[idx]
    print(f'\n   {"nap":>4}  {"bounded":>12}  {"bounded_norm":>14}  {"margin":>12}')
    for nap in (1, 2, 4, 6, 8, 12, 16):
        mm = samp[:, :nap]
        dd = mm                                     # sign is irrelevant to the score
        b = _confidence_score(dd, 'bounded').mean().item()
        n = _confidence_score(dd, 'bounded_norm').mean().item()
        g = _confidence_score(dd, 'margin').mean().item()
        print(f'   {nap:>4}  {b:>12.6f}  {n:>14.6f}  {g:>12.6f}')
    print('\n   bounded falls geometrically with nap; bounded_norm is flat by construction;')
    print('   margin RISES with nap (the sum|d| factor grows linearly while prob decays).')

    print('\n' + '=' * 96)
    print('C. SHARPENED GEOMETRIC MEAN: score = exp(mean_j logsigmoid(2*beta*|d_j|))')
    print('=' * 96)
    target = stats['bounded']['p75'] / max(stats['bounded']['p25'], 1e-12)
    print(f'   target selectivity (bounded p75/p25) = {target:.2f}\n')
    print(f'   {"beta":>6}  {"mean":>9}  {"p25":>9}  {"p75":>9}  {"p75/p25":>9}  {"CV":>7}')
    for beta in (0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0):
        s = score_beta(m, beta)
        p25, _med, p75 = q(s)
        print(f'   {beta:>6.1f}  {s.mean():>9.4f}  {p25:>9.4f}  {p75:>9.4f}  '
              f'{p75 / max(p25, 1e-12):>9.2f}  {(s.std() / s.mean()).item():>7.3f}')
    print('\n   beta only sharpens; it never reintroduces the nap-compounding, because the')
    print('   mean over j is taken after the temperature.')


if __name__ == '__main__':
    main()
