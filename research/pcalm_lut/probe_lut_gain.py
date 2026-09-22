"""Bug hunt: why did the non-residual LUT stack reach a per-block gain of 12.5?

The invariant under test, as stated: "a LUT block cannot diverge by construction -- its output is a soft
mixture over table entries, so ||output|| is bounded by the largest table entry". Everything below is
measured, not argued.

Checks, in order:
  A  routing normalisation -- do the per-table blend weights sum to 1, and what does the sum over tables
     of the confidence score do? (hypothesis 4, the stated leading suspicion)
  B  homogeneity -- feed c*h for a range of c and see whether ||block|| tracks c. A bounded mixture must
     saturate; an amplifier stays proportional.
  C  the calibration -- what multiplier was actually applied per block, and what did it do to the table
     entries. (hypothesis 1)
  D  the diverged checkpoint -- did the table entries themselves blow up, or is the gain coming from
     somewhere else? (hypothesis 3)
  E  the same block under confidence_form='bounded', where the score IS bounded by 1.

Usage: python3 probe_lut_gain.py
"""
import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.expanduser('~/projects/spiky/src'))
from autoencoder import LUT_KW, Autoencoder  # noqa: E402
from data import load  # noqa: E402
from spiky.lutorch.light_multi_head_lut import LightMultiHeadLUT  # noqa: E402

R = os.path.join(HERE, 'runs_autoencoder')


def rms(t):
    return float(t.pow(2).mean().sqrt())


@torch.no_grad()
def decompose(lut, z):
    """The pieces of one LightMHL read, by hand, mirroring _blend_bag."""
    d = z[:, lut.anchor_a] - z[:, lut.anchor_b]                 # [B, T, nap]
    m = d.abs()
    score = lut.confidence_score(d)                             # [B, T]
    mv, _ = m.min(dim=-1, keepdim=True)
    w = torch.softmax(torch.cat([torch.zeros_like(m[..., :1]), -2.0 * mv / lut.read_tau], -1), -1)
    return {'blend_w_sum_per_table': float(w.sum(-1).mean()),   # must be exactly 1
            'blend_w_min': float(w.sum(-1).min()), 'blend_w_max': float(w.sum(-1).max()),
            'score_sum_over_tables': float(score.sum(-1).mean()),
            'score_mean': float(score.mean()), 'score_max': float(score.max()),
            'margin_sum_mean': float(m.sum(-1).mean()),
            'table_absmax': float(lut.tables.abs().max()), 'table_rms': rms(lut.tables),
            'h_rms': rms(z)}


@torch.no_grad()
def main():
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    xte, _ = load('fashion', train=False, device=dev)
    x = xte[:512]

    print('=' * 100)
    print('A. IS THE MIXTURE NORMALISED?  (fresh calibrated non-residual model, tph=128)')
    m = Autoencoder(784, 64, 2, 'lut', 128, dev, 0, residual=False)
    m.calibrate_init(x, verbose=False)
    h = m.enc(x)
    for i in range(m.n_blocks):
        s = decompose(m.blocks[i], h)
        u = m.raw_block(i, h)
        print(f'  block {i}: blend weights per table sum to {s["blend_w_sum_per_table"]:.6f} '
              f'(min {s["blend_w_min"]:.6f}, max {s["blend_w_max"]:.6f})')
        print(f'           SUM OF SCORES OVER THE {m.blocks[i].n_tables} TABLES = '
              f'{s["score_sum_over_tables"]:.2f}   (mean score {s["score_mean"]:.4f}, '
              f'max {s["score_max"]:.4f})')
        print(f'           sum_j|d_j| mean {s["margin_sum_mean"]:.4f} | table |max| '
              f'{s["table_absmax"]:.4e} | ||out||/||h|| = {rms(u) / rms(h):.4f}')
        h = m.block(i, h)

    print()
    print('B. HOMOGENEITY: feed c*h. A bounded mixture must SATURATE; an amplifier stays proportional.')
    h0 = m.enc(x)
    lut0 = m.blocks[0]
    print(f'  {"c":>8s} {"||c h||":>10s} {"||block(c h)||":>15s} {"ratio out/in":>13s} '
          f'{"sum score":>11s} {"table |max|":>12s}')
    for c in (0.25, 0.5, 1.0, 2.0, 4.0, 8.0):
        z = c * h0
        u = m.raw_block(0, z)
        s = decompose(lut0, z)
        print(f'  {c:>8.2f} {rms(z):>10.4f} {rms(u):>15.4f} {rms(u) / rms(z):>13.4f} '
              f'{s["score_sum_over_tables"]:>11.2f} {s["table_absmax"]:>12.4e}')
    print(f'  the largest table entry anywhere in block 0 is {float(lut0.tables.abs().max()):.4e};')
    print('  if the invariant held, ||block(c h)|| could never exceed that, at any c.')

    print()
    print('C. WHAT THE CALIBRATION DID (multiplier per block, table norms before and after)')
    m2 = Autoencoder(784, 64, 2, 'lut', 128, dev, 0, residual=False)
    before = [(float(b.tables.abs().max()), rms(b.tables)) for b in m2.blocks]
    h = m2.enc(x)
    mult = []
    for i in range(m2.n_blocks):
        target, got = rms(h), rms(m2.raw_block(i, h))
        mult.append(target / max(got, 1e-12))
        m2.blocks[i].tables.mul_(mult[-1])
        h = m2.block(i, h)
    after = [(float(b.tables.abs().max()), rms(b.tables)) for b in m2.blocks]
    print(f'  {"block":>6s} {"multiplier":>11s} {"|max| before":>13s} {"|max| after":>12s} '
          f'{"rms before":>11s} {"rms after":>10s}')
    for i, (mu, b, a) in enumerate(zip(mult, before, after)):
        print(f'  {i:>6d} {mu:>11.3f} {b[0]:>13.4e} {a[0]:>12.4e} {b[1]:>11.4e} {a[1]:>10.4e}')

    print()
    print('D. THE DIVERGED CHECKPOINT: did the table entries blow up?')
    ck = os.path.join(R, 'lut-L2-tph128-nores-s2000', 'model.pt')
    if os.path.exists(ck):
        cfg = json.load(open(f'{R}/lut-L2-tph128-nores-s2000/run.json'))['cfg']
        md = Autoencoder(784, cfg['width'], cfg['depth_L'], 'lut', cfg['tables'], dev, cfg['seed'],
                         residual=False)
        md.load_state_dict(torch.load(ck, map_location=dev))
        h = md.enc(x)
        print(f'  {"block":>6s} {"table |max|":>12s} {"table rms":>11s} {"sum score":>12s} '
              f'{"||out||/||h||":>14s} {"||h|| in":>11s}')
        for i in range(md.n_blocks):
            s = decompose(md.blocks[i], h)
            u = md.raw_block(i, h)
            print(f'  {i:>6d} {s["table_absmax"]:>12.4e} {s["table_rms"]:>11.4e} '
                  f'{s["score_sum_over_tables"]:>12.2f} {rms(u) / rms(h):>14.4f} {rms(h):>11.4e}')
            h = md.block(i, h)
    else:
        print(f'  checkpoint not found at {ck}')

    print()
    print("E. THE SAME BLOCK WITH confidence_form='bounded' (score in (0,1], so bounded by construction)")
    for form in ('margin', 'bounded'):
        kw = dict(LUT_KW)
        kw['confidence_form'] = form
        lut = LightMultiHeadLUT(input_dim=64, n_tables=128, output_dim=64, random_seed=1,
                                device=torch.device(dev), **kw)
        lut.tables.mul_(100.0)                       # same table scale for both
        h0 = m.enc(x)
        row = []
        for c in (1.0, 4.0, 16.0):
            z = c * h0
            row.append(rms(lut(z)) / rms(z))
        s = decompose(lut, h0)
        print(f'  {form:>9s}: out/in at c=1,4,16 -> ' + ', '.join(f'{v:.4f}' for v in row)
              + f'   (sum of scores {s["score_sum_over_tables"]:.2f}, table |max| {s["table_absmax"]:.3e})')


if __name__ == '__main__':
    main()
