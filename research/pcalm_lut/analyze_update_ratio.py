"""Interior vs readout update magnitude, from any of the run directories.

The claim under test: under plain SGD, BP's interior update should be heavily attenuated relative to its
readout (the Jacobian product through depth, which is what Adam exists to undo), while PC's should be
roughly uniform, because PC's per-layer targets are local and never compose a Jacobian. Reported both on
the APPLIED update (what the optimiser did) and on the raw GRADIENT (what it was given), so the two are
not confused.

Usage: python3 analyze_update_ratio.py [--dir runs_sgd]
"""
import argparse
import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def last(h, k):
    v = [r[k] for r in h if k in r and isinstance(r[k], (int, float)) and not math.isnan(r[k])]
    return v[-1] if v else float('nan')


def mean_tail(h, k, n=5):
    v = [r[k] for r in h if k in r and isinstance(r[k], (int, float)) and not math.isnan(r[k])]
    v = v[-n:]
    return sum(v) / len(v) if v else float('nan')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', default='runs_sgd')
    a = ap.parse_args()
    root = os.path.join(HERE, a.dir)

    print(f'== {a.dir}: applied-update and gradient RMS, interior-forward vs readout')
    print('   values are the mean of the last 5 logged probes (a single probe is one batch).')
    print(f'\n   {"run":26s} {"upd interior":>13s} {"upd readout":>12s} {"upd i/r":>9s} '
          f'{"grad interior":>14s} {"grad readout":>13s} {"grad i/r":>9s} {"cos int":>8s}')
    rows = []
    for n in sorted(os.listdir(root)):
        p = os.path.join(root, n, 'run.json')
        if not os.path.exists(p):
            continue
        d = json.load(open(p))
        h = d['hist']
        ui = mean_tail(h, 'update/interior_tables_rms')
        ur = mean_tail(h, 'update/readout_tables_rms')
        gi = mean_tail(h, 'grad/interior_tables_rms')
        gr = mean_tail(h, 'grad/readout_tables_rms')
        # interior cosine to BP, mean over the interior layers (the readout is the last slot)
        arow = [r for r in h if 'align/f_tables_L0' in r]
        ci = float('nan')
        if arow:
            v, i = [], 0
            while f'align/f_tables_L{i}' in arow[-1]:
                v.append(arow[-1][f'align/f_tables_L{i}'])
                i += 1
            ci = sum(v[:-1]) / max(len(v) - 1, 1)
        print(f'   {n:26s} {ui:>13.3e} {ur:>12.3e} {ui / ur if ur else float("nan"):>9.3f} '
              f'{gi:>14.3e} {gr:>13.3e} {gi / gr if gr else float("nan"):>9.3f} {ci:>+8.3f}')
        rows.append(dict(run=n, upd_interior=ui, upd_readout=ur, grad_interior=gi, grad_readout=gr,
                         cos_interior=ci, cfg=d['cfg']))
    out = os.path.join(root, 'update_ratio.json')
    json.dump(rows, open(out, 'w'), indent=1)
    print(f'\nwrote {os.path.relpath(out, HERE)}')


if __name__ == '__main__':
    main()
