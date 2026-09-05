"""Regularity / dependency analysis across the axes: tph, nap (log2 cells), d_in, d_out, H.

METHODOLOGY. Across the 16k set bpb is dominated by the table budget
`tables = 6*H*tph*cells*d_out`, which spans 9.4M to 604M and is itself a product of four of the
five axes. A raw correlation of any axis against bpb is therefore mostly measuring table budget.
So this script:

  (1) fits the dominant effect first — bpb vs log2(tables) within CompressionMHL only — and
      reports it as the backdrop everything else is measured against;
  (2) enumerates ISO-TABLE-BUDGET groups from the data (tables equal to within 1%) and reads
      each axis only inside those groups, where the budget is held fixed by construction;
  (3) runs the multiple regression anyway, with the collinearity spelled out, and labels it
      indicative rather than inferential;
  (4) reports the two cost axes nobody charges for: gather traffic scales with H*tph (capped at
      1024 by fiat, which also makes H and tph nearly anti-collinear) and the soft-backward
      buffer scales as [tokens, H*tph, 2^nap].

Everything is read from each run's own config.json / summary.json / corrected_score.json.

    python axis_analysis.py            # full analysis
    python axis_analysis.py --json     # machine-readable, for the figure
"""
import glob
import itertools
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ANCHOR = 1.1651468950008814
NOISE = 0.002
N_EMBD, DEPTH = 384, 6


def _score(s):
    return s.get('corrected_val_bpb', s.get('bpb_fixed', s.get('proxy_val_bpb')))


def load():
    """Every scored run, 16k and 4k proxy alike, with its shape and derived quantities."""
    rows = []
    for p in sorted(glob.glob(os.path.join(HERE, '*', 'corrected_score.json'))):
        d = os.path.dirname(p)
        s = json.load(open(p))
        cfg = json.load(open(os.path.join(d, 'config.json')))
        bpb = _score(s)
        if bpb is None:
            continue
        raw = cfg.get('ffn_lut_kind') == 'fastmhl_raw'
        dense = cfg.get('ffn_type') == 'dense'
        nap = cfg.get('raw_nap') if raw else cfg.get('lut_n_anchor_pairs')
        H = cfg.get('raw_n_heads') if raw else cfg.get('lut_n_heads')
        tph = cfg.get('raw_tph') if raw else cfg.get('lut_tables_per_head')
        d_in = cfg.get('lut_inner_in_dim', cfg.get('lut_inner_dim'))
        d_out = cfg.get('lut_inner_out_dim', cfg.get('lut_inner_dim'))
        fam = ('dense' if dense else 'raw' if raw else
               'inproj' if d_out == -1 else 'compression')
        row_w = None if dense else (N_EMBD if fam in ('raw', 'inproj') else d_out)
        name = os.path.basename(d)
        rows.append(dict(
            run=name,
            tag=(name[6:10] if name.startswith('exp_n_') else name.split('_')[1].upper()),
            proxy=bool(s.get('proxy_sweep')), fam=fam, bpb=bpb,
            steps=cfg.get('n_steps'), H=H, tph=tph, cells=(2 ** nap) if nap else None,
            nap=nap, d_in=d_in, d_out=d_out,
            tables=(DEPTH * H * tph * (2 ** nap) * row_w) if row_w else None,
            gather=(H * tph) if H else None,
            params=s.get('total_params') or 0))
    return rows


def ols(xs, ys):
    """Least squares y = a + b x; returns (a, b, r2)."""
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    a = my - b * mx
    ss_res = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys))
    ss_tot = sum((y - my) ** 2 for y in ys)
    return a, b, (1 - ss_res / ss_tot if ss_tot else float('nan'))


def pearson(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = math.sqrt(sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys))
    return num / den if den else float('nan')


def iso_groups(rows, tol=0.01):
    """Groups of runs whose table budgets agree to within `tol`."""
    have = sorted((r for r in rows if r['tables']), key=lambda r: r['tables'])
    groups, cur = [], [have[0]]
    for r in have[1:]:
        if abs(r['tables'] - cur[0]['tables']) / cur[0]['tables'] <= tol:
            cur.append(r)
        else:
            groups.append(cur)
            cur = [r]
    groups.append(cur)
    return [g for g in groups if len(g) > 1]


def varying(group, axes=('H', 'tph', 'cells', 'd_in', 'd_out')):
    return [a for a in axes if len({r[a] for r in group}) > 1]


def main():
    rows = load()
    k16 = [r for r in rows if r['steps'] == 16000]
    comp16 = [r for r in k16 if r['fam'] == 'compression']
    proxy = [r for r in rows if r['proxy'] and r['fam'] == 'compression']

    if '--json' in sys.argv:
        print(json.dumps(dict(all=rows), indent=2))
        return

    print(f"LOADED  {len(rows)} scored runs: {len(k16)} at 16k, "
          f"{len([r for r in rows if r['proxy']])} proxy at 4k")
    print(f"16k CompressionMHL usable for the axis work: {len(comp16)}\n")

    # ---------- 1. the backdrop ----------
    print("=" * 78)
    print("1. THE DOMINANT EFFECT — bpb vs log2(table params), CompressionMHL 16k only")
    print("=" * 78)
    xs = [math.log2(r['tables']) for r in comp16]
    ys = [r['bpb'] for r in comp16]
    a, b, r2 = ols(xs, ys)
    print(f"   n = {len(comp16)}   bpb = {a:.6f} {b:+.6f} * log2(tables)")
    print(f"   R^2 = {r2:.4f}   slope = {b:+.6f} bpb per DOUBLING of table budget")
    print(f"   over the observed range ({min(r['tables'] for r in comp16):,} to "
          f"{max(r['tables'] for r in comp16):,}, "
          f"{math.log2(max(r['tables'] for r in comp16) / min(r['tables'] for r in comp16)):.1f} "
          f"doublings) that is {abs(b) * math.log2(max(r['tables'] for r in comp16) / min(r['tables'] for r in comp16)):.4f} bpb")
    resid = [y - (a + b * x) for x, y in zip(xs, ys)]
    sd = math.sqrt(sum(e * e for e in resid) / (len(resid) - 2))
    print(f"   residual sd = {sd:.6f}  ({sd/NOISE:.1f}x the {NOISE} noise floor) — this is the "
          f"room left\n   for every other axis combined, once the budget is accounted for.")

    # ---------- 2. iso-budget groups ----------
    print("\n" + "=" * 78)
    print("2. ISO-TABLE-BUDGET GROUPS — the only place an axis can be read cleanly")
    print("=" * 78)
    for g in iso_groups(comp16):
        g = sorted(g, key=lambda r: r['bpb'])
        spread = g[-1]['bpb'] - g[0]['bpb']
        ax = varying(g)
        print(f"\n   tables {g[0]['tables']:,}   n={len(g)}   varying: {', '.join(ax)}")
        for r in g:
            print(f"      {r['tag']}  H{r['H']} tph{r['tph']:<4} cells{r['cells']:<5} "
                  f"d_in{r['d_in']:<4} d_out{r['d_out']:<4}  bpb {r['bpb']:.6f}  "
                  f"(+{r['bpb']-g[0]['bpb']:.6f})")
        print(f"      spread {spread:.6f}  "
              f"({'ABOVE' if spread > NOISE else 'below'} the {NOISE} noise floor)")

    # ---------- 3. regression ----------
    print("\n" + "=" * 78)
    print("3. MULTIPLE REGRESSION — indicative only, read the caveats")
    print("=" * 78)
    try:
        import numpy as np
    except ImportError:
        print("   numpy unavailable; skipped")
        np = None
    if np is not None:
        names = ['log2(tables)', 'log2(tph)', 'nap', 'log2(d_in)', 'log2(d_out)', 'log2(H)']
        X = np.array([[math.log2(r['tables']), math.log2(r['tph']), r['nap'],
                       math.log2(r['d_in']), math.log2(r['d_out']), math.log2(r['H'])]
                      for r in comp16])
        y = np.array([r['bpb'] for r in comp16])
        Z = (X - X.mean(0)) / X.std(0)          # standardised, so betas are comparable
        A = np.hstack([np.ones((len(Z), 1)), Z])
        beta, *_ = np.linalg.lstsq(A, y, rcond=None)
        pred = A @ beta
        r2m = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        print(f"   n = {len(comp16)} points, 6 predictors -> {len(comp16)-7} residual df. "
              f"UNDERDETERMINED in practice.")
        print(f"   R^2 = {r2m:.4f} (unadjusted); adjusted "
              f"{1-(1-r2m)*(len(comp16)-1)/(len(comp16)-7):.4f}")
        print(f"   standardised coefficients (bpb per 1 sd of the predictor):")
        for nm, c in zip(names, beta[1:]):
            print(f"      {nm:<14} {c:+.6f}")
        print("\n   PREDICTOR CORRELATION MATRIX — the reason the above cannot be trusted:")
        print("      " + "".join(f"{n[:9]:>11}" for n in names))
        C = np.corrcoef(X.T)
        for i, nm in enumerate(names):
            print(f"      {nm[:9]:<9}" + "".join(f"{C[i][j]:>11.3f}" for j in range(len(names))))
        worst = max(((abs(C[i][j]), names[i], names[j]) for i, j in
                     itertools.combinations(range(len(names)), 2)))
        print(f"\n   strongest collinearity: |r| = {worst[0]:.3f} between {worst[1]} and "
              f"{worst[2]}")
        print("   H*tph is capped at 1024 by fiat, so log2(H) and log2(tph) are anti-correlated")
        print("   by construction; and tables = 6*H*tph*cells*d_out makes log2(tables) an exact")
        print("   linear combination of log2(H)+log2(tph)+nap+log2(d_out). The design matrix is")
        print("   therefore RANK-DEFICIENT in exact arithmetic — the fit is a projection, not an")
        print("   identification. Read section 2, not this.")

    # ---------- 4. uncounted cost axes ----------
    print("\n" + "=" * 78)
    print("4. THE COST AXES NOBODY CHARGES FOR — gather H*tph, buffer H*tph*2^nap")
    print("=" * 78)
    print(f"   {'run':<6}{'H*tph':>7}{'2^nap':>7}{'H*tph*2^nap':>14}{'tables':>14}{'bpb':>11}")
    for r in sorted(comp16, key=lambda r: r['bpb']):
        print(f"   {r['tag']:<6}{r['gather']:>7}{r['cells']:>7}"
              f"{r['gather']*r['cells']:>14,}{r['tables']:>14,}{r['bpb']:>11.6f}")
    best = min(comp16, key=lambda r: r['bpb'])
    print(f"\n   best run {best['tag']}: H*tph {best['gather']} (cap 1024), 2^nap {best['cells']},"
          f" buffer factor {best['gather']*best['cells']:,}")
    mx = max(comp16, key=lambda r: r['gather'] * r['cells'])
    print(f"   largest buffer factor: {mx['tag']} at {mx['gather']*mx['cells']:,}")
    print(f"   correlation of log2(H*tph*2^nap) with bpb across the 16k CompressionMHL runs: "
          f"{pearson([math.log2(r['gather']*r['cells']) for r in comp16], [r['bpb'] for r in comp16]):+.3f}")

    # ---------- 5. proxy vs 16k ----------
    print("\n" + "=" * 78)
    print("5. PROXY vs 16k — pairs where the same shape exists at both budgets")
    print("=" * 78)
    key = lambda r: (r['H'], r['tph'], r['cells'], r['d_in'], r['d_out'])
    pk = {key(r): r for r in proxy}
    both = [(r, pk[key(r)]) for r in comp16 if key(r) in pk]
    print(f"   {len(both)} shapes have both a 4k proxy run and a 16k run:")
    for a_, b_ in sorted(both, key=lambda t: t[0]['bpb']):
        print(f"      {a_['tag']:<6} H{a_['H']} tph{a_['tph']} c{a_['cells']} "
              f"in{a_['d_in']} out{a_['d_out']}   16k {a_['bpb']:.6f}   "
              f"4k {b_['bpb']:.6f} ({b_['tag']})")
    if len(both) > 1:
        print(f"\n   rank agreement over those {len(both)}:")
        o16 = [t[0]['tag'] for t in sorted(both, key=lambda t: t[0]['bpb'])]
        o4 = [t[0]['tag'] for t in sorted(both, key=lambda t: t[1]['bpb'])]
        print(f"      by 16k: {' < '.join(o16)}")
        print(f"      by 4k : {' < '.join(o4)}")
        print(f"      identical: {o16 == o4}")


if __name__ == '__main__':
    main()
