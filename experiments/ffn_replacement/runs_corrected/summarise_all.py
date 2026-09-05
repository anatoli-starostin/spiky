"""Combined table and axis analysis over every proxy run in runs_corrected/.

Reads each run's corrected_score.json and every sweep*_manifest.json, prints the ranking with FULL
projection-FLOPs accounting (compress AND decompress, against vanilla's whole FFN cost of
2*384*1536 = 1,179,648 MACs/token), and works the axis comparisons the two sweeps set up.

    python summarise_all.py            # table + analysis
    python summarise_all.py --json     # same, as JSON on stdout
"""
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
VANILLA_FFN_MACS = 2 * 384 * 1536


def load():
    runs = {}
    for p in sorted(glob.glob(os.path.join(HERE, 'sweep*_manifest.json'))):
        for r in json.load(open(p))['runs']:
            sp = os.path.join(HERE, r['run'], 'corrected_score.json')
            if not os.path.exists(sp):
                continue
            s = json.load(open(sp))
            H, d_in, d_out = r['H'], r['d_in'], r['d_out']
            cf = (H * 384 * d_in) if H else None
            df = (H * 384 * d_out) if H else None
            tag = r['run'].split('_')[1].upper()
            runs[tag] = dict(
                tag=tag, run=r['run'], H=H, tph=r['tph'], cells=r['cells'],
                d_in=d_in, d_out=d_out, params=s['total_params'],
                table_params=(6 * H * r['tph'] * r['cells'] * d_out) if H else None,
                bpb=s['proxy_val_bpb'], d0=s.get('delta_vs_sweep_vanilla', 0.0),
                hours=s['training_time_hours'],
                batch=f"{s['training_budget']['device_batch_size']}x"
                      f"{s['training_budget']['grad_accum']}",
                compress=cf, decompress=df,
                proj=(cf + df) if H else VANILLA_FFN_MACS,
                ratio=((cf + df) / VANILLA_FFN_MACS) if H else 1.0,
                curve=[float(l.split(',')[2]) for l in
                       open(os.path.join(HERE, r['run'], 'metrics.csv')
                            ).read().strip().split('\n')[1:]])
    return runs


def main():
    R = load()
    if '--json' in sys.argv:
        print(json.dumps(R, indent=2))
        return
    order = sorted(R.values(), key=lambda r: r['bpb'])
    S1, S0 = R['S01'], R['S00']
    print(f"ALL {len(R)} PROXY RUNS — 4,000 steps, effective batch 24 sequences, "
          f"corrected eval (bs48 x 100, skip 12).")
    print("Comparable to each other ONLY. Never to the 16k / batch-48 anchors.\n")
    print(f"{'':<5}{'shape':<33}{'params':>12}{'tables':>12}{'bpb':>10}{'vs S0':>10}"
          f"{'vs S1':>10}{'compress':>10}{'decomp':>9}{'proj/van':>10}{'batch':>7}{'h':>6}")
    for r in order:
        shape = ('dense 4x MLP' if r['H'] is None else
                 f"H{r['H']} tph{r['tph']} c{r['cells']} in{r['d_in']} out{r['d_out']}")
        tp = f"{r['table_params']:,}" if r['table_params'] else '-'
        print(f"{r['tag']:<5}{shape:<33}{r['params']:>12,}{tp:>12}{r['bpb']:>10.6f}"
              f"{r['d0']:>+10.6f}{r['bpb']-S1['bpb']:>+10.6f}"
              f"{(r['compress'] or 0):>10,}{(r['decompress'] or 0):>9,}"
              f"{r['ratio']:>9.4f}x{r['batch']:>7}{r['hours']:>6.2f}")

    def line(title, tags, key):
        print(f"\n{title}")
        prev = None
        for t in tags:
            r = R[t]
            step = '' if prev is None else f"  step {r['bpb']-R[prev]['bpb']:+.6f}"
            dp = '' if prev is None else f"  ({r['params']-R[prev]['params']:+,} params)"
            print(f"   {t:<4} {key}={r[key]:>4}  params {r['params']:>12,}  "
                  f"bpb {r['bpb']:.6f}{step}{dp}")
            prev = t

    line("(a) d_out ladder at cells 256, d_in 32", ['S04', 'S01', 'S05', 'R1', 'R2'], 'd_out')
    print(f"\n(b) iso-table 75,497,472 at ~105M: S05 {R['S05']['bpb']:.6f} vs "
          f"R3 {R['R3']['bpb']:.6f}  ->  {R['R3']['bpb']-R['S05']['bpb']:+.6f}")
    print(f"    iso-table 50,331,648 at ~80M:  S01 {R['S01']['bpb']:.6f} vs "
          f"S07 {R['S07']['bpb']:.6f}  ->  {R['S07']['bpb']-R['S01']['bpb']:+.6f}")
    print("\n(c) the d_in 32->64 effect, by table budget")
    for a, b, lbl in (('S01', 'S02', 'c256 out32, tables 50.3M'),
                      ('S07', 'R5', 'c128 out64, tables 50.3M'),
                      ('S05', 'R4', 'c256 out48, tables 75.5M'),
                      ('U1', 'U2', 'c64  out48, tables 18.9M')):
        if a in R and b in R:
            print(f"   {lbl:<28} {a} {R[a]['bpb']:.6f} -> {b} {R[b]['bpb']:.6f}  "
                  f"{R[b]['bpb']-R[a]['bpb']:+.6f}")
    if all(t in R for t in ('U1', 'U2', 'U3')):
        print("\n    the SMALL-BUDGET d_in ladder (tables pinned at 18,874,368, only d_in moves)")
        for t in ('U1', 'U2', 'U3'):
            r = R[t]
            print(f"      {t}  d_in {r['d_in']:>3}  H*d_in {r['H']*r['d_in']:>3}  "
                  f"bpb {r['bpb']:.6f}  vs U1 {r['bpb']-R['U1']['bpb']:+.6f}  "
                  f"proj FLOPs {r['ratio']:.4f}x")
        sp = max(R[t]['bpb'] for t in ('U1', 'U2', 'U3')) - \
            min(R[t]['bpb'] for t in ('U1', 'U2', 'U3'))
        print(f"      spread {sp:.6f}   (noise floor ~0.002)")
        print(f"      SIGN OF THE d_in EFFECT: 75.5M tables {R['R4']['bpb']-R['S05']['bpb']:+.6f} "
              f"(more d_in HURTS)  vs  18.9M tables "
              f"{R['U3']['bpb']-R['U1']['bpb']:+.6f} (more d_in HELPS)")
    print("\n(d) head trade H4 -> H2, and what actually drives it")
    for a, b, lbl in (('S01', 'S08', 'd_in 32, d_out 32'),
                      ('R4', 'R6', 'd_in 64, d_out 48'),
                      ('S05', 'R7', 'd_in 32, d_out 48')):
        if a in R and b in R:
            print(f"   {lbl:<20} {a} {R[a]['bpb']:.6f} -> {b} {R[b]['bpb']:.6f}  "
                  f"{R[b]['bpb']-R[a]['bpb']:+.6f}   proj FLOPs "
                  f"{R[a]['ratio']:.4f}x -> {R[b]['ratio']:.4f}x")
    quad = [t for t in ('R7', 'S05', 'R6', 'R4') if t in R]
    if len(quad) == 4:
        print("    at cells 256 / d_out 48 — the rule is the TOTAL routing code width "
              "H*d_in, not H:")
        for t in sorted(quad, key=lambda t: R[t]['H'] * R[t]['d_in']):
            r = R[t]
            print(f"      {t:<4} H{r['H']} x d_in {r['d_in']:>3} = "
                  f"{r['H']*r['d_in']:>3} total code   bpb {r['bpb']:.6f}"
                  f"{'   <- starved' if r['H']*r['d_in'] < 128 else ''}")
    print("\n(e) quality per projection-FLOP — bpb below S0 per unit of proj/vanilla ratio")
    for r in sorted((r for r in R.values() if r['H']),
                    key=lambda r: r['d0'] / r['ratio'])[:6]:
        print(f"   {r['tag']:<4} {-r['d0']:.6f} below S0 at {r['ratio']:.4f}x  "
              f"-> {-r['d0']/r['ratio']:.4f} bpb per unit FLOPs ratio  "
              f"({r['params']:,} params)")


if __name__ == '__main__':
    main()
