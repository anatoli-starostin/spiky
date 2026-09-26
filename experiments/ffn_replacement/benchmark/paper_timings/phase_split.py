#!/usr/bin/env python3
"""FFN-slot phase splits for the paper: per-phase ms and % of total.

    python paper_timings/phase_split.py --load-checkpoint

Writes results.json (consumed by make_figure.py) and prints the tables.

Phases timed, for the CompressionMHL slot:
    1. compress Linear 384->192
    2. anchor-compare -> index          } ONE kernel under --gather-impl cuda-fused,
    3. LUT gather+sum                   } so they are reported together there
    4. decompress Linear 192->384
and for the vanilla dense slot: Linear 384->1536 (+GELU), Linear 1536->384.

Phases are timed on the REAL intermediate tensors, captured by forward hooks from an
actual slot call, so no phase is timed on a synthetic input of the wrong dtype. The
phase sum is printed against the independently measured total, so any unaccounted
residual ("other": reshape/dtype glue) is visible rather than hidden.

TWO MEASUREMENT RULES THIS SCRIPT ENFORCES, both learned the hard way here:
  * a GLOBAL warm-up before any measurement. The per-call 60-iteration burn-in is NOT
    enough on its own: measured cold-first the vanilla slot reads 0.378 ms and warmed
    0.344 -- a 10% error that lands on whichever model happens to be timed first.
  * a DRIFT CHECK. The identical vanilla slot re-measured at the end of a long session
    came back ~6% slower. Absolute ms taken minutes apart are therefore not safely
    comparable; use interleaved.py for the citable totals and ratios.
"""
import argparse
import json
import os
import statistics
import subprocess
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.dirname(HERE)
REPO = os.path.dirname(os.path.dirname(os.path.dirname(BENCH)))
for p in (BENCH, os.path.join(REPO, 'src')):
    if p not in sys.path:
        sys.path.insert(0, p)

import bench            # noqa: E402
import gather_cuda      # noqa: E402
import gather_fused     # noqa: E402
import hybrid           # noqa: E402
import model as M       # noqa: E402

DEFAULT_EXPS = ('exp_n_0126_grid_H4d48_nap7_tph64',
                'exp_n_0127_grid_H4d48_nap7_tph128',
                'exp_n_0128_grid_H4d48_nap8_tph64')


def clocks():
    try:
        o = subprocess.run(['nvidia-smi', '--query-gpu=clocks.sm,clocks.max.sm',
                            '--format=csv,noheader'], capture_output=True, text=True)
        return o.stdout.strip() or 'n/a'
    except Exception:
        return 'n/a'


def capture(model, xb):
    """Grab the real tensors flowing between the slot's phases."""
    c = model.blocks[0].ffn
    got, hs = {}, []
    hs.append(c.compress.register_forward_hook(
        lambda m, i, o: got.__setitem__('xf', i[0].detach()) or
        got.__setitem__('z', o.detach())))
    hs.append(c.lut_batched.register_forward_hook(
        lambda m, i, o: got.__setitem__('zl', i[0].detach()) or
        got.__setitem__('y', o.detach())))
    hs.append(c.decompress.register_forward_hook(
        lambda m, i, o: got.__setitem__('yd', i[0].detach())))
    with torch.no_grad():
        model.blocks[0].ffn_slot(xb)
    for h in hs:
        h.remove()
    return got


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=os.path.join(REPO, 'experiments', 'ffn_replacement', 'runs'))
    ap.add_argument('--exps', default=','.join(DEFAULT_EXPS))
    ap.add_argument('--baseline', default='exp_n_0135_untied_vanilla_baseline_16k')
    ap.add_argument('--batch', type=int, default=48)
    ap.add_argument('--seq', type=int, default=512)
    ap.add_argument('--iters', type=int, default=30)
    ap.add_argument('--reps', type=int, default=7)
    ap.add_argument('--warm', type=int, default=300, help='global warm-up calls')
    ap.add_argument('--load-checkpoint', action='store_true')
    ap.add_argument('--out', default=os.path.join(HERE, 'results.json'))
    args = ap.parse_args()

    B, SEQ = args.batch, args.seq
    N = B * SEQ
    torch.set_float32_matmul_precision('high')

    def timed(fn):
        with torch.no_grad():
            for _ in range(60):                       # per-call clock burn-in
                fn()
            torch.cuda.synchronize()
            return statistics.median(
                [bench.timeit(fn, iters=args.iters) for _ in range(args.reps)])

    out = {'gpu': bench.gpu_name(), 'torch': torch.__version__,
           'batch': B, 'seq': SEQ, 'tokens': N, 'dtype': 'bf16',
           'iters': args.iters, 'reps': args.reps,
           'clock_cold': clocks(), 'models': {}}

    print(f'GPU {out["gpu"]} | torch {out["torch"]}')
    print(f'batch {B} x seq {SEQ} = {N:,} tokens/call | {args.iters} iters x '
          f'{args.reps} reps (median) | 60-iter burn-in | torch.no_grad')
    print(f'SM clock BEFORE warm-up: {out["clock_cold"]}')

    _, van = M.build(os.path.join(args.root, args.baseline))
    van = van.to(torch.bfloat16).eval()
    xb = torch.randn(B, SEQ, 384, device='cuda', dtype=torch.bfloat16)
    with torch.no_grad():                             # GLOBAL warm-up: see docstring
        for _ in range(args.warm):
            van.blocks[0].ffn_slot(xb)
    torch.cuda.synchronize()
    out['clock_warm'] = clocks()
    print(f'SM clock AFTER warm-up:  {out["clock_warm"]}\n')

    mlp = van.blocks[0].mlp
    lin1, act, lin2 = mlp[0], mlp[1], mlp[2]
    with torch.no_grad():
        h1 = act(lin1(xb))
    t_van = timed(lambda: van.blocks[0].ffn_slot(xb))
    t_v1 = timed(lambda: act(lin1(xb)))
    t_v2 = timed(lambda: lin2(h1))
    print('=== vanilla dense FFN slot (bf16) ===')
    print(f'  TOTAL                          {t_van:.4f} ms')
    print(f'  1. Linear 384->1536 + GELU     {t_v1:.4f} ms  ({t_v1/t_van*100:4.1f}%)')
    print(f'  2. Linear 1536->384            {t_v2:.4f} ms  ({t_v2/t_van*100:4.1f}%)')
    print(f'  phase sum {t_v1+t_v2:.4f} vs total {t_van:.4f} '
          f'(other {t_van-(t_v1+t_v2):+.4f})\n')
    out['models']['vanilla'] = {
        'total': t_van, 'phases': [['Linear 384->1536 + GELU', t_v1],
                                   ['Linear 1536->384', t_v2],
                                   ['other', max(t_van - t_v1 - t_v2, 0.0)]]}

    for d in args.exps.split(','):
        label = d.split('_grid_')[0]
        _, ref = M.build(os.path.join(args.root, d),
                         load_checkpoint=args.load_checkpoint)
        res = {}
        for impl in ('cuda-fused', 'cuda-bf16'):
            m = M.build(os.path.join(args.root, d),
                        load_checkpoint=args.load_checkpoint)[1]
            m.load_state_dict(ref.state_dict())
            m = hybrid.apply(m)
            n = (gather_fused.patch(m) if impl == 'cuda-fused'
                 else gather_cuda.patch(m, table_dtype='bf16'))
            assert n == len(M.lut_modules(m)), f'patched {n}'
            c = m.blocks[0].ffn
            g = capture(m, xb)
            r = dict(total=timed(lambda: m.blocks[0].ffn_slot(xb)),
                     compress=timed(lambda: c.compress(g['xf'])),
                     lut=timed(lambda: c.lut_batched(g['zl'])),
                     decompress=timed(lambda: c.decompress(g['yd'].reshape(N, -1))))
            if impl == 'cuda-bf16':                   # phases 2 and 3 are separable here
                l = M.lut_modules(m)[0]
                zf = g['zl'].reshape(N, -1).float().contiguous()
                A, Bq = l.soft_anchor_a_long, l.soft_anchor_b_long
                tbl = gather_cuda.prepare_table(l.weights.data, 'bf16')
                with torch.no_grad():
                    idx = l._native_eval_msb(zf, A, Bq, 0.0, 256)
                r['route'] = timed(lambda: l._native_eval_msb(zf, A, Bq, 0.0, 256))
                r['gather'] = timed(lambda: gather_cuda.gather_sum(
                    tbl, idx, l.n_heads, l.tables_per_head))
            res[impl] = r
            del m
            torch.cuda.empty_cache()

        f, s = res['cuda-fused'], res['cuda-bf16']
        ps = f['compress'] + f['lut'] + f['decompress']
        print(f'=== {label} | FFN slot, fused path (cuda-fused) ===')
        print(f'  TOTAL                          {f["total"]:.4f} ms   '
              f'({f["total"]/t_van:.2f}x vanilla slot)')
        print(f'  1. compress Linear 384->192    {f["compress"]:.4f} ms  '
              f'({f["compress"]/f["total"]*100:4.1f}%)')
        print(f'  2+3. routing + gather (FUSED)  {f["lut"]:.4f} ms  '
              f'({f["lut"]/f["total"]*100:4.1f}%)   <- one kernel, not separable')
        print(f'  4. decompress Linear 192->384  {f["decompress"]:.4f} ms  '
              f'({f["decompress"]/f["total"]*100:4.1f}%)')
        print(f'  phase sum {ps:.4f} vs total {f["total"]:.4f} '
              f'(other {f["total"]-ps:+.4f} = reshape/dtype glue)')
        print(f'  separable reference (cuda-bf16, native routing + separate gather):')
        print(f'    TOTAL                        {s["total"]:.4f} ms   '
              f'({s["total"]/t_van:.2f}x vanilla)')
        print(f'    1. compress                  {s["compress"]:.4f} ms')
        print(f'    2. anchor-compare -> index   {s["route"]:.4f} ms  '
              f'({s["route"]/s["total"]*100:4.1f}%)')
        print(f'    3. LUT gather+sum            {s["gather"]:.4f} ms  '
              f'({s["gather"]/s["total"]*100:4.1f}%)')
        print(f'    4. decompress                {s["decompress"]:.4f} ms')
        print(f'    (2)+(3) separate {s["route"]+s["gather"]:.4f} vs fused {f["lut"]:.4f}'
              f' -> fusion saves {s["route"]+s["gather"]-f["lut"]:+.4f} ms\n')

        out['models'][label] = {
            'total': f['total'], 'vs_vanilla': f['total'] / t_van,
            'phases': [['compress 384->192', f['compress']],
                       ['routing + gather (fused)', f['lut']],
                       ['decompress 192->384', f['decompress']],
                       ['other', max(f['total'] - ps, 0.0)]],
            'separable': s}
        del ref
        torch.cuda.empty_cache()

    t_van2 = timed(lambda: van.blocks[0].ffn_slot(xb))
    out['drift_pct'] = abs(t_van2 - t_van) / t_van * 100
    out['clock_end'] = clocks()
    print(f'DRIFT CHECK: vanilla slot first {t_van:.4f} ms, re-measured last '
          f'{t_van2:.4f} ms -> {out["drift_pct"]:.1f}% drift')
    print(f'SM clock at end: {out["clock_end"]}')
    print('  -> absolute ms taken minutes apart are NOT safely comparable; use '
          'interleaved.py for citable totals/ratios.')

    with open(args.out, 'w') as fh:
        json.dump(out, fh, indent=2)
    print(f'\nwrote {args.out}')


if __name__ == '__main__':
    main()
