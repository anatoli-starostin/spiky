#!/usr/bin/env python3
"""Interleaved FFN-slot totals -- THE CITABLE NUMBERS, plus the torch.compile check.

    python paper_timings/interleaved.py --load-checkpoint

Why interleaved: phase_split.py's drift check shows ~6% drift between the first and last
measurement of the SAME vanilla slot over a long sequential session. Absolute ms measured
minutes apart are therefore not safely comparable. Here every variant is timed in
ALTERNATING rounds inside one process, so drift applies to all of them equally and the
RATIOS are sound; the [min-max] spread makes the residual noise visible.

The torch.compile variants are timed in the SAME interleave, so eager-vs-compiled is not
a comparison across two different thermal states either. Compilation itself is excluded
(the global warm-up calls every variant, compiled ones included, before timing starts).

NOTE for anyone quoting these: torch.compile speeds up the vanilla BASELINE (~1.10x here)
while doing nothing for the LUT models (it graph-breaks at the pybind11 custom op). If you
quote compiled-vanilla as the reference, the LUT ratios get correspondingly worse. Say
which baseline you used.
"""
import argparse
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
import gather_fused     # noqa: E402
import hybrid           # noqa: E402
import model as M       # noqa: E402

DEFAULT_EXPS = ('exp_n_0126_grid_H4d48_nap7_tph64',
                'exp_n_0127_grid_H4d48_nap7_tph128',
                'exp_n_0128_grid_H4d48_nap8_tph64')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=os.path.join(REPO, 'experiments', 'ffn_replacement', 'runs'))
    ap.add_argument('--exps', default=','.join(DEFAULT_EXPS))
    ap.add_argument('--baseline', default='exp_n_0135_untied_vanilla_baseline_16k')
    ap.add_argument('--compile-exp', default='exp_n_0127',
                    help='which LUT model to also run under torch.compile')
    ap.add_argument('--batch', type=int, default=48)
    ap.add_argument('--seq', type=int, default=512)
    ap.add_argument('--iters', type=int, default=30)
    ap.add_argument('--rounds', type=int, default=11)
    ap.add_argument('--warm', type=int, default=120)
    ap.add_argument('--no-compile', action='store_true')
    ap.add_argument('--load-checkpoint', action='store_true')
    args = ap.parse_args()

    B, SEQ, N = args.batch, args.seq, args.batch * args.seq
    torch.set_float32_matmul_precision('high')
    xb = torch.randn(B, SEQ, 384, device='cuda', dtype=torch.bfloat16)

    fns, keep = {}, []
    _, van = M.build(os.path.join(args.root, args.baseline))
    van = van.to(torch.bfloat16).eval()
    fns['vanilla (eager)'] = van.blocks[0].ffn_slot
    keep.append(van)

    compile_target = None
    for d in args.exps.split(','):
        label = d.split('_grid_')[0]
        _, ref = M.build(os.path.join(args.root, d),
                         load_checkpoint=args.load_checkpoint)
        m = M.build(os.path.join(args.root, d),
                    load_checkpoint=args.load_checkpoint)[1]
        m.load_state_dict(ref.state_dict())
        m = hybrid.apply(m)
        assert gather_fused.patch(m) == len(M.lut_modules(m))
        fns[f'{label} (fused, eager)'] = m.blocks[0].ffn_slot
        if label == args.compile_exp:
            compile_target = m
        keep.append(m)
        del ref

    if not args.no_compile:
        fns['vanilla (compiled)'] = torch.compile(van.blocks[0].ffn_slot)
        if compile_target is not None:
            fns[f'{args.compile_exp} (compiled)'] = torch.compile(
                compile_target.blocks[0].ffn_slot)

    # warm everything: clock ramp AND torch.compile compilation, both excluded from timing
    with torch.no_grad():
        for _ in range(args.warm):
            for f in fns.values():
                f(xb)
    torch.cuda.synchronize()

    try:
        clk = subprocess.run(['nvidia-smi', '--query-gpu=clocks.sm,clocks.max.sm',
                              '--format=csv,noheader'],
                             capture_output=True, text=True).stdout.strip()
    except Exception:
        clk = 'n/a'
    print(f'GPU {bench.gpu_name()} | torch {torch.__version__} | SM clock after warm: {clk}')
    print(f'batch {B} x seq {SEQ} = {N:,} tokens | {args.iters} iters x {args.rounds} '
          f'interleaved rounds | torch.no_grad | bf16\n')

    acc = {k: [] for k in fns}
    with torch.no_grad():
        for _ in range(args.rounds):
            for k, f in fns.items():
                acc[k].append(bench.timeit(lambda _f=f: _f(xb), iters=args.iters,
                                           warmup=2))

    base = statistics.median(acc['vanilla (eager)'])
    print(f'{"variant":<28}{"median ms":>11}{"[min-max]":>19}{"vs vanilla":>12}')
    for k, v in acc.items():
        md = statistics.median(v)
        print(f'{k:<28}{md:>11.4f}{f"[{min(v):.4f}-{max(v):.4f}]":>19}{md/base:>11.2f}x')

    if 'vanilla (compiled)' in acc:
        vc = statistics.median(acc['vanilla (compiled)'])
        print(f'\ntorch.compile on the baseline: {base:.4f} -> {vc:.4f} '
              f'({base/vc:.3f}x faster). Against the COMPILED vanilla, the LUT ratios are:')
        for k, v in acc.items():
            if 'fused, eager' in k:
                print(f'  {k.replace(" (fused, eager)", ""):<12} '
                      f'{statistics.median(v)/vc:.2f}x')


if __name__ == '__main__':
    main()
