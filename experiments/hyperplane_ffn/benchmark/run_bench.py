#!/usr/bin/env python3
"""Run the FFN-slot inference benchmark for one experiment against a baseline.

  python run_bench.py --exp exp_n_0126_grid_H4d48_nap7_tph64 \
                      --baseline exp_n_0135_untied_vanilla_baseline_16k

Reports, in order: model facts, the bit-exactness check, the FFN-slot breakdown, and
the interleaved end-to-end ladder with the vs-baseline ratio. Timing is never printed
before correctness passes.
"""
import argparse
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
# HERE = <repo>/experiments/hyperplane_ffn/benchmark, so the repo root is three up
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
for p in (HERE, os.path.join(REPO, 'src')):
    if p not in sys.path:
        sys.path.insert(0, p)

import bench            # noqa: E402
import gather           # noqa: E402
import gather_ternary   # noqa: E402
import hybrid           # noqa: E402
import model as M       # noqa: E402


def family_of(m):
    """'fastmhl' (CompressionMHL/anchor-pair), 'ternary', or 'dense'.

    Dispatching on this matters: gather.patch() matches only FastMultiHeadLut and
    would silently patch NOTHING on a ternary model, reporting an unoptimized run as
    if it were optimized.
    """
    if M.lut_modules(m):
        return 'fastmhl'
    if gather_ternary.ternary_modules(m):
        return 'ternary'
    return 'dense'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--exp', required=True, help='experiment dir name or path')
    ap.add_argument('--baseline', default='exp_n_0135_untied_vanilla_baseline_16k')
    ap.add_argument('--root', default=os.path.join(REPO, 'experiments', 'hyperplane_ffn'))
    ap.add_argument('--batches', default='12,48,96')
    ap.add_argument('--seq', type=int, default=512)
    ap.add_argument('--rounds', type=int, default=11)
    ap.add_argument('--iters', type=int, default=30)
    ap.add_argument('--load-checkpoint', action='store_true',
                    help='load trained weights (needed only if values matter, '
                         'e.g. realized ternary sparsity)')
    ap.add_argument('--no-warmup', action='store_true',
                    help='DISABLE the clock burn-in. Do not use: see README.')
    ap.add_argument('--addr-dtype', default='fp32', choices=('fp32', 'bf16'),
                    help='ternary family only: addressing GEMM precision. fp32 is '
                         'bit-exact; bf16 is faster but changes routing decisions.')
    ap.add_argument('--tune-gather', action='store_true',
                    help='sweep the Triton gather config on this GPU first')
    args = ap.parse_args()

    def resolve(name):
        return name if os.path.isdir(name) else os.path.join(args.root, name)

    exp_dir, base_dir = resolve(args.exp), resolve(args.baseline)
    batches = tuple(int(b) for b in args.batches.split(','))
    torch.set_float32_matmul_precision('high')
    print(f'GPU: {bench.gpu_name()} | torch {torch.__version__}')
    print(f'exp:      {os.path.basename(exp_dir)}')
    print(f'baseline: {os.path.basename(base_dir)}')

    cfg, ref = M.build(exp_dir, load_checkpoint=args.load_checkpoint)
    luts = M.lut_modules(ref)
    print(f'params {sum(p.numel() for p in ref.parameters()):,}'
          + (f' | tph {luts[0].tables_per_head} heads {luts[0].n_heads}'
             f' | table {tuple(luts[0].weights.shape)}'
             f' = {luts[0].weights.numel() * 4 / 2**20:.1f} MiB/slot fp32' if luts else ''))

    if args.tune_gather and luts:
        l = luts[0]
        width = getattr(l, '_fwd_input_dim', l.input_dim)
        x = torch.randn(48 * args.seq, width, device=ref.tok_emb.weight.device)
        idx = l._native_eval_msb(x, l.soft_anchor_a_long, l.soft_anchor_b_long, 0.0, 256)
        best = gather.tune(l.weights.data.contiguous(), idx, l.n_heads, l.tables_per_head)
        print(f'gather tune: {best[0]:.4f} ms at BLOCK_N={best[1]} warps={best[2]} '
              f'stages={best[3]} (defaults {gather.BLOCK_N}/{gather.NUM_WARPS}/'
              f'{gather.NUM_STAGES})')
        gather.BLOCK_N, gather.NUM_WARPS, gather.NUM_STAGES = best[1], best[2], best[3]

    fam = family_of(ref)
    n_slots = len(luts) if fam == 'fastmhl' else \
        len(gather_ternary.ternary_modules(ref)) if fam == 'ternary' else 0
    print(f'family: {fam} ({n_slots} LUT slots)')

    def do_patch(m):
        if fam == 'fastmhl':
            return gather.patch(m)
        if fam == 'ternary':
            return gather_ternary.patch(m, addr_dtype=args.addr_dtype)
        return 0

    if fam == 'dense':
        print('the --exp model has no LUT: nothing to optimize, aborting.')
        return 1

    # ---- correctness before any timing
    print('\nCORRECTNESS: Triton gather vs the unpatched fp32 model')
    cand = M.build(exp_dir, load_checkpoint=args.load_checkpoint)[1]
    cand.load_state_dict(ref.state_dict())
    n = do_patch(cand)
    assert n == n_slots, f'patched {n} of {n_slots} slots'
    ok, diffs = bench.check_bit_exact(ref, cand, seq=args.seq)
    for B, d in diffs.items():
        print(f'  batch {B:>3}: max|logit diff| {d:.3e}  '
              f'{"EXACT" if d == 0 else "DIFFERS"}')
    if not ok:
        if fam == 'ternary' and args.addr_dtype == 'bf16':
            print('\n  bf16 addressing is NOT bit-exact by construction: near a~0 the'
                  '\n  rounding flips the sign bit and selects a different table row.'
                  '\n  Re-run with --addr-dtype fp32 for an exact comparison.')
        else:
            print('\nNOT bit-exact -- refusing to report timings.')
            return 1
    else:
        print(f'  patched {n} LUT slots, all bit-exact')

    # ---- build the timed models. Load the reference weights BEFORE converting to
    # hybrid-v2 storage, so the timed model is the same model, not just the same shape.
    opt = M.build(exp_dir, load_checkpoint=args.load_checkpoint)[1]
    opt.load_state_dict(ref.state_dict())
    opt = hybrid.apply(opt)
    assert do_patch(opt) == n_slots, 'patch count mismatch'
    counter = hybrid.count_native_calls(opt)
    with torch.no_grad():
        opt(torch.randint(0, cfg['tokenizer_vocab_size'], (2, args.seq),
                          device=next(opt.parameters()).device))
    if fam == 'fastmhl':
        print(f'  native bit-pack kernel calls per forward: {counter["n"]}/{n_slots} '
              f'-> {"alive" if counter["n"] == n_slots else "LOST (fell back to the "
                   "compiled path -- check the fp32 input hook)"}')

    van = M.build(base_dir)[1].to(torch.bfloat16).eval()

    print(f'\nFFN SLOT, one block, batch 48 x {args.seq}')
    slot = bench.slot_breakdown({'exp shipped (fp32)': ref,
                                 'exp hybrid-v2 + Triton gather': opt,
                                 'baseline slot (bf16)': van},
                                batch=48, seq=args.seq, n_embd=cfg['n_embd'])
    for k, v in slot.items():
        print(f'  {k:<34} {v:.4f} ms')
    print(f'  -> optimized slot vs baseline slot: '
          f'{slot["exp hybrid-v2 + Triton gather"] / slot["baseline slot (bf16)"]:.2f}x')

    print(f'\nEND TO END, interleaved, {args.rounds} rounds'
          + ('  [WARM-UP DISABLED -- numbers unreliable]' if args.no_warmup else ''))
    res = bench.interleaved_ab({'exp optimized': opt, 'baseline': van},
                               batches=batches, seq=args.seq, rounds=args.rounds,
                               iters=args.iters, vocab=cfg['tokenizer_vocab_size'],
                               warm=not args.no_warmup)
    bench.report(res, 'baseline', batches)
    return 0


if __name__ == '__main__':
    sys.exit(main())
