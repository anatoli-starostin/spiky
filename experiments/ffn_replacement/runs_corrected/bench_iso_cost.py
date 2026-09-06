"""Iso-cost: what does Light's speed advantage buy, in bpb, at equal training / inference cost?

The budget law from the 16k runs is bpb = 1.369488 - 0.007455 * log2(table_params), i.e.
-0.00746 bpb per doubling of table budget. So a speed advantage converts into quality only
through however many doublings it pays for.

Rather than ASSUME time scales linearly with table count, this measures it: the same layer at
tph = 128 / 256 / 512 / 1024, and fits the exponent. Then it converts.

Two framings, both reported:
  * equal TRAINING wall-clock -- how many extra doublings Light's faster step buys;
  * equal INFERENCE cost -- which runs the other way, because Light's eval forward is
    SLOWER than Fast's native path, so it must give budget back.

CAVEAT stated up front: the budget law was fitted on 16k-step runs and the arms here are 4k
proxies. Applying its slope at this budget is an extrapolation. It is the best estimate we
have, not a measurement of these runs.

    python bench_iso_cost.py
"""
import math
import os
import statistics
import sys
import time

import torch

sys.path.insert(0, os.path.expanduser('~/projects/spiky/src'))
from spiky.lutorch.compression_mhl import CompressionMultiHeadLUT      # noqa: E402

DEV = torch.device('cuda:0')
TOKENS = 12 * 512
SLOPE = 0.007455          # bpb per doubling of table params (16k budget law)

# measured, whole model, fp32, this GPU (bench_model_step.py)
MODEL = {'dense': (19.9, 6.8), 'fast_off': (217.8, 10.3),
         'fast_gate': (282.2, 11.6), 'light': (68.3, 21.3)}
BPB = {'fast_off': 1.434572, 'light': 1.477708, 'dense': 1.474749}
SD = 0.0096


def timed(fn, reps=20, warmup=8):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(ts)


def layer(impl, tph, **over):
    torch.manual_seed(0)
    kw = dict(input_dim=384, output_dim=384, inner_in_dim=32, inner_out_dim=48,
              nap=8, tph=tph, n_heads=4, joint_head_compression=False,
              initial_weights_noise=1e-3, random_seed=1000)
    if impl == 'fast':
        kw.update(forward_mode='hard', learnable_temps=True, use_bf16=False)
    m = CompressionMultiHeadLUT(**kw, lut_impl=impl, device=DEV, **over)
    x = torch.randn(TOKENS, 384, device=DEV) * 0.6
    g = torch.randn(TOKENS, 384, device=DEV)

    def fwd():
        with torch.no_grad():
            m(x)

    def fwd_bwd():
        m.zero_grad(set_to_none=True)
        xx = x.detach().requires_grad_(True)
        m(xx).backward(g)

    t_f, t_fb = timed(fwd), timed(fwd_bwd)
    del m, x, g
    torch.cuda.empty_cache()
    return t_f, t_fb


def fit_exponent(tphs, times):
    """time ~ tph**k; return k by least squares in log-log."""
    lx = [math.log2(t) for t in tphs]
    ly = [math.log2(v) for v in times]
    mx, my = sum(lx) / len(lx), sum(ly) / len(ly)
    num = sum((a - mx) * (b - my) for a, b in zip(lx, ly))
    den = sum((a - mx) ** 2 for a in lx)
    return num / den


def main():
    print('HOW TRAINING/EVAL TIME SCALES WITH TABLE BUDGET (measured, not assumed)')
    print('=' * 92)
    tphs = [128, 256, 512, 1024]
    print(f"   {'tph':>6}{'light fwd':>12}{'light f+b':>12}{'fast f+b':>12}")
    lf, lfb, ffb = [], [], []
    for t in tphs:
        a, b = layer('light', t, forward_confidence=True, confidence_form='bounded_norm')
        try:
            _, c = layer('fast', t, forward_confidence=True, confidence_form='bounded_norm')
        except torch.cuda.OutOfMemoryError:
            c = float('nan')
            torch.cuda.empty_cache()
        lf.append(a), lfb.append(b), ffb.append(c)
        print(f'   {t:>6}{a:>12.3f}{b:>12.3f}{c:>12.3f}')
    k_train = fit_exponent(tphs, lfb)
    k_eval = fit_exponent(tphs, lf)
    ok = [(t, c) for t, c in zip(tphs, ffb) if c == c]
    k_fast = fit_exponent([t for t, _ in ok], [c for _, c in ok]) if len(ok) > 1 else float('nan')
    print(f'\n   light train time ~ tph^{k_train:.3f}     light eval time ~ tph^{k_eval:.3f}'
          f'     fast train time ~ tph^{k_fast:.3f}')
    print('   (k = 1 would mean a doubling of tables costs a doubling of time)')

    print('\n' + '=' * 92)
    print('EQUAL TRAINING WALL-CLOCK: what does Light\'s faster step buy?')
    print('=' * 92)
    for ref, name in (('fast_gate', 'gated Fast (arms A\', C, D)'),
                      ('fast_off', 'gate-off Fast (baseline S5)')):
        speedup = MODEL[ref][0] / MODEL['light'][0]
        doublings = math.log2(speedup) / max(k_train, 1e-9)
        gain = doublings * SLOPE
        print(f'   vs {name:<28} step {MODEL[ref][0]:6.1f} -> {MODEL["light"][0]:.1f} ms'
              f'  = {speedup:.2f}x')
        print(f'      -> {doublings:.2f} doublings of table budget'
              f'  -> {gain:.4f} bpb recovered')
    gap = BPB['light'] - BPB['fast_off']
    sp = MODEL['fast_off'][0] / MODEL['light'][0]
    dbl = math.log2(sp) / max(k_train, 1e-9)
    print(f'\n   gap to close (light - fast_off) = {gap:+.4f}')
    print(f'   recovered at equal train cost   = {dbl * SLOPE:.4f}  '
          f'({dbl * SLOPE / gap:.0%} of it)')
    print(f'   remaining                        = {gap - dbl * SLOPE:+.4f}')

    print('\n' + '=' * 92)
    print('EQUAL INFERENCE COST: the comparison this line actually cares about')
    print('=' * 92)
    for ref, name in (('fast_off', 'gate-off Fast'), ('dense', 'vanilla dense FFN')):
        ratio = MODEL['light'][1] / MODEL[ref][1]          # light is SLOWER -> ratio > 1
        doublings = -math.log2(ratio) / max(k_eval, 1e-9)  # negative: budget must SHRINK
        cost = -doublings * SLOPE
        adj = BPB['light'] + cost
        print(f'   vs {name:<20} eval {MODEL[ref][1]:5.1f} vs {MODEL["light"][1]:.1f} ms'
              f'  = light is {ratio:.2f}x MORE expensive')
        print(f'      -> must give back {-doublings:.2f} doublings  -> {cost:+.4f} bpb')
        print(f'      -> light at iso-inference-cost: {adj:.4f}   vs {name} '
              f'{BPB[ref]:.4f}   ({adj - BPB[ref]:+.4f})')

    print(f'\n   (seed sd {SD}; the budget-law slope is extrapolated from 16k runs)')


if __name__ == '__main__':
    main()
