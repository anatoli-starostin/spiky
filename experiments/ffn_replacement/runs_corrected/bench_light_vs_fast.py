"""Wall-clock: is LightMultiHeadLUT actually faster than FastMultiHeadLut?

Params and projection FLOPs were matched by construction; speed never was. Light's backward
is pure autograd through the score only -- no hand-written directional surrogate, no
soft/temperature machinery -- so it *should* be cheaper. This measures whether it is, and by
how much, at the sizes we actually train at.

Measured at the anchor sizing (H=4, tph=256, nap=8, d_in=32, d_out=48, n_embd=384) with the
training batch shape (device_batch 12 x seq_len 512 = 6,144 tokens).

NOTE ON DTYPE: the anchor config has lut_use_bf16=False, so the runs this compares against
trained in fp32. fp32 is therefore the headline number; bf16 is reported alongside for
completeness rather than substituted for it.

Three layers, each isolated (CompressionMHL alone) and then in the real model:
    fast, gate off        -- the baseline arm
    fast + bounded_norm   -- arms A', C, D
    light + bounded_norm  -- arm B

Eval is measured separately, because Fast has a native CUDA bit-pack kernel for the hard
forward that is DISABLED whenever forward_confidence is on ("native kernel has no score
gate"). That disabling may cost more at inference than anything the gate does at training.

    python bench_light_vs_fast.py
"""
import json
import os
import statistics
import sys
import time

import torch

FR = os.path.expanduser('~/projects/spiky/experiments/ffn_replacement')
RC = os.path.join(FR, 'runs_corrected')
sys.path.insert(0, os.path.join(FR, 'tools'))
sys.path.insert(0, os.path.expanduser('~/projects/nanochat'))
sys.path.insert(0, os.path.expanduser('~/projects/spiky/src'))

from spiky.lutorch.compression_mhl import CompressionMultiHeadLUT   # noqa: E402

DEV = torch.device('cuda:0')
TOKENS = 12 * 512                       # device_batch x seq_len, the real step shape
KW = dict(input_dim=384, output_dim=384, inner_in_dim=32, inner_out_dim=48,
          nap=8, tph=256, n_heads=4, joint_head_compression=False,
          initial_weights_noise=1e-3, random_seed=1000)
REPS, WARMUP = 40, 12


def timed(fn, reps=REPS, warmup=WARMUP):
    """Median ms/call with proper warmup and CUDA sync. Median, not mean: the first
    post-warmup calls still catch occasional recompiles and clock ramps."""
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
    return statistics.median(ts), statistics.stdev(ts) if len(ts) > 1 else 0.0


def build(impl, use_bf16, **over):
    torch.manual_seed(0)
    kw = dict(KW)
    if impl == 'fast':
        kw.update(forward_mode='hard', learnable_temps=True, use_bf16=use_bf16)
    m = CompressionMultiHeadLUT(**kw, lut_impl=impl, device=DEV, **over)
    with torch.no_grad():                    # decompress is zero-init: give it real scale
        m.decompress.weight.normal_(0, 2.3 / (384 * 192) ** 0.5)
    return m


def bench_layer(name, impl, use_bf16, **over):
    m = build(impl, use_bf16, **over)
    x = torch.randn(TOKENS, 384, device=DEV) * 0.6
    g = torch.randn(TOKENS, 384, device=DEV)

    def fwd_only():
        with torch.no_grad():
            m(x)

    def fwd_bwd():
        m.zero_grad(set_to_none=True)
        xx = x.detach().requires_grad_(True)
        m(xx).backward(g)

    def fwd_train():                         # forward with grad graph, no backward
        xx = x.detach().requires_grad_(True)
        m(xx)

    torch.cuda.reset_peak_memory_stats()
    f_eval, _ = timed(fwd_only)
    f_train, _ = timed(fwd_train)
    fb, fb_sd = timed(fwd_bwd)
    peak = torch.cuda.max_memory_allocated() / 2 ** 20
    del m, x, g
    torch.cuda.empty_cache()
    return dict(name=name, fwd_eval=f_eval, fwd_train=f_train, fwd_bwd=fb,
                bwd=fb - f_train, fb_sd=fb_sd, peak_mib=peak)


def main():
    print(f'device: {torch.cuda.get_device_name(0)}   tokens/call: {TOKENS:,}')

    for use_bf16 in (False, True):
        tag = 'bf16' if use_bf16 else 'fp32  <-- the dtype the runs actually used'
        print('\n' + '=' * 104)
        print(f'1. LAYER MICRO-BENCHMARK, {tag}')
        print('=' * 104)
        rows = [
            bench_layer('fast, gate off  (baseline S5)', 'fast', use_bf16),
            bench_layer("fast + bounded_norm  (A', C, D)", 'fast', use_bf16,
                        forward_confidence=True, confidence_form='bounded_norm'),
            bench_layer('light + bounded_norm  (arm B)', 'light', use_bf16,
                        forward_confidence=True, confidence_form='bounded_norm'),
        ]
        print(f"   {'':<34}{'fwd(eval)':>11}{'fwd(train)':>12}{'fwd+bwd':>10}"
              f"{'bwd only':>10}{'peak MiB':>11}")
        for r in rows:
            print(f"   {r['name']:<34}{r['fwd_eval']:>11.3f}{r['fwd_train']:>12.3f}"
                  f"{r['fwd_bwd']:>10.3f}{r['bwd']:>10.3f}{r['peak_mib']:>11.1f}")
        base, gated, light = rows
        print(f"\n   train step (fwd+bwd) — light vs gated fast : "
              f"{gated['fwd_bwd'] / light['fwd_bwd']:.2f}x faster")
        print(f"   train step (fwd+bwd) — light vs gate-off fast: "
              f"{base['fwd_bwd'] / light['fwd_bwd']:.2f}x faster")
        print(f"   backward alone      — light vs gated fast    : "
              f"{gated['bwd'] / max(light['bwd'], 1e-9):.2f}x faster")
        print(f"   EVAL forward        — gated fast vs gate-off : "
              f"{gated['fwd_eval'] / base['fwd_eval']:.2f}x SLOWER "
              f"(native CUDA kernel disabled by the gate)")
        print(f"   EVAL forward        — light vs gate-off fast : "
              f"{light['fwd_eval'] / base['fwd_eval']:.2f}x")
        if not use_bf16:
            json.dump([{k: v for k, v in r.items()} for r in rows],
                      open('/tmp/bench_layer_fp32.json', 'w'), indent=2)


if __name__ == '__main__':
    main()
