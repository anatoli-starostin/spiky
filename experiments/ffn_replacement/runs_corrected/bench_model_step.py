"""Whole-model step cost: how much of a training step is the LUT FFN at all?

The layer micro-benchmark says Light's backward is much cheaper. That only matters end-to-end
in proportion to the FFN's share of the step, so measure the real model: 6 blocks, attention,
embeddings, the actual device_batch 12 x seq_len 512 shape, fp32 (lut_use_bf16=False in the
anchor config).

A dense-FFN model of the same shape gives the floor -- everything that is NOT the LUT layer --
so the ceiling on any LUT-side speedup can be stated rather than guessed.

    python bench_model_step.py
"""
import copy
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

from nanochat.common import get_base_dir                     # noqa: E402
from nanochat.tokenizer import RustBPETokenizer              # noqa: E402
from model_build import build_model                          # noqa: E402

DEV = 'cuda'
REPS, WARMUP = 15, 6
SRC = os.path.join(RC, 'sweep_s05_dout48_H4_tph256_c256_din32', 'config.json')


def timed(fn, reps=REPS, warmup=WARMUP):
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


def bench(label, over, vocab, ids):
    cfg = json.load(open(SRC))
    cfg.update(over)
    torch.manual_seed(cfg['random_seed'])
    m = build_model(cfg, vocab, device=DEV)
    torch.cuda.reset_peak_memory_stats()

    def step():
        m.zero_grad(set_to_none=True)
        out = m(ids)
        loss = out.float().mean() if not isinstance(out, tuple) else out[0].float().mean()
        loss.backward()

    def fwd_eval():
        with torch.no_grad():
            m(ids)

    t_step = timed(step)
    t_eval = timed(fwd_eval)
    peak = torch.cuda.max_memory_allocated() / 2 ** 30
    del m
    torch.cuda.empty_cache()
    return dict(label=label, step_ms=t_step, eval_ms=t_eval, peak_gib=peak)


def main():
    cfg = json.load(open(SRC))
    vocab = RustBPETokenizer.from_directory(
        os.path.join(get_base_dir(), 'tokenizer')).get_vocab_size()
    ids = torch.randint(0, vocab, (cfg['device_batch_size'], cfg['seq_len']), device=DEV)
    print(f"model step: device_batch {cfg['device_batch_size']} x seq_len {cfg['seq_len']}"
          f" = {cfg['device_batch_size'] * cfg['seq_len']:,} tokens, fp32")

    rows = [
        bench('dense FFN (the non-LUT floor)', dict(ffn_type='dense'), vocab, ids),
        bench('fast, gate off  (baseline S5)', {}, vocab, ids),
        bench("fast + bounded_norm  (A', C, D)",
              dict(lut_forward_confidence=True, lut_confidence_form='bounded_norm'),
              vocab, ids),
        bench('light + bounded_norm  (arm B)',
              dict(lut_impl='light', lut_forward_confidence=True,
                   lut_confidence_form='bounded_norm'), vocab, ids),
    ]
    print(f"\n   {'':<34}{'train step':>12}{'eval fwd':>11}{'peak GiB':>11}")
    for r in rows:
        print(f"   {r['label']:<34}{r['step_ms']:>10.1f}ms{r['eval_ms']:>9.1f}ms"
              f"{r['peak_gib']:>11.2f}")

    dense, base, gated, light = rows
    print(f"\n   The LUT FFN's share of a training step:")
    for r in (base, gated, light):
        share = (r['step_ms'] - dense['step_ms']) / r['step_ms']
        print(f"      {r['label']:<34} {share:6.1%} of the step is LUT-attributable "
              f"({r['step_ms'] - dense['step_ms']:.1f}ms of {r['step_ms']:.1f}ms)")
    print(f"\n   Ceiling on any LUT-side speedup, from Amdahl: even an infinitely fast LUT")
    print(f"   layer leaves the {dense['step_ms']:.1f}ms floor, so the best possible speedup")
    print(f"   over the gated arm is {gated['step_ms'] / dense['step_ms']:.2f}x.")
    print(f"   Light actually achieves {gated['step_ms'] / light['step_ms']:.2f}x "
          f"vs gated, {base['step_ms'] / light['step_ms']:.2f}x vs gate-off.")
    print(f"\n   EVAL forward (what inference costs):")
    for r in rows:
        print(f"      {r['label']:<34} {r['eval_ms']:7.1f}ms   "
              f"{r['eval_ms'] / base['eval_ms']:5.2f}x vs gate-off fast")

    json.dump(rows, open('/tmp/bench_model_step.json', 'w'), indent=2)


if __name__ == '__main__':
    main()
