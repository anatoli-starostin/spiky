"""Generate the SECOND proxy sweep — 6 runs that probe what the first 11 left ambiguous.

Identical setup to sweep 1 in every respect that matters for comparability: 4,000 steps,
effective batch 24 sequences (12,288 tokens), lr 3e-4 with 10% linear warmup and cosine decay
to 0.1x peak parameterised by n_steps=4000, and the corrected eval protocol
(`evaluate_bpb_fixed`, bs48 x 100, skip 12, 2,451,456 val tokens) every 500 steps. All 17
proxy runs are therefore mutually comparable — and, as always, comparable to NOTHING on the
16k / batch-48 line.

What each run is for:

  R3  cells 128, d_out 96   — iso-param with S5 (~105M). If they tie inside the ~0.002 noise
                              floor, the driver is TABLE PARAMS, not d_out specifically, and
                              sweep 1's headline needs revising.
  R4  d_in 64, d_out 48     — the recommended stack. Do the d_in and d_out gains add?
  R1  d_out 64              — d_out ladder rung 4. Has the increasing return turned over?
  R5  cells 128, d_in 64,
      d_out 64              — the same stacking question at the 80M size class (vs S7).
  R6  H2, d_in 64, d_out 48 — head trade at the winning config. H2 halves BOTH projections,
                              so a tie here is a free efficiency win.
  R2  d_out 96              — d_out ladder rung 5, upper bracket (~181M).

    python make_sweep2.py
"""
import json
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
FR = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(FR, 'tools'))
sys.path.insert(0, os.path.expanduser('~/projects/nanochat'))

from make_sweep import build_cfg, SEQS, N_STEPS, EVAL_EVERY   # noqa: E402  same generator

VANILLA_FFN_MACS = 2 * 384 * 1536      # 1,179,648 MACs/token for the dense 4x MLP

# name, order, H, tph, cells, d_in, d_out, expected, note
SWEEP2 = [
    ('r3_isoparam_c128_dout96', 1, 4, 256, 128, 32, 96, 105_000_000,
     'ISO-PARAM WITH S5 (~105M) BY A DIFFERENT ROUTE: cells 128 / d_out 96 against S5\'s '
     'cells 256 / d_out 48. Same table budget, opposite split. A tie inside the noise floor '
     'would mean sweep 1 measured TABLE PARAMS, not d_out, and the headline changes.'),
    ('r4_stack_din64_dout48', 2, 4, 256, 256, 64, 48, 105_200_000,
     'THE RECOMMENDED STACK: sweep 1\'s two independent real effects together, d_in 32->64 '
     '(-0.005 for +0.3M params) and d_out 32->48 (-0.020 for +25.3M). Tests whether they add.'),
    ('r1_dout64', 3, 4, 256, 256, 32, 64, 130_000_000,
     'd_out LADDER RUNG 4 (16/32/48/64). Sweep 1 found an INCREASING return across 16->32->48; '
     'this is where it either keeps going or turns over.'),
    ('r5_stack80m_c128_din64_dout64', 4, 4, 256, 128, 64, 64, 80_200_000,
     'STACKING AT THE 80M SIZE CLASS — S7 (cells 128, d_out 64) plus d_in 64. Pairs with R4 '
     'to say whether the d_in/d_out interaction depends on scale.'),
    ('r6_H2_din64_dout48', 5, 2, 512, 256, 64, 48, 105_000_000,
     'HEAD TRADE AT THE WINNING CONFIG: R4 with H 4->2. H2 halves BOTH projections, so if the '
     'H4-vs-H2 gap has closed (it was decaying to the noise floor by step 4000 in sweep 1) '
     'this is a free halving of projection FLOPs.'),
    ('r2_dout96', 6, 4, 256, 256, 32, 96, 180_900_000,
     'd_out LADDER RUNG 5, upper bracket (~181M). The far end, to bound where d_out stops '
     'paying at all.'),
]


def main():
    from nanochat.common import get_base_dir
    from nanochat.tokenizer import RustBPETokenizer
    from model_build import build_model
    tok = RustBPETokenizer.from_directory(os.path.join(get_base_dir(), 'tokenizer'))
    vocab = tok.get_vocab_size()

    print(f'{"run":<40} {"built":>12} {"expected":>12} {"dev":>4} {"dev%":>7}  status')
    order, bad = [], []
    for name, idx, H, tph, cells, d_in, d_out, expect, note in SWEEP2:
        d = os.path.join(HERE, f'sweep_{name}')
        os.makedirs(d, exist_ok=True)
        cfg = build_cfg(name, H, tph, cells, d_in, d_out, note)
        with open(os.path.join(d, 'config.json'), 'w') as f:
            json.dump(cfg, f, indent=2)
        shutil.copy(os.path.join(FR, 'train_fixed.py'), os.path.join(d, 'train.py'))
        m = build_model(cfg, vocab, device='cpu')
        tot = sum(p.numel() for p in m.parameters())
        del m
        dev = (tot - expect) / expect
        ok = abs(dev) <= 0.01
        if not ok:
            bad.append((name, tot, expect, dev))
        print(f'sweep_{name:<34} {tot:>12,} {expect:>12,} '
              f'{cfg["device_batch_size"]:>4} {100*dev:>+6.2f}%  '
              f'{"OK" if ok else "*** OUT OF TOLERANCE ***"}')
        cf, df = H * 384 * d_in, H * 384 * d_out
        order.append(dict(idx=idx, run=f'sweep_{name}', params=tot, expected=expect,
                          deviation=dev, device_batch_size=cfg['device_batch_size'],
                          grad_accum=SEQS // cfg['device_batch_size'],
                          H=H, tph=tph, cells=cells, d_in=d_in, d_out=d_out,
                          compress_flops=cf, decompress_flops=df,
                          projection_flops_total=cf + df,
                          compress_flops_ratio=cf / 589824,
                          projection_flops_ratio_vs_vanilla_ffn=(cf + df) / VANILLA_FFN_MACS))
    order.sort(key=lambda r: r['idx'])
    with open(os.path.join(HERE, 'sweep2_manifest.json'), 'w') as f:
        json.dump(dict(n_steps=N_STEPS, effective_batch_sequences=SEQS,
                       effective_batch_tokens=SEQS * 512, eval_every=EVAL_EVERY,
                       vanilla_ffn_macs_per_token=VANILLA_FFN_MACS, runs=order), f, indent=2)
    print(f'\nrun order: {" -> ".join(r["run"].split("_")[1] for r in order)}')
    print(f'wrote {HERE}/sweep2_manifest.json')
    if bad:
        print('\n*** STOP: param counts out of the 1% tolerance ***')
        for n, t, e, dv in bad:
            print(f'   {n}: built {t:,} vs expected {e:,} ({100*dv:+.2f}%)')
        sys.exit(1)
    print('all 6 param counts within 1% of the brief — clear to run')


if __name__ == '__main__':
    main()
