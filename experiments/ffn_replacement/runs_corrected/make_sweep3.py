"""Generate R7 — the one cell neither sweep covered: the H2 counterpart of S5.

S5 (H4, tph256, cells256, d_in 32, d_out 48) is the best of the 17 proxy runs. R6 showed that
halving the heads at d_in 64 / d_out 48 costs +0.000103 bpb — twenty times below the noise
floor — while halving both dense projections. Whether that also holds at S5's own d_in 32 does
NOT follow from R6: halving heads at d_in 32 also halves the TOTAL routing code width
(4x32 = 128 -> 2x32 = 64), which is a different change from the one R6 tested
(4x64 = 256 -> 2x64 = 128). R7 measures it directly.

If R7 ties S5 inside the ~0.002 noise floor it dominates both standing recommendations: S5's
quality at 0.0521x of vanilla's FFN projection cost instead of 0.1042x.

Identical budget and recipe to all 17 (make_sweep.build_cfg is the shared generator).

    python make_sweep3.py
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

from make_sweep import build_cfg, SEQS, N_STEPS, EVAL_EVERY   # noqa: E402

VANILLA_FFN_MACS = 2 * 384 * 1536

SWEEP3 = [
    ('r7_H2_din32_dout48', 1, 2, 512, 256, 32, 48, 104_800_000,
     'THE H2 COUNTERPART OF S5 — the cell neither sweep covered. S5 (H4 tph256 c256 in32 '
     'out48) is the best of the 17 proxy runs; R6 showed H4->H2 is free at d_in 64 / d_out 48 '
     '(+0.000103) while halving both projections. This asks the same question at S5\'s own '
     'd_in 32, where halving heads ALSO halves the total routing code width (4x32=128 -> '
     '2x32=64) rather than keeping it at 128 as R6 did. Tables 6*2*512*256*48 = 75,497,472, '
     'identical to S5, R4 and R6.'),
]


def main():
    from nanochat.common import get_base_dir
    from nanochat.tokenizer import RustBPETokenizer
    from model_build import build_model
    tok = RustBPETokenizer.from_directory(os.path.join(get_base_dir(), 'tokenizer'))
    vocab = tok.get_vocab_size()

    order, bad = [], []
    for name, idx, H, tph, cells, d_in, d_out, expect, note in SWEEP3:
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
        cf, df = H * 384 * d_in, H * 384 * d_out
        print(f'sweep_{name}')
        print(f'   H*tph = {H*tph} (<=1024 OK)   d_in = {d_in} (>=32 OK)   '
              f'tables 6*{H}*{tph}*{cells}*{d_out} = {6*H*tph*cells*d_out:,}')
        print(f'   built params {tot:,}  expected {expect:,}  '
              f'dev {100*dev:+.2f}%  {"OK" if ok else "*** OUT OF TOLERANCE ***"}')
        print(f'   device_batch {cfg["device_batch_size"]} x grad_accum '
              f'{SEQS // cfg["device_batch_size"]}  (effective {SEQS} sequences)')
        print(f'   projection FLOPs: compress {cf:,} + decompress {df:,} = {cf+df:,}  '
              f'-> {(cf+df)/VANILLA_FFN_MACS:.4f}x vanilla FFN')
        order.append(dict(idx=idx, run=f'sweep_{name}', params=tot, expected=expect,
                          deviation=dev, device_batch_size=cfg['device_batch_size'],
                          grad_accum=SEQS // cfg['device_batch_size'],
                          H=H, tph=tph, cells=cells, d_in=d_in, d_out=d_out,
                          compress_flops=cf, decompress_flops=df,
                          projection_flops_total=cf + df,
                          compress_flops_ratio=cf / 589824,
                          projection_flops_ratio_vs_vanilla_ffn=(cf + df) / VANILLA_FFN_MACS))
    with open(os.path.join(HERE, 'sweep3_manifest.json'), 'w') as f:
        json.dump(dict(n_steps=N_STEPS, effective_batch_sequences=SEQS,
                       effective_batch_tokens=SEQS * 512, eval_every=EVAL_EVERY,
                       vanilla_ffn_macs_per_token=VANILLA_FFN_MACS, runs=order), f, indent=2)
    print(f'\nwrote {HERE}/sweep3_manifest.json')
    if bad:
        print('*** STOP: param count out of the 1% tolerance ***')
        sys.exit(1)
    print('param count within 1% of the brief — clear to run')


if __name__ == '__main__':
    main()
