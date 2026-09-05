"""Generate V2 (exp_n_0173) — V1's shape with the dims TIED, landing on the 37.7M budget.

    H = 8   tph = 64   nap = 8 (cells 256)   d_in = 48   d_out = 48

    tables            6*8*64*256*48 = 37,748,736  — the SAME budget as the historical
                                                    0084 / 0121 / 0131 / 0132 / 0137 slice
    projection FLOPs  8*384*(48+48) =    294,912  = 0.2500x vanilla's 1,179,648
                                                    *** DOUBLE the 0.125x most of the grid runs at ***
    H*tph = 512 (under the 1024 cap)   H*d_in = 384
    soft-backward buffer [tokens, H*tph=512, cells=256] fp32 — identical to V1, 3.22 GB at bs12

WHAT IT TESTS. Against the historical 0132 this is the closest thing to a clean d_out slice that
exists anywhere:

    0132  H8 tph128 cells256 d_in=d_out=24    37,748,736 tables   16k bpb 1.183174
    V2    H8 tph64  cells256 d_in=d_out=48    37,748,736 tables

H fixed at 8, cells fixed at 256, budget fixed — only tph (128->64) trades against d_out
(24->48). The axis analysis concluded d_out has NO clean slice because it enters the table count
linearly; this is the nearest approach, because the compensating change is confined to tph alone
instead of being smeared across several axes. It is still NOT a pure isolate: the tph-over-cells
rule was worth +0.0028..+0.0095 per step, so halving tph is not negligible, and the two effects
push in OPPOSITE directions here (more d_out good, less tph bad), which means a null result would
be ambiguous rather than informative.

Against V1 it is a clean d_out doubling at fixed H/tph/cells/d_in — but that DOUBLES the table
budget, so it is a budget comparison. The budget law predicts -0.00746 bpb per doubling; whether
V2 beats or misses that is the informative part.

    python make_sweep6.py
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
H, TPH, CELLS, D_IN, D_OUT = 8, 64, 256, 48, 48
TABLES = 6 * H * TPH * CELLS * D_OUT
EXPECT = 68_237_580
NAME = 'exp_n_0173_v2_proxy_H8_tph64_c256_din48_dout48'
NOTE = (
    'V2 — V1\'s shape with the dims TIED (d_in = d_out = 48), which doubles the table budget to '
    '37,748,736 and lands it in the SAME iso-budget group as the historical 0084 / 0121 / 0131 / '
    '0132 / 0137 slice. Against 0132 (H8 tph128 cells256 d 24/24) this is the closest thing to a '
    'clean d_out slice anywhere: H fixed at 8, cells fixed at 256, budget fixed, only tph '
    '(128->64) trading against d_out (24->48). Still NOT a pure isolate — halving tph is worth '
    '+0.0028..+0.0095 by the tph-over-cells rule, and the two effects push in OPPOSITE '
    'directions, so a null result here would be ambiguous rather than informative. NOTE the '
    'cost: projection FLOPs 8*384*96 = 294,912 = 0.2500x vanilla, DOUBLE the 0.125x most of the '
    'grid runs at.')


def main():
    from nanochat.common import get_base_dir
    from nanochat.tokenizer import RustBPETokenizer
    from model_build import build_model

    ok = True
    for label, got, want in (
            ('table params = 6*8*64*256*48', TABLES, 37_748_736),
            ('projection FLOPs = H*384*(d_in+d_out)', H * 384 * (D_IN + D_OUT), 294_912),
            ('H*tph', H * TPH, 512),
            ('H*d_in', H * D_IN, 384)):
        good = got == want
        ok &= good
        print(f'   {label:<40} {got:>12,}  expected {want:>12,}  '
              f'{"OK" if good else "*** MISMATCH ***"}')
    ratio = H * 384 * (D_IN + D_OUT) / VANILLA_FFN_MACS
    good = abs(ratio - 0.25) < 1e-9
    ok &= good
    print(f'   {"projection FLOPs vs vanilla":<40} {ratio:>12.4f}x  expected       0.2500x  '
          f'{"OK" if good else "*** MISMATCH ***"}   <-- DOUBLE the grid\'s usual 0.125x')
    if not ok:
        print('*** STOP: a pre-launch identity failed ***')
        sys.exit(1)

    cfg = build_cfg('v2_H8_tph64_c256_din48_dout48', H, TPH, CELLS, D_IN, D_OUT, NOTE)
    cfg['exp_name'] = NAME
    cfg['_sweep_tag'] = 'proxy-sweep-v2'
    d = os.path.join(HERE, NAME)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, 'config.json'), 'w') as f:
        json.dump(cfg, f, indent=2)
    shutil.copy(os.path.join(FR, 'train_fixed.py'), os.path.join(d, 'train.py'))

    tok = RustBPETokenizer.from_directory(os.path.join(get_base_dir(), 'tokenizer'))
    m = build_model(cfg, tok.get_vocab_size(), device='cpu')
    tot = sum(p.numel() for p in m.parameters())
    del m
    dev = (tot - EXPECT) / EXPECT
    print(f'   {"built params":<40} {tot:>12,}  expected {EXPECT:>12,}  '
          f'dev {100*dev:+.2f}%  {"OK" if abs(dev) <= 0.01 else "*** OUT OF TOLERANCE ***"}')
    print(f'   {"device_batch x grad_accum":<40} {cfg["device_batch_size"]} x '
          f'{SEQS // cfg["device_batch_size"]}  (effective {SEQS} sequences)')
    print(f'   {"soft-backward buffer":<40} '
          f'{cfg["device_batch_size"]*512*H*TPH*CELLS*4/1e9:>12.2f} GB  (same shape as V1)')

    with open(os.path.join(HERE, 'sweep6_manifest.json'), 'w') as f:
        json.dump(dict(n_steps=N_STEPS, effective_batch_sequences=SEQS,
                       effective_batch_tokens=SEQS * 512, eval_every=EVAL_EVERY,
                       vanilla_ffn_macs_per_token=VANILLA_FFN_MACS,
                       runs=[dict(idx=1, run=NAME, tag='V2', params=tot, expected=EXPECT,
                                  deviation=dev,
                                  device_batch_size=cfg['device_batch_size'],
                                  grad_accum=SEQS // cfg['device_batch_size'],
                                  H=H, tph=TPH, cells=CELLS, d_in=D_IN, d_out=D_OUT,
                                  tables=TABLES,
                                  compress_flops=H * 384 * D_IN,
                                  decompress_flops=H * 384 * D_OUT,
                                  projection_flops_total=H * 384 * (D_IN + D_OUT),
                                  compress_flops_ratio=H * 384 * D_IN / 589824,
                                  projection_flops_ratio_vs_vanilla_ffn=ratio)]), f, indent=2)
    print(f'\nwrote {HERE}/sweep6_manifest.json')
    if abs(dev) > 0.01:
        sys.exit(1)
    print('all identities hold and the param count is within 1% — clear to run')


if __name__ == '__main__':
    main()
