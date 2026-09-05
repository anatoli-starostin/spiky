"""Generate V1 (exp_n_0172) — a deliberate probe at a corner the iso-budget analysis rejects.

WHAT THIS TESTS. Two regularities established at 16k both point AGAINST this shape:

  * "more tph, fewer cells at fixed budget" holds 5/5, and at this exact 18,874,368 budget
    0128 (tph64, cells256) LOST to 0127 (tph128, cells128) by 0.005454. V1 uses tph 64, the
    lowest in the whole set, with cells 256.
  * H8 was one of the two bad extremes in the 37.7M H-slice (0132 H8 1.183174 vs H2 1.179273).

BUT both were measured with d_in TIED to d_out. V1 breaks the tie: it doubles the input routing
width (d_in 48, so H*d_in = 384) while halving the output width (d_out 24). The question is
whether untying rescues a shape the tied rules reject. Reported straight either way — a win here
would be a genuine surprise and more interesting than a confirmation.

    H = 8   tph = 64   nap = 8 (cells 256)   d_in = 48   d_out = 24

    tables            6*8*64*256*24 = 18,874,368   (identical to 0127 / 0128 / 0169 / U1-U3)
    projection FLOPs  8*384*(48+24)  = 221,184     = 0.1875x vanilla's 1,179,648
    H*tph = 512 (under the 1024 cap)   H*d_in = 384
    soft-backward buffer [tokens, H*tph=512, cells=256] fp32 = 3.2 GiB at device_batch 12

NOT a clean d_out isolate: it is the first run at this budget with d_out 24 rather than 48, but
cells and tph move with it to hold the budget, so d_out remains confounded exactly as the axis
analysis said.

    python make_sweep5.py
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
H, TPH, CELLS, D_IN, D_OUT = 8, 64, 256, 48, 24
TABLES = 6 * H * TPH * CELLS * D_OUT          # must be 18,874,368
EXPECT = 48_920_844
NAME = 'exp_n_0172_v1_proxy_H8_tph64_c256_din48_dout24'
NOTE = (
    'V1 — a deliberate probe at a corner the iso-budget analysis REJECTS. Both established '
    'regularities point against it: "more tph, fewer cells at fixed budget" holds 5/5 at 16k '
    '(and at this exact 18,874,368 budget 0128 tph64/c256 lost to 0127 tph128/c128 by '
    '0.005454), and H8 was one of the two bad extremes in the 37.7M H-slice (0132 1.183174 vs '
    'H2 0131 1.179273). BUT both were measured with d_in TIED to d_out. V1 breaks the tie — '
    'd_in 48 (H*d_in = 384, double the U1 rung) with d_out halved to 24 — to ask whether '
    'untying rescues a shape the tied rules reject. NOT a clean d_out isolate: cells and tph '
    'move with d_out to hold the budget, so d_out stays confounded exactly as the axis '
    'analysis said.')


def main():
    from nanochat.common import get_base_dir
    from nanochat.tokenizer import RustBPETokenizer
    from model_build import build_model

    checks = [
        ('table params = 6*8*64*256*24', TABLES, 18_874_368),
        ('projection FLOPs = H*384*(d_in+d_out)', H * 384 * (D_IN + D_OUT), 221_184),
        ('H*tph', H * TPH, 512),
        ('H*d_in', H * D_IN, 384),
    ]
    ok = True
    for label, got, want in checks:
        good = got == want
        ok &= good
        print(f'   {label:<40} {got:>12,}  expected {want:>12,}  '
              f'{"OK" if good else "*** MISMATCH ***"}')
    print(f'   {"projection FLOPs vs vanilla":<40} '
          f'{H*384*(D_IN+D_OUT)/VANILLA_FFN_MACS:>12.4f}x  expected       0.1875x  '
          f'{"OK" if abs(H*384*(D_IN+D_OUT)/VANILLA_FFN_MACS - 0.1875) < 1e-9 else "*** MISMATCH ***"}')
    if not ok:
        print('*** STOP: a pre-launch identity failed ***')
        sys.exit(1)

    cfg = build_cfg('v1_H8_tph64_c256_din48_dout24', H, TPH, CELLS, D_IN, D_OUT, NOTE)
    cfg['exp_name'] = NAME
    cfg['_sweep_tag'] = 'proxy-sweep-v1'
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
    print(f'   {"device_batch x grad_accum":<40} '
          f'{cfg["device_batch_size"]} x {SEQS // cfg["device_batch_size"]}  '
          f'(effective {SEQS} sequences)')
    print(f'   {"soft-backward buffer at that batch":<40} '
          f'{cfg["device_batch_size"]*512*H*TPH*CELLS*4/1e9:>12.2f} GB')

    manifest = dict(n_steps=N_STEPS, effective_batch_sequences=SEQS,
                    effective_batch_tokens=SEQS * 512, eval_every=EVAL_EVERY,
                    vanilla_ffn_macs_per_token=VANILLA_FFN_MACS,
                    runs=[dict(idx=1, run=NAME, tag='V1', params=tot, expected=EXPECT,
                               deviation=dev,
                               device_batch_size=cfg['device_batch_size'],
                               grad_accum=SEQS // cfg['device_batch_size'],
                               H=H, tph=TPH, cells=CELLS, d_in=D_IN, d_out=D_OUT,
                               tables=TABLES,
                               compress_flops=H * 384 * D_IN,
                               decompress_flops=H * 384 * D_OUT,
                               projection_flops_total=H * 384 * (D_IN + D_OUT),
                               compress_flops_ratio=H * 384 * D_IN / 589824,
                               projection_flops_ratio_vs_vanilla_ffn=(
                                   H * 384 * (D_IN + D_OUT) / VANILLA_FFN_MACS))])
    with open(os.path.join(HERE, 'sweep5_manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f'\nwrote {HERE}/sweep5_manifest.json')
    if abs(dev) > 0.01:
        sys.exit(1)
    print('all identities hold and the param count is within 1% — clear to run')


if __name__ == '__main__':
    main()
