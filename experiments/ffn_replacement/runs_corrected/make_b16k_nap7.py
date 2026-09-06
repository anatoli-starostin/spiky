"""exp_n_0186: the third rung of the table-budget ladder — nap 8 -> 7, at tph=128.

The ladder, all Light + bounded_norm, 16k, seed 1, everything else held:

    exp_n_0184  tph 256, nap 8   75,497,472 table params   total 105,100,416   1.201075
    exp_n_0185  tph 128, nap 8   37,748,736                total  67,351,680   1.206222
    exp_n_0186  tph 128, nap 7   18,874,368                total  48,477,312   <- this run

End to end that is a 4x reduction in table parameters. 0184 -> 0185 cost only +0.005147 for
the first halving, against the budget law's -0.00746 per doubling; if the second halving is
similarly cheap, the case that this architecture is not capacity-limited gets much stronger.

CAVEAT, and it is the reason this run is NOT simply "0185 halved again": nap is not the same
axis as tph. Halving tph removes whole tables and leaves each remaining table intact. Dropping
nap from 8 to 7 halves the rows per table (2^8 -> 2^7) AND removes one anchor comparison per
table, i.e. it takes away a routing DECISION, not just storage. The two therefore cost the
same parameters but are not interchangeable, and the budget law — fitted across cells and tph
together — cannot tell them apart. A result here that differs from the tph halving is
informative precisely because of that.

Config is copied from exp_n_0185 with lut_n_anchor_pairs the ONLY change; an assertion aborts
on any other drift.

    python make_b16k_nap7.py
"""
import copy
import json
import os
import shutil
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
FR = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(FR, 'tools'))
sys.path.insert(0, os.path.expanduser('~/projects/nanochat'))

SRC = os.path.join(HERE, 'exp_n_0185_B16k_light_bnorm_tph128_seed1')
RUN = 'exp_n_0186_B16k_light_bnorm_tph128_nap7_seed1'
EXPECT_TOTAL = 48_477_312
EXPECT_TABLES = 18_874_368

NOTE = (
    'ARM B AT 16K, THIRD RUNG OF THE TABLE-BUDGET LADDER (tph=128, nap=7) — '
    'LightMultiHeadLUT with the normalised confidence gate (lut_impl="light", '
    'lut_forward_confidence=true, lut_confidence_form="bounded_norm") on the full '
    '16,000-step schedule. Copied from exp_n_0185 with lut_n_anchor_pairs 8 -> 7 as the ONLY '
    'change (asserted; any other drift aborts the build). Table parameters: 6 layers x H(4) '
    'x tph(128) x 2^7(128) cells x d_out(48) = 18,874,368, against exp_n_0185\'s 37,748,736 '
    'and exp_n_0184\'s 75,497,472 — a 4x reduction end to end. Non-table parameters are '
    'unchanged at 29,602,944, so the total goes 105,100,416 -> 67,351,680 -> 48,477,312. '
    'THE LADDER SO FAR: exp_n_0184 (tph 256, nap 8) 1.201075; exp_n_0185 (tph 128, nap 8) '
    '1.206222, i.e. the first halving of the table budget cost only +0.005147, against the '
    '16k budget law\'s -0.00746 per doubling. CAVEAT — nap IS NOT THE SAME AXIS AS tph: '
    'halving tph removes whole tables and leaves each remaining table intact, whereas '
    'dropping nap from 8 to 7 halves the rows per table (2^8 -> 2^7) AND removes one anchor '
    'comparison per table, taking away a routing DECISION rather than only storage. The two '
    'cost identical parameters but are not interchangeable, and the budget law — fitted '
    'across cells and tph together — cannot distinguish them. A result here that departs from '
    'the tph halving is informative precisely for that reason. Everything else is held: '
    'd_in=d_out=48, H=4, joint_head_compression false, forward_mode hard, lut_use_bf16 false '
    '(tables stay fp32 — _embedding_bag_per_sample_weights_backward_cuda is not implemented '
    'for bf16), random_seed 1, lut_base_seed 1000, device_batch 12 / grad_accum 4 / effective '
    'batch 24,576, lr 3e-4 with 0.1 warmup, eval_every 200, trained with train_fixed.py so '
    'the in-run eval IS the corrected protocol (bs48 x 100, skip 12, 2,451,456 val tokens of '
    'the held-out shard_06542.parquet). REFERENCES: exp_n_0129 Fast gate-off tph=256 '
    '1.170961; vanilla dense exp_n_0135 1.165147 and exp_n_0176 1.161798 (vanilla seed spread '
    '0.00335). NOTE that exp_n_0129 has the confidence gate OFF, so a Light-vs-Fast reading '
    'against it mixes the implementation with the gate and is not single-variable; the ladder '
    'comparisons within 0184/0185/0186 are.')


def main():
    from nanochat.common import get_base_dir
    from nanochat.tokenizer import RustBPETokenizer
    from model_build import build_model

    base = json.load(open(os.path.join(SRC, 'config.json')))
    cfg = copy.deepcopy(base)
    cfg['lut_n_anchor_pairs'] = 7
    cfg['exp_name'] = RUN
    cfg['_arch_note'] = NOTE
    cfg['_sweep_tag'] = 'lookupffn-arm-b-16k-tph128-nap7'

    drift = [k for k in set(cfg) | set(base)
             if k not in ('exp_name', 'lut_n_anchor_pairs', '_arch_note', '_sweep_tag')
             and cfg.get(k) != base.get(k)]
    if drift:
        print(f'*** STOP: unintended drift from exp_n_0185: {drift}')
        sys.exit(1)
    print(f'config diff vs exp_n_0185: lut_n_anchor_pairs '
          f'{base["lut_n_anchor_pairs"]} -> {cfg["lut_n_anchor_pairs"]}  '
          f'(+ exp_name/_arch_note/_sweep_tag). No other field differs.')

    d = os.path.join(HERE, RUN)
    assert not os.path.exists(d), f'{d} exists -- never overwrite a prior run'
    os.makedirs(d)
    with open(os.path.join(d, 'config.json'), 'w') as f:
        json.dump(cfg, f, indent=2)
    src_train = os.path.join(FR, 'train_fixed.py')
    dst_train = os.path.join(d, 'train.py')
    shutil.copy(src_train, dst_train)
    assert open(src_train, 'rb').read() == open(dst_train, 'rb').read(), \
        'train.py is not byte-identical to train_fixed.py'
    print('train.py byte-identical to train_fixed.py: OK')

    vocab = RustBPETokenizer.from_directory(
        os.path.join(get_base_dir(), 'tokenizer')).get_vocab_size()
    m = build_model(cfg, vocab, device='cpu')
    tot = sum(p.numel() for p in m.parameters())
    tbl = sum(b.ffn.lut_light.tables.numel() for b in m.blocks)
    ffn = m.blocks[0].ffn
    light = ffn.lut_light

    checks = {
        'total params': (tot, EXPECT_TOTAL),
        'table params': (tbl, EXPECT_TABLES),
        'non-table params': (tot - tbl, 29_602_944),
        'lut_light present': (light is not None, True),
        'lut_batched absent': (not hasattr(ffn, 'lut_batched'), True),
        'multi_head_input ON': (light.multi_head_input, True),
        'tables_per_head': (light.tables_per_head, 128),
        'n_heads': (light.n_heads, 4),
        'n_anchor_pairs': (light.n_anchor_pairs, 7),
        'cells per table (2^nap)': (light.table_size, 128),
        'table tensor shape': (tuple(light.tables.shape), (512, 128, 48)),
        'confidence_form': (light.confidence_form, 'bounded_norm'),
        'confidence_gain': (light.confidence_gain, 1.0),
        'tables dtype fp32': (light.tables.dtype, torch.float32),
        'no temperature params': ([n for n, _ in m.named_parameters() if 'temp' in n], []),
    }
    ok = True
    print(f'\n{RUN}')
    for name, (got, want) in checks.items():
        good = got == want
        ok &= good
        print(f'   {name:<28}{str(got)[:26]:>28}   expected {str(want)[:22]:<24}'
              f'{"OK" if good else "*** MISMATCH ***"}')
    print(f'\n   ladder: 105,100,416 (0184) -> 67,351,680 (0185) -> {tot:,} (0186)')
    print(f'   tables:  75,497,472        ->  37,748,736       -> {tbl:,}')

    # Smoke: decompress is zero-initialised (model_build.py:144), so on a pristine model every
    # gradient through the LUT is exactly 0 and a naive check proves nothing. Drive it first.
    with torch.no_grad():
        for b in m.blocks:
            b.ffn.decompress.weight.normal_(0, 2.3 / b.ffn.decompress.weight.numel() ** 0.5)
    ids = torch.randint(0, vocab, (2, 64))
    loss = m(ids).float().mean()
    loss.backward()
    gt, gc = light.tables.grad, ffn.compress.weight.grad
    gd = ffn.decompress.weight.grad
    print(f'\n   smoke fwd+bwd (CPU, decompress driven to trained scale first):')
    print(f'      loss {loss.item():.6g} finite={torch.isfinite(loss).item()}')
    print(f'      tables grad     {gt.norm():.6g}  {gt.dtype}  finite={torch.isfinite(gt).all().item()}')
    print(f'      compress grad   {gc.norm():.6g}  finite={torch.isfinite(gc).all().item()}')
    print(f'      decompress grad {gd.norm():.6g}')
    ok &= bool(torch.isfinite(gt).all() and torch.isfinite(gc).all()
               and gt.norm() > 0 and gc.norm() > 0)
    del m
    if not ok:
        print('\n*** STOP — not launching ***')
        sys.exit(1)
    print(f'\nwrote {d}/  — verified, NOT launched')


if __name__ == '__main__':
    main()
