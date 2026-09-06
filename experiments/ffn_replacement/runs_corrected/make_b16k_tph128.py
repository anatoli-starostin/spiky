"""The tph=128 half-budget point: exactly one variable away from exp_n_0184.

exp_n_0184 (16K Light, tph=256) landed at 1.201075 — +0.030114 against exp_n_0129 (same
sizing, Fast, gate off, 1.170961) and +0.035928 against vanilla dense (1.165147). This
halves the table budget and changes NOTHING else, so 0184 vs this run is a clean read on how
Light responds to table budget at the full 16K schedule.

    tables = 6 layers x H(4) x tph x cells(2^8=256) x d_out(48)
      tph 256 -> 75,497,472      tph 128 -> 37,748,736      delta -37,748,736
      total  105,100,416   ->    67,351,680

One doubling of table budget is worth -0.00746 bpb under the 16k budget law, so the naive
expectation is this run lands ~+0.0075 above 0184. A materially different answer is itself
the result — it would say Light's budget response differs from Fast's, which the law was
fitted on.

Config follows exp_n_0184 exactly (lut_impl light, forward_confidence, bounded_norm,
lut_use_bf16 false, random_seed 1, d_in/d_out 48, nap 8, H 4, 16000 steps, eval_every 200,
device_batch 12 / grad_accum 4, same lr schedule) with tables_per_head 256 -> 128.

    python make_b16k_tph128.py
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

SRC = os.path.join(HERE, 'exp_n_0184_B16k_light_bnorm_seed1')
RUN = 'exp_n_0185_B16k_light_bnorm_tph128_seed1'
EXPECT = 67_351_680

NOTE = (
    'ARM B AT 16K, HALF TABLE BUDGET (tph=128) — LightMultiHeadLUT with the normalised '
    'confidence gate (lut_impl="light", lut_forward_confidence=true, '
    'lut_confidence_form="bounded_norm") on the full 16,000-step schedule. Config follows '
    'exp_n_0184 exactly; the ONLY delta is lut_tables_per_head 256 -> 128, so 0184 vs this '
    'run is a SINGLE-VARIABLE read on Light\'s response to table budget. Table parameters '
    'halve: 6 layers x H(4) x tph x 2^8 cells x d_out(48) = 75,497,472 -> 37,748,736, taking '
    'the total from 105,100,416 to 67,351,680. Everything else is held: d_in=d_out=48, nap=8, '
    'H=4, joint_head_compression false, forward_mode hard, lut_use_bf16 false (tables stay '
    'fp32 — _embedding_bag_per_sample_weights_backward_cuda is not implemented for bf16), '
    'random_seed 1, lut_base_seed 1000, device_batch 12 / grad_accum 4 / effective batch '
    '24,576, lr 3e-4 with 0.1 warmup, eval_every 200. Light detaches the routing address '
    '(sign(d).detach(), no STE), so x receives gradient ONLY through the confidence score; '
    'multi_head_input is auto-enabled and there are no learnable temperature scalars. '
    'REFERENCES (all corrected protocol, bs48 x 100, skip 12, 2,451,456 val tokens): '
    'exp_n_0184 Light tph=256 1.201075; exp_n_0129 Fast gate-off tph=256 1.170961; vanilla '
    'dense exp_n_0135 1.165147 and exp_n_0176 1.161798 (vanilla seed spread 0.00335). Under '
    'the 16k budget law (-0.00746 bpb per doubling of table budget) halving the budget '
    'predicts roughly +0.0075 vs exp_n_0184; a materially different result would say Light\'s '
    "budget response differs from Fast's, which is what that law was fitted on.")


def main():
    from nanochat.common import get_base_dir
    from nanochat.tokenizer import RustBPETokenizer
    from model_build import build_model

    base = json.load(open(os.path.join(SRC, 'config.json')))
    cfg = copy.deepcopy(base)
    cfg['lut_tables_per_head'] = 128
    cfg['exp_name'] = RUN
    cfg['_arch_note'] = NOTE
    cfg['_sweep_tag'] = 'lookupffn-arm-b-16k-tph128'

    drift = [k for k in set(cfg) | set(base)
             if k not in ('exp_name', 'lut_tables_per_head', '_arch_note', '_sweep_tag')
             and cfg.get(k) != base.get(k)]
    assert not drift, f'unintended drift from exp_n_0184: {drift}'
    for k, v in (('lut_impl', 'light'), ('lut_forward_confidence', True),
                 ('lut_confidence_form', 'bounded_norm'), ('lut_use_bf16', False),
                 ('random_seed', 1), ('lut_inner_in_dim', 48), ('lut_inner_out_dim', 48),
                 ('lut_n_anchor_pairs', 8), ('lut_n_heads', 4), ('n_steps', 16000),
                 ('eval_every', 200)):
        assert cfg[k] == v, f'{k}={cfg[k]!r} expected {v!r}'

    d = os.path.join(HERE, RUN)
    assert not os.path.exists(d), f'{d} exists -- never overwrite a prior run'
    os.makedirs(d)
    with open(os.path.join(d, 'config.json'), 'w') as f:
        json.dump(cfg, f, indent=2)
    shutil.copy(os.path.join(FR, 'train_fixed.py'), os.path.join(d, 'train.py'))

    vocab = RustBPETokenizer.from_directory(
        os.path.join(get_base_dir(), 'tokenizer')).get_vocab_size()
    m = build_model(cfg, vocab, device='cpu')
    tot = sum(p.numel() for p in m.parameters())
    ffn = m.blocks[0].ffn
    light = getattr(ffn, 'lut_light', None)
    tbl_params = sum(b.ffn.lut_light.tables.numel() for b in m.blocks)

    checks = {
        'total params': (tot, EXPECT),
        'table params (all 6 blocks)': (tbl_params, 37_748_736),
        'lut_light present': (light is not None, True),
        'lut_batched absent': (not hasattr(ffn, 'lut_batched'), True),
        'multi_head_input ON': (getattr(light, 'multi_head_input', None), True),
        'tables_per_head': (getattr(light, 'tables_per_head', None), 128),
        'n_heads': (getattr(light, 'n_heads', None), 4),
        'confidence_form': (getattr(light, 'confidence_form', None), 'bounded_norm'),
        'tables dtype fp32': (light.tables.dtype, torch.float32),
        'no temperature params': ([n for n, _ in m.named_parameters() if 'temp' in n], []),
    }
    ok = True
    print(f'{RUN}\n')
    for name, (got, want) in checks.items():
        good = got == want
        ok &= good
        print(f'   {name:<32}{str(got)[:24]:>26}   expected {str(want)[:20]:<22}'
              f'{"OK" if good else "*** MISMATCH ***"}')
    print(f'\n   vs exp_n_0184: {105_100_416:,} -> {tot:,}   delta {tot - 105_100_416:,}')
    print(f'   forms across blocks: { {b.ffn.lut_light.confidence_form for b in m.blocks} }  '
          f'tph: { {b.ffn.lut_light.tables_per_head for b in m.blocks} }')

    # Real smoke: decompress is zero-init by design, so drive it to trained scale first or
    # the gradients vanish and the check proves nothing.
    with torch.no_grad():
        for b in m.blocks:
            b.ffn.decompress.weight.normal_(0, 2.3 / b.ffn.decompress.weight.numel() ** 0.5)
    ids = torch.randint(0, vocab, (2, 64))
    out = m(ids)
    out.float().mean().backward()
    gt = ffn.lut_light.tables.grad
    gc = ffn.compress.weight.grad
    print(f'\n   smoke fwd+bwd (CPU): loss finite, tables grad {gt.norm():.4g} {gt.dtype}, '
          f'compress grad {gc.norm():.4g}, finite={torch.isfinite(gt).all().item()}')
    ok &= bool(torch.isfinite(gt).all()) and gt.norm() > 0
    del m
    if not ok:
        print('*** STOP ***')
        sys.exit(1)
    print(f'\nwrote {d}/config.json and train.py')


if __name__ == '__main__':
    main()
