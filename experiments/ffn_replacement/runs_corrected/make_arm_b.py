"""Arm B: LightMultiHeadLUT (detached routing) with the normalised confidence gate.

Light's defining property is that the routing address is `sign(d).detach()` — no STE, no
temperature surrogate — so x receives gradient ONLY through the confidence score. Arm B asks
whether that is enough to train the anchor configuration.

MEASURED FIRST (diag_light_vs_fast.py), which is why this run is worth an hour of GPU:

    fast + bounded_norm    grad_x 0.14138   grad_tables 195.89   grad_com 2.9347
    light + bounded_norm   grad_x 0.023045  grad_tables 195.89   grad_com 0.47296

  * table and decompress gradients are IDENTICAL to Fast's — the 75.5M table parameters are
    not handicapped at all, and the forward outputs match (0.035242 both);
  * the handicap is confined to the input side: 16.3% of Fast's grad_x;
  * but under AdamW a uniform rescale is largely absorbed, so the statistic that matters is
    direction: cos(light, fast) = +0.576 on compress.weight. Substantially aligned, but not
    a rescaled copy — the surrogate carries information Light cannot see.

Prediction on that evidence: arm B trains rather than collapses, and lags. This is NOT arm
A's failure mode — there is no forward attenuation here.

Config is CLONED FROM DISK (sweep_s05_dout48_H4_tph256_c256_din32/config.json); the only
deltas are `lut_impl`, the two gate keys and the seed. Multi-head input is ON automatically
(CompressionMHL turns it on when has_compress and not joint_head_compression and n_heads>1),
which is what makes Light carry Fast's exact projections and table budget.

PARAM COUNT: Light has NO learnable temperatures — 2 per layer, 6 layers — so it is exactly
12 parameters below the baseline's 104,952,588 -> 104,952,576. Expected and asserted, not a
drift. (The single-layer test in test_light_multi_head_input.py sees a difference of 2; the
model has six of them, which is what the guard here caught.)

    python make_arm_b.py
"""
import copy
import json
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
FR = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(FR, 'tools'))
sys.path.insert(0, os.path.expanduser('~/projects/nanochat'))

SRC = os.path.join(HERE, 'sweep_s05_dout48_H4_tph256_c256_din32')
RUNS = [('exp_n_0181_B_light_bnorm_seed1', 'B1', 1)]
BASELINE_PARAMS = 104_952_588
EXPECT = BASELINE_PARAMS - 12         # 2 learnable temperatures per layer x 6 layers
FORM = 'bounded_norm'
NOTE = (
    'ARM B — LightMultiHeadLUT with the normalised confidence gate '
    '(lut_impl="light", confidence_form="bounded_norm"). Config cloned verbatim from '
    'sweep_s05_dout48_H4_tph256_c256_din32/config.json; the ONLY deltas are lut_impl, '
    'lut_forward_confidence=true, lut_confidence_form="bounded_norm" and the seed. Light '
    'detaches the routing address (sign(d).detach(), no STE, no temperature surrogate), so '
    'x receives gradient ONLY through the confidence score. Multi-head input is on, so the '
    'projections and table budget match Fast exactly and the ONLY structural difference is '
    'the missing directional surrogate — Light has no learnable temperatures, hence exactly '
    '12 parameters fewer than the baseline (2 per layer, 6 layers). Measured before launch: '
    'Light\'s table and '
    'decompress gradients are IDENTICAL to Fast\'s and the forward outputs match, but it '
    'keeps only 16.3% of Fast\'s grad_x, with cos(light, fast) = +0.576 on compress.weight '
    '— substantially aligned yet not a rescaled copy, so the surrogate carries information '
    'Light cannot see. One seed, paired against baseline S5 (1.434572).')


def main():
    from nanochat.common import get_base_dir
    from nanochat.tokenizer import RustBPETokenizer
    from model_build import build_model
    base = json.load(open(os.path.join(SRC, 'config.json')))
    vocab = RustBPETokenizer.from_directory(
        os.path.join(get_base_dir(), 'tokenizer')).get_vocab_size()

    order, bad = [], []
    for name, tag, seed in RUNS:
        cfg = copy.deepcopy(base)
        cfg['random_seed'] = seed
        cfg['lut_impl'] = 'light'
        cfg['lut_forward_confidence'] = True
        cfg['lut_confidence_form'] = FORM
        cfg['exp_name'] = name
        cfg['_sweep_tag'] = 'lookupffn-arm-b'
        cfg['_arch_note'] = NOTE + f' This is {tag}, random_seed {seed}.'
        assert cfg['lut_base_seed'] == 1000, cfg['lut_base_seed']
        d = os.path.join(HERE, name)
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, 'config.json'), 'w') as f:
            json.dump(cfg, f, indent=2)
        shutil.copy(os.path.join(FR, 'train_fixed.py'), os.path.join(d, 'train.py'))

        drift = [k for k in set(cfg) | set(base)
                 if k not in ('random_seed', 'exp_name', '_arch_note', '_sweep_tag',
                              'lut_impl', 'lut_forward_confidence', 'lut_confidence_form')
                 and cfg.get(k) != base.get(k)]
        m = build_model(cfg, vocab, device='cpu')
        tot = sum(p.numel() for p in m.parameters())
        inner = [b.ffn.lut_light for b in m.blocks]
        forms = {l.confidence_form for l in inner}
        mh = {bool(getattr(l, 'multi_head_input', False)) for l in inner}
        temps = [n for n, _ in m.named_parameters() if 'temp' in n]
        # the light path must carry Fast's projections, not a degenerate shared one
        proj = {n: p.numel() for n, p in m.blocks[0].ffn.named_parameters()
                if 'compress' in n}
        del m
        ok = (tot == EXPECT and forms == {FORM} and mh == {True} and not temps
              and proj.get('compress.weight') == 49152
              and proj.get('decompress.weight') == 73728)
        if not ok or drift:
            bad.append((name, tot, drift, forms, mh, temps, proj))
        print(f'   {name}  seed {seed}  params {tot:,}  expected {EXPECT:,} '
              f'(baseline {BASELINE_PARAMS:,} minus Light\'s 12 temperature scalars)  '
              f'{"OK" if ok else "*** MISMATCH ***"}')
        print(f'      light layer: form={forms}  multi_head_input={mh}  '
              f'temperature params={temps or "none (expected)"}')
        print(f'      projections: {proj}')
        print(f'      drift beyond seed+impl+gate: {drift or "none"}')
        order.append(dict(idx=len(order) + 1, run=name, tag=tag, params=tot, expected=EXPECT,
                          deviation=(tot - EXPECT) / EXPECT,
                          baseline_params=BASELINE_PARAMS,
                          params_vs_baseline=tot - BASELINE_PARAMS,
                          device_batch_size=cfg['device_batch_size'],
                          grad_accum=cfg['total_batch_size'] //
                          (cfg['device_batch_size'] * cfg['seq_len']),
                          H=cfg['lut_n_heads'], tph=cfg['lut_tables_per_head'],
                          cells=2 ** cfg['lut_n_anchor_pairs'],
                          d_in=cfg['lut_inner_in_dim'], d_out=cfg['lut_inner_out_dim'],
                          random_seed=seed, lut_impl='light', forward_confidence=True,
                          confidence_form=FORM,
                          compress_flops=cfg['lut_n_heads'] * 384 * cfg['lut_inner_in_dim'],
                          decompress_flops=cfg['lut_n_heads'] * 384 * cfg['lut_inner_out_dim'],
                          projection_flops_total=cfg['lut_n_heads'] * 384 *
                          (cfg['lut_inner_in_dim'] + cfg['lut_inner_out_dim']),
                          compress_flops_ratio=cfg['lut_n_heads'] * 384 *
                          cfg['lut_inner_in_dim'] / 589824,
                          projection_flops_ratio_vs_vanilla_ffn=cfg['lut_n_heads'] * 384 *
                          (cfg['lut_inner_in_dim'] + cfg['lut_inner_out_dim']) /
                          (2 * 384 * 1536)))
    with open(os.path.join(HERE, 'sweep_armb_manifest.json'), 'w') as f:
        json.dump(dict(n_steps=base['n_steps'], effective_batch_sequences=24,
                       effective_batch_tokens=base['total_batch_size'],
                       eval_every=base['eval_every'],
                       vanilla_ffn_macs_per_token=2 * 384 * 1536,
                       cloned_from='sweep_s05_dout48_H4_tph256_c256_din32',
                       baseline_pair=dict(run='sweep_s05_dout48_H4_tph256_c256_din32',
                                          corrected_val_bpb=1.434572, random_seed=1),
                       runs=order), f, indent=2)
    print(f'\nwrote {HERE}/sweep_armb_manifest.json')
    if bad:
        print('*** STOP ***')
        sys.exit(1)
    print('arm B matches Fast exactly but for Light\'s 12 missing temperature scalars '
          '— clear to run')


if __name__ == '__main__':
    main()
