"""Arm D: the THIRD point on the selectivity axis -- `margin`, at the same forward scale.

A', C and D hold the score's MEAN fixed at 0.6838 and vary only how much it discriminates
between confident and unconfident routings. That turns a yes/no question into a curve:

    arm      form                  gain     mean     p75/p25   within-token CV
    A'       bounded_norm          1.00     0.6838      1.09        0.067
    C        bounded              12.61     0.6839      2.06        0.584
    D        margin                2.99     0.6836      3.04        0.946

Every gain is measured on the same 6,291,456 real margin vectors, not guessed: bounded_norm
means 0.683836, bounded 0.054236 (ratio 12.609) and margin 0.228620 (ratio 2.991).

`margin` is also the EXACT LookupFFN kernel form, (sum_j |d_j|) * prod_j sigmoid(2|d_j|), so
arm D is the closest thing in this series to "what the paper actually does", corrected only
for the scale that our nap=8 breaks. It is the most selective of the three, so if selectivity
matters at all in either direction, D is where it should show up largest.

Reading the three together:
    monotone improvement A' > C > D   ->  selectivity helps; keep the gate, tune it up
    monotone degradation A' < C < D   ->  selectivity hurts; the gate is a liability and A'
                                          only survived by being nearly inert
    all three within a seed sd        ->  the score mechanism is irrelevant on our geometry
                                          at this budget, whatever its shape

One further caveat specific to `margin`: it SELF-NORMALISES during training -- its sum|d|
factor grows with the margins, so its mean rises from 0.229 to 0.944 over 4,000 steps in the
gate-off trajectory. The gain is fixed at its init value, so arm D's effective scale will
drift upward as it trains, unlike A' (0.684 -> 0.759) and C. That is a real confound and is
recorded here rather than discovered later.

The gain adds no parameters and is absorbable into decompress, so the param count is exactly
the baseline's -- any difference is mechanism, not capacity.

    python make_arm_d.py
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
RUNS = [('exp_n_0183_D_margin_gain_seed1', 'D1', 1)]
EXPECT = 104_952_588
FORM = 'margin'
GAIN = 2.99
NOTE = (
    'ARM D — the THIRD point on the selectivity axis: confidence_form="margin" with '
    'confidence_gain=2.99. Config cloned verbatim from '
    'sweep_s05_dout48_H4_tph256_c256_din32/config.json; the ONLY deltas are '
    'lut_forward_confidence=true, lut_confidence_form="margin", lut_confidence_gain=2.99 and '
    'the seed. A\', C and D hold the score\'s MEAN fixed at 0.6838 and vary only its SPREAD, '
    'turning "does selectivity help?" into a curve: bounded_norm (gain 1.00, p75/p25 1.09, '
    'within-token CV 0.067), bounded (gain 12.61, 2.06, 0.584), margin (gain 2.99, 3.04, '
    '0.946). Every gain is measured on the same 6,291,456 real margin vectors, not guessed: '
    'bounded_norm means 0.683836, bounded 0.054236 (ratio 12.609), margin 0.228620 (ratio '
    '2.991). "margin" is also the EXACT LookupFFN kernel form, (sum_j |d_j|) * prod_j '
    'sigmoid(2|d_j|), so this arm is the closest in the series to what the paper actually '
    'does, corrected only for the scale our nap=8 breaks — and it is the most selective of '
    'the three, so if selectivity matters in either direction it should show up largest here. '
    'CONFOUND, recorded up front: margin SELF-NORMALISES during training (its sum|d| factor '
    'grows with the margins, 0.229 -> 0.944 over 4,000 gate-off steps) while the gain stays '
    'fixed at its init value, so D\'s effective scale drifts upward as it trains in a way A\' '
    'and C\'s do not. The gain is a constant, hence absorbable into the linear decompress, so '
    'it adds no parameters and no expressive power — the param count is exactly the '
    'baseline\'s and any difference is mechanism rather than capacity. One seed, paired '
    'against baseline S5 (1.434572).')


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
        cfg['lut_forward_confidence'] = True
        cfg['lut_confidence_form'] = FORM
        cfg['lut_confidence_gain'] = GAIN
        cfg['exp_name'] = name
        cfg['_sweep_tag'] = 'lookupffn-arm-d'
        cfg['_arch_note'] = NOTE + f' This is {tag}, random_seed {seed}.'
        assert cfg['lut_base_seed'] == 1000, cfg['lut_base_seed']
        d = os.path.join(HERE, name)
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, 'config.json'), 'w') as f:
            json.dump(cfg, f, indent=2)
        shutil.copy(os.path.join(FR, 'train_fixed.py'), os.path.join(d, 'train.py'))

        drift = [k for k in set(cfg) | set(base)
                 if k not in ('random_seed', 'exp_name', '_arch_note', '_sweep_tag',
                              'lut_forward_confidence', 'lut_confidence_form',
                              'lut_confidence_gain')
                 and cfg.get(k) != base.get(k)]
        m = build_model(cfg, vocab, device='cpu')
        tot = sum(p.numel() for p in m.parameters())
        inner = [b.ffn.lut_batched for b in m.blocks]
        forms = {l.confidence_form for l in inner}
        gains = {l.confidence_gain for l in inner}
        gates = {l.forward_confidence for l in inner}
        del m
        ok = (tot == EXPECT and forms == {FORM} and gains == {GAIN} and gates == {True})
        if not ok or drift:
            bad.append((name, tot, drift, forms, gains, gates))
        print(f'   {name}  seed {seed}  params {tot:,}  expected {EXPECT:,}  '
              f'{"OK" if ok else "*** MISMATCH ***"}')
        print(f'      gate in every block: forward_confidence={gates}  form={forms}  '
              f'gain={gains}')
        print(f'      drift beyond seed+gate+gain: {drift or "none"}')
        order.append(dict(idx=len(order) + 1, run=name, tag=tag, params=tot, expected=EXPECT,
                          deviation=(tot - EXPECT) / EXPECT,
                          device_batch_size=cfg['device_batch_size'],
                          grad_accum=cfg['total_batch_size'] //
                          (cfg['device_batch_size'] * cfg['seq_len']),
                          H=cfg['lut_n_heads'], tph=cfg['lut_tables_per_head'],
                          cells=2 ** cfg['lut_n_anchor_pairs'],
                          d_in=cfg['lut_inner_in_dim'], d_out=cfg['lut_inner_out_dim'],
                          random_seed=seed, forward_confidence=True,
                          confidence_form=FORM, confidence_gain=GAIN,
                          measured_score_mean=0.683644,
                          measured_score_p75_over_p25=3.04,
                          compress_flops=cfg['lut_n_heads'] * 384 * cfg['lut_inner_in_dim'],
                          decompress_flops=cfg['lut_n_heads'] * 384 * cfg['lut_inner_out_dim'],
                          projection_flops_total=cfg['lut_n_heads'] * 384 *
                          (cfg['lut_inner_in_dim'] + cfg['lut_inner_out_dim']),
                          compress_flops_ratio=cfg['lut_n_heads'] * 384 *
                          cfg['lut_inner_in_dim'] / 589824,
                          projection_flops_ratio_vs_vanilla_ffn=cfg['lut_n_heads'] * 384 *
                          (cfg['lut_inner_in_dim'] + cfg['lut_inner_out_dim']) /
                          (2 * 384 * 1536)))
    with open(os.path.join(HERE, 'sweep_armd_manifest.json'), 'w') as f:
        json.dump(dict(n_steps=base['n_steps'], effective_batch_sequences=24,
                       effective_batch_tokens=base['total_batch_size'],
                       eval_every=base['eval_every'],
                       vanilla_ffn_macs_per_token=2 * 384 * 1536,
                       cloned_from='sweep_s05_dout48_H4_tph256_c256_din32',
                       baseline_pair=dict(run='sweep_s05_dout48_H4_tph256_c256_din32',
                                          corrected_val_bpb=1.434572, random_seed=1),
                       isolates='selectivity, at matched forward scale vs arm A\'',
                       runs=order), f, indent=2)
    print(f'\nwrote {HERE}/sweep_armd_manifest.json')
    if bad:
        print('*** STOP ***')
        sys.exit(1)
    print('arm D matches the baseline param count exactly (the gain adds none) — clear to run')


if __name__ == '__main__':
    main()
