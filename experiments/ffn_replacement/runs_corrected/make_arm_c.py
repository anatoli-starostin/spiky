"""Arm C: does the confidence gate's SELECTIVITY buy anything, once its scale is right?

Arm A and arm A' each changed two things at once, so neither can answer this.

    arm A   bounded                score mean 0.0542   p75/p25 2.06   -> failed badly
    arm A'  bounded_norm           score mean 0.6838   p75/p25 1.09   -> ~baseline

A was both badly scaled AND selective; A' fixed the scale but ALSO flattened the gate to
nearly a constant multiplier -- and a constant multiplier is exactly absorbable into the
linear decompress that follows, so it cannot change what the model expresses. If A' lands at
parity, that says a near-constant gate is harmless. It does NOT say gating helps.

Arm C is the missing cell: `bounded` -- selectivity fully intact -- with a constant
`confidence_gain` chosen so its MEAN matches bounded_norm's exactly.

    arm C   bounded x 12.61        score mean 0.6839   p75/p25 2.06

Measured on the same 6,291,456 real margin vectors: bounded mean 0.05423589, bounded_norm
mean 0.68383604, ratio 12.6086 -> gain 12.61 gives 0.683915, matching to 0.011%. So C vs A'
is a ONE-VARIABLE comparison: identical forward scale, identical everything else, and the
only difference is whether the gate discriminates between confident and unconfident routings
(p75/p25 2.06 vs 1.09; within-token CV 0.536 vs 0.061).

Reading the result:
    C ~ A'  ->  selectivity is inert here; the gate only ever needed a sane scale, and the
                whole LookupFFN score mechanism buys nothing on our geometry.
    C < A'  ->  selectivity helps, and arm A failed purely on scale -- the gate is worth
                keeping, with the gain as a required companion knob.
    C > A'  ->  selectivity actively hurts (down-weighting uncertain rows costs more than
                it saves), which would be the most interesting outcome of the three.

The gain adds no parameters and is absorbable into decompress, so the param count is exactly
the baseline's -- any difference is mechanism, not capacity.

    python make_arm_c.py
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
RUNS = [('exp_n_0182_C_bounded_gain_seed1', 'C1', 1)]
EXPECT = 104_952_588
FORM = 'bounded'
GAIN = 12.61
NOTE = (
    'ARM C — the confidence gate with "bounded"\'s full selectivity and a corrected scale: '
    'confidence_form="bounded" with confidence_gain=12.61. Config cloned verbatim from '
    'sweep_s05_dout48_H4_tph256_c256_din32/config.json; the ONLY deltas are '
    'lut_forward_confidence=true, lut_confidence_form="bounded", lut_confidence_gain=12.61 '
    'and the seed. The gain was chosen by measurement, not guessed: on 6,291,456 real margin '
    'vectors at this sizing, bounded scores 0.05423589 mean and bounded_norm 0.68383604, a '
    'ratio of 12.6086, so gain 12.61 puts bounded\'s mean at 0.683915 — within 0.011% of arm '
    'A\'\'s. That makes C vs A\' a ONE-VARIABLE comparison isolating SELECTIVITY: identical '
    'forward scale, but p75/p25 2.06 vs 1.09 and within-token CV 0.536 vs 0.061. Arm A '
    '(bounded, no gain) and arm A\' (bounded_norm) each moved two variables at once and so '
    'cannot separate "the scale was wrong" from "the selectivity is useless". The gain is a '
    'constant, hence exactly absorbable into the linear decompress, so it adds no parameters '
    'and no expressive power — the param count is exactly the baseline\'s and any difference '
    'is mechanism rather than capacity. One seed, paired against baseline S5 (1.434572).')


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
        cfg['_sweep_tag'] = 'lookupffn-arm-c'
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
                          measured_score_mean=0.683915,
                          measured_score_p75_over_p25=2.06,
                          compress_flops=cfg['lut_n_heads'] * 384 * cfg['lut_inner_in_dim'],
                          decompress_flops=cfg['lut_n_heads'] * 384 * cfg['lut_inner_out_dim'],
                          projection_flops_total=cfg['lut_n_heads'] * 384 *
                          (cfg['lut_inner_in_dim'] + cfg['lut_inner_out_dim']),
                          compress_flops_ratio=cfg['lut_n_heads'] * 384 *
                          cfg['lut_inner_in_dim'] / 589824,
                          projection_flops_ratio_vs_vanilla_ffn=cfg['lut_n_heads'] * 384 *
                          (cfg['lut_inner_in_dim'] + cfg['lut_inner_out_dim']) /
                          (2 * 384 * 1536)))
    with open(os.path.join(HERE, 'sweep_armc_manifest.json'), 'w') as f:
        json.dump(dict(n_steps=base['n_steps'], effective_batch_sequences=24,
                       effective_batch_tokens=base['total_batch_size'],
                       eval_every=base['eval_every'],
                       vanilla_ffn_macs_per_token=2 * 384 * 1536,
                       cloned_from='sweep_s05_dout48_H4_tph256_c256_din32',
                       baseline_pair=dict(run='sweep_s05_dout48_H4_tph256_c256_din32',
                                          corrected_val_bpb=1.434572, random_seed=1),
                       isolates='selectivity, at matched forward scale vs arm A\'',
                       runs=order), f, indent=2)
    print(f'\nwrote {HERE}/sweep_armc_manifest.json')
    if bad:
        print('*** STOP ***')
        sys.exit(1)
    print('arm C matches the baseline param count exactly (the gain adds none) — clear to run')


if __name__ == '__main__':
    main()
