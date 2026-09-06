"""Arm A': the confidence gate again, with the NORMALISED score (geometric mean).

Arm A (`confidence_form="bounded"`) was stopped at step 3,200 with a monotonically
WIDENING gap to the baseline (see exp_n_0178_A_fwdconf_seed1/STOPPED.md). The measured
cause: score = prod_j sigmoid(2|d_j|) is a product over NAP=8 factors, so at our margin
scale (median |d| = 0.381) it lands at 0.054 -- an 18.4x forward attenuation that divides
the gradient reaching the LUT tables by ~19x.

A' keeps everything and changes only the FORM to "bounded_norm" = that same product raised
to 1/NAP, i.e. the geometric mean of the per-anchor sigmoids. Same ordering (it is a
monotone transform), NAP-independent scale. Re-measured on real activations at this exact
sizing before launch:

    score  bounded       p25 0.0326  median 0.0465  p75 0.0670  mean 0.0542   (18.4x down)
    score  bounded_norm  p25 0.652   median 0.681   p75 0.713   mean 0.684    ( 1.46x down)
    grad_tables vs gate-off:  bounded 0.052x    margin 0.262x    bounded_norm 0.671x

ONE SEED. Arm A's deficit was 0.135 by step 500 and 0.225 by step 3000, an order of
magnitude above the 0.0096 seed sd, so a single run resolves a repeat of that failure.
It does NOT resolve a small win -- if A' lands near the baseline, that is a "measure more
seeds" result, not a verdict.

Config is CLONED FROM DISK (sweep_s05_dout48_H4_tph256_c256_din32/config.json); the only
deltas are the two gate keys and the seed. The gate adds no parameters, so the param count
must be exactly the baseline's.

    python make_arm_a_prime.py
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
RUNS = [('exp_n_0180_Aprime_fwdconf_norm_seed1', "A'1", 1)]
EXPECT = 104_952_588
FORM = 'bounded_norm'
NOTE = (
    "ARM A' — FastMultiHeadLut with the forward_confidence score gate in its NORMALISED "
    'form (confidence_form "bounded_norm"): score = (prod_j sigmoid(2|d_j|)) ** (1/NAP), the '
    'geometric mean of the per-anchor sigmoids. Config cloned verbatim from '
    'sweep_s05_dout48_H4_tph256_c256_din32/config.json; the ONLY deltas are '
    'lut_forward_confidence=true, lut_confidence_form="bounded_norm" and the seed. This is '
    'the follow-up to arm A, which used the un-normalised "bounded" form and was stopped at '
    'step 3,200 with a widening gap: the product over NAP=8 factors measured 0.054 on real '
    'activations (18.4x forward attenuation, ~19x cut to the table gradient). The geometric '
    'mean has the SAME ordering but a NAP-independent scale — re-measured at this sizing it '
    'is 0.684 mean, with the table gradient at 0.671x of gate-off. The gate introduces NO '
    'parameters, so this arm has exactly the baseline param count and any difference is '
    'mechanism rather than capacity. One seed, paired against baseline S5 (1.434572): arm A '
    "failed by 0.135 at step 500, an order of magnitude above the 0.0096 seed sd, so one run "
    'resolves a repeat of that failure (it would not resolve a small win).')


def main():
    from nanochat.common import get_base_dir
    from nanochat.tokenizer import RustBPETokenizer
    from model_build import build_model
    base = json.load(open(os.path.join(SRC, 'config.json')))
    print(f"anchor seeds: random_seed {base['random_seed']}, lut_base_seed "
          f"{base['lut_base_seed']}  (A' uses random_seed 1, pairing with S5)")
    vocab = RustBPETokenizer.from_directory(
        os.path.join(get_base_dir(), 'tokenizer')).get_vocab_size()

    order, bad = [], []
    for name, tag, seed in RUNS:
        cfg = copy.deepcopy(base)
        cfg['random_seed'] = seed
        cfg['lut_forward_confidence'] = True
        cfg['lut_confidence_form'] = FORM
        cfg['exp_name'] = name
        cfg['_sweep_tag'] = 'lookupffn-arm-a-prime'
        cfg['_arch_note'] = NOTE + f' This is {tag}, random_seed {seed}.'
        assert cfg['lut_base_seed'] == 1000, cfg['lut_base_seed']
        d = os.path.join(HERE, name)
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, 'config.json'), 'w') as f:
            json.dump(cfg, f, indent=2)
        shutil.copy(os.path.join(FR, 'train_fixed.py'), os.path.join(d, 'train.py'))

        drift = [k for k in set(cfg) | set(base)
                 if k not in ('random_seed', 'exp_name', '_arch_note', '_sweep_tag',
                              'lut_forward_confidence', 'lut_confidence_form')
                 and cfg.get(k) != base.get(k)]
        m = build_model(cfg, vocab, device='cpu')
        tot = sum(p.numel() for p in m.parameters())
        # the form must have actually reached every block, not silently defaulted
        forms = {b.ffn.lut_batched.confidence_form for b in m.blocks}
        gates = {b.ffn.lut_batched.forward_confidence for b in m.blocks}
        del m
        ok = tot == EXPECT and forms == {FORM} and gates == {True}
        if not ok or drift:
            bad.append((name, tot, drift, forms, gates))
        print(f'   {name}  seed {seed}  params {tot:,}  expected {EXPECT:,}  '
              f'{"OK" if ok else "*** MISMATCH ***"}')
        print(f'      gate reached every block: forward_confidence={gates}  form={forms}')
        print(f'      drift beyond seed+gate: {drift or "none"}')
        order.append(dict(idx=len(order) + 1, run=name, tag=tag, params=tot, expected=EXPECT,
                          deviation=(tot - EXPECT) / EXPECT,
                          device_batch_size=cfg['device_batch_size'],
                          grad_accum=cfg['total_batch_size'] //
                          (cfg['device_batch_size'] * cfg['seq_len']),
                          H=cfg['lut_n_heads'], tph=cfg['lut_tables_per_head'],
                          cells=2 ** cfg['lut_n_anchor_pairs'],
                          d_in=cfg['lut_inner_in_dim'], d_out=cfg['lut_inner_out_dim'],
                          random_seed=seed, forward_confidence=True,
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
    with open(os.path.join(HERE, 'sweep_armaprime_manifest.json'), 'w') as f:
        json.dump(dict(n_steps=base['n_steps'], effective_batch_sequences=24,
                       effective_batch_tokens=base['total_batch_size'],
                       eval_every=base['eval_every'],
                       vanilla_ffn_macs_per_token=2 * 384 * 1536,
                       cloned_from='sweep_s05_dout48_H4_tph256_c256_din32',
                       baseline_pair=dict(run='sweep_s05_dout48_H4_tph256_c256_din32',
                                          corrected_val_bpb=1.434572, random_seed=1),
                       runs=order), f, indent=2)
    print(f'\nwrote {HERE}/sweep_armaprime_manifest.json')
    if bad:
        print('*** STOP ***')
        sys.exit(1)
    print("A' matches the baseline param count exactly (the gate adds none) — clear to run")


if __name__ == '__main__':
    main()
