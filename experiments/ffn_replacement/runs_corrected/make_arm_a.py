"""Arm A: FastMultiHeadLut + forward confidence, at two seeds paired with the baseline.

Config is CLONED FROM DISK (sweep_s05_dout48_H4_tph256_c256_din32/config.json) with only
`lut_forward_confidence` / `lut_confidence_form` added and `random_seed` set — nothing retyped.

SEED PAIRING. The three existing anchor runs use random_seed 1 / 2 / 3 with lut_base_seed 1000
throughout, so A reuses seeds 1 and 2 and pairs directly against:

    S5  (seed 1) 1.434572      A-seed1  ->  exp_n_0178
    S5b (seed 2) 1.452477      A-seed2  ->  exp_n_0179
    S5c (seed 3) 1.449728      (baseline third seed, no A counterpart)

Two seeds because the measured 4k seed sd is 0.0096: one run per arm cannot resolve the
expected effect. Note the gate adds NO parameters, so both arms have identical param counts —
any difference is mechanism, not capacity.

    python make_arm_a.py
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
RUNS = [('exp_n_0178_A_fwdconf_seed1', 'A1', 1), ('exp_n_0179_A_fwdconf_seed2', 'A2', 2)]
EXPECT = 104_952_588
NOTE = (
    'ARM A — FastMultiHeadLut with the LookupFFN-style forward_confidence score gate '
    '(confidence_form "bounded"). Config cloned verbatim from '
    'sweep_s05_dout48_H4_tph256_c256_din32/config.json; the ONLY deltas are '
    'lut_forward_confidence=true, lut_confidence_form="bounded" and the seed. The gate '
    'multiplies each gathered table row by score = prod_j sigmoid(2|d_j|) over the routing '
    'margins d = x[anchor_a] - x[anchor_b], applied identically in train and eval; the hard '
    'sign address is unchanged and the existing directional surrogate backward is kept intact '
    'with the score path ADDED. It introduces NO parameters, so this arm has exactly the '
    'baseline param count and any difference is mechanism rather than capacity. Two seeds (1 '
    'and 2) paired against the baseline seeds S5 (1.434572) and S5b (1.452477), because the '
    'measured seed sd at this 4k budget is 0.0096 and one run per arm cannot resolve the '
    'expected effect.')


def main():
    from nanochat.common import get_base_dir
    from nanochat.tokenizer import RustBPETokenizer
    from model_build import build_model
    base = json.load(open(os.path.join(SRC, 'config.json')))
    print(f"anchor seeds: random_seed {base['random_seed']}, lut_base_seed "
          f"{base['lut_base_seed']}  (baseline family uses random_seed 1/2/3)")
    vocab = RustBPETokenizer.from_directory(
        os.path.join(get_base_dir(), 'tokenizer')).get_vocab_size()

    order, bad = [], []
    for name, tag, seed in RUNS:
        cfg = copy.deepcopy(base)
        cfg['random_seed'] = seed
        cfg['lut_forward_confidence'] = True
        cfg['lut_confidence_form'] = 'bounded'
        cfg['exp_name'] = name
        cfg['_sweep_tag'] = f'lookupffn-arm-a-{tag.lower()}'
        cfg['_arch_note'] = NOTE + f' This is {tag}, random_seed {seed}.'
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
        del m
        ok = tot == EXPECT
        if not ok or drift:
            bad.append((name, tot, drift))
        print(f'   {name}  seed {seed}  params {tot:,}  expected {EXPECT:,}  '
              f'{"OK" if ok else "*** PARAM MISMATCH ***"}   '
              f'drift beyond seed+gate: {drift or "none"}')
        order.append(dict(idx=len(order) + 1, run=name, tag=tag, params=tot, expected=EXPECT,
                          deviation=(tot - EXPECT) / EXPECT,
                          device_batch_size=cfg['device_batch_size'],
                          grad_accum=cfg['total_batch_size'] //
                          (cfg['device_batch_size'] * cfg['seq_len']),
                          H=cfg['lut_n_heads'], tph=cfg['lut_tables_per_head'],
                          cells=2 ** cfg['lut_n_anchor_pairs'],
                          d_in=cfg['lut_inner_in_dim'], d_out=cfg['lut_inner_out_dim'],
                          random_seed=seed, forward_confidence=True,
                          confidence_form='bounded',
                          compress_flops=cfg['lut_n_heads'] * 384 * cfg['lut_inner_in_dim'],
                          decompress_flops=cfg['lut_n_heads'] * 384 * cfg['lut_inner_out_dim'],
                          projection_flops_total=cfg['lut_n_heads'] * 384 *
                          (cfg['lut_inner_in_dim'] + cfg['lut_inner_out_dim']),
                          compress_flops_ratio=cfg['lut_n_heads'] * 384 *
                          cfg['lut_inner_in_dim'] / 589824,
                          projection_flops_ratio_vs_vanilla_ffn=cfg['lut_n_heads'] * 384 *
                          (cfg['lut_inner_in_dim'] + cfg['lut_inner_out_dim']) /
                          (2 * 384 * 1536)))
    with open(os.path.join(HERE, 'sweep_arma_manifest.json'), 'w') as f:
        json.dump(dict(n_steps=base['n_steps'], effective_batch_sequences=24,
                       effective_batch_tokens=base['total_batch_size'],
                       eval_every=base['eval_every'],
                       vanilla_ffn_macs_per_token=2 * 384 * 1536,
                       cloned_from='sweep_s05_dout48_H4_tph256_c256_din32',
                       runs=order), f, indent=2)
    print(f'\nwrote {HERE}/sweep_arma_manifest.json')
    if bad:
        print('*** STOP ***')
        sys.exit(1)
    print('both A runs match the baseline param count exactly (the gate adds none) '
          '— clear to run')


if __name__ == '__main__':
    main()
