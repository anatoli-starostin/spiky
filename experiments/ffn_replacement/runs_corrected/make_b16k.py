"""The 16K Light run: arm B's mechanism at the full budget, paired with exp_n_0129.

Arm B (4K, exp_n_0181) put LightMultiHeadLUT at 1.477708 against a 1.474749 vanilla-dense
zero-line -- a +0.003 difference, INSIDE the 0.0096 seed sd, i.e. Light matched dense quality
with a 5.6x cheaper backward. This asks whether that survives at the 16K budget, where the
reference points are real rather than proxies:

    exp_n_0129  same LUT sizing, Fast, gate off   1.170961
    exp_n_0135  vanilla dense                     1.165147
    exp_n_0176  vanilla dense, seed 2             1.161798   (vanilla seed spread 0.00335)

Config is CLONED VERBATIM from exp_n_0129/config.json -- the 16K member of this LUT sizing --
with only four deltas: lut_impl, the two gate keys, and exp_name. random_seed is already 1.

ONE DELIBERATE DEVIATION FROM ARM B, decided by Anatoli: `lut_inner_in_dim` stays at the 16K
family's 48 rather than arm B's 32. That buys a like-for-like comparison against exp_n_0129
at the cost of exact faithfulness to the 4K arm. d_in affects only the compress projection,
not the table budget, so the table-budget law is unaffected.

Two keys in the cloned config are VESTIGIAL and kept only because the clone is verbatim:
  "compute_dtype": "bf16"  -- written by make_sweep.py, read by NO code (grep confirms a
                              single hit, the line that writes it). The field that reaches
                              the LUT is lut_use_bf16, which is false.
  "eval_steps": 10         -- the legacy batch-coupled eval knob. eval_config() reads only
                              an optional `fixed_eval` sub-dict and deliberately ignores
                              this key, so scoring is the standard bs48 x 100, skip 12.
Neither can silently change the run; both are noted so nobody has to re-derive that.

    python make_b16k.py
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

SRC = os.path.join(HERE, 'exp_n_0129_grid_H4d48_nap8_tph256')
RUN = 'exp_n_0184_B16k_light_bnorm_seed1'
EXPECT = 105_100_428 - 12          # exp_n_0129's count minus Light's 12 temperature scalars


def main():
    from nanochat.common import get_base_dir
    from nanochat.tokenizer import RustBPETokenizer
    from model_build import build_model

    base = json.load(open(os.path.join(SRC, 'config.json')))
    cfg = copy.deepcopy(base)
    cfg['lut_impl'] = 'light'
    cfg['lut_forward_confidence'] = True
    cfg['lut_confidence_form'] = 'bounded_norm'
    cfg['random_seed'] = 1
    cfg['exp_name'] = RUN

    drift = [k for k in set(cfg) | set(base)
             if k not in ('exp_name', 'lut_impl', 'lut_forward_confidence',
                          'lut_confidence_form', 'random_seed')
             and cfg.get(k) != base.get(k)]
    assert not drift, f'unintended drift from exp_n_0129: {drift}'
    assert base['random_seed'] == 1, base['random_seed']
    assert cfg['lut_inner_in_dim'] == 48 and cfg['lut_use_bf16'] is False

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

    # Inspect the INSTANTIATED modules -- do not infer capability from the config.
    ffn = m.blocks[0].ffn
    light = getattr(ffn, 'lut_light', None)
    checks = {
        'params': (tot, EXPECT),
        'lut_light module present': (light is not None, True),
        'lut_batched absent (Fast path unused)': (not hasattr(ffn, 'lut_batched'), True),
        'multi_head_input ON': (getattr(light, 'multi_head_input', None), True),
        'n_heads': (getattr(light, 'n_heads', None), cfg['lut_n_heads']),
        'tables_per_head': (getattr(light, 'tables_per_head', None),
                            cfg['lut_tables_per_head']),
        'confidence_form': (getattr(light, 'confidence_form', None), 'bounded_norm'),
        'confidence_gain': (getattr(light, 'confidence_gain', None), 1.0),
        'tables dtype fp32': (light.tables.dtype if light else None, __import__('torch').float32),
        'no temperature params': ([n for n, _ in m.named_parameters() if 'temp' in n], []),
        'compress out (H*d_in)': (ffn.compress.out_features,
                                  cfg['lut_n_heads'] * cfg['lut_inner_in_dim']),
    }
    ok = True
    print(f'{RUN}\n')
    for name, (got, want) in checks.items():
        good = got == want
        ok &= good
        print(f'   {name:<40}{str(got)[:26]:>28}   expected {str(want)[:22]:<24}'
              f'{"OK" if good else "*** MISMATCH ***"}')
    print(f'\n   forms across all {len(m.blocks)} blocks: '
          f'{ {b.ffn.lut_light.confidence_form for b in m.blocks} }   '
          f'multi_head_input: { {b.ffn.lut_light.multi_head_input for b in m.blocks} }')
    del m
    if not ok:
        print('*** STOP ***')
        sys.exit(1)
    print(f'\nwrote {d}/config.json and train.py -- READY, NOT LAUNCHED')


if __name__ == '__main__':
    main()
