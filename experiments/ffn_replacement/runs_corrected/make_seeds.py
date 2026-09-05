"""Clone the S5 proxy run at two further RNG seeds, to MEASURE the noise floor.

The "~0.002 noise floor" quoted on every marginal comparison in SWEEP_RESULTS.md is an
inherited assumption, never a measurement. S5b and S5c turn it into a measured quantity.

The config is CLONED FROM DISK (`sweep_s05_dout48_H4_tph256_c256_din32/config.json`) and only
`random_seed` is changed — nothing is retyped.

WHAT random_seed ACTUALLY CONTROLS — verified empirically by building the model at seed 1 and
seed 2 and diffing the state dicts, not inferred from reading the code:

  RE-DRAWN (28,118,016 params, 26.8%):  tok_emb, head, blocks.attn.qkv,
                                        blocks.ffn.compress.{weight,bias},
                                        blocks.ffn.decompress.bias
  UNAFFECTED (76,951,356 params):       blocks.ffn.lut_batched.* — the ENTIRE 75,497,472-param
                                        table budget, its anchor pairs and its temperatures —
                                        plus attn.proj and ffn.decompress.weight (both zeroed
                                        by construction) and every LayerNorm

The LUT's anchors and tables are drawn from their own `torch.Generator().manual_seed(
lut_base_seed + layer_idx)`, which is independent of the global `torch.manual_seed(random_seed)`
that train_fixed.py sets. And the training loader walks the shard deterministically from token 0
with no shuffle and no seed, so **data order is identical across all three runs**.

    => THE MEASURED SPREAD IS A LOWER BOUND ON TRUE RUN-TO-RUN VARIANCE.
       A fuller measurement would also vary `lut_base_seed` (re-drawing the anchors and tables)
       and the data order. This measures the variance from re-drawing the dense parameters only.

The runs are NOT degenerate: 28.1M parameters genuinely differ, so these are three distinct
optimisation trajectories, not three copies of one.

    python make_seeds.py
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
RUNS = [('exp_n_0174_s5b_seed2', 'S5b', 2), ('exp_n_0175_s5c_seed3', 'S5c', 3)]
NOTE = (
    'SEED REPLICATE of the S5 proxy run — config cloned verbatim from '
    'sweep_s05_dout48_H4_tph256_c256_din32/config.json with ONLY random_seed changed, to turn '
    'the assumed "~0.002 noise floor" into a measured quantity. VERIFIED EMPIRICALLY (built at '
    'two seeds and diffed the state dicts): random_seed re-draws 28,118,016 params (26.8%) — '
    'tok_emb, head, attn.qkv, ffn.compress.{weight,bias}, ffn.decompress.bias — and leaves '
    '76,951,356 unaffected, INCLUDING the entire 75,497,472-param LUT table budget and its '
    'anchor pairs, which are seeded from lut_base_seed via their own Generator. The training '
    'loader is deterministic from token 0 with no shuffle, so data order is identical too. THE '
    'MEASURED SPREAD IS THEREFORE A LOWER BOUND on true run-to-run variance; a fuller '
    'measurement would also vary lut_base_seed and the data order.')


def main():
    from nanochat.common import get_base_dir
    from nanochat.tokenizer import RustBPETokenizer
    from model_build import build_model
    base = json.load(open(os.path.join(SRC, 'config.json')))
    print(f"cloned from {os.path.basename(SRC)}   random_seed {base['random_seed']}, "
          f"lut_base_seed {base['lut_base_seed']}")
    vocab = RustBPETokenizer.from_directory(
        os.path.join(get_base_dir(), 'tokenizer')).get_vocab_size()

    order = []
    for name, tag, seed in RUNS:
        if seed == base['random_seed']:
            print(f'*** STOP: {name} would reuse seed {seed} — identical to S5 ***')
            sys.exit(1)
        cfg = copy.deepcopy(base)
        cfg['random_seed'] = seed
        cfg['exp_name'] = name
        cfg['_sweep_tag'] = f'proxy-seed-{tag.lower()}'
        cfg['_arch_note'] = NOTE + f' This is {tag}, random_seed {seed} (S5 used '
        cfg['_arch_note'] += f"{base['random_seed']}); everything else is byte-identical."
        d = os.path.join(HERE, name)
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, 'config.json'), 'w') as f:
            json.dump(cfg, f, indent=2)
        shutil.copy(os.path.join(FR, 'train_fixed.py'), os.path.join(d, 'train.py'))

        # every key except random_seed / exp_name / the notes must match S5 exactly
        drift = [k for k in set(cfg) | set(base)
                 if k not in ('random_seed', 'exp_name', '_arch_note', '_sweep_tag')
                 and cfg.get(k) != base.get(k)]
        m = build_model(cfg, vocab, device='cpu')
        tot = sum(p.numel() for p in m.parameters())
        del m
        print(f'   {name}  seed {seed}  params {tot:,}  '
              f'config drift beyond the seed: {drift or "none"}')
        if drift:
            print('*** STOP: the clone differs from S5 in more than the seed ***')
            sys.exit(1)
        order.append(dict(idx=len(order) + 1, run=name, tag=tag, params=tot,
                          expected=104_952_588, deviation=0.0,
                          device_batch_size=cfg['device_batch_size'],
                          grad_accum=cfg['total_batch_size'] //
                          (cfg['device_batch_size'] * cfg['seq_len']),
                          H=cfg['lut_n_heads'], tph=cfg['lut_tables_per_head'],
                          cells=2 ** cfg['lut_n_anchor_pairs'],
                          d_in=cfg['lut_inner_in_dim'], d_out=cfg['lut_inner_out_dim'],
                          random_seed=seed,
                          compress_flops=cfg['lut_n_heads'] * 384 * cfg['lut_inner_in_dim'],
                          decompress_flops=cfg['lut_n_heads'] * 384 * cfg['lut_inner_out_dim'],
                          projection_flops_total=cfg['lut_n_heads'] * 384 *
                          (cfg['lut_inner_in_dim'] + cfg['lut_inner_out_dim']),
                          compress_flops_ratio=cfg['lut_n_heads'] * 384 *
                          cfg['lut_inner_in_dim'] / 589824,
                          projection_flops_ratio_vs_vanilla_ffn=cfg['lut_n_heads'] * 384 *
                          (cfg['lut_inner_in_dim'] + cfg['lut_inner_out_dim']) / (2 * 384 * 1536)))
    with open(os.path.join(HERE, 'sweep_seeds_manifest.json'), 'w') as f:
        json.dump(dict(n_steps=base['n_steps'], effective_batch_sequences=24,
                       effective_batch_tokens=base['total_batch_size'],
                       eval_every=base['eval_every'],
                       vanilla_ffn_macs_per_token=2 * 384 * 1536,
                       cloned_from='sweep_s05_dout48_H4_tph256_c256_din32',
                       runs=order), f, indent=2)
    print(f'\nwrote {HERE}/sweep_seeds_manifest.json')
    print('both clones differ from S5 in the seed alone — clear to run')


if __name__ == '__main__':
    main()
