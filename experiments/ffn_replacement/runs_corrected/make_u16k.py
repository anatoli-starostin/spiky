"""Generate the two full-length untying runs: U1-16k and U3-16k.

WHAT THIS TESTS. Every run in the historical 16k grid has `d_in == d_out`, so untying has never
been tested at full length at a small table budget. The direct anchor is **exp_n_0127 =
1.187011**, which carries the SAME table budget — 6*4*128*128*48 = 18,874,368 — with the dims
tied at 48. These two hold that budget and untie in opposite directions:

    U1-16k  d_in 32, d_out 48   H*d_in = 128   projection FLOPs 122,880  0.1042x vanilla
    U3-16k  d_in 96, d_out 48   H*d_in = 384   projection FLOPs 221,184  0.1875x vanilla
    (0127   d_in 48, d_out 48                  projection FLOPs 147,456  0.1250x vanilla)

so U1 unties DOWNWARD — cheaper routing than 0127 — and U3 unties UPWARD.

PROTOCOL is sourced from `../runs/exp_n_0127/config.json`, read-only, so the only thing that
differs from the anchor is the FFN configuration: 16,000 steps, effective batch 24,576 tokens
(device_batch 12 x grad_accum 4), lr 3e-4, warmup_frac 0.1, cosine to the 0.1x floor, wd 0.1,
seed 1, seq 512, depth 6, n_embd 384, n_head 6, untied unembedder, bf16. The ONE deliberate
departure is the eval: `evaluate_bpb_fixed` (bs48 x 100, skip 12, batch-size independent) every
500 steps, since 0127's `eval_every 200 / eval_steps 10` is the batch-coupled eval that
FIXED_EVAL.md replaced. That is what makes these numbers comparable to 0127's *corrected* score
rather than to its original one.

    python make_u16k.py
"""
import json
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
FR = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(FR, 'tools'))
sys.path.insert(0, os.path.expanduser('~/projects/nanochat'))

ANCHOR_CFG = os.path.join(FR, 'runs', 'exp_n_0127_grid_H4d48_nap7_tph128', 'config.json')
VANILLA_FFN_MACS = 2 * 384 * 1536
TABLES = 6 * 4 * 256 * 64 * 48        # 18,874,368 — must equal 0127's 6*4*128*128*48

RUNS = [
    ('exp_n_0169_u1_16k_H4_c64_din32_dout48', 1, 32, 48_329_484,
     'U1 AT FULL LENGTH. Unties DOWNWARD from the exp_n_0127 anchor: same table budget '
     '(18,874,368), d_in 48 -> 32 with d_out held at 48, so routing gets CHEAPER '
     '(0.1042x vanilla projection FLOPs against 0127\'s 0.1250x). H*d_in = 128.'),
    ('exp_n_0170_u3_16k_H4_c64_din96_dout48', 2, 96, 48_920_844,
     'U3 AT FULL LENGTH. Unties UPWARD from the exp_n_0127 anchor: same table budget '
     '(18,874,368), d_in 48 -> 96 with d_out held at 48, so routing gets more EXPENSIVE '
     '(0.1875x vanilla projection FLOPs against 0127\'s 0.1250x). H*d_in = 384. At 4k the '
     'proxy put U3 ahead of U1 by 0.006548; the proxy has been reliable for ranking and has '
     'overstated magnitude by roughly 3x.'),
]
H, TPH, CELLS, NAP, D_OUT = 4, 256, 64, 6, 48


def main():
    from nanochat.common import get_base_dir
    from nanochat.tokenizer import RustBPETokenizer
    from model_build import build_model
    anchor = json.load(open(ANCHOR_CFG))          # read-only; runs/ is never written
    tok = RustBPETokenizer.from_directory(os.path.join(get_base_dir(), 'tokenizer'))
    vocab = tok.get_vocab_size()

    anchor_tables = (6 * anchor['lut_n_heads'] * anchor['lut_tables_per_head']
                     * 2 ** anchor['lut_n_anchor_pairs'] * anchor['lut_inner_out_dim'])
    print(f"anchor exp_n_0127: tables {anchor_tables:,}  (ours {TABLES:,})  "
          f"match {anchor_tables == TABLES}")
    if anchor_tables != TABLES:
        print('*** STOP: table budget does not match the 0127 anchor ***')
        sys.exit(1)

    order, bad = [], []
    for name, idx, d_in, expect, note in RUNS:
        cfg = {k: anchor[k] for k in (
            'gamma', 'ffn_type', 'tie_unembedder', 'depth', 'n_embd', 'n_head', 'seq_len',
            'device_batch_size', 'total_batch_size', 'n_steps', 'lr', 'weight_decay',
            'lr_warmup_fraction', 'random_seed', 'compute_dtype', 'tokenizer_vocab_size',
            'lut_joint_head_compression', 'lut_forward_mode', 'lut_use_bf16',
            'lut_init_weights_noise', 'lut_base_seed', 'lut_learnable_temps')}
        cfg['exp_name'] = name
        cfg['eval_every'] = 500          # the ONE deliberate change: fixed-protocol eval cadence
        cfg.update(lut_inner_in_dim=d_in, lut_inner_out_dim=D_OUT,
                   lut_n_anchor_pairs=NAP, lut_tables_per_head=TPH, lut_n_heads=H)
        cf, df = H * 384 * d_in, H * 384 * D_OUT
        cfg['_arch_note'] = (
            f'{note} H={H} tph={TPH} cells={CELLS} (nap{NAP}) d_in={d_in} d_out={D_OUT}. '
            f'Constraints: H*tph = {H*TPH} (exactly at the 1024 cap, not over); d_in >= 32; '
            f'H*input_dim*d_in = {cf:,} vs input_dim^2*4 = {589824:,} — well under. Tables '
            f'6*{H}*{TPH}*{CELLS}*{D_OUT} = {TABLES:,}, identical to exp_n_0127 and to the 4k '
            f'proxy runs of the same names. Projection FLOPs compress {cf:,} + decompress '
            f'{df:,} = {cf+df:,} -> {(cf+df)/VANILLA_FFN_MACS:.4f}x vanilla FFN. PROTOCOL '
            f'mirrors exp_n_0127 exactly (16,000 steps, effective batch 24,576 tokens via '
            f'device_batch 12 / grad_accum 4, lr 3e-4, warmup_frac 0.1, cosine, wd 0.1, seed 1, '
            f'seq 512, 6L d=384 untied, bf16) with ONE deliberate change: eval is the corrected '
            f'evaluate_bpb_fixed (bs48 x 100, skip 12, 2,451,456 val tokens, batch-size '
            f'independent) every 500 steps instead of 0127\'s batch-coupled eval_every 200 / '
            f'eval_steps 10. This run is therefore directly comparable to 0127\'s CORRECTED '
            f'score of 1.187011, and to the rest of the corrected 16k grid.')
        cfg['_sweep_tag'] = f'untying-16k-{name.split("_")[3]}'

        d = os.path.join(HERE, name)
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, 'config.json'), 'w') as f:
            json.dump(cfg, f, indent=2)
        shutil.copy(os.path.join(FR, 'train_fixed.py'), os.path.join(d, 'train.py'))
        m = build_model(cfg, vocab, device='cpu')
        tot = sum(p.numel() for p in m.parameters())
        del m
        dev = (tot - expect) / expect
        ok = abs(dev) <= 0.01
        if not ok:
            bad.append((name, tot, expect, dev))
        ga = cfg['total_batch_size'] // (cfg['device_batch_size'] * cfg['seq_len'])
        print(f'\n{name}')
        print(f'   H*tph {H*TPH} (cap 1024)   d_in {d_in} (>=32)   d_out {D_OUT}   '
              f'H*d_in {H*d_in}')
        print(f'   tables {TABLES:,}   built params {tot:,}  expected {expect:,}  '
              f'dev {100*dev:+.2f}%  {"OK" if ok else "*** OUT OF TOLERANCE ***"}')
        print(f'   projection FLOPs {cf:,} + {df:,} = {cf+df:,} -> '
              f'{(cf+df)/VANILLA_FFN_MACS:.4f}x vanilla   '
              f'(H*384*d_in {cf:,} << 384*384*4 {589824:,})')
        print(f'   protocol: {cfg["n_steps"]} steps, device_batch '
              f'{cfg["device_batch_size"]} x grad_accum {ga} = '
              f'{cfg["total_batch_size"]:,} tokens, lr {cfg["lr"]}, '
              f'warmup {cfg["lr_warmup_fraction"]}, seed {cfg["random_seed"]}, '
              f'eval_every {cfg["eval_every"]}')
        order.append(dict(idx=idx, run=name, params=tot, expected=expect, deviation=dev,
                          device_batch_size=cfg['device_batch_size'], grad_accum=ga,
                          H=H, tph=TPH, cells=CELLS, d_in=d_in, d_out=D_OUT,
                          tables=TABLES, compress_flops=cf, decompress_flops=df,
                          projection_flops_total=cf + df,
                          compress_flops_ratio=cf / 589824,
                          projection_flops_ratio_vs_vanilla_ffn=(cf + df) / VANILLA_FFN_MACS))
    order.sort(key=lambda r: r['idx'])
    with open(os.path.join(HERE, 'u16k_manifest.json'), 'w') as f:
        json.dump(dict(anchor='exp_n_0127_grid_H4d48_nap7_tph128',
                       anchor_corrected_bpb=1.1870110778691643,
                       table_params_shared=TABLES,
                       vanilla_ffn_macs_per_token=VANILLA_FFN_MACS, runs=order), f, indent=2)
    print(f'\nwrote {HERE}/u16k_manifest.json')
    if bad:
        print('*** STOP: param counts out of the 1% tolerance ***')
        sys.exit(1)
    print('both within 1% of expectation, table budgets identical to the 0127 anchor '
          '— clear to run')


if __name__ == '__main__':
    main()
