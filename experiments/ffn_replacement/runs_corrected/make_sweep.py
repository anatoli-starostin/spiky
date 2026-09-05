"""Generate the 11-run short proxy sweep over the shape of the compression FFN.

BUDGET (shared by every run, and 8x cheaper than the 16k line): 4,000 steps, effective batch
24 sequences = 12,288 tokens, eval every 500 steps. Everything else is exp_n_0118's recipe.

EVAL is NOT scaled down: every run uses the corrected `evaluate_bpb_fixed` — bs48 x 100
batches, leading 12 rows skipped, 2,451,456 val tokens — identical for all 11, only less
often. So the runs are comparable to EACH OTHER to full eval precision.

    *** THESE PROXY bpb NUMBERS ARE NOT COMPARABLE TO THE 16k / batch-48 ANCHORS. ***
    A 4k-step run at half the batch is nowhere near the 16k runs' loss; comparing a sweep
    number to exp_n_0135 (1.165147), exp_n_0136 (1.192926), exp_n_0118 (1.164939) or
    exp_n_0129 (1.170961) is meaningless. S0, the vanilla dense control, is the sweep's own
    zero-line and the ONLY anchor these numbers may be read against.

d_in / d_out. The brief asked for the LUT's input width and its table-row width to be untied.
They ALREADY are: `CompressionMultiHeadLUT(inner_in_dim=..., inner_out_dim=...)` has been the
signature since PR #109, `model_build.py` passes `lut_inner_in_dim` / `lut_inner_out_dim`
straight through, and `param_count` computes the LUT budget as n_heads*tph*2^nap*eff_out —
i.e. tables scale with cells*d_out only, while d_in touches nothing but the compress
projection. No library change was needed; S1 (d_in == d_out == 32) is the tied control that
reproduces exp_n_0164 and shows the untied path is a strict generalisation.

    python make_sweep.py          # writes the 11 run folders and checks every param count
"""
import copy
import json
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
FR = os.path.dirname(HERE)                       # experiments/ffn_replacement
sys.path.insert(0, os.path.join(FR, 'tools'))
sys.path.insert(0, os.path.expanduser('~/projects/nanochat'))

N_STEPS = 4000
SEQS = 24                     # effective batch, in sequences
SEQ_LEN = 512
EVAL_EVERY = 500
BASE_PARAMS = 28_714_752      # embeddings + attention + norms + untied head

# name, order, H, tph, cells, d_in, d_out, expected (from the brief), note
SWEEP = [
    ('s00_vanilla_dense',                   0, None, None, None, None, None, 35_800_000,
     'VANILLA DENSE CONTROL — standard 4x MLP FFN, no LUT. The sweep zero-line: every other '
     'run is read as a delta against this, on the same 4k/24-seq budget.'),
    ('s01_tied_H4_tph256_c256_din32_dout32', 1, 4, 256, 256, 32, 32, 79_600_000,
     'TIED CONTROL — d_in == d_out == 32, i.e. exactly exp_n_0164 on the proxy budget. Doubles '
     'as the regression check that the untied d_in/d_out path reproduces the tied one.'),
    ('s02_din64_H4_tph256_c256_dout32',      2, 4, 256, 256, 64, 32, 80_200_000,
     'd_in LADDER, rung 2 of 3 (S1 32 -> S2 64 -> S3 96). Widens ONLY the code the sign '
     'comparisons are computed on; the table budget is untouched. Isolates routing width.'),
    ('s03_din96_H4_tph256_c256_dout32',      3, 4, 256, 256, 96, 32, 80_800_000,
     'd_in LADDER, rung 3 of 3. Same table budget as S1/S2, widest routing code.'),
    ('s09_pure_H1_tph1024_c256_din128_dout32', 4, 1, 1024, 256, 128, 32, 79_900_000,
     'PURE FastMHL END of the head/routing trade (S1 H4 -> S8 H2 -> S9 H1) at constant table '
     'budget: one head, all 1024 tables, and the whole 128-wide code routed jointly.'),
    ('s08_H2_tph512_c256_din64_dout32',      5, 2, 512, 256, 64, 32, 79_800_000,
     'HEADS DOWN, ROUTING UP — midpoint of the S1/S8/S9 trade: half the heads, twice the '
     'tables each, twice the code width per head. Same table budget, same total FLOPs.'),
    ('s06_isoparam_deep_c512_dout16',        6, 4, 256, 512, 32, 16, 79_600_000,
     'ISO-PARAM SHAPE LINE, deep/narrow end: twice the cells per table (nap9), half the row '
     'width. Same 50.3M table budget as S1 and S7, spent on depth instead of width.'),
    ('s07_isoparam_shallow_c128_dout64',     7, 4, 256, 128, 32, 64, 80_600_000,
     'ISO-PARAM SHAPE LINE, shallow/wide end: half the cells (nap7), twice the row width. '
     'Same 50.3M table budget as S1 and S6, spent on width instead of depth.'),
    ('s04_dout16_H4_tph256_c256_din32',      8, 4, 256, 256, 32, 16, 54_400_000,
     'd_out LADDER, low rung (S4 16 -> S1 32 -> S5 48). Halves the table budget outright — '
     'the capacity axis, against S2/S3 which move routing at constant capacity.'),
    ('s05_dout48_H4_tph256_c256_din32',      9, 4, 256, 256, 32, 48, 104_900_000,
     'd_out LADDER, high rung. 1.5x the table budget of S1.'),
    ('s10_scaled_H2_tph512_c512_din64_dout32', 10, 2, 512, 512, 64, 32, 130_100_000,
     'SCALED CANDIDATE — S8 with twice the cells. The largest run in the sweep; a preview of '
     'what the shape would look like at the size class the 16k runs live in.'),
]


def build_cfg(name, H, tph, cells, d_in, d_out, note):
    nap = {64: 6, 128: 7, 256: 8, 512: 9, 1024: 10}[cells] if cells else None
    # soft-backward buffer is [tokens, H*tph, cells] fp32 -> 12.9 GiB at bs12 when
    # H*tph*cells = 524,288, which OOMs the 5090 (see exp_n_0162's train_oom_bs12.log)
    dbs = 6 if (cells and H * tph * cells >= 524_288) else 12
    cfg = {
        'exp_name': f'sweep_{name}',
        'gamma': 0,
        'ffn_type': 'dense' if H is None else 'compression',
        'tie_unembedder': False,
        'depth': 6, 'n_embd': 384, 'n_head': 6, 'seq_len': SEQ_LEN,
        'device_batch_size': dbs,
        'total_batch_size': SEQS * SEQ_LEN,
        'n_steps': N_STEPS,
        'lr': 0.0003, 'weight_decay': 0.1, 'lr_warmup_fraction': 0.1,
        'eval_every': EVAL_EVERY,
        'random_seed': 1,
        'compute_dtype': 'bf16',
        'tokenizer_vocab_size': 32768,
    }
    if H is not None:
        cfg.update({
            'lut_inner_in_dim': d_in, 'lut_inner_out_dim': d_out,
            'lut_n_anchor_pairs': nap, 'lut_tables_per_head': tph, 'lut_n_heads': H,
            'lut_joint_head_compression': False,
            'lut_forward_mode': 'hard', 'lut_use_bf16': False,
            'lut_init_weights_noise': 0.001, 'lut_base_seed': 1000,
            'lut_learnable_temps': True,
        })
        shape = (f'H={H} tph={tph} cells={cells} (nap{nap}) d_in={d_in} d_out={d_out} | '
                 f'H*tph={H*tph} (<=1024 OK), d_in>=32 OK | '
                 f'compress-projection FLOPs H*384*d_in = {H*384*d_in:,} vs vanilla '
                 f'384*384*4 = {589824:,} -> {H*384*d_in/589824:.4f}x')
    else:
        shape = 'dense 4x MLP (384->1536->384), no LUT'
    cfg['_arch_note'] = (
        f'[SHORT PROXY SWEEP] {note} {shape}. Budget shared by all 11 runs: {N_STEPS} steps, '
        f'effective batch {SEQS} sequences = {SEQS*SEQ_LEN:,} tokens '
        f'(device_batch {dbs} / grad_accum {SEQS//dbs}), eval every {EVAL_EVERY} steps. '
        f'Otherwise exp_n_0118\'s recipe: untied unembedder, hard forward with learnable '
        f'temps, independent per-head compression, 6L d=384 on ClimbMix, lr 3e-4, '
        f'warmup_frac 0.1, cosine. EVAL IS NOT SCALED DOWN — the corrected protocol '
        f'(evaluate_bpb_fixed, bs48 x 100, skip 12, 2,451,456 val tokens), identical for '
        f'every run. *** THE bpb FROM THIS SWEEP IS COMPARABLE ONLY TO THE OTHER SWEEP RUNS, '
        f'NEVER TO THE 16k/batch-48 ANCHORS (exp_n_0135 1.165147, exp_n_0136 1.192926, '
        f'exp_n_0118 1.164939, exp_n_0129 1.170961). S0 (sweep_s00_vanilla_dense) is this '
        f'sweep\'s own zero-line. ***')
    cfg['_sweep_tag'] = f'proxy-sweep-{name.split("_")[0]}'
    return cfg


def main():
    from nanochat.common import get_base_dir
    from nanochat.tokenizer import RustBPETokenizer
    from model_build import build_model
    tok = RustBPETokenizer.from_directory(os.path.join(get_base_dir(), 'tokenizer'))
    vocab = tok.get_vocab_size()

    print(f'{"run":<42} {"built":>12} {"expected":>12} {"dev":>4} {"dev%":>7}  status')
    order, bad = [], []
    for name, idx, H, tph, cells, d_in, d_out, expect, note in SWEEP:
        d = os.path.join(HERE, f'sweep_{name}')
        os.makedirs(d, exist_ok=True)
        cfg = build_cfg(name, H, tph, cells, d_in, d_out, note)
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
        print(f'sweep_{name:<36} {tot:>12,} {expect:>12,} '
              f'{cfg["device_batch_size"]:>4} {100*dev:>+6.2f}%  '
              f'{"OK" if ok else "*** OUT OF TOLERANCE ***"}')
        order.append(dict(idx=idx, run=f'sweep_{name}', params=tot, expected=expect,
                          deviation=dev, device_batch_size=cfg['device_batch_size'],
                          grad_accum=SEQS // cfg['device_batch_size'],
                          H=H, tph=tph, cells=cells, d_in=d_in, d_out=d_out,
                          compress_flops=(H * 384 * d_in) if H else 589824,
                          compress_flops_ratio=(H * 384 * d_in / 589824) if H else 1.0))
    order.sort(key=lambda r: r['idx'])
    with open(os.path.join(HERE, 'sweep_manifest.json'), 'w') as f:
        json.dump(dict(n_steps=N_STEPS, effective_batch_sequences=SEQS,
                       effective_batch_tokens=SEQS * SEQ_LEN, eval_every=EVAL_EVERY,
                       base_params=BASE_PARAMS, runs=order), f, indent=2)
    print(f'\nrun order: {" -> ".join(r["run"].split("_")[1] for r in order)}')
    print(f'wrote {HERE}/sweep_manifest.json')
    if bad:
        print('\n*** STOP: param counts out of the 1% tolerance ***')
        for n, t, e, d in bad:
            print(f'   {n}: built {t:,} vs expected {e:,} ({100*d:+.2f}%)')
        sys.exit(1)
    print('all 11 param counts within 1% of the brief — clear to run')


if __name__ == '__main__':
    main()
