"""Generate the small-budget d_in ladder — U1, U2, U3.

WHY THESE EXIST. Every run in the historical 16k grid has `d_in == d_out`, so untying has
never been tested at a SMALL table budget. The `H*d_in >= 128` floor was derived from the 4k
sweep at 50.3M and 75.5M tables. The hypothesis here is that when tables are SCARCE the
routing has to work harder, so the floor RISES above 128. That is informative either way: if
it rises, small models need more code width and therefore more multiplies, which closes off
the small end of the design space on the paper's own FLOPs axis.

The table budget is held FIXED at 6*4*256*64*48 = 18,874,368 across all three — a quarter of
S5's — and only `d_in` moves:

    U1  H4 tph256 cells64 d_in 32 d_out 48   H*d_in = 128   proj FLOPs 122,880  0.1042x
    U2  H4 tph256 cells64 d_in 64 d_out 48   H*d_in = 256   proj FLOPs 147,456  0.1250x
    U3  H4 tph256 cells64 d_in 96 d_out 48   H*d_in = 384   proj FLOPs 172,032  0.1458x

CAVEAT to carry into the reading: unlike a table-split change, moving d_in MOVES FLOPs. These
three are not iso-cost, and the comparison against the 75.5M-budget pair (S5 d_in 32 vs R4
d_in 64, where more d_in HURT) has to be read with that in mind.

Same budget and recipe as the other 18 proxy runs, from the same generator, so all 21 are
mutually comparable — and comparable to nothing on the 16k line.

    python make_sweep4.py
"""
import json
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
FR = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(FR, 'tools'))
sys.path.insert(0, os.path.expanduser('~/projects/nanochat'))

from make_sweep import build_cfg, SEQS, N_STEPS, EVAL_EVERY   # noqa: E402

VANILLA_FFN_MACS = 2 * 384 * 1536
TABLES = 6 * 4 * 256 * 64 * 48        # 18,874,368 — identical for all three

SWEEP4 = [
    ('u1_small_din32_dout48', 1, 4, 256, 64, 32, 48, 48_300_000,
     'SMALL-BUDGET d_in LADDER, rung 1 of 3. Table budget pinned at 18,874,368 — a quarter of '
     "S5's — with cells cut to 64 (nap6, only 6 comparisons per address). H*d_in = 128, exactly "
     'the floor derived from the 4k sweep. If the floor rises when tables are scarce, THIS is '
     'the run that should suffer.'),
    ('u2_small_din64_dout48', 2, 4, 256, 64, 64, 48, 48_600_000,
     'SMALL-BUDGET d_in LADDER, rung 2 of 3. H*d_in = 256, twice the floor. Same table budget '
     'as U1; only the compress projection grows.'),
    ('u3_small_din96_dout48', 3, 4, 256, 64, 96, 48, 48_900_000,
     'SMALL-BUDGET d_in LADDER, rung 3 of 3. H*d_in = 384, three times the floor. At the 75.5M '
     'budget the equivalent step (S5 -> R4) HURT by +0.0072; this asks whether the sign flips '
     'when tables are scarce.'),
]


def main():
    from nanochat.common import get_base_dir
    from nanochat.tokenizer import RustBPETokenizer
    from model_build import build_model
    tok = RustBPETokenizer.from_directory(os.path.join(get_base_dir(), 'tokenizer'))
    vocab = tok.get_vocab_size()

    order, bad, tabs = [], [], set()
    for name, idx, H, tph, cells, d_in, d_out, expect, note in SWEEP4:
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
        tables = 6 * H * tph * cells * d_out
        tabs.add(tables)
        cf, df = H * 384 * d_in, H * 384 * d_out
        print(f'sweep_{name}')
        print(f'   H*tph = {H*tph} (<=1024 OK)   d_in {d_in} (>=32 OK)   d_out {d_out}   '
              f'H*d_in = {H*d_in}')
        print(f'   tables 6*{H}*{tph}*{cells}*{d_out} = {tables:,}')
        print(f'   built params {tot:,}  expected {expect:,}  dev {100*dev:+.2f}%  '
              f'{"OK" if ok else "*** OUT OF TOLERANCE ***"}')
        print(f'   projection FLOPs compress {cf:,} + decompress {df:,} = {cf+df:,}  '
              f'-> {(cf+df)/VANILLA_FFN_MACS:.4f}x vanilla FFN')
        print(f'   device_batch {cfg["device_batch_size"]} x grad_accum '
              f'{SEQS // cfg["device_batch_size"]}')
        order.append(dict(idx=idx, run=f'sweep_{name}', params=tot, expected=expect,
                          deviation=dev, device_batch_size=cfg['device_batch_size'],
                          grad_accum=SEQS // cfg['device_batch_size'],
                          H=H, tph=tph, cells=cells, d_in=d_in, d_out=d_out,
                          compress_flops=cf, decompress_flops=df,
                          projection_flops_total=cf + df,
                          compress_flops_ratio=cf / 589824,
                          projection_flops_ratio_vs_vanilla_ffn=(cf + df) / VANILLA_FFN_MACS))
    order.sort(key=lambda r: r['idx'])
    with open(os.path.join(HERE, 'sweep4_manifest.json'), 'w') as f:
        json.dump(dict(n_steps=N_STEPS, effective_batch_sequences=SEQS,
                       effective_batch_tokens=SEQS * 512, eval_every=EVAL_EVERY,
                       vanilla_ffn_macs_per_token=VANILLA_FFN_MACS,
                       table_params_shared=TABLES, runs=order), f, indent=2)
    print(f'\ntable budget identical across all three: {tabs == {TABLES}} ({tabs})')
    print(f'wrote {HERE}/sweep4_manifest.json')
    if tabs != {TABLES}:
        print('*** STOP: table budgets are not identical ***')
        sys.exit(1)
    if bad:
        print('*** STOP: param counts out of the 1% tolerance ***')
        for n, t, e, dv in bad:
            print(f'   {n}: built {t:,} vs expected {e:,} ({100*dv:+.2f}%)')
        sys.exit(1)
    print('all three within 1% of the brief, table budgets identical — clear to run')


if __name__ == '__main__':
    main()
