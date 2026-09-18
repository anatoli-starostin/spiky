"""P4: pooled vs per-dimension (whitened) FVU for saved distilled students, on the harness's held-out slab.

pooled FVU   = sum_d MSE_d / sum_d Var_d          (what the harness reports; MSE weights every dim equally, so
                                                   high-variance dims dominate)
whitened FVU = mean_d (MSE_d / Var_d)            (each output channel counts equally)
Also: FVU on the top-16 variance dims vs the rest, per layer.

    python analyze_perdim_fvu.py run1 run2 ...   [--device cpu]
"""
import argparse, json, os, sys
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import distill_ffn as D  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('runs', nargs='+')
ap.add_argument('--device', default='cuda')
a = ap.parse_args()
dev = a.device
R = os.path.join(HERE, 'runs')
tok = D.RustBPETokenizer.from_directory(os.path.join(D.get_base_dir(), 'tokenizer'))
teacher, tcfg = D.load_teacher(D.DEF_TEACHER, tok.get_vocab_size(), dev)
val_idx = D.take_rows(D.tokenizing_distributed_data_loader_bos_bestfit(tok, 48, 512, split='val', device=dev), 32,
                      skip_rows=12)
io = D.teacher_ffn_io(teacher, val_idx, list(range(6)))
base = json.load(open(D.DEF_STUDENT))
out = {}
for run in a.runs:
    man = json.load(open(os.path.join(R, run, 'manifest.json')))
    ov = man['student_overrides']
    rows = []
    for li in range(6):
        h, o = io[li]
        s = D.build_student({**base, **ov}, 384, 6, li, dev)
        s.load_state_dict(torch.load(os.path.join(R, run, f'student_L{li}.pt'), map_location=dev))
        s.eval()
        with torch.no_grad():
            pred = torch.cat([D.student_forward(s, h[i:i + 2048]) for i in range(0, h.shape[0], 2048)])
        e2 = (pred - o).double().pow(2).mean(0)                   # per-dim MSE
        v = o.double().var(0, unbiased=False)                     # per-dim var
        top = v.argsort(descending=True)[:16]
        rest = v.argsort(descending=True)[16:]
        rows.append(dict(pooled=(e2.sum() / v.sum()).item(), whitened=(e2 / v).mean().item(),
                         top16=(e2[top].sum() / v[top].sum()).item(), rest=(e2[rest].sum() / v[rest].sum()).item()))
        del s
    out[run] = rows
    print(f'--- {run}')
    for li, r in enumerate(rows):
        print(f'  L{li}: pooled {r["pooled"]:.4f} | whitened {r["whitened"]:.4f} | top-16-var dims {r["top16"]:.4f} | '
              f'other 368 dims {r["rest"]:.4f}')
    print(f'  mean: pooled {sum(r["pooled"] for r in rows)/6:.4f} | whitened {sum(r["whitened"] for r in rows)/6:.4f}')
json.dump(out, open(os.path.join(R, 'perdim_fvu_' + '_'.join(a.runs)[:120] + '.json'), 'w'), indent=1)
