"""Per-cell-bias menu runs: FVU, gate slope, addressing drift, tau, usage, and the LUT-half vs matrix-half split.

    python analyze_menu_bias.py runA runB ... [--compare ref1 ref2 ...] [--device cuda]

For each layer of each run, on the harness's held-out slab (hard read, eval mode):
  matrix part  y_W = sum_m a_m x W_m       bias part  y_b = sum_t s_t b[t, c_t]
  reports ||y_b|| / ||y_W|| (rms over tokens and dims, pre-decompress) and the FVU of the student with each half
  ablated (bias zeroed / matrix zeroed) to show which half carries the fit.
"""
import argparse, csv, json, math, os, sys
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import distill_ffn as D  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('runs', nargs='+')
ap.add_argument('--compare', nargs='*', default=['ladder4k_H8', 'menu4k_H8_tph16', 'menu4k_H8_tph8',
                                                  'menu4k_H8_tph16_M128'])
ap.add_argument('--device', default='cuda')
a = ap.parse_args()
dev = a.device
R = os.path.join(HERE, 'runs')
FFN = 2 * 384 * 1536


def final_fvu(run):
    j = json.load(open(os.path.join(R, run, 'results.json')))
    return [j['final'][str(i)]['fvu'] for i in range(6)], j


def slope(run):
    rows = list(csv.DictReader(open(os.path.join(R, run, 'curves.csv'))))
    f = {(int(r['step']), int(r['layer'])): float(r['val_fvu']) for r in rows}
    last = max(k[0] for k in f)
    return [(f[(last - 1000, l)] - f[(last, l)]) / f[(last, l)] for l in range(6)]


def eff(c):
    p = c.double() / c.sum(); p = p[p > 0]
    return math.exp(-(p * p.log()).sum().item())


print('=== final held-out FVU (step 4000) ===')
for run in a.compare + a.runs:
    if os.path.exists(os.path.join(R, run, 'results.json')):
        f, j = final_fvu(run)
        s = slope(run)
        print(f'{run:30s} ' + ' '.join(f'{v:.4f}' for v in f) + f' | mean {sum(f)/6:.4f} | last-1K ' +
              ' '.join(f'{100*x:+.2f}' for x in s) + f' (mean {100*sum(s)/6:+.2f}%) | {j["wall_clock_s"]/60:.1f} min')

tok = D.RustBPETokenizer.from_directory(os.path.join(D.get_base_dir(), 'tokenizer'))
teacher, _ = D.load_teacher(D.DEF_TEACHER, tok.get_vocab_size(), dev)
val_idx = D.take_rows(D.tokenizing_distributed_data_loader_bos_bestfit(tok, 48, 512, split='val', device=dev), 32,
                      skip_rows=12)
io = D.teacher_ffn_io(teacher, val_idx, list(range(6)))
base = json.load(open(D.DEF_STUDENT))
out = {}
for run in a.runs:
    ov = json.load(open(os.path.join(R, run, 'manifest.json')))['student_overrides']
    print(f'=== {run} ===')
    rows = []
    for li in range(6):
        h, o = io[li]
        var = o.double().var(0, unbiased=False).mean().item()
        s = D.build_student({**base, **ov}, 384, 6, li, dev)
        s0 = D.build_student({**base, **ov}, 384, 6, li, dev)          # the untrained init (same seeds)
        s.load_state_dict(torch.load(os.path.join(R, run, f'student_L{li}.pt'), map_location=dev)); s.eval()
        lut = s.lut_light
        H = lut.n_heads
        with torch.no_grad():
            yW, yb, A = [], [], []
            fv = {}
            for i in range(0, h.shape[0], 4096):
                z = s.compress(h[i:i + 4096]).view(-1, H, lut.input_dim)
                am, bias = lut._menu_reads(z)
                w_part = lut._apply_menu(am, z)
                yW.append(w_part); yb.append(bias if bias is not None else torch.zeros_like(w_part)); A.append(am)
            yW, yb, A = torch.cat(yW), torch.cat(yb), torch.cat(A)
            N = yW.shape[0]
            dec = s.decompress

            def fvu_of(y):
                pred = dec(y.reshape(N, -1)) if s.has_decompress else y.sum(1)
                return ((pred - o).double().pow(2).mean().item()) / var
            fv['full'], fv['matrix_only'], fv['bias_only'] = fvu_of(yW + yb), fvu_of(yW), fvu_of(yb)
        ratio = (yb.pow(2).mean().sqrt() / yW.pow(2).mean().sqrt()).item()
        L0, L1 = s0.lut_light.menu_logits.detach(), lut.menu_logits.detach()
        tau = lut.menu_tau.detach()
        eff_tok = sum(eff(A[:, hh].abs().sum(0)) for hh in range(H)) / H
        rows.append(dict(ratio=ratio, fvu=fv, argmax_changed=(L0.argmax(-1) != L1.argmax(-1)).float().mean().item(),
                         dlogit_rms=(L1 - L0).pow(2).mean().sqrt().item(), tau=tau.tolist(), eff_tok=eff_tok,
                         bias_rms=lut.menu_bias.detach().pow(2).mean().sqrt().item() if lut.menu_bias is not None else 0))
        r = rows[-1]
        print(f'  L{li}: ||bias part||/||matrix part|| {ratio:.3f} | FVU full {fv["full"]:.4f}, matrix-only '
              f'{fv["matrix_only"]:.4f}, bias-only {fv["bias_only"]:.4f} | argmax≠init {100*r["argmax_changed"]:.1f}% '
              f'| logit rms Δ {r["dlogit_rms"]:.3e} (init std 0.01) | tau {tau.min():.3f}-{tau.max():.3f} | '
              f'eff items (token) {eff_tok:.1f}/{lut.menu_size}')
        del s, s0
    lut_params = None
    s = D.build_student({**base, **ov}, 384, 6, 0, 'cpu')
    lut = s.lut_light
    macs = lut.inference_macs_per_token()
    n_lut = sum(p.numel() for p in lut.parameters())
    store = (lut.menu.numel() + (lut.menu_bias.numel() if lut.menu_bias is not None else 0)) * 4 + lut.n_tables * 256 * 6 / 8
    print(f'  params: LUT {n_lut:,} (student {sum(p.numel() for p in s.parameters()):,}) | inference LUT storage '
          f'{store/1e6:.2f} MB fp32 | hard-read MACs/FFN {(2*384*384 + macs["sparse_hard"])/FFN:.3f}')
    out[run] = rows
json.dump(out, open(os.path.join(R, 'menu_bias_analysis.json'), 'w'), indent=1)
