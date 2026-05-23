"""Magnitude-leakage probe: histogram of |p| in the trained softmax MatmulMHL.

p = d/(T_soft+|d|) is the SOFT SIGN. If exp493/494 only used the discrete
sign pattern (orthant), training would push |p|->1 (saturate). If it leaks
information into magnitudes, |p| sits well below 1. This measures it on real
val data, per module type and overall.
"""
import os, sys, math
import torch
HERE = os.path.dirname(os.path.abspath(__file__))
NANOCHAT_ROOT = os.environ.get('NANOCHAT_ROOT', '/home/starost/nanochat')
if NANOCHAT_ROOT not in sys.path:
    sys.path.insert(0, NANOCHAT_ROOT)
from spiky.lutorch.tiny_multi_head_lut import MatmulMultiHeadLut
from nanochat.common import get_base_dir
from nanochat.tokenizer import RustBPETokenizer
from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit
from hardening_eval import build_model  # same arch builder, gate_mode=softmax

DEVICE = 'cuda'
CKPT = os.path.join(HERE, 'checkpoint.pt')
N_BATCHES = 4

# kind label per module name
KIND = {'qkv_lut': 'qkv', 'v_lut': 'v', 'out_proj': 'out', 'residual_lut': 'residual'}


def main():
    ck = torch.load(CKPT, map_location=DEVICE, weights_only=False)
    cfg = ck['config']
    model = build_model(cfg)
    model.load_state_dict(ck['model_state_dict'], strict=False)
    model.eval()

    # collect |p| per kind via forward-pre-hooks that recompute the soft sign
    buckets = {k: [] for k in ('qkv', 'v', 'out', 'residual')}

    def mk_hook(kind):
        def hook(mod, args):
            x = args[0]
            with torch.no_grad():
                T_soft = mod.log_soft_score_temp.exp()
                d = x[:, mod.soft_anchor_a_long] - x[:, mod.soft_anchor_b_long]
                p = d / (T_soft + d.abs())
                buckets[kind].append(p.abs().flatten().float().cpu())
        return hook

    for layer in model.layers:
        for name, kind in KIND.items():
            getattr(layer, name).register_forward_pre_hook(mk_hook(kind))

    tok = RustBPETokenizer.from_directory(os.path.join(get_base_dir(), 'tokenizer'))
    loader = tokenizing_distributed_data_loader_bos_bestfit(
        tok, cfg['device_batch_size'], cfg['context_size'], split='val', device=DEVICE)
    with torch.no_grad():
        for _ in range(N_BATCHES):
            x, _y = next(loader)
            model(x.clone())

    def report(name, vals):
        v = torch.cat(vals)
        vs = v if v.numel() <= 2_000_000 else v[torch.randperm(v.numel())[:2_000_000]]
        qs = torch.quantile(vs, torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9]))
        frac_soft = (v < 0.5).float().mean().item()
        frac_sat = (v > 0.9).float().mean().item()
        print(f"{name:9s} n={v.numel():>11,d}  mean|p|={v.mean():.3f}  "
              f"p10={qs[0]:.3f} p25={qs[1]:.3f} p50={qs[2]:.3f} p75={qs[3]:.3f} p90={qs[4]:.3f}  "
              f"|frac<0.5={frac_soft:.2f}  frac>0.9={frac_sat:.2f}")
        return v

    print(f"T_soft mean = {sum(l.qkv_lut.log_soft_score_temp.exp().item() for l in model.layers)/len(model.layers):.3f} (qkv); "
          f"checkpoint = exp494 (== exp493 model)\n")
    allv = []
    for k in ('qkv', 'v', 'out', 'residual'):
        allv.append(report(k, buckets[k]))
    overall = report('ALL', allv)

    # ascii histogram of |p| overall, 20 bins in [0,1]
    print("\n|p| histogram (overall):")
    hist = torch.histc(overall, bins=20, min=0.0, max=1.0)
    hist = hist / hist.sum()
    for i, h in enumerate(hist.tolist()):
        lo = i / 20
        bar = '#' * int(round(h * 200))
        print(f"  {lo:.2f}-{lo+0.05:.2f} | {bar} {h*100:.1f}%")
    print("\nInterpretation: mass below ~0.9 => magnitude is in use (analog channel);")
    print("a pile at >0.9 would mean the model is effectively sign-only (hardenable).")


if __name__ == '__main__':
    main()
