"""Measure the forward-confidence gate on real activations at anchor sizing."""
import copy, json, math, os, sys, torch
FR = os.path.expanduser('~/projects/spiky/experiments/ffn_replacement')
RC = os.path.join(FR, 'runs_corrected')
sys.path.insert(0, os.path.join(FR, 'tools'))
sys.path.insert(0, os.path.expanduser('~/projects/nanochat'))
sys.path.insert(0, os.path.expanduser('~/projects/spiky/src'))
from nanochat.common import get_base_dir
from nanochat.tokenizer import RustBPETokenizer
from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit
from model_build import build_model
from spiky.lutorch.fast_multi_head_lut import _confidence_score

DEV = 'cpu'   # arm A owns the GPU; this must not disturb it
base = json.load(open(os.path.join(RC, 'sweep_s05_dout48_H4_tph256_c256_din32', 'config.json')))
tok = RustBPETokenizer.from_directory(os.path.join(get_base_dir(), 'tokenizer'))
vocab = tok.get_vocab_size()

# one real batch of real text
ld = tokenizing_distributed_data_loader_bos_bestfit(tok, 2, 512, split='val', device=DEV)
x_ids, _ = next(iter(ld))
x_ids = x_ids.clone()


def capture(cfg, ckpt=None, label=''):
    """Return per-block (margins |d|, score_bounded, score_margin) on real activations."""
    torch.manual_seed(cfg['random_seed'])
    m = build_model(cfg, vocab, device=DEV)
    if ckpt:
        m.load_state_dict(torch.load(ckpt, map_location=DEV), strict=False)
    m.eval()
    out = []
    for blk in m.blocks:
        ffn = blk.ffn
        rec = {}

        def hook(mod, inp, _out, rec=rec, ffn=ffn):
            z = ffn.compress(inp[0]).view(inp[0].shape[0], ffn.n_heads, ffn.inner_in_dim)
            lut = ffn.lut_batched
            # block-diagonal anchors: [H, tph, NAP] indices within each head's slice
            a, b = lut.soft_anchor_a_long, lut.soft_anchor_b_long
            zz = z if a.dim() == 3 else z.reshape(z.shape[0], -1)
            if a.dim() == 3:
                H, T, NAP = a.shape
                ia = a.reshape(1, H, T * NAP).expand(zz.shape[0], H, T * NAP)
                ib = b.reshape(1, H, T * NAP).expand(zz.shape[0], H, T * NAP)
                d = (torch.gather(zz, 2, ia) - torch.gather(zz, 2, ib)).view(-1, H, T, NAP)
            else:
                d = zz[:, a] - zz[:, b]
            rec['m'] = d.abs().detach().float().flatten()
            rec['sb'] = _confidence_score(d, 'bounded').detach().float().flatten()
            rec['sm'] = _confidence_score(d, 'margin').detach().float().flatten()
        h = ffn.register_forward_hook(hook)
        with torch.no_grad():
            m(x_ids)
        h.remove()
        out.append(rec)
    dn = {n: p.norm().item() for n, p in m.named_parameters() if 'decompress.weight' in n}
    return out, dn, m


def stats(t):
    t = t[torch.randperm(t.numel())[:200000]] if t.numel() > 200000 else t
    q = torch.quantile(t, torch.tensor([0.0, .25, .5, .75, 1.0]))
    return (f"min {q[0]:.6g}  p25 {q[1]:.6g}  median {q[2]:.6g}  p75 {q[3]:.6g}  "
            f"max {q[4]:.6g}  mean {t.mean():.6g}")


print('=' * 78)
print('1+5. SCORE AND MARGIN DISTRIBUTION AT INIT, anchor sizing, REAL activations')
print('=' * 78)
recs, dn0, _ = capture(base, label='init')
allm = torch.cat([r['m'] for r in recs])
allb = torch.cat([r['sb'] for r in recs])
allg = torch.cat([r['sm'] for r in recs])
print(f"   |d| (margins, nap=8)   {stats(allm)}")
print(f"   score BOUNDED          {stats(allb)}")
print(f"   score MARGIN           {stats(allg)}")
print(f"\n   0.5^8 = {0.5**8:.6g}  (the value if every margin were exactly 0)")
print(f"   bounded mean {allb.mean():.6g}  ->  attenuation factor "
      f"{1/allb.mean():.1f}x on every gathered row")
print(f"   margin  mean {allg.mean():.6g}  ->  factor {allg.mean():.3f}x "
      f"({'amplifies' if allg.mean() > 1 else 'attenuates'})")
print('\n   per block (bounded mean / margin mean / |d| median):')
for i, r in enumerate(recs):
    print(f'      block {i}: {r["sb"].mean():.6g}   {r["sm"].mean():.6g}   '
          f'{r["m"].median():.6g}')

print('\n' + '=' * 78)
print('2. decompress.weight NORM: init vs the TRAINED baseline checkpoint')
print('=' * 78)
ck = os.path.join(RC, 'sweep_s05_dout48_H4_tph256_c256_din32', 'checkpoint.pt')
print('   at init (decompress is zero-initialised by design):')
for k, v in dn0.items():
    print(f'      {k:<34} {v:.6f}')
if os.path.exists(ck):
    _, dnT, _ = capture(base, ckpt=ck, label='trained')
    print('   trained baseline (S5, 4000 steps):')
    for k, v in dnT.items():
        print(f'      {k:<34} {v:.6f}')
