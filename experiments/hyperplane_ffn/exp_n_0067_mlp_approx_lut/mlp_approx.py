"""exp_n_0067 — MLP-approximates-LUT ablation.

How well can a standard 2-layer widening GELU MLP approximate (a) a whole trained
CompressionMHL block (384->384) and (b) a single FastMHL head (48->48), as a
function of MLP hidden width? Probes the function-class gap: if even wide MLPs
can't fit the LUT, the discrete hyperplane routing is a fundamentally different
(spiky/non-smooth) function than a smooth GELU MLP represents.

Targets = the TRAINED LUTs from the end-to-end LUT baseline exp_n_0052 (val_bpb
1.2285517): load its checkpoint, take block 0 (easiest) and block 5 (hardest),
and a single head of block 0. Inputs = real block-FFN-input activations streamed
through the frozen exp_n_0052 model; targets = the frozen LUT's outputs. Fit
2-layer GELU MLPs Linear(in,w*dim)->GELU->Linear(w*dim,out) for w in {4,8,16,32,64}
with MSE; report train/val MSE, R2, MLP params (marking where they cross the LUT's
own param count), and MSE-vs-width / MSE-vs-params curves. No shared-module edits.
"""
import sys, os, json, math, time
try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass
import torch, torch.nn as nn, torch.nn.functional as F
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

NANOCHAT_ROOT = os.environ.get('NANOCHAT_ROOT', os.path.expanduser('~/projects/nanochat'))
if NANOCHAT_ROOT not in sys.path:
    sys.path.insert(0, NANOCHAT_ROOT)
from nanochat.common import get_base_dir
from nanochat.tokenizer import RustBPETokenizer
from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit
from spiky.lutorch.compression_mhl import CompressionMultiHeadLUT

EXP_DIR = os.path.dirname(os.path.abspath(__file__)); HFF = os.path.dirname(EXP_DIR)
cfg = json.load(open(os.path.join(EXP_DIR, 'config.json')))
DEV = 'cuda'; torch.manual_seed(cfg['random_seed'])
DEPTH, N_EMBD, N_HEAD, SEQ_LEN, DBS = cfg['depth'], cfg['n_embd'], cfg['n_head'], cfg['seq_len'], cfg['device_batch_size']
WIDTHS = cfg['mlp_widths']; N_COLLECT = int(cfg['n_collect']); VAL_FRAC = cfg['val_frac']
FIT_STEPS, FIT_LR, FIT_BATCH = cfg['fit_steps'], cfg['fit_lr'], cfg['fit_batch']
STUDY = cfg['study_blocks']
if os.environ.get('SMOKE'):
    WIDTHS = [4, 16]; N_COLLECT = 24576; FIT_STEPS = 150
    print('*** SMOKE ***')

BASE = get_base_dir(); tok = RustBPETokenizer.from_directory(os.path.join(BASE, 'tokenizer'))
VOCAB = tok.get_vocab_size()
loader = tokenizing_distributed_data_loader_bos_bestfit(tok, DBS, SEQ_LEN, split='train', device=DEV)

# ---- model (compression-FFN, matches exp_n_0052 / the shared exp043+ trainer) ----
class Rope(nn.Module):
    def __init__(s, hd, msl, base=10000.0):
        super().__init__()
        inv = 1.0/(base**(torch.arange(0,hd,2,dtype=torch.float32)/hd)); t=torch.arange(msl,dtype=torch.float32)
        e=torch.cat([torch.outer(t,inv)]*2,-1); s.register_buffer('cos',e.cos(),persistent=False); s.register_buffer('sin',e.sin(),persistent=False)
def rh(x): a,b=x.chunk(2,-1); return torch.cat([-b,a],-1)
def rope(q,k,c,s): c=c[None,None]; s=s[None,None]; return q*c+rh(q)*s, k*c+rh(k)*s
class Attn(nn.Module):
    def __init__(s,d,h): super().__init__(); s.n_head=h; s.qkv=nn.Linear(d,3*d,bias=False); s.proj=nn.Linear(d,d,bias=False)
    def forward(s,x,c,sn):
        B,T,C=x.size(); q,k,v=s.qkv(x).split(C,2)
        q=q.view(B,T,s.n_head,C//s.n_head).transpose(1,2); k=k.view(B,T,s.n_head,C//s.n_head).transpose(1,2); v=v.view(B,T,s.n_head,C//s.n_head).transpose(1,2)
        q,k=rope(q,k,c[:T],sn[:T]); y=F.scaled_dot_product_attention(q,k,v,is_causal=True)
        return s.proj(y.transpose(1,2).contiguous().view(B,T,C))
def mk_cmhl(seed):
    return CompressionMultiHeadLUT(input_dim=N_EMBD, output_dim=N_EMBD, inner_in_dim=cfg['lut_inner_in_dim'],
        inner_out_dim=cfg['lut_inner_out_dim'], nap=cfg['lut_n_anchor_pairs'], tph=cfg['lut_tables_per_head'],
        n_heads=cfg['lut_n_heads'], joint_head_compression=cfg['lut_joint_head_compression'],
        batched_multi_head_input=cfg['lut_batched_multi_head_input'], forward_mode=cfg['lut_forward_mode'],
        use_bf16=cfg['lut_use_bf16'], initial_weights_noise=cfg['lut_init_weights_noise'],
        learnable_temps=cfg['lut_learnable_temps'], random_seed=cfg['lut_base_seed'] + seed)
class Block(nn.Module):
    def __init__(s,d,h,i):
        super().__init__(); s.ln1=nn.LayerNorm(d); s.attn=Attn(d,h); s.ln2=nn.LayerNorm(d); s.ffn=mk_cmhl(i)
    def forward(s,x,c,sn):
        x=x+s.attn(s.ln1(x),c,sn); h=s.ln2(x); B,T,C=h.shape
        return x+s.ffn(h.reshape(B*T,C)).reshape(B,T,C).to(h.dtype)
class GPT(nn.Module):
    def __init__(s):
        super().__init__(); s.tok_emb=nn.Embedding(VOCAB,N_EMBD); s.rope=Rope(N_EMBD//N_HEAD,SEQ_LEN)
        s.blocks=nn.ModuleList([Block(N_EMBD,N_HEAD,i) for i in range(DEPTH)]); s.ln_f=nn.LayerNorm(N_EMBD); s.head=nn.Linear(N_EMBD,VOCAB,bias=False)
        if cfg['tie_unembedder']: s.head.weight=s.tok_emb.weight
    def forward(s,idx):
        x=s.tok_emb(idx)
        for b in s.blocks: x=b(x,s.rope.cos,s.rope.sin)
        return x

model = GPT().to(DEV)
sd = torch.load(os.path.join(HFF, cfg['target_ckpt']), map_location=DEV)
missing, unexpected = model.load_state_dict(sd, strict=False)
print(f'loaded exp_n_0052 ckpt: missing={len(missing)} unexpected={len(unexpected)}')
for p in model.parameters(): p.requires_grad_(False)
model.eval()
lut_params_block = sum(p.numel() for p in model.blocks[0].ffn.parameters())
lut_params_head = cfg['lut_tables_per_head'] * (1 << cfg['lut_n_anchor_pairs']) * cfg['lut_inner_out_dim']
print(f'LUT params: whole block ~{lut_params_block:,} | single head ~{lut_params_head:,}')

# ---- collect real-activation (input, target) pairs ----
caps = {}
def blk_hook(bi):
    def h(m, inp, out): caps.setdefault(f'block{bi}', []).append((inp[0].reshape(-1, N_EMBD).float(), out.reshape(-1, N_EMBD).float()))
    return h
def head_hook(m, inp, out):
    z = inp[0]                                  # [N,H,d] fed to the batched FastMHL
    z = z.reshape(z.shape[0], cfg['lut_n_heads'], cfg['lut_inner_in_dim']) if z.dim()==2 else z
    caps.setdefault('head0', []).append((z[:, 0, :].float(), out[:, 0, :].float()))
hs = [model.blocks[b].ffn.register_forward_hook(blk_hook(b)) for b in STUDY]
hs.append(model.blocks[STUDY[0]].ffn.lut_batched.register_forward_hook(head_hook))
collected = 0
with torch.no_grad():
    while collected < N_COLLECT:
        x, _ = next(loader); model(x); collected += x.numel()
for h in hs: h.remove()
def stack(key):
    xs = torch.cat([a for a, _ in caps[key]], 0)[:N_COLLECT]
    ys = torch.cat([b for _, b in caps[key]], 0)[:N_COLLECT]
    return xs, ys
targets = {}
for b in STUDY: targets[f'block{b}'] = stack(f'block{b}') + (N_EMBD, N_EMBD, lut_params_block)
targets['head0'] = stack('head0') + (cfg['lut_inner_in_dim'], cfg['lut_inner_out_dim'], lut_params_head)
print('collected:', {k: tuple(v[0].shape) for k, v in targets.items()})

def fit(X, Y, din, dout, width):
    n = X.shape[0]; nval = int(n * VAL_FRAC); Xtr, Ytr, Xv, Yv = X[nval:], Y[nval:], X[:nval], Y[:nval]
    hid = width * din
    mlp = nn.Sequential(nn.Linear(din, hid), nn.GELU(), nn.Linear(hid, dout)).to(DEV)
    opt = torch.optim.Adam(mlp.parameters(), lr=FIT_LR)
    ntr = Xtr.shape[0]
    g = torch.Generator(device=DEV).manual_seed(0)
    for st in range(FIT_STEPS):
        idx = torch.randint(0, ntr, (min(FIT_BATCH, ntr),), device=DEV, generator=g)
        loss = F.mse_loss(mlp(Xtr[idx]), Ytr[idx]); opt.zero_grad(); loss.backward(); opt.step()
    mlp.eval()
    with torch.no_grad():
        tr = F.mse_loss(mlp(Xtr[:8192]), Ytr[:8192]).item()
        vp = mlp(Xv); vmse = F.mse_loss(vp, Yv).item()
        var = Yv.var(0).mean().item(); r2 = 1 - vmse / (var + 1e-12)
    return {'width': width, 'hid': hid, 'params': sum(p.numel() for p in mlp.parameters()),
            'train_mse': round(tr, 6), 'val_mse': round(vmse, 6), 'r2': round(r2, 5), 'target_var': round(var, 6)}

results = {}
for name, (X, Y, din, dout, lp) in targets.items():
    results[name] = {'lut_params': lp, 'sweep': []}
    print(f'--- fitting MLPs to {name} (in={din} out={dout}, LUT~{lp:,} params) ---')
    for w in WIDTHS:
        r = fit(X, Y, din, dout, w); results[name]['sweep'].append(r)
        cross = 'MLP>LUT' if r['params'] > lp else 'MLP<LUT'
        print(f'   w={w:2d}x hid={r["hid"]:5d} params={r["params"]:>9,} ({cross}) | val_mse={r["val_mse"]:.5f} R2={r["r2"]:.4f}')

# ---- plots: MSE vs width, MSE vs params ----
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
for name in results:
    sw = results[name]['sweep']
    ax1.plot([r['width'] for r in sw], [r['val_mse'] for r in sw], 'o-', label=f'{name} (LUT {results[name]["lut_params"]/1e6:.2f}M)')
    ax2.plot([r['params'] for r in sw], [r['val_mse'] for r in sw], 'o-', label=name)
    ax2.axvline(results[name]['lut_params'], ls=':', alpha=0.4)
ax1.set(xlabel='MLP width (x input dim)', ylabel='val MSE (approx LUT)', title='MLP approximation of trained LUT vs width'); ax1.set_yscale('log'); ax1.set_xscale('log', base=2); ax1.grid(True, alpha=0.3, which='both'); ax1.legend(fontsize=8)
ax2.set(xlabel='MLP params', ylabel='val MSE', title='vs MLP params (dotted = LUT param count)'); ax2.set_yscale('log'); ax2.set_xscale('log'); ax2.grid(True, alpha=0.3, which='both'); ax2.legend(fontsize=8)
plt.tight_layout(); plt.savefig(os.path.join(EXP_DIR, 'mlp_approx_curves.png'), dpi=120); plt.close()

summary = {'exp_name': cfg['exp_name'], 'target': 'exp_n_0052 trained LUTs (val_bpb 1.2285517)',
           'n_collect': N_COLLECT, 'widths': WIDTHS, 'fit_steps': FIT_STEPS, 'results': results}
json.dump(summary, open(os.path.join(EXP_DIR, 'summary.json'), 'w'), indent=2)
print('\n=== DONE ==='); print(json.dumps({k: {'lut_params': v['lut_params'],
      'best_val_mse': min(r['val_mse'] for r in v['sweep']), 'best_r2': max(r['r2'] for r in v['sweep'])} for k, v in results.items()}, indent=2))
