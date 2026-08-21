"""exp_n_0071 — DEPTH sweep of MLP-approximates-LUT (does composition break the floor?).

exp_n_0067 showed a 2-layer (single-hidden) GELU MLP's approximation of the trained
LUT FLOORS even at 64x width (block0 R2~0.93, block5 ~0.88, single head ~0.69). But
the LUT's hard routing is piecewise-CONSTANT (hyperplane-bounded cells) and a
CONJUNCTION of sign tests — compositional/logical, which shallow nets need
exponential width for but DEPTH represents efficiently. So sweep DEPTH (with
pre-norm residual + LayerNorm so deep nets train cleanly) at fixed width and ask:
does depth drive the approximation error toward 0 (the LUT IS representable — it
just needed composition), or does it STILL floor (genuinely outside the smooth-MLP
class)? Watch the single FastMHL head (purest hard routing): if depth lifts R2 0.69
-> ~1.0, composition was the missing ingredient.

Same targets/datasets as exp_n_0067 (exp_n_0052's trained LUTs: block 0, block 5,
single head of block 0). No shared-module edits.
"""
import sys, os, json, math
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
DEPTHS = cfg['depths']; FIXED_W = cfg['fixed_width']; CORNER = (cfg['corner_depth'], cfg['corner_width'])
N_COLLECT = int(cfg['n_collect']); VAL_FRAC = cfg['val_frac']
FIT_STEPS, FIT_LR, FIT_WARM, FIT_BATCH = cfg['fit_steps'], cfg['fit_lr'], cfg['fit_warmup_frac'], cfg['fit_batch']
STUDY = cfg['study_blocks']
if os.environ.get('SMOKE'):
    DEPTHS = [2, 6]; CORNER = None; N_COLLECT = 24576; FIT_STEPS = 200; print('*** SMOKE ***')

BASE = get_base_dir(); tok = RustBPETokenizer.from_directory(os.path.join(BASE, 'tokenizer'))
VOCAB = tok.get_vocab_size()
loader = tokenizing_distributed_data_loader_bos_bestfit(tok, DBS, SEQ_LEN, split='train', device=DEV)

# ---- frozen exp_n_0052 model (compression FFN) ----
class Rope(nn.Module):
    def __init__(s, hd, msl, base=10000.0):
        super().__init__(); inv=1.0/(base**(torch.arange(0,hd,2,dtype=torch.float32)/hd)); t=torch.arange(msl,dtype=torch.float32)
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
    def __init__(s,d,h,i): super().__init__(); s.ln1=nn.LayerNorm(d); s.attn=Attn(d,h); s.ln2=nn.LayerNorm(d); s.ffn=mk_cmhl(i)
    def forward(s,x,c,sn): x=x+s.attn(s.ln1(x),c,sn); h=s.ln2(x); B,T,C=h.shape; return x+s.ffn(h.reshape(B*T,C)).reshape(B,T,C).to(h.dtype)
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
mi, un = model.load_state_dict(torch.load(os.path.join(HFF, cfg['target_ckpt']), map_location=DEV), strict=False)
print(f'loaded exp_n_0052 ckpt: missing={len(mi)} unexpected={len(un)}')
for p in model.parameters(): p.requires_grad_(False)
model.eval()

# ---- collect targets (same as exp_n_0067) ----
caps = {}
def blk_hook(bi):
    def h(m, inp, out): caps.setdefault(f'block{bi}', []).append((inp[0].reshape(-1, N_EMBD).float(), out.reshape(-1, N_EMBD).float()))
    return h
def head_hook(m, inp, out):
    z = inp[0]; z = z.reshape(z.shape[0], cfg['lut_n_heads'], cfg['lut_inner_in_dim']) if z.dim()==2 else z
    caps.setdefault('head0', []).append((z[:, 0, :].float(), out[:, 0, :].float()))
hs = [model.blocks[b].ffn.register_forward_hook(blk_hook(b)) for b in STUDY]
hs.append(model.blocks[STUDY[0]].ffn.lut_batched.register_forward_hook(head_hook))
c = 0
with torch.no_grad():
    while c < N_COLLECT:
        x, _ = next(loader); model(x); c += x.numel()
for h in hs: h.remove()
def stk(k): return torch.cat([a for a,_ in caps[k]],0)[:N_COLLECT], torch.cat([b for _,b in caps[k]],0)[:N_COLLECT]
targets = {}
for b in STUDY: targets[f'block{b}'] = stk(f'block{b}') + (N_EMBD, N_EMBD)
targets['head0'] = stk('head0') + (cfg['lut_inner_in_dim'], cfg['lut_inner_out_dim'])
print('collected:', {k: tuple(v[0].shape) for k, v in targets.items()})

# ---- deep pre-norm residual MLP (depth = number of Linear layers on the main path) ----
class DeepResMLP(nn.Module):
    def __init__(self, din, dout, H, depth):
        super().__init__(); self.depth = depth
        if depth == 2:
            self.net = nn.Sequential(nn.Linear(din, H), nn.GELU(), nn.Linear(H, dout))
        else:
            self.inp = nn.Linear(din, H)
            self.blocks = nn.ModuleList([nn.ModuleList([nn.LayerNorm(H), nn.Linear(H, H)]) for _ in range(depth - 2)])
            self.oln = nn.LayerNorm(H); self.out = nn.Linear(H, dout)
    def forward(self, x):
        if self.depth == 2: return self.net(x)
        x = self.inp(x)
        for ln, lin in self.blocks: x = x + lin(F.gelu(ln(x)))   # pre-norm residual
        return self.out(F.gelu(self.oln(x)))

def lr_at(step):
    w = int(FIT_WARM * FIT_STEPS)
    if step < w: return step / max(w, 1)
    p = (step - w) / max(FIT_STEPS - w, 1); return 0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * p))

def fit(X, Y, din, dout, depth, width):
    n = X.shape[0]; nv = int(n * VAL_FRAC); Xtr, Ytr, Xv, Yv = X[nv:], Y[nv:], X[:nv], Y[:nv]
    H = width * din; net = DeepResMLP(din, dout, H, depth).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=FIT_LR, weight_decay=0.0, betas=(0.9, 0.95))
    ntr = Xtr.shape[0]; g = torch.Generator(device=DEV).manual_seed(0)
    for st in range(FIT_STEPS):
        for gp in opt.param_groups: gp['lr'] = FIT_LR * lr_at(st)
        idx = torch.randint(0, ntr, (min(FIT_BATCH, ntr),), device=DEV, generator=g)
        loss = F.mse_loss(net(Xtr[idx]), Ytr[idx]); opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
    net.eval()
    with torch.no_grad():
        tr = F.mse_loss(net(Xtr[:8192]), Ytr[:8192]).item()
        vmse = F.mse_loss(net(Xv), Yv).item(); var = Yv.var(0).mean().item(); r2 = 1 - vmse / (var + 1e-12)
    return {'depth': depth, 'width': width, 'H': H, 'params': sum(p.numel() for p in net.parameters()),
            'train_mse': round(tr, 6), 'val_mse': round(vmse, 6), 'r2': round(r2, 5)}

results = {}
runs = [(d, FIXED_W) for d in DEPTHS] + ([CORNER] if CORNER else [])
for name, (X, Y, din, dout) in targets.items():
    results[name] = []
    print(f'--- {name} (in={din} out={dout}) ---')
    for depth, width in runs:
        r = fit(X, Y, din, dout, depth, width); results[name].append(r)
        print(f'   depth={depth} width={width}x H={r["H"]:5d} params={r["params"]:>9,} | val_mse={r["val_mse"]:.5f} R2={r["r2"]:.4f}')

# ---- plots: R2 vs depth (fixed width), and val_mse vs depth ----
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
for name in results:
    dd = [r for r in results[name] if r['width'] == FIXED_W]
    ax1.plot([r['depth'] for r in dd], [r['r2'] for r in dd], 'o-', label=f'{name}')
    ax2.plot([r['depth'] for r in dd], [r['val_mse'] for r in dd], 'o-', label=f'{name}')
    if CORNER:
        cr = [r for r in results[name] if (r['depth'], r['width']) == CORNER][0]
        ax1.plot(cr['depth'], cr['r2'], '*', ms=14, color='k')
ax1.set(xlabel=f'MLP depth (Linear layers, width {FIXED_W}x)', ylabel='val R2 (approx LUT)', title='depth vs R2 (★ = corner d6xw16)'); ax1.grid(True, alpha=0.3); ax1.legend(fontsize=9); ax1.axhline(1.0, ls='--', c='gray', alpha=0.5)
ax2.set(xlabel='MLP depth', ylabel='val MSE', title='depth vs val MSE'); ax2.set_yscale('log'); ax2.grid(True, alpha=0.3, which='both'); ax2.legend(fontsize=9)
plt.tight_layout(); plt.savefig(os.path.join(EXP_DIR, 'depth_curves.png'), dpi=120); plt.close()

summary = {'exp_name': cfg['exp_name'], 'target': 'exp_n_0052 trained LUTs', 'depths': DEPTHS, 'fixed_width': FIXED_W,
           'corner': CORNER, 'fit_steps': FIT_STEPS, 'n_collect': N_COLLECT, 'results': results,
           'best_r2': {k: max(r['r2'] for r in v) for k, v in results.items()}}
json.dump(summary, open(os.path.join(EXP_DIR, 'summary.json'), 'w'), indent=2)
print('\n=== DONE ==='); print(json.dumps(summary['best_r2'], indent=2))
