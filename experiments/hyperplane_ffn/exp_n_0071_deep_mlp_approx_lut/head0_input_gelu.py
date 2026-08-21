"""exp_n_0071 (head-0, input-GELU, single config, 40k) — does an extra input
nonlinearity + longer training beat the ~0.79 depth-3 floor for head 0?

Single config: depth 3, width 32 (H=1536), approximating block-0 head 0 of
exp_n_0052 LUTs. GELU after BOTH linear layers (the one change vs the 0.7884
baseline):
    h   = GELU(Linear(48->1536)(x))
    h   = h + Linear(1536->1536)(GELU(LayerNorm(h)))
    out = Linear(1536->48)(LayerNorm(h))          # NO GELU before output
Same setup as the head-0 sweep (input z[:,0,:], target out[:,0,:], ~320k tokens,
80/20 val, per-element MSE, val R2=1-vmse/var, AdamW + cosine + 5% warmup +
grad-clip). Train 40k steps; log val R2 every 2k steps (trajectory is the point).
No shared-module edits.
"""
import sys, os, json, math
try: sys.stdout.reconfigure(line_buffering=True)
except Exception: pass
import torch, torch.nn as nn, torch.nn.functional as F
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.expanduser('~/projects/nanochat'))
from nanochat.common import get_base_dir
from nanochat.tokenizer import RustBPETokenizer
from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit
from spiky.lutorch.compression_mhl import CompressionMultiHeadLUT

EXP_DIR = os.path.dirname(os.path.abspath(__file__)); HFF = os.path.dirname(EXP_DIR)
cfg = json.load(open(os.path.join(EXP_DIR, 'config.json')))
DEV = 'cuda'; torch.manual_seed(cfg['random_seed'])
N_EMBD, N_HEAD, SEQ, DBS, DEPTH, VOCAB = cfg['n_embd'], cfg['n_head'], cfg['seq_len'], cfg['device_batch_size'], cfg['depth'], cfg['tokenizer_vocab_size']
DIN = cfg['lut_inner_in_dim']; DOUT = cfg['lut_inner_out_dim']
N_COLLECT = int(cfg['n_collect']); VAL_FRAC = cfg['val_frac']
FIT_LR, FIT_WARM, FIT_BATCH = cfg['fit_lr'], cfg['fit_warmup_frac'], cfg['fit_batch']
CONFIG_DEPTH, CONFIG_WIDTH, FIT_STEPS, EVAL_EVERY = 3, 32, 40000, 2000
if os.environ.get('SMOKE'):
    N_COLLECT = 24576; FIT_STEPS = 400; EVAL_EVERY = 200; print('*** SMOKE ***')

base = get_base_dir(); tok = RustBPETokenizer.from_directory(os.path.join(base, 'tokenizer')); assert tok.get_vocab_size() == VOCAB
loader = tokenizing_distributed_data_loader_bos_bestfit(tok, DBS, SEQ, split='train', device=DEV)

class Rope(nn.Module):
    def __init__(s, hd, msl, base=10000.0):
        super().__init__(); inv=1.0/(base**(torch.arange(0,hd,2,dtype=torch.float32)/hd)); t=torch.arange(msl,dtype=torch.float32)
        e=torch.cat([torch.outer(t,inv)]*2,-1); s.register_buffer('cos',e.cos(),persistent=False); s.register_buffer('sin',e.sin(),persistent=False)
def rh(x): a,b=x.chunk(2,-1); return torch.cat([-b,a],-1)
def rope(q,k,c,s): c=c[None,None]; s=s[None,None]; return q*c+rh(q)*s,k*c+rh(k)*s
class Attn(nn.Module):
    def __init__(s,d,h): super().__init__(); s.n_head=h; s.qkv=nn.Linear(d,3*d,bias=False); s.proj=nn.Linear(d,d,bias=False)
    def forward(s,x,c,sn):
        B,T,C=x.size(); q,k,v=s.qkv(x).split(C,2)
        q=q.view(B,T,s.n_head,C//s.n_head).transpose(1,2); k=k.view(B,T,s.n_head,C//s.n_head).transpose(1,2); v=v.view(B,T,s.n_head,C//s.n_head).transpose(1,2)
        q,k=rope(q,k,c[:T],sn[:T]); y=F.scaled_dot_product_attention(q,k,v,is_causal=True)
        return s.proj(y.transpose(1,2).contiguous().view(B,T,C))
def mk_cmhl(seed):
    return CompressionMultiHeadLUT(input_dim=N_EMBD,output_dim=N_EMBD,inner_in_dim=DIN,inner_out_dim=DOUT,nap=cfg['lut_n_anchor_pairs'],
        tph=cfg['lut_tables_per_head'],n_heads=cfg['lut_n_heads'],joint_head_compression=cfg['lut_joint_head_compression'],
        batched_multi_head_input=cfg['lut_batched_multi_head_input'],forward_mode=cfg['lut_forward_mode'],use_bf16=cfg['lut_use_bf16'],
        initial_weights_noise=cfg['lut_init_weights_noise'],learnable_temps=cfg['lut_learnable_temps'],random_seed=cfg['lut_base_seed']+seed)
class Block(nn.Module):
    def __init__(s,d,h,i): super().__init__(); s.ln1=nn.LayerNorm(d); s.attn=Attn(d,h); s.ln2=nn.LayerNorm(d); s.ffn=mk_cmhl(i)
    def forward(s,x,c,sn): x=x+s.attn(s.ln1(x),c,sn); h=s.ln2(x); B,T,C=h.shape; return x+s.ffn(h.reshape(B*T,C)).reshape(B,T,C).to(h.dtype)
class GPT(nn.Module):
    def __init__(s):
        super().__init__(); s.tok_emb=nn.Embedding(VOCAB,N_EMBD); s.rope=Rope(N_EMBD//N_HEAD,SEQ)
        s.blocks=nn.ModuleList([Block(N_EMBD,N_HEAD,i) for i in range(DEPTH)]); s.ln_f=nn.LayerNorm(N_EMBD); s.head=nn.Linear(N_EMBD,VOCAB,bias=False)
        if cfg['tie_unembedder']: s.head.weight=s.tok_emb.weight
    def forward(s,idx):
        x=s.tok_emb(idx)
        for b in s.blocks: x=b(x,s.rope.cos,s.rope.sin)
        return x

model=GPT().to(DEV)
mi,un=model.load_state_dict(torch.load(os.path.join(HFF,cfg['target_ckpt']),map_location=DEV),strict=False)
print(f'loaded exp_n_0052 ckpt: missing={len(mi)} unexpected={len(un)}')
for p in model.parameters(): p.requires_grad_(False)
model.eval()
cap=[]
def hh(m,inp,out):
    z=inp[0]; z=z.reshape(z.shape[0],cfg['lut_n_heads'],DIN) if z.dim()==2 else z
    cap.append((z[:,0,:].float(), out[:,0,:].float()))
h=model.blocks[0].ffn.lut_batched.register_forward_hook(hh)
n=0
with torch.no_grad():
    while n<N_COLLECT:
        x,_=next(loader); model(x); n+=x.numel()
h.remove()
X=torch.cat([a for a,_ in cap],0)[:N_COLLECT]; Y=torch.cat([b for _,b in cap],0)[:N_COLLECT]
print(f'head0 collected: X={tuple(X.shape)} Y={tuple(Y.shape)}')

class DeepResMLP(nn.Module):   # depth 3: input Linear + 1 residual block + output; GELU after BOTH linears, none before out
    def __init__(self, din, dout, H, depth):
        super().__init__(); self.inp=nn.Linear(din,H)
        self.blocks=nn.ModuleList([nn.ModuleList([nn.LayerNorm(H),nn.Linear(H,H)]) for _ in range(depth-2)])
        self.oln=nn.LayerNorm(H); self.out=nn.Linear(H,dout)
    def forward(self,x):
        h=F.gelu(self.inp(x))                          # GELU after input projection
        for ln,lin in self.blocks: h=h+lin(F.gelu(ln(h)))   # pre-norm residual
        return self.out(self.oln(h))                   # output projection, NO GELU
def lr_at(st):
    w=int(FIT_WARM*FIT_STEPS)
    if st<w: return st/max(w,1)
    p=(st-w)/max(FIT_STEPS-w,1); return 0.05+0.95*0.5*(1+math.cos(math.pi*p))

nv=int(X.shape[0]*VAL_FRAC); Xtr,Ytr,Xv,Yv=X[nv:],Y[nv:],X[:nv],Y[:nv]
var=Yv.var(0).mean().item()
H=CONFIG_WIDTH*DIN; net=DeepResMLP(DIN,DOUT,H,CONFIG_DEPTH).to(DEV)
opt=torch.optim.AdamW(net.parameters(),lr=FIT_LR,weight_decay=0.0,betas=(0.9,0.95))
ntr=Xtr.shape[0]; g=torch.Generator(device=DEV).manual_seed(0)
def val_r2():
    net.eval()
    with torch.no_grad(): vmse=F.mse_loss(net(Xv),Yv).item()
    net.train(); return vmse, 1-vmse/(var+1e-12)
traj=[]
print(f'config: depth={CONFIG_DEPTH} width={CONFIG_WIDTH}x H={H} params={sum(p.numel() for p in net.parameters()):,} | {FIT_STEPS} steps, eval every {EVAL_EVERY}')
net.train()
for st in range(1, FIT_STEPS+1):
    for gp in opt.param_groups: gp['lr']=FIT_LR*lr_at(st)
    idx=torch.randint(0,ntr,(min(FIT_BATCH,ntr),),device=DEV,generator=g)
    loss=F.mse_loss(net(Xtr[idx]),Ytr[idx]); opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(net.parameters(),1.0); opt.step()
    if st%EVAL_EVERY==0 or st==FIT_STEPS:
        vm,r2=val_r2(); traj.append({'step':st,'val_mse':round(vm,6),'r2':round(r2,5)})
        print(f'   step {st:6d} | lr={FIT_LR*lr_at(st):.2e} | val_mse={vm:.6f} R2={r2:.5f}')
final=traj[-1]

plt.figure(figsize=(9,5))
plt.plot([t['step'] for t in traj],[t['r2'] for t in traj],'o-',label='input-GELU depth3xw32')
plt.axhline(0.7884, ls='--', c='gray', label='baseline depth3xw32 (no input GELU) 0.7884')
plt.axhline(1.0, ls=':', c='green', alpha=0.5)
plt.xlabel('training step'); plt.ylabel('val R2 (approx head0)'); plt.title('head0 input-GELU depth3xw32 — R2 vs step (40k)')
plt.legend(fontsize=9); plt.grid(True,alpha=0.3); plt.tight_layout(); plt.savefig(os.path.join(EXP_DIR,'head0_gelu_curves.png'),dpi=120); plt.close()

summary={'exp_name':'exp_n_0071_head0_input_gelu_d3w32_40k','target':'exp_n_0052 block0 head0 (48->48)',
         'config':{'depth':CONFIG_DEPTH,'width':CONFIG_WIDTH,'H':H,'fit_steps':FIT_STEPS,'input_gelu':True,'output_gelu':False},
         'baseline_no_input_gelu_depth3w32':0.7884,'final_r2':final['r2'],'final_val_mse':final['val_mse'],'trajectory':traj}
json.dump(summary,open(os.path.join(EXP_DIR,'head0_gelu_summary.json'),'w'),indent=2)
print('\n=== DONE ==='); print(f"final R2={final['r2']} (baseline no-input-GELU d3w32 = 0.7884)")
