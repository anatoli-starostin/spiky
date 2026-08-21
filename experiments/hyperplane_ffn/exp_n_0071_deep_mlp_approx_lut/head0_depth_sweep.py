"""exp_n_0071 (head-0 focus) — DEPTH x WIDTH sweep of MLPs approximating a SINGLE
FastMHL head (head 0 of block 0) of exp_n_0052's trained LUTs.

Head 0 was the least-saturated target in exp_n_0067 (2-layer MLP floored at
R2 0.585->0.689 across widths 4->64x). Here: same target/input/metric as
exp_n_0067's single-head case, but sweep DEPTH {2,3,4,6,8} x WIDTH {16,32,64}
with deep pre-norm residual MLPs, cosine LR, and longer training (10k steps).
Question: does depth break past ~0.69 toward 1.0 (a deep MLP CAN represent the
hard hyperplane-routing head) or does it also floor?  No shared-module edits.
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
DIN = cfg['lut_inner_in_dim']; DOUT = cfg['lut_inner_out_dim']   # 48, 48
DEPTHS = cfg['head0_depths']; WIDTHS = cfg['head0_widths']
N_COLLECT = int(cfg['n_collect']); VAL_FRAC = cfg['val_frac']
FIT_STEPS, FIT_LR, FIT_WARM, FIT_BATCH = cfg['head0_fit_steps'], cfg['fit_lr'], cfg['fit_warmup_frac'], cfg['fit_batch']
if os.environ.get('SMOKE'):
    DEPTHS = [2, 8]; WIDTHS = [16]; N_COLLECT = 24576; FIT_STEPS = 200; print('*** SMOKE ***')

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
# collect head-0 (input z[:,0,:], target out[:,0,:]) -- identical to exp_n_0067 single-head
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
print(f'head0 collected: X={tuple(X.shape)} Y={tuple(Y.shape)} | LUT head params ~{cfg["lut_tables_per_head"]*(1<<cfg["lut_n_anchor_pairs"])*DOUT:,}')

class DeepResMLP(nn.Module):
    def __init__(self, din, dout, H, depth):
        super().__init__(); self.depth=depth
        if depth==2: self.net=nn.Sequential(nn.Linear(din,H),nn.GELU(),nn.Linear(H,dout))
        else:
            self.inp=nn.Linear(din,H); self.blocks=nn.ModuleList([nn.ModuleList([nn.LayerNorm(H),nn.Linear(H,H)]) for _ in range(depth-2)])
            self.oln=nn.LayerNorm(H); self.out=nn.Linear(H,dout)
    def forward(self,x):
        if self.depth==2: return self.net(x)
        x=self.inp(x)
        for ln,lin in self.blocks: x=x+lin(F.gelu(ln(x)))
        return self.out(F.gelu(self.oln(x)))
def lr_at(st):
    w=int(FIT_WARM*FIT_STEPS)
    if st<w: return st/max(w,1)
    p=(st-w)/max(FIT_STEPS-w,1); return 0.05+0.95*0.5*(1+math.cos(math.pi*p))
def fit(depth,width):
    nv=int(X.shape[0]*VAL_FRAC); Xtr,Ytr,Xv,Yv=X[nv:],Y[nv:],X[:nv],Y[:nv]
    H=width*DIN; net=DeepResMLP(DIN,DOUT,H,depth).to(DEV)
    opt=torch.optim.AdamW(net.parameters(),lr=FIT_LR,weight_decay=0.0,betas=(0.9,0.95)); ntr=Xtr.shape[0]
    g=torch.Generator(device=DEV).manual_seed(0)
    for st in range(FIT_STEPS):
        for gp in opt.param_groups: gp['lr']=FIT_LR*lr_at(st)
        idx=torch.randint(0,ntr,(min(FIT_BATCH,ntr),),device=DEV,generator=g)
        loss=F.mse_loss(net(Xtr[idx]),Ytr[idx]); opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(),1.0); opt.step()
    net.eval()
    with torch.no_grad():
        tr=F.mse_loss(net(Xtr[:8192]),Ytr[:8192]).item(); vmse=F.mse_loss(net(Xv),Yv).item()
        var=Yv.var(0).mean().item(); r2=1-vmse/(var+1e-12)
    return {'depth':depth,'width':width,'H':H,'params':sum(p.numel() for p in net.parameters()),'train_mse':round(tr,6),'val_mse':round(vmse,6),'r2':round(r2,5)}

results=[]
for depth in DEPTHS:
    for width in WIDTHS:
        r=fit(depth,width); results.append(r)
        print(f'   depth={depth} width={width}x H={r["H"]:4d} params={r["params"]:>8,} | val_mse={r["val_mse"]:.5f} R2={r["r2"]:.4f}')

plt.figure(figsize=(9,6))
for width in WIDTHS:
    rr=[r for r in results if r['width']==width]
    plt.plot([r['depth'] for r in rr],[r['r2'] for r in rr],'o-',label=f'width {width}x')
plt.axhline(0.689, ls='--', c='gray', label='exp_n_0067 2-layer head0 best (0.689)')
plt.axhline(1.0, ls=':', c='green', alpha=0.5)
plt.xlabel('MLP depth (Linear layers)'); plt.ylabel('val R2 (approx FastMHL head0)'); plt.ylim(0.5,1.02)
plt.title('exp_n_0071: deep MLP approx of a single FastMHL head (block0 head0)\ndoes depth break the ~0.69 floor?')
plt.legend(fontsize=9); plt.grid(True,alpha=0.3); plt.tight_layout(); plt.savefig(os.path.join(EXP_DIR,'head0_depth_curves.png'),dpi=120); plt.close()

summary={'exp_name':'exp_n_0071_head0_depth_sweep','target':'exp_n_0052 block0 head0 (48->48)','depths':DEPTHS,'widths':WIDTHS,
         'fit_steps':FIT_STEPS,'n_collect':N_COLLECT,'exp_n_0067_2layer_head0_floor':{4:0.5853,8:0.6299,16:0.6585,32:0.6786,64:0.6890},
         'results':results,'best_r2':max(r['r2'] for r in results)}
json.dump(summary,open(os.path.join(EXP_DIR,'head0_summary.json'),'w'),indent=2)
print('\n=== DONE ==='); print('best head0 R2 across depth x width:', summary['best_r2'])
