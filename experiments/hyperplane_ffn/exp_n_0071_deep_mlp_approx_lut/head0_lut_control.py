"""exp_n_0071 (head-0 CONTROL) — train an actual same-structure FastMultiHeadLut
(copied+frozen anchoring, zero-init tables) to fit the head-0 target, and measure
how many steps to reach R2>0.99 / >0.999 (should approach ~1.0 since routing is
identical to the target).

Setup:
  - Fresh FastMultiHeadLut: n_heads=1, tables_per_head=64, n_anchor_pairs=6,
    input_dim=48, n_outputs=48, forward_mode='hard', fp32 weights, use_bf16=False,
    learnable_temps=False (temps are buffers -> only self.weights trains).
  - COPY anchors from frozen exp_n_0052 block0 head-0: lut_batched.soft_anchor_a_long[:64]
    / soft_anchor_b_long[:64] (indices in [0,48)), write into the fresh LUT's anchor
    buffers (buffers => already requires_grad=False => routing frozen & identical).
  - Zero-init the table weights [64,64,48].
  - Target: SAME head-0 target/inputs/val split as all prior runs (input z[:,0,:],
    target out[:,0,:], ~320k tokens, 80/20 val, per-element MSE, R2=1-vmse/var).
  - Train: AdamW, constant LR 1e-3, batch 8192, grad-clip 1.0, up to 30k steps, eval
    val R2 every 1k, early-stop once R2>0.9995. Report steps to cross 0.99 and 0.999.
Do NOT modify fast_multi_head_lut.py / compression_mhl.py. No shared-module edits.
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
from spiky.lutorch.fast_multi_head_lut import FastMultiHeadLut

EXP_DIR = os.path.dirname(os.path.abspath(__file__)); HFF = os.path.dirname(EXP_DIR)
cfg = json.load(open(os.path.join(EXP_DIR, 'config.json')))
DEV = 'cuda'; torch.manual_seed(cfg['random_seed'])
N_EMBD, N_HEAD, SEQ, DBS, DEPTH, VOCAB = cfg['n_embd'], cfg['n_head'], cfg['seq_len'], cfg['device_batch_size'], cfg['depth'], cfg['tokenizer_vocab_size']
DIN = cfg['lut_inner_in_dim']; DOUT = cfg['lut_inner_out_dim']
N_COLLECT = int(cfg['n_collect']); VAL_FRAC = cfg['val_frac']
TPH, NAP = cfg['lut_tables_per_head'], cfg['lut_n_anchor_pairs']
FIT_LR, FIT_BATCH, FIT_STEPS, EVAL_EVERY = 1e-3, 8192, 30000, 250
EARLY_EVALS = {20, 50, 100, 150}   # extra fine evals to resolve fast crossings
if os.environ.get('SMOKE'):
    N_COLLECT = 24576; FIT_STEPS = 2000; EVAL_EVERY = 500; print('*** SMOKE ***')

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

lb = model.blocks[0].ffn.lut_batched
A = lb.soft_anchor_a_long[:TPH].contiguous().to(DEV)   # [64,6] head-0 tables
B_ = lb.soft_anchor_b_long[:TPH].contiguous().to(DEV)  # [64,6]
assert A.shape == (TPH, NAP) and int(A.max()) < DIN and int(B_.max()) < DIN
print(f'head-0 anchors copied: A={tuple(A.shape)} range[{int(A.min())},{int(A.max())}], B range[{int(B_.min())},{int(B_.max())}]')

cap=[]
def hh(m,inp,out):
    z=inp[0]; z=z.reshape(z.shape[0],cfg['lut_n_heads'],DIN) if z.dim()==2 else z
    cap.append((z[:,0,:].float(), out[:,0,:].float()))
h=lb.register_forward_hook(hh)
n=0
with torch.no_grad():
    while n<N_COLLECT:
        x,_=next(loader); model(x); n+=x.numel()
h.remove()
X=torch.cat([a for a,_ in cap],0)[:N_COLLECT]; Y=torch.cat([b for _,b in cap],0)[:N_COLLECT]
print(f'head0 collected: X={tuple(X.shape)} Y={tuple(Y.shape)}')
nv=int(X.shape[0]*VAL_FRAC); Xtr,Ytr,Xv,Yv=X[nv:],Y[nv:],X[:nv],Y[:nv]; var=Yv.var(0).mean().item(); ntr=Xtr.shape[0]

# --- fresh same-structure single-head FastMHL, copied+frozen anchors, zero-init tables ---
lut = FastMultiHeadLut(input_dim=DIN, n_heads=1, n_outputs=DOUT, n_anchor_pairs=NAP, tables_per_head=TPH,
                       forward_mode='hard', weight_dtype=torch.float32, use_bf16=False,
                       learnable_temps=False, random_seed=0, initial_weights_noise=0.0, device=torch.device(DEV)).to(DEV)
with torch.no_grad():
    lut.soft_anchor_a_long.copy_(A); lut.soft_anchor_b_long.copy_(B_)   # identical routing to target
    nn.init.zeros_(lut.weights)                                          # zero-init tables
lut.soft_anchor_a_long.requires_grad_(False); lut.soft_anchor_b_long.requires_grad_(False)
trainable=[n for n,p in lut.named_parameters() if p.requires_grad]
print(f'LUT trainable params: {trainable} | weights shape={tuple(lut.weights.shape)} | total={sum(p.numel() for p in lut.parameters() if p.requires_grad):,}')

opt=torch.optim.AdamW([p for p in lut.parameters() if p.requires_grad], lr=FIT_LR, weight_decay=0.0, betas=(0.9,0.95))
g=torch.Generator(device=DEV).manual_seed(0)
def val_r2():
    lut.eval()
    with torch.no_grad(): vm=F.mse_loss(lut(Xv)[:,0,:], Yv).item()
    lut.train(); return vm, 1-vm/(var+1e-12)
traj=[]; cross99=None; cross999=None
print(f'--- LUT control: constant LR {FIT_LR:.0e}, batch {FIT_BATCH}, up to {FIT_STEPS} steps ---')
lut.train()
for st in range(1,FIT_STEPS+1):
    idx=torch.randint(0,ntr,(min(FIT_BATCH,ntr),),device=DEV,generator=g)
    loss=F.mse_loss(lut(Xtr[idx])[:,0,:], Ytr[idx]); opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(lut.parameters(),1.0); opt.step()
    if st%EVAL_EVERY==0 or st in EARLY_EVALS or st==FIT_STEPS:
        vm,r2=val_r2(); traj.append({'step':st,'val_mse':round(vm,8),'r2':round(r2,6)})
        if cross99 is None and r2>0.99: cross99=st
        if cross999 is None and r2>0.999: cross999=st
        print(f'   step {st:6d} | val_mse={vm:.8f} R2={r2:.6f}')
        if r2>0.9995: print('   >>> reached R2>0.9995, early stop'); break
final=traj[-1]; peak=max(traj,key=lambda t:t['r2'])
print(f'steps to R2>0.99: {cross99} | >0.999: {cross999} | final R2={final["r2"]}')

plt.figure(figsize=(9,5))
plt.plot([t['step'] for t in traj],[t['r2'] for t in traj],'o-',c='tab:green',label='same-structure LUT (frozen anchors, zero-init tables)')
plt.axhline(1.0, ls=':', c='green', alpha=0.6, label='R2=1.0')
plt.axhline(0.99, ls='--', c='gray'); plt.axhline(0.9375, ls='--', c='tab:orange', label='best MLP approx (300k) 0.938')
plt.xlabel('training step'); plt.ylabel('val R2 (approx head0)'); plt.title('head0 CONTROL: same-structure LUT to ~100% — R2 vs step')
plt.legend(fontsize=8); plt.grid(True,alpha=0.3); plt.tight_layout(); plt.savefig(os.path.join(EXP_DIR,'head0_lut_control_curves.png'),dpi=120); plt.close()

summary={'exp_name':'exp_n_0071_head0_lut_control','target':'exp_n_0052 block0 head0 (48->48)',
         'setup':'fresh FastMHL n_heads=1 tph=64 nap=6, anchors copied+frozen from ckpt, zero-init tables, only weights trained',
         'opt':{'lr':FIT_LR,'batch':FIT_BATCH,'steps_max':FIT_STEPS},
         'steps_to_R2_gt_0.99':cross99,'steps_to_R2_gt_0.999':cross999,
         'final_r2':final['r2'],'peak_r2':peak['r2'],'peak_step':peak['step'],'final_val_mse':final['val_mse'],'trajectory':traj}
json.dump(summary,open(os.path.join(EXP_DIR,'head0_lut_control_summary.json'),'w'),indent=2)
print('\n=== DONE ==='); print(f"steps>0.99={cross99} steps>0.999={cross999} final R2={final['r2']}")
