"""exp_n_0076 — CELL-LEVEL (direction-preserving) momentum for the LUT table params.

Distinct from exp_n_0073 (lazy/sparse Adam) and exp_n_0074 (element-wise β1 sweep).
Element-wise Adam keeps m,v PER SCALAR (each of a cell's 48 dims rescaled independently),
distorting the cell's gradient DIRECTION. Cell-level momentum treats each table cell (one
[48] row) as a UNIT:
  - m: per-cell VECTOR [.,48], EMA of the gradient vector (β1).
  - v: per-cell SCALAR [.], EMA of the cell's gradient mean-square (β2).
  - update = lr * m_hat / (sqrt(v_hat)+eps), the SAME scalar sqrt(v_hat) scaling the whole
    48-dim vector -> gradient DIRECTION is preserved (per-cell LAMB/LARS-style normalization).
Only the LUT table params use this; all other params stay on AdamW (0.9,0.95). Table wd=0.

Base recipe = exp_n_0052. Runs at equal 73.7M-token budget (3000 steps @1× batch):
  OPT_MODE=elementwise (baseline, table Adam β1=0.9,β2=0.95); OPT_MODE=cell with
  B1_TABLE in {0.9,0.95,0.98}. No shared-src edits.
"""
import sys, os, json, math, time, csv
try: sys.stdout.reconfigure(line_buffering=True)
except Exception: pass
import torch, torch.nn as nn, torch.nn.functional as F
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

NANOCHAT_ROOT = os.environ.get('NANOCHAT_ROOT', os.path.expanduser('~/projects/nanochat'))
if NANOCHAT_ROOT not in sys.path: sys.path.insert(0, NANOCHAT_ROOT)
from nanochat.common import get_base_dir
from nanochat.tokenizer import RustBPETokenizer, get_token_bytes
from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit
from nanochat.loss_eval import evaluate_bpb
from spiky.lutorch.fast_multi_head_lut import FastMultiHeadLut
from spiky.lutorch.compression_mhl import CompressionMultiHeadLUT

EXP_DIR = os.path.dirname(os.path.abspath(__file__)); cfg = json.load(open(os.path.join(EXP_DIR,'config.json')))
DEVICE='cuda' if torch.cuda.is_available() else 'cpu'; torch.manual_seed(cfg['random_seed'])
OPT_MODE=os.environ.get('OPT_MODE','elementwise')            # 'elementwise' | 'cell'
B1_TABLE=float(os.environ.get('B1_TABLE','0.9')); B2=cfg['adam_beta2']
RUN_TAG=os.environ.get('RUN_TAG', f'{OPT_MODE}_b{B1_TABLE}')
DEPTH,N_EMBD,N_HEAD,SEQ_LEN=cfg['depth'],cfg['n_embd'],cfg['n_head'],cfg['seq_len']
DEVICE_BS,TOTAL_BS,N_STEPS=cfg['device_batch_size'],cfg['total_batch_size'],cfg['n_steps']
LR,WD,WARMUP_FRAC=cfg['lr'],cfg['weight_decay'],cfg['lr_warmup_fraction']
EVAL_EVERY,EVAL_STEPS=cfg['eval_every'],cfg['eval_steps']; TIE=bool(cfg['tie_unembedder'])
LUT_IN,LUT_OUT=cfg['lut_inner_in_dim'],cfg['lut_inner_out_dim']
LUT_NAP,LUT_TPH,LUT_HEADS=cfg['lut_n_anchor_pairs'],cfg['lut_tables_per_head'],cfg['lut_n_heads']
REF_DENSE,REF_LUT=cfg['tied_dense_ref_bpb'],cfg['e2e_lut_ref_bpb']
if os.environ.get('SMOKE_STEPS'):
    N_STEPS=int(os.environ['SMOKE_STEPS']); EVAL_EVERY=max(1,N_STEPS); print(f'*** SMOKE_STEPS={N_STEPS} ({RUN_TAG}) ***')

BASE_DIR=get_base_dir(); tokenizer=RustBPETokenizer.from_directory(os.path.join(BASE_DIR,'tokenizer'))
VOCAB_SIZE=tokenizer.get_vocab_size(); assert VOCAB_SIZE==cfg['tokenizer_vocab_size']
train_loader=tokenizing_distributed_data_loader_bos_bestfit(tokenizer,DEVICE_BS,SEQ_LEN,split='train',device=DEVICE)
val_loader_factory=lambda: tokenizing_distributed_data_loader_bos_bestfit(tokenizer,DEVICE_BS,SEQ_LEN,split='val',device=DEVICE)
token_bytes=get_token_bytes(device=DEVICE)

class RotaryEmbedding(nn.Module):
    def __init__(self,head_dim,max_seq_len,base=10000.0,device=None):
        super().__init__(); inv=1.0/(base**(torch.arange(0,head_dim,2,dtype=torch.float32,device=device)/head_dim))
        t=torch.arange(max_seq_len,device=device,dtype=torch.float32); emb=torch.cat([torch.outer(t,inv)]*2,-1)
        self.register_buffer('cos',emb.cos(),persistent=False); self.register_buffer('sin',emb.sin(),persistent=False)
def _rotate_half(x): x1,x2=x.chunk(2,-1); return torch.cat([-x2,x1],-1)
def apply_rope(q,k,cos,sin): cos=cos[None,None]; sin=sin[None,None]; return q*cos+_rotate_half(q)*sin, k*cos+_rotate_half(k)*sin
class MinimalAttention(nn.Module):
    def __init__(self,n_embd,n_head): super().__init__(); self.n_head=n_head; self.qkv=nn.Linear(n_embd,3*n_embd,bias=False); self.proj=nn.Linear(n_embd,n_embd,bias=False)
    def forward(self,x,cos,sin):
        B,T,C=x.size(); q,k,v=self.qkv(x).split(C,2)
        q=q.view(B,T,self.n_head,C//self.n_head).transpose(1,2); k=k.view(B,T,self.n_head,C//self.n_head).transpose(1,2); v=v.view(B,T,self.n_head,C//self.n_head).transpose(1,2)
        q,k=apply_rope(q,k,cos[:T],sin[:T]); y=F.scaled_dot_product_attention(q,k,v,is_causal=True)
        return self.proj(y.transpose(1,2).contiguous().view(B,T,C))
class MinimalBlock(nn.Module):
    def __init__(self,n_embd,n_head,layer_idx):
        super().__init__(); self.ln1=nn.LayerNorm(n_embd); self.attn=MinimalAttention(n_embd,n_head); self.ln2=nn.LayerNorm(n_embd)
        self.ffn=CompressionMultiHeadLUT(input_dim=n_embd,output_dim=n_embd,inner_in_dim=LUT_IN,inner_out_dim=LUT_OUT,
            nap=LUT_NAP,tph=LUT_TPH,n_heads=LUT_HEADS,joint_head_compression=cfg['lut_joint_head_compression'],
            batched_multi_head_input=cfg['lut_batched_multi_head_input'],forward_mode=cfg['lut_forward_mode'],
            use_bf16=cfg['lut_use_bf16'],initial_weights_noise=cfg['lut_init_weights_noise'],
            learnable_temps=cfg['lut_learnable_temps'],random_seed=cfg['lut_base_seed']+layer_idx)
    def forward(self,x,cos,sin):
        x=x+self.attn(self.ln1(x),cos,sin); h=self.ln2(x); B,T,C=h.shape
        return x+self.ffn(h.reshape(B*T,C)).reshape(B,T,C).to(h.dtype)
class MinimalGPT(nn.Module):
    def __init__(self,vocab_size,n_embd,n_head,n_layer,seq_len):
        super().__init__(); self.tok_emb=nn.Embedding(vocab_size,n_embd); self.rope=RotaryEmbedding(n_embd//n_head,seq_len)
        self.blocks=nn.ModuleList([MinimalBlock(n_embd,n_head,i) for i in range(n_layer)]); self.ln_f=nn.LayerNorm(n_embd); self.head=nn.Linear(n_embd,vocab_size,bias=False)
        self.apply(self._init_weights)
        for block in self.blocks:
            nn.init.zeros_(block.attn.proj.weight)
            if block.ffn.has_decompress: nn.init.zeros_(block.ffn.decompress.weight)
        if TIE: self.head.weight=self.tok_emb.weight
    @staticmethod
    def _init_weights(m):
        if isinstance(m,(nn.Linear,nn.Embedding)): nn.init.normal_(m.weight,std=0.02)
    def get_device(self): return self.tok_emb.weight.device
    def forward(self, idx, targets=None, loss_reduction='mean'):
        x=self.tok_emb(idx)
        for block in self.blocks: x=block(x,self.rope.cos,self.rope.sin)
        logits=self.head(self.ln_f(x))
        if targets is not None: return F.cross_entropy(logits.view(-1,logits.size(-1)),targets.view(-1),reduction=loss_reduction,ignore_index=-1)
        return logits

class CellLevelMomentum:
    """Direction-preserving per-cell Adam: vector 1st moment, SCALAR (per-cell RMS^2) 2nd
    moment so one scalar scales the whole 48-dim cell (LAMB/LARS-style). Dense: all cells
    updated every step (matches how the tables actually receive gradient)."""
    def __init__(self, params, betas, eps=1e-8):
        self.params=list(params); self.b1,self.b2=betas; self.eps=eps; self.t=0
        self.m=[torch.zeros_like(p) for p in self.params]                 # [.,48] vector
        self.v=[torch.zeros(p.shape[:-1],device=p.device) for p in self.params]  # [.] scalar
    @torch.no_grad()
    def step(self, lr):
        self.t+=1; bc1=1-self.b1**self.t; bc2=1-self.b2**self.t
        for p,m,v in zip(self.params,self.m,self.v):
            g=p.grad
            if g is None: continue
            m.mul_(self.b1).add_(g,alpha=1-self.b1)                       # per-cell vector EMA
            gms=g.pow(2).mean(dim=-1)                                     # per-cell mean-square [.]
            v.mul_(self.b2).add_(gms,alpha=1-self.b2)
            denom=(v/bc2).sqrt().add_(self.eps).unsqueeze(-1)            # scalar per cell
            p.add_(m/bc1/denom, alpha=-lr)                               # same scalar scales whole vector

model=MinimalGPT(VOCAB_SIZE,N_EMBD,N_HEAD,DEPTH,SEQ_LEN).to(DEVICE)
total_params=sum(p.numel() for p in model.parameters())
lut_mods=[m for m in model.modules() if isinstance(m,FastMultiHeadLut)]
table_ids={id(m.weights) for m in lut_mods}; lut_param_ids={id(p) for m in lut_mods for p in m.parameters(recurse=False)}
table_params=[m.weights for m in lut_mods]
decay,nodecay_rest,tables=[],[],[]
for p in model.parameters():
    if not p.requires_grad: continue
    if id(p) in table_ids: tables.append(p)
    elif id(p) in lut_param_ids or p.ndim<2: nodecay_rest.append(p)
    else: decay.append(p)
groups=[dict(params=decay,lr=LR,betas=(0.9,B2),eps=1e-8,weight_decay=WD),
        dict(params=nodecay_rest,lr=LR,betas=(0.9,B2),eps=1e-8,weight_decay=0.0)]
cell_opt=None
if OPT_MODE=='cell':
    cell_opt=CellLevelMomentum(table_params,betas=(B1_TABLE,B2))
    print(f'[{RUN_TAG}] tables ({sum(p.numel() for p in table_params):,}) on CELL-LEVEL momentum β1={B1_TABLE} β2={B2}; rest AdamW')
else:  # elementwise: tables in their own AdamW group with (B1_TABLE,B2)
    groups.append(dict(params=table_params,lr=LR,betas=(B1_TABLE,B2),eps=1e-8,weight_decay=0.0))
    print(f'[{RUN_TAG}] tables on ELEMENT-WISE AdamW β1={B1_TABLE} β2={B2}; rest AdamW')
optimizer=torch.optim.AdamW(groups)
for g in optimizer.param_groups: g['initial_lr']=g['lr']
grad_accum=max(1,TOTAL_BS//(DEVICE_BS*SEQ_LEN))
print(f'[{RUN_TAG}] params={total_params:,} | n_steps={N_STEPS} grad_accum={grad_accum}')

def get_lr_scale(step,n_steps,warmup_frac):
    w=int(warmup_frac*n_steps)
    if step<w: return step/max(w,1)
    p=(step-w)/max(n_steps-w,1); return 0.1+0.9*0.5*(1+math.cos(math.pi*p))

csv_f=open(os.path.join(EXP_DIR,f'metrics_{RUN_TAG}.csv'),'w',newline=''); csv_w=csv.writer(csv_f); csv_w.writerow(['step','tokens','train_loss','val_bpb','grad_norm'])
val_bpbs,val_toks,ema,best_bpb,t0=[],[],None,float('inf'),time.time(); maxgn=0.0
model.train()
for step in range(1,N_STEPS+1):
    lr_scale=get_lr_scale(step,N_STEPS,WARMUP_FRAC); lr_now=LR*lr_scale
    for g in optimizer.param_groups: g['lr']=g['initial_lr']*lr_scale
    optimizer.zero_grad(set_to_none=True)
    if cell_opt is not None:
        for p in cell_opt.params:
            if p.grad is not None: p.grad=None
    accum=0.0
    for _ in range(grad_accum):
        x,y=next(train_loader); loss=model(x,y); (loss/grad_accum).backward(); accum+=loss.item()/grad_accum
    gn=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)); maxgn=max(maxgn,gn)
    optimizer.step()
    if cell_opt is not None: cell_opt.step(lr_now)
    ema=accum if ema is None else 0.99*ema+0.01*accum
    if step%100==0 or step==1: print(f'[{RUN_TAG}] step {step:5d} | loss={ema:.4f} | grad_norm={gn:.2f} | lr={lr_now:.2e}')
    if step%EVAL_EVERY==0 or step==N_STEPS:
        model.eval(); bpb=evaluate_bpb(model,val_loader_factory(),EVAL_STEPS,token_bytes); model.train()
        best_bpb=min(best_bpb,bpb); toks=step*TOTAL_BS; val_bpbs.append(bpb); val_toks.append(toks)
        print(f'[{RUN_TAG}] [VAL] step {step} ({toks/1e6:.1f}M): bpb={bpb:.4f}')
        csv_w.writerow([step,toks,f'{ema:.6f}',f'{bpb:.6f}',f'{gn:.4f}']); csv_f.flush()
csv_f.close(); elapsed=time.time()-t0
summary={'exp_name':cfg['exp_name'],'run_tag':RUN_TAG,'opt_mode':OPT_MODE,'b1_table':B1_TABLE,'b2':B2,'n_steps':N_STEPS,
         'best_val_bpb':best_bpb,'final_val_bpb':val_bpbs[-1] if val_bpbs else None,'max_grad_norm':round(maxgn,3),
         'refs':{'tied_dense':REF_DENSE,'e2e_lut':REF_LUT},'total_params':total_params,'training_time_hours':round(elapsed/3600,3)}
json.dump(summary,open(os.path.join(EXP_DIR,f'summary_{RUN_TAG}.json'),'w'),indent=2)
print(f'[{RUN_TAG}] DONE best_bpb={best_bpb:.4f} final={summary["final_val_bpb"]} max_gn={maxgn:.2f}')

import glob as _glob
def load_csv(p):
    import csv as _c; rows=list(_c.DictReader(open(p)))
    return [(int(r['tokens']),float(r['val_bpb'])) for r in rows if r['val_bpb']]
files=sorted(_glob.glob(os.path.join(EXP_DIR,'metrics_*.csv')))
if len(files)>=2:
    plt.figure(figsize=(9,5))
    for p in files:
        tag=os.path.basename(p)[len('metrics_'):-len('.csv')]; pts=load_csv(p)
        if pts: plt.plot([t/1e6 for t,_ in pts],[v for _,v in pts],'o-',ms=3,label=tag)
    plt.axhline(REF_DENSE,ls='--',c='k',label=f'tied dense {REF_DENSE}'); plt.axhline(REF_LUT,ls='--',c='gray',label=f'e2e LUT {REF_LUT}')
    plt.xlabel('tokens (M)'); plt.ylabel('val bpb'); plt.title('exp_n_0076: cell-level vs element-wise momentum on LUT tables (equal budget)')
    plt.legend(fontsize=8); plt.grid(True,alpha=0.3); plt.tight_layout(); plt.savefig(os.path.join(EXP_DIR,'compare_bpb.png'),dpi=120); plt.close()
    print('wrote compare_bpb.png')
