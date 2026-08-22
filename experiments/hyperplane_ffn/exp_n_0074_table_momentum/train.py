"""exp_n_0074 — does higher MOMENTUM (β1) on the LUT TABLE params recover the
bigger-batch benefit at fixed 1x batch / fixed token budget?

Hypothesis: the LUT-vs-dense gap at 1x batch is a gradient-VARIANCE problem — each
table row's gradient is averaged over only ~380 tokens/step (24576 tokens / ~64 cells
per table spread across... ) so it's much noisier than a dense weight. 1.5x batch is
known to close the gap to ~1.19. Test whether raising β1 on the table params (longer
temporal averaging of the noisy per-cell gradient) recovers some of that gain WITHOUT
more tokens.

Base recipe = exp_n_0052 (6L/384/H8/d48/tph64/nap6 batched hard LUT, tied dense, AdamW
lr3e-4 wd0.1 cosine, betas (0.9,0.95)). Only this folder; wraps modules, no shared-src edit.

RUNS (env): B1_TABLE = β1 on the table param group (others fixed at (0.9,0.95));
BATCH_SCALE (1.0 or 1.5). Token budget is FIXED (n_steps*total_batch @1x); each run's
step count is derived so total tokens match. Table params get their OWN AdamW group with
betas (B1_TABLE, β2=0.95), wd=0 (as recipe); everything else stays (0.9,0.95). Logs
val_bpb vs TOKENS so runs at different batch sizes compare at equal budget.
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

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
cfg = json.load(open(os.path.join(EXP_DIR, 'config.json')))
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(cfg['random_seed'])

B1_TABLE = float(os.environ.get('B1_TABLE', '0.9'))
BATCH_SCALE = float(os.environ.get('BATCH_SCALE', '1.0'))
RUN_TAG = os.environ.get('RUN_TAG', f'b1_{B1_TABLE}')
DEPTH, N_EMBD, N_HEAD, SEQ_LEN = cfg['depth'], cfg['n_embd'], cfg['n_head'], cfg['seq_len']
BASE_DBS, BASE_TBS = cfg['device_batch_size'], cfg['total_batch_size']
DEVICE_BS = int(round(BASE_DBS*BATCH_SCALE)); TOTAL_BS = int(round(BASE_TBS*BATCH_SCALE))
TOKEN_BUDGET = cfg['n_steps']*BASE_TBS
N_STEPS = round(TOKEN_BUDGET/TOTAL_BS)
LR, WD, WARMUP_FRAC = cfg['lr'], cfg['weight_decay'], cfg['lr_warmup_fraction']
EVAL_EVERY, EVAL_STEPS = cfg['eval_every'], cfg['eval_steps']
TIE = bool(cfg['tie_unembedder']); B2 = cfg['adam_beta2']
LUT_IN, LUT_OUT = cfg['lut_inner_in_dim'], cfg['lut_inner_out_dim']
LUT_NAP, LUT_TPH, LUT_HEADS = cfg['lut_n_anchor_pairs'], cfg['lut_tables_per_head'], cfg['lut_n_heads']
REF_DENSE, REF_LUT = cfg['tied_dense_ref_bpb'], cfg['e2e_lut_ref_bpb']
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
        if targets is not None:
            return F.cross_entropy(logits.view(-1,logits.size(-1)), targets.view(-1), reduction=loss_reduction, ignore_index=-1)
        return logits

model=MinimalGPT(VOCAB_SIZE,N_EMBD,N_HEAD,DEPTH,SEQ_LEN).to(DEVICE)
total_params=sum(p.numel() for p in model.parameters())

# optimizer: table params in their OWN group with betas (B1_TABLE, B2); rest as recipe
lut_mods=[m for m in model.modules() if isinstance(m,FastMultiHeadLut)]
table_ids={id(m.weights) for m in lut_mods}
lut_param_ids={id(p) for m in lut_mods for p in m.parameters(recurse=False)}   # tables + temps
decay,nodecay_rest,tables=[],[],[]
for p in model.parameters():
    if not p.requires_grad: continue
    if id(p) in table_ids: tables.append(p)
    elif id(p) in lut_param_ids or p.ndim<2: nodecay_rest.append(p)
    else: decay.append(p)
groups=[dict(params=decay,lr=LR,betas=(0.9,B2),eps=1e-8,weight_decay=WD),
        dict(params=nodecay_rest,lr=LR,betas=(0.9,B2),eps=1e-8,weight_decay=0.0),
        dict(params=tables,lr=LR,betas=(B1_TABLE,B2),eps=1e-8,weight_decay=0.0)]
optimizer=torch.optim.AdamW(groups)
for g in optimizer.param_groups: g['initial_lr']=g['lr']
grad_accum=max(1,TOTAL_BS//(DEVICE_BS*SEQ_LEN))
print(f'[{RUN_TAG}] params={total_params:,} | table params={sum(p.numel() for p in tables):,} beta1={B1_TABLE} (eff window ~{1/(1-B1_TABLE):.0f}) | rest beta1=0.9 | batch_scale={BATCH_SCALE} DBS={DEVICE_BS} TBS={TOTAL_BS} steps={N_STEPS} grad_accum={grad_accum} | token_budget={TOKEN_BUDGET:,}')

def get_lr_scale(step,n_steps,warmup_frac):
    w=int(warmup_frac*n_steps)
    if step<w: return step/max(w,1)
    p=(step-w)/max(n_steps-w,1); return 0.1+0.9*0.5*(1+math.cos(math.pi*p))

csv_f=open(os.path.join(EXP_DIR,f'metrics_{RUN_TAG}.csv'),'w',newline=''); csv_w=csv.writer(csv_f); csv_w.writerow(['step','tokens','train_loss','val_bpb'])
val_bpbs,val_toks,ema,best_bpb,t0=[],[],None,float('inf'),time.time()
model.train()
for step in range(1,N_STEPS+1):
    lr_scale=get_lr_scale(step,N_STEPS,WARMUP_FRAC)
    for g in optimizer.param_groups: g['lr']=g['initial_lr']*lr_scale
    optimizer.zero_grad(set_to_none=True); accum=0.0
    for _ in range(grad_accum):
        x,y=next(train_loader); loss=model(x,y); (loss/grad_accum).backward(); accum+=loss.item()/grad_accum
    gn=torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); optimizer.step()
    ema=accum if ema is None else 0.99*ema+0.01*accum
    if step%100==0 or step==1: print(f'[{RUN_TAG}] step {step:5d} | loss={ema:.4f} | grad_norm={gn:.2f} | lr={lr_scale*LR:.2e}')
    if step%EVAL_EVERY==0 or step==N_STEPS:
        model.eval(); bpb=evaluate_bpb(model,val_loader_factory(),EVAL_STEPS,token_bytes); model.train()
        best_bpb=min(best_bpb,bpb); toks=step*TOTAL_BS; val_bpbs.append(bpb); val_toks.append(toks)
        print(f'[{RUN_TAG}] [VAL] step {step} ({toks/1e6:.1f}M tok): bpb={bpb:.4f}')
        csv_w.writerow([step,toks,f'{ema:.6f}',f'{bpb:.6f}']); csv_f.flush()
csv_f.close(); elapsed=time.time()-t0
summary={'exp_name':cfg['exp_name'],'run_tag':RUN_TAG,'b1_table':B1_TABLE,'batch_scale':BATCH_SCALE,
         'device_batch':DEVICE_BS,'total_batch':TOTAL_BS,'n_steps':N_STEPS,'token_budget':TOKEN_BUDGET,
         'best_val_bpb':best_bpb,'final_val_bpb':val_bpbs[-1] if val_bpbs else None,
         'refs':{'tied_dense':REF_DENSE,'e2e_lut':REF_LUT},'total_params':total_params,'training_time_hours':round(elapsed/3600,3)}
json.dump(summary,open(os.path.join(EXP_DIR,f'summary_{RUN_TAG}.json'),'w'),indent=2)
print(f'[{RUN_TAG}] DONE best_bpb={best_bpb:.4f} final={summary["final_val_bpb"]}')

# combined plot (val_bpb vs tokens) over all runs present
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
    plt.xlabel('tokens (M)'); plt.ylabel('val bpb'); plt.title('exp_n_0074: table-param β1 momentum + 1.5x-batch ref (equal token budget)')
    plt.legend(fontsize=8); plt.grid(True,alpha=0.3); plt.tight_layout(); plt.savefig(os.path.join(EXP_DIR,'compare_bpb.png'),dpi=120); plt.close()
    print('wrote compare_bpb.png')
