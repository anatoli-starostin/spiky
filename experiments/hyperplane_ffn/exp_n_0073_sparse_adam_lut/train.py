"""exp_n_0073 — does dense AdamW mishandle the LUT tables' SPARSE (embedding_bag)
gradients? Diagnostic (per-cell gap-between-hits, momentum staleness, wd shrink) +
lazy/sparse-Adam A/B on the LUT table params.

Base recipe = exp_n_0052 (6L/384/H8/d48/tph64/nap6 batched hard LUT, tied dense,
AdamW lr3e-4 wd0.1 cosine, betas (0.9,0.95)). Only this exp folder; wraps
FastMultiHeadLut/CompressionMultiHeadLUT, does NOT modify shared src.

MODES (env OPT_MODE): 'baseline' = all params incl LUT tables under dense AdamW
(exactly exp_n_0052). 'sparse' = LUT TABLE params under a LAZY/SPARSE Adam whose
moments advance ONLY on steps a row is actually selected (frozen between hits, per-row
bias-correction); all other params under the normal AdamW. LR/schedule matched.

PART A diagnostic (env DIAG=1, run on the baseline): each step compute the set of LUT
cells that receive a nonzero gradient (= are selected by the real batch), and track
per-cell the GAP in optimizer steps between successive hits -> gap distribution
(mean/median/p90/p99/max), frequent vs tail cells, distinct-cells-per-step, and the
Adam moment decay (β^gap) + hypothetical wd shrink implied by the gaps.
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

OPT_MODE = os.environ.get('OPT_MODE', 'baseline')   # 'baseline' | 'sparse'
DIAG = bool(int(os.environ.get('DIAG', '0')))
DEPTH, N_EMBD, N_HEAD, SEQ_LEN = cfg['depth'], cfg['n_embd'], cfg['n_head'], cfg['seq_len']
DEVICE_BS, TOTAL_BS, N_STEPS = cfg['device_batch_size'], cfg['total_batch_size'], int(os.environ.get('N_STEPS', cfg['n_steps']))
LR, WD, WARMUP_FRAC = cfg['lr'], cfg['weight_decay'], cfg['lr_warmup_fraction']
EVAL_EVERY, EVAL_STEPS = cfg['eval_every'], cfg['eval_steps']
TIE = bool(cfg['tie_unembedder'])
LUT_IN, LUT_OUT = cfg['lut_inner_in_dim'], cfg['lut_inner_out_dim']
LUT_NAP, LUT_TPH, LUT_HEADS = cfg['lut_n_anchor_pairs'], cfg['lut_tables_per_head'], cfg['lut_n_heads']
B1, B2 = cfg['adam_betas']
REF_DENSE, REF_LUT = cfg['tied_dense_ref_bpb'], cfg['e2e_lut_ref_bpb']
RUN_TAG = OPT_MODE
if os.environ.get('SMOKE_STEPS'):
    N_STEPS=int(os.environ['SMOKE_STEPS']); EVAL_EVERY=max(1,N_STEPS); print(f'*** SMOKE_STEPS={N_STEPS} ({RUN_TAG} DIAG={DIAG}) ***')

BASE_DIR = get_base_dir(); tokenizer = RustBPETokenizer.from_directory(os.path.join(BASE_DIR,'tokenizer'))
VOCAB_SIZE = tokenizer.get_vocab_size(); assert VOCAB_SIZE == cfg['tokenizer_vocab_size']
train_loader = tokenizing_distributed_data_loader_bos_bestfit(tokenizer, DEVICE_BS, SEQ_LEN, split='train', device=DEVICE)
val_loader_factory = lambda: tokenizing_distributed_data_loader_bos_bestfit(tokenizer, DEVICE_BS, SEQ_LEN, split='val', device=DEVICE)
token_bytes = get_token_bytes(device=DEVICE)

class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim, max_seq_len, base=10000.0, device=None):
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

model = MinimalGPT(VOCAB_SIZE, N_EMBD, N_HEAD, DEPTH, SEQ_LEN).to(DEVICE)
total_params=sum(p.numel() for p in model.parameters())

# --- identify LUT table params (the FastMHL .weights) ---
lut_mods=[m for m in model.modules() if isinstance(m,FastMultiHeadLut)]
lut_weight_ids={id(m.weights) for m in lut_mods}
TABLE_ROWS_PER = lut_mods[0].weights.shape[0]*lut_mods[0].weights.shape[1]   # tables*table_dim per block LUT
TOTAL_CELLS = DEPTH * TABLE_ROWS_PER

# --- lazy/sparse Adam for the table params: moments advance ONLY on hit rows ---
class LazyAdamTables:
    def __init__(self, params, betas, eps=1e-8):
        self.params=list(params); self.b1,self.b2=betas; self.eps=eps
        self.m=[torch.zeros_like(p) for p in self.params]
        self.v=[torch.zeros_like(p) for p in self.params]
        self.cnt=[torch.zeros(p.shape[:-1], device=p.device) for p in self.params]  # per-row update count
    @torch.no_grad()
    def step(self, lr):
        for p,m,v,cnt in zip(self.params,self.m,self.v,self.cnt):
            g=p.grad
            if g is None: continue
            hit=(g!=0).any(-1)                       # [tables,rows] rows selected this step
            if not bool(hit.any()): continue
            cnt[hit]+=1
            gh=g[hit]                                 # [K,dim]
            m[hit]=self.b1*m[hit]+(1-self.b1)*gh
            v[hit]=self.b2*v[hit]+(1-self.b2)*gh*gh
            c=cnt[hit].unsqueeze(-1)                  # per-row update count -> per-row bias correction
            mhat=m[hit]/(1-self.b1**c); vhat=v[hit]/(1-self.b2**c)
            p[hit]=p[hit]-lr*mhat/(vhat.sqrt()+self.eps)

def setup_dense_optimizer(params_iter, lr, weight_decay):
    lut_ids={id(p) for m in model.modules() if isinstance(m,FastMultiHeadLut) for p in m.parameters(recurse=False)}
    decay,nodecay=[],[]
    for p in params_iter:
        if not p.requires_grad: continue
        (nodecay if (id(p) in lut_ids or p.ndim<2) else decay).append(p)
    groups=[dict(params=decay,lr=lr,betas=(B1,B2),eps=1e-8,weight_decay=weight_decay),
            dict(params=nodecay,lr=lr,betas=(B1,B2),eps=1e-8,weight_decay=0.0)]
    opt=torch.optim.AdamW(groups)
    for g in opt.param_groups: g['initial_lr']=g['lr']
    return opt

if OPT_MODE=='sparse':
    table_params=[m.weights for m in lut_mods]
    rest=[p for p in model.parameters() if id(p) not in lut_weight_ids]
    dense_opt=setup_dense_optimizer(rest, LR, WD)
    lazy_opt=LazyAdamTables(table_params, betas=(B1,B2))
    print(f'[sparse] tables under LazyAdam ({sum(p.numel() for p in table_params):,} params); rest under AdamW')
else:
    dense_opt=setup_dense_optimizer(model.parameters(), LR, WD); lazy_opt=None
    print(f'[baseline] all params under dense AdamW (LUT tables in the wd=0 nodecay group per exp_n_0052)')
print(f'[{RUN_TAG}] params={total_params:,} | total LUT cells={TOTAL_CELLS:,} | betas=({B1},{B2}) | n_steps={N_STEPS} | DIAG={DIAG}')

def get_lr_scale(step,n_steps,warmup_frac):
    w=int(warmup_frac*n_steps)
    if step<w: return step/max(w,1)
    p=(step-w)/max(n_steps-w,1); return 0.1+0.9*0.5*(1+math.cos(math.pi*p))

# --- diagnostic: per-cell hit gaps ---
def hit_cells_of_batch(model, x):
    """Global unique cell ids selected by batch x across all LUTs (hard routing)."""
    caps={}; handles=[]
    for bi,blk in enumerate(model.blocks):
        def mk(bi):
            def hook(mod,inp,out): caps[bi]=inp[0].detach()
            return hook
        handles.append(blk.ffn.lut_batched.register_forward_hook(mk(bi)))
    with torch.no_grad(): model(x)
    for h in handles: h.remove()
    outs=[]
    for bi,blk in enumerate(model.blocks):
        m=blk.ffn.lut_batched; zf=caps[bi]
        if zf.dim()==3: zf=zf.reshape(zf.shape[0],-1)
        d=zf[:,m.soft_anchor_a_long]-zf[:,m.soft_anchor_b_long]
        idx=((d>0).long()*m.soft_powers.view(1,1,-1)).sum(-1)      # [N,ntab]
        ntab=idx.shape[1]; tab=torch.arange(ntab,device=idx.device).view(1,-1)
        outs.append(torch.unique(((bi*ntab+tab)*m.table_dim+idx).reshape(-1)))
    return torch.unique(torch.cat(outs))

if DIAG:
    last_hit=torch.zeros(TOTAL_CELLS,dtype=torch.int64,device=DEVICE)
    hit_count=torch.zeros(TOTAL_CELLS,dtype=torch.int64,device=DEVICE)
    sum_gap=torch.zeros(TOTAL_CELLS,dtype=torch.int64,device=DEVICE)
    max_gap=torch.zeros(TOTAL_CELLS,dtype=torch.int64,device=DEVICE)
    distinct_per_step=[]

csv_f=open(os.path.join(EXP_DIR,f'metrics_{RUN_TAG}.csv'),'w',newline=''); csv_w=csv.writer(csv_f); csv_w.writerow(['step','train_loss','val_bpb'])
val_bpbs, val_steps, ema, best_bpb, t0 = [], [], None, float('inf'), time.time()
grad_accum=max(1,TOTAL_BS//(DEVICE_BS*SEQ_LEN))
model.train()
for step in range(1,N_STEPS+1):
    lr_scale=get_lr_scale(step,N_STEPS,WARMUP_FRAC); lr_now=LR*lr_scale
    for g in dense_opt.param_groups: g['lr']=g['initial_lr']*lr_scale
    dense_opt.zero_grad(set_to_none=True)
    if lazy_opt is not None:
        for p in lazy_opt.params:
            if p.grad is not None: p.grad=None
    accum=0.0; last_x=None
    for _ in range(grad_accum):
        x,y=next(train_loader); last_x=x
        loss=model(x,y); (loss/grad_accum).backward(); accum+=loss.item()/grad_accum
    torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
    dense_opt.step()
    if lazy_opt is not None: lazy_opt.step(lr_now)
    ema=accum if ema is None else 0.99*ema+0.01*accum
    if DIAG:
        hc=hit_cells_of_batch(model,last_x); distinct_per_step.append(int(hc.numel()))
        prev=last_hit[hc]; mask=prev>0; gap=(step-prev)
        if mask.any():
            hcm=hc[mask]; gm=gap[mask]
            sum_gap.index_add_(0,hcm,gm); max_gap[hcm]=torch.maximum(max_gap[hcm],gm)
        hit_count[hc]+=1; last_hit[hc]=step
    if step%100==0 or step==1: print(f'[{RUN_TAG}] step {step:5d} | loss={ema:.4f} | lr={lr_now:.2e}')
    if step%EVAL_EVERY==0 or step==N_STEPS:
        model.eval(); bpb=evaluate_bpb(model,val_loader_factory(),EVAL_STEPS,token_bytes); model.train()
        best_bpb=min(best_bpb,bpb); val_bpbs.append(bpb); val_steps.append(step)
        print(f'[{RUN_TAG}] [VAL] step {step}: bpb={bpb:.4f} (dense {REF_DENSE}, lut {REF_LUT})')
        csv_w.writerow([step,f'{ema:.6f}',f'{bpb:.6f}']); csv_f.flush()
csv_f.close()
elapsed=time.time()-t0

diag_out={}
if DIAG:
    import numpy as np
    hc_np=hit_count.cpu().numpy(); ever=hc_np>0; ge2=hc_np>=2
    mean_gap=(sum_gap.cpu().numpy()[ge2]/(hc_np[ge2]-1)); maxg=max_gap.cpu().numpy()[ge2]
    def pct(a):
        a=np.asarray(a,dtype=float); return {'mean':round(float(a.mean()),3),'median':round(float(np.median(a)),3),
            'p90':round(float(np.percentile(a,90)),3),'p99':round(float(np.percentile(a,99)),3),'max':round(float(a.max()),3)}
    # frequent vs tail by hit_count among ever-hit cells
    hc_ever=hc_np[ever]; order=np.argsort(hc_ever); n=len(hc_ever)
    tail_idx=np.where(ever)[0][order[:max(1,n//10)]]; freq_idx=np.where(ever)[0][order[-max(1,n//10):]]
    def gap_of(idxs):
        gg=[];
        for i in idxs:
            if hc_np[i]>=2: gg.append(sum_gap[i].item()/(hc_np[i]-1))
        return pct(gg) if gg else {}
    gaps_pct=pct(mean_gap); maxg_pct=pct(maxg)
    def decay(gap,b): return round(float(b**gap),4)
    diag_out={'total_cells':TOTAL_CELLS,'cells_ever_hit':int(ever.sum()),'cells_never_hit':int((~ever).sum()),
        'distinct_cells_per_step_mean':round(float(np.mean(distinct_per_step)),1),
        'distinct_cells_per_step_frac':round(float(np.mean(distinct_per_step))/TOTAL_CELLS,4),
        'per_cell_mean_gap':gaps_pct,'per_cell_max_gap':maxg_pct,
        'mean_gap_frequent_decile':gap_of(freq_idx),'mean_gap_tail_decile':gap_of(tail_idx),
        'momentum_decay_at_gap':{
            'median_gap':gaps_pct['median'],'p99_gap':gaps_pct['p99'],'max_gap':maxg_pct['max'],
            'beta1_0.9__at_median':decay(gaps_pct['median'],0.9),'beta1_0.9__at_p99':decay(gaps_pct['p99'],0.9),'beta1_0.9__at_max':decay(gaps_pct['max'],0.9),
            'beta2_0.95(recipe)__at_median':decay(gaps_pct['median'],0.95),'beta2_0.95(recipe)__at_p99':decay(gaps_pct['p99'],0.95),'beta2_0.95(recipe)__at_max':decay(maxg_pct['max'],0.95),
            'beta2_0.999(task)__at_p99':decay(gaps_pct['p99'],0.999),'beta2_0.999(task)__at_max':decay(maxg_pct['max'],0.999)},
        'wd_note':'exp_n_0052 puts LUT tables in the wd=0 nodecay group -> ZERO shrink between hits in the actual recipe',
        'hypothetical_wd0.1_shrink':{'per_step_at_peakLR':round(LR*0.1,7),
            'between_hits_at_p99gap':round(1-(1-LR*0.1)**gaps_pct['p99'],6),'between_hits_at_maxgap':round(1-(1-LR*0.1)**maxg_pct['max'],6)}}
    json.dump(diag_out,open(os.path.join(EXP_DIR,'diagnostic_gaps.json'),'w'),indent=2)
    print('[DIAG]',json.dumps(diag_out,indent=2))

summary={'exp_name':cfg['exp_name'],'opt_mode':OPT_MODE,'n_steps':N_STEPS,'betas':[B1,B2],
         'best_val_bpb':best_bpb,'final_val_bpb':val_bpbs[-1] if val_bpbs else None,
         'refs':{'tied_dense':REF_DENSE,'e2e_lut':REF_LUT},'diagnostic':diag_out if DIAG else None,
         'total_params':total_params,'training_time_hours':round(elapsed/3600,3)}
json.dump(summary,open(os.path.join(EXP_DIR,f'summary_{RUN_TAG}.json'),'w'),indent=2)
print(f'[{RUN_TAG}] DONE best_bpb={best_bpb:.4f} final={summary["final_val_bpb"]}')

# combined plot if both modes present
import glob as _glob
def load_csv(p):
    import csv as _c; rows=list(_c.DictReader(open(p)))
    return [(int(r['step']),float(r['val_bpb'])) for r in rows if r['val_bpb']]
files=sorted(_glob.glob(os.path.join(EXP_DIR,'metrics_*.csv')))
if len(files)>=2:
    plt.figure(figsize=(9,5))
    for p in files:
        tag=os.path.basename(p)[len('metrics_'):-len('.csv')]; pts=load_csv(p)
        if pts: plt.plot([s for s,_ in pts],[v for _,v in pts],'o-',ms=3,label=tag)
    plt.axhline(REF_DENSE,ls='--',c='k',label=f'tied dense {REF_DENSE}'); plt.axhline(REF_LUT,ls='--',c='gray',label=f'e2e LUT {REF_LUT}')
    plt.xlabel('step'); plt.ylabel('val bpb'); plt.title('exp_n_0073 sparse/lazy-Adam vs dense-AdamW (LUT tables), equal budget')
    plt.legend(fontsize=8); plt.grid(True,alpha=0.3); plt.tight_layout(); plt.savefig(os.path.join(EXP_DIR,'compare_bpb.png'),dpi=120); plt.close()
    print('wrote compare_bpb.png')
