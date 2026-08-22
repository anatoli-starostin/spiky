"""exp_n_0075 — Issue #108 diagnostic: per-cell hit-gap distribution for the LUT
tables, and whether rarely-hit cells are the underperformers (justifying a
per-cell-hit optimizer or not).

Approach (honest & simple): LOAD the trained exp_n_0052 checkpoint (16k, near-
converged, so routing is ~stationary) and run a WINDOW of real steps at the cosine-end
LR (3e-5, ~frozen). Over the window, for each of the 196,608 table cells record:
  - hit stats: #steps hit, fraction of steps hit, gaps between consecutive hits
    (median / p90 / p99 / max).
  - token load: total tokens routed to the cell (embedding_bag is mode=sum, so the raw
    row gradient scales with token count — must normalize).
  - convergence proxy: per-cell residual gradient PER TOKEN = sum(row grad-norm) /
    sum(tokens). Normalizing by token count removes the frequency confound (a frequent
    cell has a bigger raw grad just from summing more tokens). Larger per-token residual
    = the cell's rows are less well-fit for the tokens landing on it.
Then bucket cells by hit-fraction (every-step / frequent / rare / very-rare) and test
whether the rare buckets have systematically larger per-token residual gradient.

Cell indexing matches exp_n_0072/0073: per LUT, d = zf[:,anchor_a]-zf[:,anchor_b];
idx = sign-pack in [0,64); global cell = block*32768 + table*64 + idx. No shared-src edits.
"""
import sys, os, json, math
try: sys.stdout.reconfigure(line_buffering=True)
except Exception: pass
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

NANOCHAT_ROOT = os.environ.get('NANOCHAT_ROOT', os.path.expanduser('~/projects/nanochat'))
if NANOCHAT_ROOT not in sys.path: sys.path.insert(0, NANOCHAT_ROOT)
from nanochat.common import get_base_dir
from nanochat.tokenizer import RustBPETokenizer
from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit
from spiky.lutorch.fast_multi_head_lut import FastMultiHeadLut
from spiky.lutorch.compression_mhl import CompressionMultiHeadLUT

EXP_DIR = os.path.dirname(os.path.abspath(__file__)); HFF = os.path.dirname(EXP_DIR)
CKPT = os.path.join(HFF, 'exp_n_0052_H8_d48_tph64_batched_control_16k', 'checkpoint.pt')
C052 = json.load(open(os.path.join(HFF, 'exp_n_0052_H8_d48_tph64_batched_control_16k', 'config.json')))
DEV='cuda'; torch.manual_seed(C052['random_seed'])
N_EMBD,N_HEAD,SEQ,DBS,DEPTH,VOCAB = C052['n_embd'],C052['n_head'],C052['seq_len'],C052['device_batch_size'],C052['depth'],C052['tokenizer_vocab_size']
DIN,DOUT = C052['lut_inner_in_dim'],C052['lut_inner_out_dim']
NAP,TPH,HEADS = C052['lut_n_anchor_pairs'],C052['lut_tables_per_head'],C052['lut_n_heads']
WINDOW = int(os.environ.get('WINDOW','500')); LR = 3e-5
if os.environ.get('SMOKE'): WINDOW=8; print('*** SMOKE ***')

base=get_base_dir(); tok=RustBPETokenizer.from_directory(os.path.join(base,'tokenizer')); assert tok.get_vocab_size()==VOCAB
loader=tokenizing_distributed_data_loader_bos_bestfit(tok,DBS,SEQ,split='train',device=DEV)

class Rope(nn.Module):
    def __init__(s,hd,msl,base=10000.0):
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
    return CompressionMultiHeadLUT(input_dim=N_EMBD,output_dim=N_EMBD,inner_in_dim=DIN,inner_out_dim=DOUT,nap=NAP,
        tph=TPH,n_heads=HEADS,joint_head_compression=C052['lut_joint_head_compression'],
        batched_multi_head_input=C052['lut_batched_multi_head_input'],forward_mode=C052['lut_forward_mode'],
        use_bf16=C052['lut_use_bf16'],initial_weights_noise=C052['lut_init_weights_noise'],
        learnable_temps=C052['lut_learnable_temps'],random_seed=C052['lut_base_seed']+seed)
class Block(nn.Module):
    def __init__(s,d,h,i): super().__init__(); s.ln1=nn.LayerNorm(d); s.attn=Attn(d,h); s.ln2=nn.LayerNorm(d); s.ffn=mk_cmhl(i)
    def forward(s,x,c,sn): x=x+s.attn(s.ln1(x),c,sn); h=s.ln2(x); B,T,C=h.shape; return x+s.ffn(h.reshape(B*T,C)).reshape(B,T,C).to(h.dtype)
class GPT(nn.Module):
    def __init__(s):
        super().__init__(); s.tok_emb=nn.Embedding(VOCAB,N_EMBD); s.rope=Rope(N_EMBD//N_HEAD,SEQ)
        s.blocks=nn.ModuleList([Block(N_EMBD,N_HEAD,i) for i in range(DEPTH)]); s.ln_f=nn.LayerNorm(N_EMBD); s.head=nn.Linear(N_EMBD,VOCAB,bias=False)
        if C052['tie_unembedder']: s.head.weight=s.tok_emb.weight
    def forward(s,idx,targets=None):
        x=s.tok_emb(idx)
        for b in s.blocks: x=b(x,s.rope.cos,s.rope.sin)
        logits=s.head(s.ln_f(x))
        if targets is not None: return F.cross_entropy(logits.view(-1,logits.size(-1)),targets.view(-1),ignore_index=-1)
        return logits

model=GPT().to(DEV)
mi,un=model.load_state_dict(torch.load(CKPT,map_location=DEV),strict=False)
print(f'loaded exp_n_0052 ckpt: missing={len(mi)} unexpected={len(un)}')
lut_mods=[b.ffn.lut_batched for b in model.blocks]
NTAB=lut_mods[0].soft_anchor_a_long.shape[0]; TDIM=lut_mods[0].table_dim   # 512, 64
PER=NTAB*TDIM; TOTAL=DEPTH*PER
print(f'cells: {DEPTH} blocks x {NTAB} tables x {TDIM} rows = {TOTAL:,}')
opt=torch.optim.AdamW([m.weights for m in lut_mods]+[p for n,p in model.named_parameters() if 'lut_batched.weights' not in n and p.requires_grad],
                      lr=LR,betas=(0.9,0.95),weight_decay=0.0)

hit_count=torch.zeros(TOTAL,dtype=torch.int64,device=DEV); last_hit=torch.zeros(TOTAL,dtype=torch.int64,device=DEV)
sum_gap=torch.zeros(TOTAL,dtype=torch.int64,device=DEV); max_gap=torch.zeros(TOTAL,dtype=torch.int64,device=DEV)
sum_tokens=torch.zeros(TOTAL,dtype=torch.int64,device=DEV); sum_gradnorm=torch.zeros(TOTAL,dtype=torch.float64,device=DEV)

model.train()
for step in range(1,WINDOW+1):
    caps={}; handles=[]
    for bi,blk in enumerate(model.blocks):
        def mk(bi):
            def hook(m,inp,out): caps[bi]=inp[0].detach()
            return hook
        handles.append(blk.ffn.lut_batched.register_forward_hook(mk(bi)))
    opt.zero_grad(set_to_none=True); x,y=next(loader); loss=model(x,y); loss.backward()
    for h in handles: h.remove()
    for bi,blk in enumerate(model.blocks):
        m=blk.ffn.lut_batched; zf=caps[bi]
        if zf.dim()==3: zf=zf.reshape(zf.shape[0],-1)
        d=zf[:,m.soft_anchor_a_long]-zf[:,m.soft_anchor_b_long]
        idx=((d>0).long()*m.soft_powers.view(1,1,-1)).sum(-1)        # [N,ntab]
        tab=torch.arange(NTAB,device=DEV).view(1,-1)
        local=(tab*TDIM+idx).reshape(-1)                            # per (token,table) local cell id
        counts=torch.bincount(local,minlength=PER)                  # tokens per cell this step
        rn=m.weights.grad.norm(dim=-1).reshape(-1).double()         # [PER] row grad-norm
        goff=bi*PER
        sum_tokens[goff:goff+PER]+=counts; sum_gradnorm[goff:goff+PER]+=rn
        hit_local=(counts>0); gidx=(goff+torch.nonzero(hit_local,as_tuple=True)[0])
        prev=last_hit[gidx]; mask=prev>0; gap=step-prev
        if mask.any():
            sum_gap.index_add_(0,gidx[mask],gap[mask]); max_gap[gidx[mask]]=torch.maximum(max_gap[gidx[mask]],gap[mask])
        hit_count[gidx]+=1; last_hit[gidx]=step
    opt.step()
    if step%100==0 or step==1: print(f'step {step}/{WINDOW} loss={loss.item():.4f}')

# ---- analysis ----
hc=hit_count.cpu().numpy(); frac=hc/WINDOW; tokens=sum_tokens.cpu().numpy(); gn=sum_gradnorm.cpu().numpy()
ever=hc>0; ge2=hc>=2
mean_gap=sum_gap.cpu().numpy()[ge2]/(hc[ge2]-1); maxg=max_gap.cpu().numpy()[ever]
resid_per_tok=np.where(tokens>0, gn/np.maximum(tokens,1), np.nan)   # per-cell per-token residual grad
def pct(a):
    a=np.asarray(a,float); a=a[~np.isnan(a)]
    return {'mean':round(float(a.mean()),4),'median':round(float(np.median(a)),4),'p90':round(float(np.percentile(a,90)),4),
            'p99':round(float(np.percentile(a,99)),4),'max':round(float(a.max()),4)} if len(a) else {}
# hit-fraction buckets
buckets={'every_step(>0.99)':(frac>0.99),'frequent(0.5-0.99)':((frac>0.5)&(frac<=0.99)),
         'rare(0.05-0.5)':((frac>0.05)&(frac<=0.5)),'very_rare(0<..<=0.05)':((frac>0)&(frac<=0.05)),'never(0)':(frac==0)}
bucket_stats={}
for name,msk in buckets.items():
    n=int(msk.sum());
    if n==0: bucket_stats[name]={'n':0}; continue
    rpt=resid_per_tok[msk]; rpt=rpt[~np.isnan(rpt)]
    bucket_stats[name]={'n':n,'frac_of_cells':round(n/TOTAL,4),'mean_hit_frac':round(float(frac[msk].mean()),4),
        'mean_tokens_over_window':round(float(tokens[msk].mean()),1),
        'resid_per_token_mean':round(float(rpt.mean()),5) if len(rpt) else None,
        'resid_per_token_median':round(float(np.median(rpt)),5) if len(rpt) else None}
out={'window_steps':WINDOW,'lr':LR,'total_cells':TOTAL,'note':'measured at the exp_n_0052 checkpoint (near-converged); LR=3e-5 (~frozen)',
     'cells_ever_hit':int(ever.sum()),'cells_never_hit':int((~ever).sum()),
     'frac_hit_every_step':round(float((frac>0.99).mean()),4),'frac_rare_le0.5':round(float(((frac>0)&(frac<=0.5)).mean()),4),
     'frac_very_rare_le0.05':round(float(((frac>0)&(frac<=0.05)).mean()),4),
     'per_cell_hit_fraction':pct(frac[ever]),'per_cell_mean_gap':pct(mean_gap),'per_cell_max_gap':pct(maxg),
     'resid_per_token_overall':pct(resid_per_tok),'buckets':bucket_stats}
json.dump(out,open(os.path.join(EXP_DIR,'cell_hit_gap_diag.json'),'w'),indent=2)
print(json.dumps(out,indent=2))

# plots
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(13,5))
ax1.hist(frac[ever],bins=50,color='tab:blue'); ax1.set(xlabel='per-cell hit fraction (steps hit / window)',ylabel='# cells',title=f'Hit-fraction histogram ({WINDOW} steps)'); ax1.grid(True,alpha=0.3)
names=[k for k in buckets if bucket_stats[k].get('n',0)>0 and bucket_stats[k].get('resid_per_token_median') is not None]
vals=[bucket_stats[k]['resid_per_token_median'] for k in names]; ns=[bucket_stats[k]['n'] for k in names]
ax2.bar(range(len(names)),vals,color='tab:red'); ax2.set_xticks(range(len(names))); ax2.set_xticklabels([n.split('(')[0] for n in names],rotation=20,ha='right')
ax2.set(ylabel='median residual grad / token',title='Under-convergence vs hit-frequency bucket')
for i,(v,c) in enumerate(zip(vals,ns)): ax2.text(i,v,f'n={c}',ha='center',va='bottom',fontsize=7)
ax2.grid(True,alpha=0.3); plt.tight_layout(); plt.savefig(os.path.join(EXP_DIR,'cell_hit_gap_diag.png'),dpi=120); plt.close()
print('=== DONE ===')
