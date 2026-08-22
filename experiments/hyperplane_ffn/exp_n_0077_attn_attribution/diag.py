"""exp_n_0077 — Is the 1×→1.5×-batch LUT gain (val_bpb ~1.22 → 1.1969) mostly
ATTENTION or the LUT-FFN? Cross-model weight-graft causal attribution + attention-
pattern comparison. DIAGNOSTIC: no training — load two checkpoints, graft weight
groups, eval val_bpb on a fixed val set. No shared-src edits.

Checkpoints (same arch, 27,343,200 params: 6L/384/H6 attn, LUT H8/d48/tph64/nap6, tied):
  1x   : exp_n_0052_H8_d48_tph64_batched_control_16k/checkpoint.pt  (~1.22)
  1.5x : exp_n_0046_H8_d48_tph64_bs1p5x_16k/checkpoint.pt           (1.196862)

Weight groups (by state_dict key):
  attn = blocks.N.attn.{qkv,proj}.weight + blocks.N.ln1.{weight,bias}   (pre-attn norm)
  ffn  = blocks.N.ffn.*  + blocks.N.ln2.{weight,bias}                   (LUT+compress+temps+anchors + pre-ffn norm)
  other= tok_emb.weight, ln_f.{weight,bias}, head.weight
PART A: eval val_bpb for the 2x2 graft (+symmetric reverse grafts). PART B: per-head
attention entropy / distance / local vs long-range / BOS-sink on a fixed val batch.
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
from nanochat.tokenizer import RustBPETokenizer, get_token_bytes
from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit
from nanochat.loss_eval import evaluate_bpb
from spiky.lutorch.compression_mhl import CompressionMultiHeadLUT

EXP_DIR=os.path.dirname(os.path.abspath(__file__)); HFF=os.path.dirname(EXP_DIR)
CK1X=os.path.join(HFF,'exp_n_0052_H8_d48_tph64_batched_control_16k','checkpoint.pt')
CK15=os.path.join(HFF,'exp_n_0046_H8_d48_tph64_bs1p5x_16k','checkpoint.pt')
C=json.load(open(os.path.join(HFF,'exp_n_0052_H8_d48_tph64_batched_control_16k','config.json')))
DEV='cuda'; torch.manual_seed(C['random_seed'])
N_EMBD,N_HEAD,SEQ,VOCAB,DEPTH=C['n_embd'],C['n_head'],C['seq_len'],C['tokenizer_vocab_size'],C['depth']
DIN,DOUT=C['lut_inner_in_dim'],C['lut_inner_out_dim']
NAP,TPH,HEADS=C['lut_n_anchor_pairs'],C['lut_tables_per_head'],C['lut_n_heads']
EVAL_STEPS=int(os.environ.get('EVAL_STEPS','100')); EVAL_DBS=48
if os.environ.get('SMOKE'): EVAL_STEPS=3; print('*** SMOKE ***')

base=get_base_dir(); tok=RustBPETokenizer.from_directory(os.path.join(base,'tokenizer')); assert tok.get_vocab_size()==VOCAB
val_factory=lambda: tokenizing_distributed_data_loader_bos_bestfit(tok,EVAL_DBS,SEQ,split='val',device=DEV)
token_bytes=get_token_bytes(device=DEV)

class Rope(nn.Module):
    def __init__(s,hd,msl,base=10000.0):
        super().__init__(); inv=1.0/(base**(torch.arange(0,hd,2,dtype=torch.float32)/hd)); t=torch.arange(msl,dtype=torch.float32)
        e=torch.cat([torch.outer(t,inv)]*2,-1); s.register_buffer('cos',e.cos(),persistent=False); s.register_buffer('sin',e.sin(),persistent=False)
def rh(x): a,b=x.chunk(2,-1); return torch.cat([-b,a],-1)
def rope(q,k,c,s): c=c[None,None]; s=s[None,None]; return q*c+rh(q)*s,k*c+rh(k)*s
class Attn(nn.Module):
    def __init__(s,d,h): super().__init__(); s.n_head=h; s.qkv=nn.Linear(d,3*d,bias=False); s.proj=nn.Linear(d,d,bias=False); s.store=False; s.last=None
    def forward(s,x,c,sn):
        B,T,Cc=x.size(); q,k,v=s.qkv(x).split(Cc,2); hd=Cc//s.n_head
        q=q.view(B,T,s.n_head,hd).transpose(1,2); k=k.view(B,T,s.n_head,hd).transpose(1,2); v=v.view(B,T,s.n_head,hd).transpose(1,2)
        q,k=rope(q,k,c[:T],sn[:T])
        if s.store:
            sc=(q@k.transpose(-2,-1))/math.sqrt(hd)
            m=torch.triu(torch.ones(T,T,device=x.device,dtype=torch.bool),1); sc=sc.masked_fill(m[None,None],float('-inf'))
            a=torch.softmax(sc,-1); s.last=a.detach(); y=a@v
        else:
            y=F.scaled_dot_product_attention(q,k,v,is_causal=True)
        return s.proj(y.transpose(1,2).contiguous().view(B,T,Cc))
def mk_cmhl(seed):
    return CompressionMultiHeadLUT(input_dim=N_EMBD,output_dim=N_EMBD,inner_in_dim=DIN,inner_out_dim=DOUT,nap=NAP,tph=TPH,n_heads=HEADS,
        joint_head_compression=C['lut_joint_head_compression'],batched_multi_head_input=C['lut_batched_multi_head_input'],
        forward_mode=C['lut_forward_mode'],use_bf16=C['lut_use_bf16'],initial_weights_noise=C['lut_init_weights_noise'],
        learnable_temps=C['lut_learnable_temps'],random_seed=C['lut_base_seed']+seed)
class Block(nn.Module):
    def __init__(s,d,h,i): super().__init__(); s.ln1=nn.LayerNorm(d); s.attn=Attn(d,h); s.ln2=nn.LayerNorm(d); s.ffn=mk_cmhl(i)
    def forward(s,x,c,sn): x=x+s.attn(s.ln1(x),c,sn); h=s.ln2(x); B,T,Cc=h.shape; return x+s.ffn(h.reshape(B*T,Cc)).reshape(B,T,Cc).to(h.dtype)
class GPT(nn.Module):
    def __init__(s):
        super().__init__(); s.tok_emb=nn.Embedding(VOCAB,N_EMBD); s.rope=Rope(N_EMBD//N_HEAD,SEQ)
        s.blocks=nn.ModuleList([Block(N_EMBD,N_HEAD,i) for i in range(DEPTH)]); s.ln_f=nn.LayerNorm(N_EMBD); s.head=nn.Linear(N_EMBD,VOCAB,bias=False)
        if C['tie_unembedder']: s.head.weight=s.tok_emb.weight
    def get_device(s): return s.tok_emb.weight.device
    def forward(s,idx,targets=None,loss_reduction='mean'):
        x=s.tok_emb(idx)
        for b in s.blocks: x=b(x,s.rope.cos,s.rope.sin)
        lg=s.head(s.ln_f(x))
        if targets is not None: return F.cross_entropy(lg.view(-1,lg.size(-1)),targets.view(-1),reduction=loss_reduction,ignore_index=-1)
        return lg

sd1=torch.load(CK1X,map_location=DEV); sd15_loop=torch.load(CK15,map_location=DEV)
def to_batched(sd):
    """exp_n_0046 stores the FFN as a per-head loop (ffn.luts.0..7, each 64 tables,
    anchors in [0,48)); convert to the batched layout (ffn.lut_batched, 512 tables,
    head h's anchors offset by h*inner_in_dim into the flat 384-space). Forward-
    equivalent in hard mode (temps/constants left to the model's deterministic init)."""
    out={k:v for k,v in sd.items() if '.ffn.luts.' not in k}
    for L in range(DEPTH):
        pre=f'blocks.{L}.ffn'; W=[]; A=[]; B=[]
        for h in range(HEADS):
            lp=f'{pre}.luts.{h}'
            W.append(sd[f'{lp}.weights']); A.append(sd[f'{lp}.soft_anchor_a_long']+h*DIN); B.append(sd[f'{lp}.soft_anchor_b_long']+h*DIN)
        out[f'{pre}.lut_batched.weights']=torch.cat(W,0)
        out[f'{pre}.lut_batched.soft_anchor_a_long']=torch.cat(A,0)
        out[f'{pre}.lut_batched.soft_anchor_b_long']=torch.cat(B,0)
    return out
sd15=to_batched(sd15_loop)
print(f'converted exp_n_0046 loop->batched: {len([k for k in sd15 if "lut_batched" in k])} lut_batched keys')
def group(k):
    if '.attn.' in k or '.ln1.' in k: return 'attn'
    if '.ffn.' in k or '.ln2.' in k: return 'ffn'
    return 'other'
ATTN_KEYS=[k for k in sd1 if group(k)=='attn']; FFN_KEYS=[k for k in sd1 if group(k)=='ffn']
print(f'attn keys={len(ATTN_KEYS)} ffn keys={len(FFN_KEYS)} other={len(sd1)-len(ATTN_KEYS)-len(FFN_KEYS)}')

def build(base_sd, graft_sd, graft_group):
    """base_sd everywhere except keys in graft_group taken from graft_sd (fall back to
    base for keys the graft sd doesn't carry, i.e. the hard-forward-irrelevant temps/
    constants — so grafting the FFN moves tables+anchors+compress, not the temps)."""
    out={}
    for k in base_sd:
        out[k]=graft_sd.get(k, base_sd[k]) if group(k)==graft_group else base_sd[k]
    return out
SAFE_MISS=('rope','log_soft_score_temp','log_select_temp','soft_bit_matrix','soft_powers','_table_offset')
def evalmodel(state):
    m=GPT().to(DEV); mi,un=m.load_state_dict(state,strict=False); m.eval()
    bad=[k for k in mi if not any(s in k for s in SAFE_MISS)]
    assert not bad, f'unexpected missing keys: {bad[:5]}'
    assert not un, f'unexpected extra keys: {un[:5]}'
    with torch.no_grad(): bpb=evaluate_bpb(m,val_factory(),EVAL_STEPS,token_bytes)
    return float(bpb)

# ---- PART A: 2x2 graft + symmetric ----
runs={}
runs['1x_pure(1xA+1xF)']       = evalmodel(sd1)
runs['1.5x_pure(1.5A+1.5F)']   = evalmodel(sd15)
runs['1x_base_ATTN<-1.5x']     = evalmodel(build(sd1, sd15, 'attn'))
runs['1x_base_FFN<-1.5x']      = evalmodel(build(sd1, sd15, 'ffn'))
runs['1.5x_base_ATTN<-1x(sym)']= evalmodel(build(sd15, sd1, 'attn'))
runs['1.5x_base_FFN<-1x(sym)'] = evalmodel(build(sd15, sd1, 'ffn'))
for k,v in runs.items(): print(f'  {k:28s} val_bpb={v:.5f}')
b1=runs['1x_pure(1xA+1xF)']; b15=runs['1.5x_pure(1.5A+1.5F)']; gap=b1-b15
attnrec=(b1-runs['1x_base_ATTN<-1.5x'])/gap if gap else None
ffnrec =(b1-runs['1x_base_FFN<-1.5x'])/gap if gap else None
print(f'gap(1x-1.5x)={gap:.5f} | attn-graft recovers {attnrec:.1%} | ffn-graft recovers {ffnrec:.1%}')

# ---- PART B: attention patterns on a fixed val batch ----
def attn_metrics(state, xb, W=8):
    m=GPT().to(DEV); m.load_state_dict(state,strict=False); m.eval()
    for blk in m.blocks: blk.attn.store=True
    with torch.no_grad(): m(xb)
    T=xb.shape[1]; ii=torch.arange(T,device=DEV); dist_ij=(ii[:,None]-ii[None,:]).clamp(min=0).float()
    local_mask=((ii[:,None]-ii[None,:]).abs()<=W).float()
    out=[]
    for li,blk in enumerate(m.blocks):
        a=blk.attn.last  # [B,H,T,T]
        ent=(-(a.clamp_min(1e-9)*a.clamp_min(1e-9).log()).sum(-1))     # [B,H,T]
        dist=(a*dist_ij[None,None]).sum(-1)                            # [B,H,T]
        loc=(a*local_mask[None,None]).sum(-1)                          # [B,H,T]
        bos=a[...,0]                                                   # [B,H,T]
        valid=slice(1,None)  # skip query 0 (only attends to itself)
        for h in range(a.shape[1]):
            out.append({'layer':li,'head':h,
                'entropy':round(float(ent[:,h,valid].mean()),3),
                'distance':round(float(dist[:,h,valid].mean()),2),
                'local_frac':round(float(loc[:,h,valid].mean()),3),
                'bos_frac':round(float(bos[:,h,valid].mean()),3)})
    return out
vb=next(iter(val_factory())); xb=vb[0][:8]   # fixed 8-seq val batch
mA=attn_metrics(sd1,xb); mB=attn_metrics(sd15,xb)
def agg(ms):
    import numpy as _n
    return {k:round(float(_n.mean([r[k] for r in ms])),3) for k in ['entropy','distance','local_frac','bos_frac']}
aggA,aggB=agg(mA),agg(mB)
print('PART B aggregate (1x):',aggA); print('PART B aggregate (1.5x):',aggB)
# biggest per-head diffs
diffs=sorted([{'layer':a['layer'],'head':a['head'],
    'd_entropy':round(b['entropy']-a['entropy'],3),'d_distance':round(b['distance']-a['distance'],2),
    'd_local':round(b['local_frac']-a['local_frac'],3),'d_bos':round(b['bos_frac']-a['bos_frac'],3)}
    for a,b in zip(mA,mB)], key=lambda r:-abs(r['d_distance']))

out={'eval_steps':EVAL_STEPS,'eval_dbs':EVAL_DBS,
     'partA_val_bpb':{k:round(v,5) for k,v in runs.items()},
     'gap_1x_minus_1.5x':round(gap,5),'attn_graft_recovers_frac':round(attnrec,4),'ffn_graft_recovers_frac':round(ffnrec,4),
     'partB_aggregate':{'model_1x':aggA,'model_1.5x':aggB},'partB_top_distance_diffs':diffs[:8],
     'partB_per_head_1x':mA,'partB_per_head_1.5x':mB}
json.dump(out,open(os.path.join(EXP_DIR,'attn_attribution.json'),'w'),indent=2)

# plots
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(13,5))
names=list(runs.keys()); vals=[runs[n] for n in names]
cols=['tab:blue','tab:green','tab:orange','tab:red','navy','darkred']
ax1.barh(range(len(names)),vals,color=cols); ax1.set_yticks(range(len(names))); ax1.set_yticklabels(names,fontsize=7)
ax1.axvline(b1,ls='--',c='gray'); ax1.axvline(b15,ls='--',c='k'); ax1.set_xlim(min(vals)-0.005,max(vals)+0.005)
ax1.set(xlabel='val_bpb',title=f'PART A graft attribution (attn recovers {attnrec:.0%}, ffn {ffnrec:.0%})')
for i,v in enumerate(vals): ax1.text(v,i,f'{v:.4f}',va='center',fontsize=7)
xa=[r['distance'] for r in mA]; xb_=[r['distance'] for r in mB]
ax2.scatter(xa,xb_,s=18,c='tab:purple'); lim=[0,max(max(xa),max(xb_))*1.1]; ax2.plot(lim,lim,'k--',alpha=0.4)
ax2.set(xlabel='1x attn distance (per head)',ylabel='1.5x attn distance',title='Per-head attention look-back distance')
ax2.grid(True,alpha=0.3); plt.tight_layout(); plt.savefig(os.path.join(EXP_DIR,'attn_attribution.png'),dpi=120); plt.close()
print('=== DONE ==='); print(json.dumps({'partA':out['partA_val_bpb'],'attn_rec':out['attn_graft_recovers_frac'],'ffn_rec':out['ffn_graft_recovers_frac']},indent=2))
