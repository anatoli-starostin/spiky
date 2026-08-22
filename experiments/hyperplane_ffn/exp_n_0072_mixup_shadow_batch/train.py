"""exp_n_0072 — "mixup shadow-batch" gradient densification for the LUT model.

Base recipe = exp_n_0052 (CompressionMultiHeadLUT FFN, 6L/384/H8/d48/tph64/nap6,
tied dense, batched hard LUT). Reference: tied-dense val_bpb 1.196646; e2e LUT 1.2286.

IDEA: at FIXED real batch / real-token budget, add an equal-size SHADOW batch each
step built by convex-interpolating existing real points (mixup), and fold its loss
into total = real_loss + lambda*shadow_loss. The interpolated points route to
intermediate LUT cells and near-boundary anchors, densifying gradient WITHOUT adding
real tokens.

INTERPOLATION LEVEL: token-EMBEDDING level (manifold mixup at the embedding), chosen
because it is the single cleanest interception that flows through the entire existing
forward (attention + all 6 blocks' LUTs) and produces genuinely interpolated hidden
states at every LUT — maximizing intermediate-cell coverage. Per step: pair the batch
with a random permutation, draw a per-sequence alpha~Beta(a,a) (a=mixup_alpha, near
the endpoints so mixing stays mild and doesn't blunt the sharp routing), form
  e_mix = alpha*e + (1-alpha)*e[perm]
run the transformer from e_mix, and use the mixup CE identity
  shadow_loss = mean_s[ alpha_s*CE(logits_mix[s], y[s]) + (1-alpha_s)*CE(logits_mix[s], y[perm][s]) ].

RUN MODE: env RUN_TAG in {baseline, mixup} selects real-only vs real+shadow at the
SAME n_steps / real batch (equal real-token budget). Writes metrics_<tag>.csv and
summary_<tag>.json; when both exist, writes a combined comparison plot.

INSTRUMENTATION (every routing_probe_every steps, no_grad probe batch):
  - val_bpb (primary).
  - shadow off-distribution fraction: of distinct (block,table,cell) routed by the
    shadow batch, the fraction NOT routed by the real batch that step.
  - gradient coverage: # distinct LUT cells that receive gradient — real-only vs
    real+shadow (union) — out of the 6*512*64 total cells.

Only touches this exp folder; does NOT modify fast_multi_head_lut.py / compression_mhl.py.
"""
import sys, os, json, math, time, csv
try: sys.stdout.reconfigure(line_buffering=True)
except Exception: pass
import torch, torch.nn as nn, torch.nn.functional as F
from torch.distributions import Beta
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

RUN_TAG = os.environ.get('RUN_TAG', 'mixup')   # 'baseline' | 'mixup'
MIXUP = (RUN_TAG == 'mixup')
DEPTH, N_EMBD, N_HEAD, SEQ_LEN = cfg['depth'], cfg['n_embd'], cfg['n_head'], cfg['seq_len']
DEVICE_BS, TOTAL_BS, N_STEPS = cfg['device_batch_size'], cfg['total_batch_size'], cfg['n_steps']
LR, WD, WARMUP_FRAC = cfg['lr'], cfg['weight_decay'], cfg['lr_warmup_fraction']
EVAL_EVERY, EVAL_STEPS = cfg['eval_every'], cfg['eval_steps']
TIE = bool(cfg['tie_unembedder'])
LUT_IN, LUT_OUT = cfg['lut_inner_in_dim'], cfg['lut_inner_out_dim']
LUT_NAP, LUT_TPH, LUT_HEADS = cfg['lut_n_anchor_pairs'], cfg['lut_tables_per_head'], cfg['lut_n_heads']
MIX_ALPHA, MIX_LAMBDA = cfg['mixup_alpha'], cfg['mixup_lambda']
PROBE_EVERY = cfg['routing_probe_every']
REF_DENSE, REF_LUT = cfg['tied_dense_ref_bpb'], cfg['e2e_lut_ref_bpb']

if os.environ.get('SMOKE_STEPS'):
    N_STEPS = int(os.environ['SMOKE_STEPS']); EVAL_EVERY = max(1, N_STEPS); PROBE_EVERY = max(1, N_STEPS//2)
    print(f'*** SMOKE_STEPS={N_STEPS} ({RUN_TAG}) ***')

BASE_DIR = get_base_dir(); tokenizer = RustBPETokenizer.from_directory(os.path.join(BASE_DIR, 'tokenizer'))
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
    def embed(self, idx): return self.tok_emb(idx)
    def forward_from_emb(self, e):
        x=e
        for block in self.blocks: x=block(x,self.rope.cos,self.rope.sin)
        return self.head(self.ln_f(x))
    def forward(self, idx, targets=None, loss_reduction='mean'):
        logits=self.forward_from_emb(self.embed(idx))
        if targets is not None:
            return F.cross_entropy(logits.view(-1,logits.size(-1)), targets.view(-1), reduction=loss_reduction, ignore_index=-1)
        return logits

def setup_optimizer(model, lr, weight_decay):
    lut_ids={id(p) for m in model.modules() if isinstance(m,FastMultiHeadLut) for p in m.parameters(recurse=False)}
    decay,nodecay=[],[]
    for p in model.parameters():
        if not p.requires_grad: continue
        (nodecay if (id(p) in lut_ids or p.ndim<2) else decay).append(p)
    groups=[dict(params=decay,lr=lr,betas=(0.9,0.95),eps=1e-8,weight_decay=weight_decay),
            dict(params=nodecay,lr=lr,betas=(0.9,0.95),eps=1e-8,weight_decay=0.0)]
    opt=torch.optim.AdamW(groups)
    for g in opt.param_groups: g['initial_lr']=g['lr']
    return opt
def get_lr_scale(step,n_steps,warmup_frac):
    w=int(warmup_frac*n_steps)
    if step<w: return step/max(w,1)
    p=(step-w)/max(n_steps-w,1); return 0.1+0.9*0.5*(1+math.cos(math.pi*p))

# --- mixup shadow batch (embedding level) ---
def make_shadow(model, x, y):
    """Return (e_mix, y, y_perm, alpha[B]) for the shadow batch."""
    e = model.embed(x)                          # [B,T,C]
    B = x.shape[0]
    perm = torch.randperm(B, device=x.device)
    alpha = Beta(MIX_ALPHA, MIX_ALPHA).sample((B,)).to(x.device)   # per-sequence
    a = alpha.view(B,1,1)
    e_mix = a*e + (1.0-a)*e[perm]
    return e_mix, y, y[perm], alpha
def shadow_loss_fn(model, x, y):
    e_mix, y_a, y_b, alpha = make_shadow(model, x, y)
    logits = model.forward_from_emb(e_mix)      # [B,T,V]
    B,T,V = logits.shape
    ce_a = F.cross_entropy(logits.view(-1,V), y_a.reshape(-1), ignore_index=-1, reduction='none').view(B,T).mean(1)
    ce_b = F.cross_entropy(logits.view(-1,V), y_b.reshape(-1), ignore_index=-1, reduction='none').view(B,T).mean(1)
    return (alpha*ce_a + (1.0-alpha)*ce_b).mean()

# --- routing instrumentation ---
def cells_hit(zf, m, bi):
    if zf.dim()==3: zf=zf.reshape(zf.shape[0],-1)
    d = zf[:, m.soft_anchor_a_long] - zf[:, m.soft_anchor_b_long]   # [N,ntab,nap]
    bits=(d>0).long(); idx=(bits*m.soft_powers.view(1,1,-1)).sum(-1)  # [N,ntab]
    ntab=idx.shape[1]; tab=torch.arange(ntab,device=idx.device).view(1,-1)
    g=((bi*ntab+tab)*m.table_dim + idx).reshape(-1)
    return torch.unique(g)
@torch.no_grad()
def routing_stats(model, x, y):
    caps={}; handles=[]
    for bi,blk in enumerate(model.blocks):
        def mk(bi):
            def hook(mod,inp,out): caps[bi]=inp[0].detach()
            return hook
        handles.append(blk.ffn.lut_batched.register_forward_hook(mk(bi)))
    model.eval()
    e=model.embed(x); model.forward_from_emb(e); caps_real={k:v for k,v in caps.items()}
    B=x.shape[0]; perm=torch.randperm(B,device=x.device); a=Beta(MIX_ALPHA,MIX_ALPHA).sample((B,)).to(x.device).view(B,1,1)
    e_mix=a*e+(1.0-a)*e[perm]; caps.clear(); model.forward_from_emb(e_mix); caps_shadow=dict(caps)
    for h in handles: h.remove()
    model.train()
    real=set(); shadow=set()
    for bi,blk in enumerate(model.blocks):
        m=blk.ffn.lut_batched
        real|=set(cells_hit(caps_real[bi],m,bi).tolist())
        shadow|=set(cells_hit(caps_shadow[bi],m,bi).tolist())
    total_cells = DEPTH * model.blocks[0].ffn.lut_batched.soft_anchor_a_long.shape[0] * model.blocks[0].ffn.lut_batched.table_dim
    off = len(shadow-real)/max(len(shadow),1)
    return {'real_cells':len(real),'shadow_cells':len(shadow),'union_cells':len(real|shadow),
            'total_cells':total_cells,'off_dist_frac':round(off,4),
            'cov_real_frac':round(len(real)/total_cells,4),'cov_union_frac':round(len(real|shadow)/total_cells,4)}

model = MinimalGPT(VOCAB_SIZE, N_EMBD, N_HEAD, DEPTH, SEQ_LEN).to(DEVICE)
total_params=sum(p.numel() for p in model.parameters())
print(f'[{RUN_TAG}] MinimalGPT depth={DEPTH} dim={N_EMBD} | params={total_params:,} | mixup={MIXUP} alpha={MIX_ALPHA} lambda={MIX_LAMBDA} | n_steps={N_STEPS}')
optimizer=setup_optimizer(model,LR,WD)
tokens_per_step=DEVICE_BS*SEQ_LEN; grad_accum=max(1,TOTAL_BS//tokens_per_step)
print(f'tokens/micro={tokens_per_step:,} grad_accum={grad_accum} eff_batch={grad_accum*tokens_per_step:,} real tokens/step')

csv_f=open(os.path.join(EXP_DIR,f'metrics_{RUN_TAG}.csv'),'w',newline=''); csv_w=csv.writer(csv_f)
csv_w.writerow(['step','train_loss','shadow_loss','val_bpb','off_dist_frac','cov_real_frac','cov_union_frac'])
val_bpbs, val_steps, ema, best_bpb, t0 = [], [], None, float('inf'), time.time()
last_probe={}
model.train()
for step in range(1,N_STEPS+1):
    lr_scale=get_lr_scale(step,N_STEPS,WARMUP_FRAC)
    for g in optimizer.param_groups: g['lr']=g['initial_lr']*lr_scale
    optimizer.zero_grad(set_to_none=True)
    accum_real=0.0; accum_shadow=0.0
    for _ in range(grad_accum):
        x,y=next(train_loader)
        real_loss=model(x,y)
        if MIXUP:
            sh=shadow_loss_fn(model,x,y)
            loss=real_loss + MIX_LAMBDA*sh; accum_shadow+=sh.item()/grad_accum
        else:
            loss=real_loss
        (loss/grad_accum).backward(); accum_real+=real_loss.item()/grad_accum
    torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); optimizer.step()
    ema=accum_real if ema is None else 0.99*ema+0.01*accum_real
    if step%100==0 or step==1:
        print(f'[{RUN_TAG}] step {step:5d} | real={ema:.4f} | shadow={accum_shadow:.4f} | lr={lr_scale*LR:.2e}')
    if step%PROBE_EVERY==0 or step==N_STEPS:
        last_probe=routing_stats(model,x,y)
        print(f'[{RUN_TAG}] probe step {step}: off_dist={last_probe["off_dist_frac"]} cov_real={last_probe["cov_real_frac"]} cov_union={last_probe["cov_union_frac"]} ({last_probe["real_cells"]}->{last_probe["union_cells"]}/{last_probe["total_cells"]})')
    if step%EVAL_EVERY==0 or step==N_STEPS:
        model.eval(); bpb=evaluate_bpb(model,val_loader_factory(),EVAL_STEPS,token_bytes); model.train()
        best_bpb=min(best_bpb,bpb); val_bpbs.append(bpb); val_steps.append(step)
        print(f'[{RUN_TAG}] [VAL] step {step}: bpb={bpb:.4f} (dense ref {REF_DENSE}, lut ref {REF_LUT})')
        csv_w.writerow([step,f'{ema:.6f}',f'{accum_shadow:.6f}',f'{bpb:.6f}',
                        last_probe.get('off_dist_frac',''),last_probe.get('cov_real_frac',''),last_probe.get('cov_union_frac','')]); csv_f.flush()
csv_f.close()
elapsed=time.time()-t0
summary={'exp_name':cfg['exp_name'],'run_tag':RUN_TAG,'mixup':MIXUP,'n_steps':N_STEPS,
         'mixup_alpha':MIX_ALPHA,'mixup_lambda':MIX_LAMBDA,'mixup_level':cfg['mixup_level'],
         'best_val_bpb':best_bpb,'final_val_bpb':val_bpbs[-1] if val_bpbs else None,
         'refs':{'tied_dense':REF_DENSE,'e2e_lut':REF_LUT},'last_routing_probe':last_probe,
         'total_params':total_params,'training_time_hours':round(elapsed/3600,3)}
json.dump(summary,open(os.path.join(EXP_DIR,f'summary_{RUN_TAG}.json'),'w'),indent=2)
print(f'[{RUN_TAG}] DONE best_bpb={best_bpb:.4f} final={summary["final_val_bpb"]}')

# combined plot when both runs present
def load_csv(tag):
    p=os.path.join(EXP_DIR,f'metrics_{tag}.csv')
    if not os.path.exists(p): return None
    import csv as _c; rows=list(_c.DictReader(open(p)))
    return [(int(r['step']),float(r['val_bpb'])) for r in rows if r['val_bpb']]
b=load_csv('baseline'); m=load_csv('mixup')
if b and m:
    plt.figure(figsize=(9,5))
    plt.plot([s for s,_ in b],[v for _,v in b],'o-',c='tab:blue',label='baseline (real only)')
    plt.plot([s for s,_ in m],[v for _,v in m],'o-',c='tab:red',label='mixup shadow-batch')
    plt.axhline(REF_DENSE,ls='--',c='k',label=f'tied dense {REF_DENSE}')
    plt.axhline(REF_LUT,ls='--',c='gray',label=f'e2e LUT {REF_LUT}')
    plt.xlabel('step'); plt.ylabel('val bpb'); plt.title('exp_n_0072 mixup shadow-batch vs baseline (equal real-token budget)')
    plt.legend(fontsize=8); plt.grid(True,alpha=0.3); plt.tight_layout(); plt.savefig(os.path.join(EXP_DIR,'compare_bpb.png'),dpi=120); plt.close()
    print('wrote compare_bpb.png')
