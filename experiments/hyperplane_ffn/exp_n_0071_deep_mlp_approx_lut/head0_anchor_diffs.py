"""exp_n_0071 (head-0, EXACT ANCHOR diffs) — hand the MLP the exact 384 signed
differences the trained head actually routes on (not all 1128 pairs). Does that
push R2 toward 1.0?

Head 0 of block0.ffn.lut_batched uses tables 0..63 (tph=64) of the 512-table
batched module; each table has nap=6 anchor sign-tests -> 64*6=384 anchor pairs
(anchor_a, anchor_b), indices in [0,48) into the head-0 inner input. We read those
exact index pairs from the checkpoint buffers soft_anchor_a_long/soft_anchor_b_long
and build a fixed (non-learned) feature map:
    x[.,48] --exact-anchor-diffs--> [.,384] -> Linear(384->4096) -> GELU -> Linear(4096->48)
bias=True, no norm, no residual, transform has no learnable params. Same target/
setup as prior head-0 runs (block0 head0 of exp_n_0052, input z[:,0,:], target
out[:,0,:], ~320k tokens, 80/20 val, per-element MSE, val R2=1-vmse/var, AdamW+
cosine+5% warmup+grad-clip, batch 8192, 40k steps, val R2 every 2k). No shared edits.
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
TPH, NAP = cfg['lut_tables_per_head'], cfg['lut_n_anchor_pairs']
HID, FIT_STEPS, EVAL_EVERY = 4096, 40000, 2000
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

# --- extract head-0 exact anchor pairs from the trained batched module ---
lb = model.blocks[0].ffn.lut_batched
A = lb.soft_anchor_a_long[:TPH].reshape(-1).to(DEV)   # tables 0..63 -> [384]
B_ = lb.soft_anchor_b_long[:TPH].reshape(-1).to(DEV)  # [384]
NFEAT = A.numel()
assert NFEAT == TPH*NAP == 384, NFEAT
assert int(A.max()) < DIN and int(B_.max()) < DIN, (int(A.max()), int(B_.max()))   # head-0 anchors index [0,48)
print(f'head-0 exact anchor pairs: {NFEAT} (tph={TPH} x nap={NAP}); a in [{int(A.min())},{int(A.max())}], b in [{int(B_.min())},{int(B_.max())}]')
def anchor_diffs(x): return x[:, A] - x[:, B_]   # [N,48] -> [N,384], the exact routing diffs

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

def lr_at(st):
    w=int(FIT_WARM*FIT_STEPS)
    if st<w: return st/max(w,1)
    p=(st-w)/max(FIT_STEPS-w,1); return 0.05+0.95*0.5*(1+math.cos(math.pi*p))
nv=int(X.shape[0]*VAL_FRAC); Xtr,Ytr,Xv,Yv=X[nv:],Y[nv:],X[:nv],Y[:nv]; var=Yv.var(0).mean().item(); ntr=Xtr.shape[0]
Fv=anchor_diffs(Xv)

net=nn.Sequential(nn.Linear(NFEAT,HID),nn.GELU(),nn.Linear(HID,DOUT)).to(DEV)
opt=torch.optim.AdamW(net.parameters(),lr=FIT_LR,weight_decay=0.0,betas=(0.9,0.95)); g=torch.Generator(device=DEV).manual_seed(0)
traj=[]; print(f'--- anchor-diffs {DIN}->{NFEAT}->{HID}->{DOUT} params={sum(p.numel() for p in net.parameters()):,} | {FIT_STEPS} steps ---')
net.train()
for st in range(1,FIT_STEPS+1):
    for gp in opt.param_groups: gp['lr']=FIT_LR*lr_at(st)
    idx=torch.randint(0,ntr,(min(FIT_BATCH,ntr),),device=DEV,generator=g)
    loss=F.mse_loss(net(anchor_diffs(Xtr[idx])),Ytr[idx]); opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(net.parameters(),1.0); opt.step()
    if st%EVAL_EVERY==0 or st==FIT_STEPS:
        net.eval()
        with torch.no_grad(): vm=F.mse_loss(net(Fv),Yv).item()
        net.train(); r2=1-vm/(var+1e-12); traj.append({'step':st,'val_mse':round(vm,6),'r2':round(r2,5)})
        print(f'   step {st:6d} | val_mse={vm:.6f} R2={r2:.5f}')
final=traj[-1]; peak=max(traj,key=lambda t:t['r2'])

plt.figure(figsize=(9,5))
plt.plot([t['step'] for t in traj],[t['r2'] for t in traj],'o-',c='tab:blue',label='EXACT anchor-diffs (384) -> Linear(4096)')
plt.axhline(0.7884, ls='--', c='tab:red', label='deep depth3xw32 (H1536) 0.7884')
plt.axhline(0.780, ls='--', c='tab:purple', label='funnel 4096->512 0.780 peak')
plt.axhline(0.758, ls='--', c='tab:green', label='all-1128 dominance 0.758')
plt.axhline(1.0, ls=':', c='green', alpha=0.5)
plt.xlabel('training step'); plt.ylabel('val R2 (approx head0)'); plt.title('head0: EXACT anchor-diffs (384) -> MLP — R2 vs step (40k)')
plt.legend(fontsize=8); plt.grid(True,alpha=0.3); plt.tight_layout(); plt.savefig(os.path.join(EXP_DIR,'head0_anchor_diffs_curves.png'),dpi=120); plt.close()

summary={'exp_name':'exp_n_0071_head0_anchor_diffs_384_4096_40k','target':'exp_n_0052 block0 head0 (48->48)',
         'arch':f'exact-anchor-diffs({DIN}->{NFEAT}) -> Linear({NFEAT}->{HID})->GELU->Linear({HID}->{DOUT})','fit_steps':FIT_STEPS,
         'n_anchor_pairs':NFEAT,'refs':{'shallow_depth2':0.71,'wide_H4096':0.752,'all1128_dominance':0.758,'funnel_4096_512_peak':0.780,'deep_depth3_w32_H1536':0.7884},
         'final_r2':final['r2'],'peak_r2':peak['r2'],'peak_step':peak['step'],'final_val_mse':final['val_mse'],'trajectory':traj}
json.dump(summary,open(os.path.join(EXP_DIR,'head0_anchor_diffs_summary.json'),'w'),indent=2)
print('\n=== DONE ==='); print(f"final R2={final['r2']} peak={peak['r2']}@{peak['step']} | refs: 1128-dom 0.758, funnel 0.780, depth3xw32 0.7884")
