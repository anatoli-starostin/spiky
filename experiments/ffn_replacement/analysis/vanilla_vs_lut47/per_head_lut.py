"""Per-head characterization of the LUT47 (H8/tph64) FFN LightMHL heads as maps R^48 -> R^48,
across all layers (0..5) and heads (0..7). Captures each head's input code slice, its top_n-blended
read output (before decompress), and the selected cell index per table (compile disabled + _pack_index
wrapped). Computes rank, codebook usage, discreteness, magnitude, sparsity per (layer,head)."""
import json, os, sys, types
import numpy as np, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT="/home/astarostin/projects/ffn_fix_wt"
sys.path.insert(0, ROOT+"/experiments/ffn_replacement/tools"); sys.path.insert(0, ROOT+"/src")
from model_build import build_model
from spiky.lutorch.light_multi_head_lut import LightMultiHeadLUT
from nanochat.common import get_base_dir
from nanochat.tokenizer import RustBPETokenizer
from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit
OUT=ROOT+"/experiments/ffn_replacement/analysis/vanilla_vs_lut47"
D=ROOT+"/experiments/ffn_replacement/lut_ablation/exp_n_abl_47_B48k_light_lmfrozeng_n2_tau0p5_tv10_tph64_h8_seed1_noquant_headdrop20"
DEV="cuda"; V=32768; B,T=16,256
torch.manual_seed(0)
c=json.load(open(D+"/config.json")); m=build_model(c,V,device=DEV)
m.load_state_dict(torch.load(D+"/checkpoint.pt",map_location=DEV),strict=False); m.eval()
L=len(m.blocks); luts=[b.ffn.lut_light for b in m.blocks]
H=luts[0].n_heads; TPH=luts[0].n_tables//H; NAP=int(np.log2(luts[0].table_size)); DIN=luts[0].input_dim; DOUT=luts[0].output_dim
print("DIMS: layers",L,"heads",H,"tph",TPH,"nap",NAP,"cells/table",2**NAP,"input_dim/head",DIN,"output_dim/head",DOUT)

cap_in=[None]*L; cap_out=[None]*L; cap_idx=[None]*L
for i,lt in enumerate(luts):
    lt._compile_enabled=False                       # force eager so _pack_index is a real python call
    orig=lt._pack_index
    def mk(i,orig):
        def wrapped(self,x_flat,d):
            r=orig(x_flat,d); cap_idx[i]=r.detach()   # [N, n_tables]
            return r
        return wrapped
    lt._pack_index=types.MethodType(mk(i,orig), lt)
    lt.register_forward_pre_hook(lambda mo,inp,i=i: cap_in.__setitem__(i, inp[0].detach()))
    lt.register_forward_hook(lambda mo,inp,out,i=i: cap_out.__setitem__(i,(out[0] if isinstance(out,tuple) else out).detach()))

tok=RustBPETokenizer.from_directory(get_base_dir()+"/tokenizer")
dl=tokenizing_distributed_data_loader_bos_bestfit(tok,B,T,split="val",device=DEV); x,y=next(dl)
with torch.no_grad(): m(x)

def effrank(M):   # participation ratio of singular values of centered [N,d]
    M=M.float()-M.float().mean(0,keepdim=True)
    s=torch.linalg.svdvals(M.cpu()).numpy()
    return float((s.sum()**2)/((s**2).sum()+1e-12)), s

orank=np.zeros((L,H)); irank=np.zeros((L,H)); onorm=np.zeros((L,H)); ospars=np.zeros((L,H))
distinct=np.zeros((L,H)); cell_entropy=np.zeros((L,H)); top10share=np.zeros((L,H)); discrete_frac=np.zeros((L,H))
spectra={}
for l in range(L):
    zin=cap_in[l]; out=cap_out[l]                      # [N,H,48]
    idx=cap_idx[l].view(-1,H,TPH)                      # [N,H,tph] cell address per table
    N=out.shape[0]
    for h in range(H):
        oh=out[:,h,:]; ih=zin[:,h,:]
        pr,s=effrank(oh); orank[l,h]=pr; spectra[(l,h)]=s
        irank[l,h]=effrank(ih)[0]
        onorm[l,h]=oh.norm(dim=-1).mean().item()
        ospars[l,h]=(oh.abs()<1e-3).float().mean().item()
        # codebook usage: (table, cell) selections for this head across N tokens
        ih_idx=idx[:,h,:]                              # [N,tph] cell address in [0,2^nap)
        pair=(torch.arange(TPH,device=DEV).view(1,-1)*(2**NAP)+ih_idx).reshape(-1)  # unique (table,cell)
        vals,counts=torch.unique(pair,return_counts=True)
        distinct[l,h]=len(vals)
        p=counts.float()/counts.sum(); cell_entropy[l,h]=float(-(p*p.clamp_min(1e-12).log2()).sum())
        top=torch.sort(counts,descending=True).values.float(); top10share[l,h]=float(top[:max(1,len(top)//10)].sum()/top.sum())
        # discreteness: fraction of per-token output variance explained by the top-1 cell identity of table0
        # (group the head output by the selected address of a representative table, ANOVA between/total)
        g=idx[:,h,0]                                   # address of table 0 per token
        tot=oh.var(0,unbiased=False).sum().item()+1e-9
        betw=0.0
        for gv in torch.unique(g):
            sub=oh[g==gv]
            if len(sub)>1: betw+=len(sub)*((sub.mean(0)-oh.mean(0))**2).sum().item()
        discrete_frac[l,h]=betw/(N*tot)

def hm(A,title,fn,cmap="viridis"):
    fig,ax=plt.subplots(figsize=(7,4)); im=ax.imshow(A,aspect="auto",cmap=cmap)
    ax.set_xlabel("head"); ax.set_ylabel("layer"); ax.set_title(title); plt.colorbar(im,ax=ax,fraction=.046)
    for l in range(L):
        for h in range(H): ax.text(h,l,f"{A[l,h]:.0f}" if A[l,h]>=1 else f"{A[l,h]:.2f}",ha="center",va="center",color="w",fontsize=7)
    fig.tight_layout(); fig.savefig(OUT+"/"+fn,dpi=130); plt.close()
hm(orank,"LUT FFN per-head OUTPUT effective rank (of 48)","ph_output_rank.png")
hm(distinct,f"LUT FFN per-head DISTINCT (table,cell) selections used (max {TPH*2**NAP})","ph_distinct_cells.png","magma")
hm(onorm,"LUT FFN per-head mean OUTPUT norm","ph_output_norm.png","cividis")
hm(discrete_frac,"LUT FFN per-head discreteness (var explained by cell identity)","ph_discreteness.png","coolwarm")

# representative SV spectra: layer0 head0, layer3 head0, layer5 head0
fig,ax=plt.subplots(figsize=(7,4))
for (l,h) in [(0,0),(2,0),(5,0)]:
    ax.plot(spectra[(l,h)],label=f"L{l}H{h} (effrank {orank[l,h]:.1f})")
ax.set_yscale("log"); ax.set_title("Representative per-head output singular values"); ax.set_xlabel("index"); ax.set_ylabel("sv (log)"); ax.legend()
fig.tight_layout(); fig.savefig(OUT+"/ph_svspectra.png",dpi=130); plt.close()

print("=== per-layer means (over 8 heads) ===")
for l in range(L):
    print(f"L{l}: outrank {orank[l].mean():.1f} inrank {irank[l].mean():.1f} distinct {distinct[l].mean():.0f} entropy {cell_entropy[l].mean():.2f} onorm {onorm[l].mean():.3f} sparsity {ospars[l].mean():.3f} discreteFrac {discrete_frac[l].mean():.2f}")
print("=== heterogeneity across heads (min..max distinct per layer) ===")
for l in range(L): print(f"L{l}: distinct {int(distinct[l].min())}..{int(distinct[l].max())} | outrank {orank[l].min():.1f}..{orank[l].max():.1f} | onorm {onorm[l].min():.3f}..{onorm[l].max():.3f}")
# stash arrays for the report text
np.savez(OUT+"/ph_stats.npz", orank=orank, irank=irank, onorm=onorm, ospars=ospars, distinct=distinct, cell_entropy=cell_entropy, discrete_frac=discrete_frac, dims=[L,H,TPH,NAP,DIN,DOUT])
print("FIGS:", [f for f in os.listdir(OUT) if f.startswith("ph_")])
