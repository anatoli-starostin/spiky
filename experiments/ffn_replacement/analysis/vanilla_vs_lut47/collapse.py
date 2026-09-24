"""Decompose WHY the LUT47 (H8) FFN residual write collapses to rank ~5-7 early despite a dense
decompress and high per-head read rank. Per layer: SVD of W_dec, SVD of the joint 384-dim head-read
code, SVD of the write, cross-head read correlation, and attribution."""
import json, os, sys
import numpy as np, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT="/home/astarostin/projects/ffn_fix_wt"
sys.path.insert(0, ROOT+"/experiments/ffn_replacement/tools"); sys.path.insert(0, ROOT+"/src")
from model_build import build_model
from nanochat.common import get_base_dir
from nanochat.tokenizer import RustBPETokenizer
from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit
OUT=ROOT+"/experiments/ffn_replacement/analysis/vanilla_vs_lut47"
D=ROOT+"/experiments/ffn_replacement/lut_ablation/exp_n_abl_47_B48k_light_lmfrozeng_n2_tau0p5_tv10_tph64_h8_seed1_noquant_headdrop20"
DEV="cuda"; V=32768; B,T=16,256
torch.manual_seed(0)
c=json.load(open(D+"/config.json")); m=build_model(c,V,device=DEV)
m.load_state_dict(torch.load(D+"/checkpoint.pt",map_location=DEV),strict=False); m.eval()
L=len(m.blocks); H=m.blocks[0].ffn.n_heads; EO=m.blocks[0].ffn.eff_out
print("DIMS: layers",L,"heads",H,"eff_out",EO,"joint code dim",H*EO,"decompress weight",tuple(m.blocks[0].ffn.decompress.weight.shape))

reads=[None]*L; writes=[None]*L
for i,b in enumerate(m.blocks):
    b.ffn.lut_light.register_forward_hook(lambda mo,inp,out,i=i: reads.__setitem__(i,(out[0] if isinstance(out,tuple) else out).detach()))
    b.ffn.register_forward_hook(lambda mo,inp,out,i=i: writes.__setitem__(i,(out if out.dim()==2 else out.reshape(-1,out.shape[-1])).detach()))
tok=RustBPETokenizer.from_directory(get_base_dir()+"/tokenizer")
dl=tokenizing_distributed_data_loader_bos_bestfit(tok,B,T,split="val",device=DEV); x,y=next(dl)
with torch.no_grad(): m(x)

def svd(M):   # centered
    M=M.float()-M.float().mean(0,keepdim=True)
    return torch.linalg.svdvals(M.cpu()).numpy()
def pr(s): return float((s.sum()**2)/((s**2).sum()+1e-12))
def stable_rank(s): return float((s**2).sum()/(s[0]**2+1e-12))
def r99(s): return int(np.searchsorted(np.cumsum(s**2)/(s**2).sum(),0.99))+1

rows=[]; spectra={}
Wd_sv={}; code_sv={}; write_sv={}; crosshead={}
for l in range(L):
    R=reads[l].reshape(-1,H*EO)            # joint 384-dim code [N,384]
    W=writes[l]                           # 384-dim write [N,384]
    Wd=m.blocks[l].ffn.decompress.weight.detach().cpu().float()   # [384,384]
    sW=torch.linalg.svdvals(Wd).numpy()
    sC=svd(R); sO=svd(W)
    Wd_sv[l]=sW; code_sv[l]=sC; write_sv[l]=sO
    # cross-head correlation: mean |cosine| between head-i and head-j per-token read vectors
    Rh=reads[l].reshape(-1,H,EO).float()
    Rn=torch.nn.functional.normalize(Rh,dim=-1)
    C=torch.zeros(H,H)
    for a in range(H):
        for b2 in range(H):
            C[a,b2]=(Rn[:,a,:]*Rn[:,b2,:]).sum(-1).abs().mean()
    crosshead[l]=C.numpy()
    rows.append(dict(l=l, code_pr=pr(sC), code_r99=r99(sC), write_pr=pr(sO), write_r99=r99(sO),
                     Wd_pr=pr(sW), Wd_stable=stable_rank(sW), Wd_r99=r99(sW),
                     offdiag_corr=float((C.sum()-C.diag().sum())/(H*H-H))))

print("=== per-layer decomposition ===")
print("L | jointCode(effrank/r99) | write(effrank/r99) | W_dec(effrank/stable/r99) | mean off-diag cross-head |cos|")
for r in rows:
    print("%d | %.1f / %d | %.1f / %d | %.1f / %.1f / %d | %.3f"%(r["l"], r["code_pr"], r["code_r99"], r["write_pr"], r["write_r99"], r["Wd_pr"], r["Wd_stable"], r["Wd_r99"], r["offdiag_corr"]))

# FIG: per-layer SVD spectra (joint code, write, W_dec) normalized
fig,axs=plt.subplots(1,3,figsize=(14,4))
for l in range(L):
    axs[0].plot(code_sv[l]/code_sv[l][0], label=f"L{l}")
    axs[1].plot(write_sv[l]/write_sv[l][0], label=f"L{l}")
    axs[2].plot(Wd_sv[l]/Wd_sv[l][0], label=f"L{l}")
for ax,t in zip(axs,["Joint 384-d head-read code","FFN residual write (384-d)","Decompress weight W_dec"]):
    ax.set_yscale("log"); ax.set_title(t); ax.set_xlabel("index"); ax.set_ylabel("sv / sv_max (log)"); ax.legend(fontsize=7,ncol=2)
fig.tight_layout(); fig.savefig(OUT+"/collapse_svd.png",dpi=130); plt.close()

# FIG: cross-head correlation heatmaps for L0 (collapsed) and L3 (rich)
fig,axs=plt.subplots(1,2,figsize=(9,4))
for ax,l in zip(axs,[0,3]):
    im=ax.imshow(crosshead[l],vmin=0,vmax=1,cmap="magma"); ax.set_title(f"L{l} cross-head read |cos| (off-diag {rows[l]['offdiag_corr']:.2f})")
    ax.set_xlabel("head"); ax.set_ylabel("head"); plt.colorbar(im,ax=ax,fraction=.046)
fig.tight_layout(); fig.savefig(OUT+"/collapse_crosshead.png",dpi=130); plt.close()

# FIG: bar of the three ranks per layer
fig,ax=plt.subplots(figsize=(7.5,4)); xs=np.arange(L); w=0.27
ax.bar(xs-w,[r["code_pr"] for r in rows],w,label="joint code effrank")
ax.bar(xs,[r["write_pr"] for r in rows],w,label="write effrank")
ax.bar(xs+w,[r["Wd_stable"] for r in rows],w,label="W_dec stable rank")
ax.set_xlabel("layer"); ax.set_ylabel("effective / stable rank"); ax.set_title("Rank decomposition per layer (LUT47 H8 FFN)"); ax.legend()
fig.tight_layout(); fig.savefig(OUT+"/collapse_ranks.png",dpi=130); plt.close()
print("FIGS: collapse_svd.png collapse_crosshead.png collapse_ranks.png")
