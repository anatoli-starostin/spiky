"""Per-table structure of the LUT47 (H8) FFN: do the 64 tables in a head span the SAME output
directions or different ones? Static (stored values) vs data-driven (real selections)."""
import json, os, sys
import numpy as np, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT="/home/astarostin/projects/ffn_fix_wt"
sys.path.insert(0, ROOT+"/experiments/ffn_replacement/tools"); sys.path.insert(0, ROOT+"/src")
from model_build import build_model
OUT=ROOT+"/experiments/ffn_replacement/analysis/vanilla_vs_lut47"
D=ROOT+"/experiments/ffn_replacement/lut_ablation/exp_n_abl_47_B48k_light_lmfrozeng_n2_tau0p5_tv10_tph64_h8_seed1_noquant_headdrop20"
DEV="cuda"
c=json.load(open(D+"/config.json")); m=build_model(c,32768,device=DEV)
m.load_state_dict(torch.load(D+"/checkpoint.pt",map_location=DEV),strict=False); m.eval()
L=len(m.blocks); H=8; TPH=64; DOUT=48; KDIR=10
obs=np.load(OUT+"/ph_stats.npz")["orank"]   # data-driven per-head OUTPUT rank measured earlier [L,H]

def pr(s): return float((s.sum()**2)/((s**2).sum()+1e-12))
per_table_rank=np.zeros((L,H)); per_table_rank_sd=np.zeros((L,H)); union_rank=np.zeros((L,H)); overlap=np.zeros((L,H))
for l in range(L):
    W=m.blocks[l].ffn.lut_light.tables.detach().float()      # [512,256,48]
    W=W.view(H,TPH,W.shape[1],DOUT)                          # [H,64,256,48]
    for h in range(H):
        Th=W[h]                                              # [64,256,48]
        # 1) per-table rank + top-KDIR value directions
        ranks=[]; dirs=[]
        for t in range(TPH):
            M=Th[t]-Th[t].mean(0,keepdim=True)               # [256,48] centered
            U,S,Vh=torch.linalg.svd(M.cpu(),full_matrices=False)
            s=S.numpy(); ranks.append(pr(s))
            dirs.append(torch.tensor(Vh[:KDIR]))             # [KDIR,48] top value directions
        per_table_rank[l,h]=float(np.mean(ranks)); per_table_rank_sd[l,h]=float(np.std(ranks))
        # 2) cross-table subspace overlap: mean off-diag ||Ui Uj^T||_F^2 / KDIR
        Dm=torch.stack(dirs)                                 # [64,KDIR,48]
        ov=0.0; n=0
        for i in range(TPH):
            for j in range(i+1,TPH):
                ov+=float((Dm[i]@Dm[j].T).pow(2).sum()/KDIR); n+=1
        overlap[l,h]=ov/n
        # 3) static union rank: all 64 tables' cell values stacked [64*256,48]
        allv=(Th.reshape(-1,DOUT)-Th.reshape(-1,DOUT).mean(0,keepdim=True)).cpu()
        union_rank[l,h]=pr(torch.linalg.svdvals(allv).numpy())

def hm(A,title,fn,cmap="viridis",fmt="%.1f"):
    fig,ax=plt.subplots(figsize=(7,4)); im=ax.imshow(A,aspect="auto",cmap=cmap); ax.set_xlabel("head"); ax.set_ylabel("layer"); ax.set_title(title); plt.colorbar(im,ax=ax,fraction=.046)
    for l in range(L):
        for h in range(H): ax.text(h,l,fmt%A[l,h],ha="center",va="center",color="w",fontsize=7)
    fig.tight_layout(); fig.savefig(OUT+"/"+fn,dpi=130); plt.close()
hm(per_table_rank,"Per-table value-matrix effective rank (of 48), mean over 64 tables","pt_pertable_rank.png")
hm(overlap,"Within-head cross-table subspace overlap (0=orthogonal,1=identical)","pt_overlap.png","magma","%.2f")
hm(union_rank,"STATIC union rank: span of all 64 tables' cell values (of 48)","pt_union_rank.png","cividis")
# static union vs data-driven observed output rank
fig,ax=plt.subplots(figsize=(7,4))
ax.plot(range(L),union_rank.mean(1),"o-",label="STATIC: union-of-tables rank (what tables CAN span)")
ax.plot(range(L),obs.mean(1),"s-",label="DATA-DRIVEN: per-head output rank on real tokens")
ax.plot(range(L),per_table_rank.mean(1),"^-",label="mean per-table rank")
ax.set_xlabel("layer"); ax.set_ylabel("effective rank (of 48)"); ax.set_title("Static table span vs realized per-head output rank"); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(OUT+"/pt_static_vs_data.png",dpi=130); plt.close()

print("=== per-layer means ===")
for l in range(L):
    print(f"L{l}: per-table rank {per_table_rank[l].mean():.1f}±{per_table_rank_sd[l].mean():.1f} | cross-table overlap {overlap[l].mean():.3f} | STATIC union rank {union_rank[l].mean():.1f} | DATA output rank {obs[l].mean():.1f}")
print("FIGS: pt_pertable_rank.png pt_overlap.png pt_union_rank.png pt_static_vs_data.png")
