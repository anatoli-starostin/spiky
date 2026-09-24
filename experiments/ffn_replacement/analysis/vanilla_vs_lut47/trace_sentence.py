"""Single-sentence side-by-side trace: vanilla (dense MLP) vs LUT47. Next-token top-5, per-layer
residual / attention-delta / FFN-delta norms, and attention at the final position. Writes a markdown
trace + a small figure."""
import json, os, sys, math
import numpy as np, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT = "/home/astarostin/projects/ffn_fix_wt"
sys.path.insert(0, ROOT + "/experiments/ffn_replacement/tools"); sys.path.insert(0, ROOT + "/src")
from model_build import build_model, apply_rope
from nanochat.common import get_base_dir
from nanochat.tokenizer import RustBPETokenizer
OUT = ROOT + "/experiments/ffn_replacement/analysis/vanilla_vs_lut47"
LA = ROOT + "/experiments/ffn_replacement"
RUNS = {"vanilla": LA + "/runs_corrected/exp_n_0216_vanilla48k_gelu_ckpt2k_seed1",
        "LUT47":   LA + "/lut_ablation/exp_n_abl_47_B48k_light_lmfrozeng_n2_tau0p5_tv10_tph64_h8_seed1_noquant_headdrop20"}
DEV="cuda"; V=32768; NH=6; HD=64
def load(d):
    c=json.load(open(d+"/config.json")); m=build_model(c,V,device=DEV)
    m.load_state_dict(torch.load(d+"/checkpoint.pt",map_location=DEV),strict=False); m.eval(); return m
models={k:load(d) for k,d in RUNS.items()}
tok=RustBPETokenizer.from_directory(get_base_dir()+"/tokenizer")
SENTS=["The capital of France is", "The cat sat on the mat."]

@torch.no_grad()
def trace(m, ids):
    T=len(ids); x=torch.tensor([ids],device=DEV)
    cap={"attn":[None]*6,"ffn":[None]*6,"blk":[None]*6,"attn_in":[None]*6}
    hs=[]
    for i,blk in enumerate(m.blocks):
        hs.append(blk.attn.register_forward_pre_hook(lambda mo,inp,i=i:cap["attn_in"].__setitem__(i,inp[0].detach())))
        hs.append(blk.attn.register_forward_hook(lambda mo,inp,out,i=i:cap["attn"].__setitem__(i,out.detach())))
        fm=blk.mlp if blk.ffn_type=="dense" else blk.ffn
        hs.append(fm.register_forward_hook(lambda mo,inp,out,i=i:cap["ffn"].__setitem__(i,(out.view(1,T,384) if out.dim()==2 else out).detach())))
        hs.append(blk.register_forward_hook(lambda mo,inp,out,i=i:cap["blk"].__setitem__(i,out.detach())))
    with torch.no_grad(): logits=m(x)[0]  # [T,V]
    for h in hs: h.remove()
    # per-layer per-position norms
    an=np.stack([cap["attn"][i][0].norm(dim=-1).cpu().numpy() for i in range(6)])  # [L,T]
    fn=np.stack([cap["ffn"][i][0].norm(dim=-1).cpu().numpy() for i in range(6)])
    rn=np.stack([cap["blk"][i][0].norm(dim=-1).cpu().numpy() for i in range(6)])
    # attention probs at final pos: which earlier tokens each head attends to
    xin=cap["attn_in"][5]  # last layer input
    q,k,_=m.blocks[5].attn.qkv(xin).split(384,dim=2)
    q=q.view(1,T,NH,HD).transpose(1,2); k=k.view(1,T,NH,HD).transpose(1,2)
    q,k=apply_rope(q,k,m.rope.cos[:T],m.rope.sin[:T])
    lg=(q@k.transpose(-2,-1))/math.sqrt(HD)
    lg=lg.masked_fill(torch.triu(torch.ones(T,T,device=DEV,dtype=torch.bool),1),float("-inf"))
    ap=torch.softmax(lg,-1)[0,:,-1,:].cpu().numpy()  # [NH,T] attention from last pos
    return logits, an, fn, rn, ap

lines=["## Single-sentence trace: vanilla (dense MLP FFN) vs LUT47 (LUT FFN)\n"]
for si,s in enumerate(SENTS):
    ids=tok.encode(s)
    toks=[tok.decode([i]) for i in ids]
    lines.append(f"\n### Sentence {si+1}: `{s}`\nTokens: {toks}\n")
    res={}
    for k,m in models.items(): res[k]=trace(m,ids)
    # next-token top-5 at the LAST position (the actual continuation prediction)
    lines.append("\n**Next-token prediction after the full prompt (top-5):**\n")
    for k in models:
        lg=res[k][0][-1]; p=torch.softmax(lg,-1); tv,ti=p.topk(5)
        pred=", ".join(f"'{tok.decode([int(t)])}'={float(v):.2f}" for v,t in zip(tv,ti))
        lines.append(f"- {k}: {pred}")
    # per-position top-1 agreement across the sentence
    lines.append("\n**Per-position top-1 next-token (each model), showing agree/disagree:**\n")
    lgV=res["vanilla"][0]; lgL=res["LUT47"][0]
    for t in range(len(ids)):
        pv=int(lgV[t].argmax()); pl=int(lgL[t].argmax())
        mark="=" if pv==pl else "X"
        lines.append(f"  [{t}] after '{toks[t]}': vanilla->'{tok.decode([pv])}'  LUT->'{tok.decode([pl])}'  {mark}")
    # per-layer FFN-delta norm at final position (how much work the FFN does)
    lines.append("\n**Per-layer norms at the FINAL position (attn delta / FFN delta / residual):**\n")
    for k in models:
        _,an,fn,rn,_=res[k]
        lines.append(f"- {k}:")
        for L in range(6):
            lines.append(f"    L{L}: attn|d|={an[L,-1]:.2f}  ffn|d|={fn[L,-1]:.2f}  resid={rn[L,-1]:.2f}  (ffn/resid={fn[L,-1]/rn[L,-1]:.2f})")
    # attention at final pos: top-3 attended tokens for each head (layer 5)
    lines.append("\n**Layer-5 attention from the final token (top-3 attended tokens per head):**\n")
    for k in models:
        ap=res[k][4]  # [NH,T]
        lines.append(f"- {k}:")
        for h in range(NH):
            top=ap[h].argsort()[::-1][:3]
            lines.append(f"    head{h}: " + ", ".join(f"'{toks[j]}'({ap[h,j]:.2f})" for j in top))
    # figure: FFN-delta norm across layers at final pos
    if si==0:
        fig,ax=plt.subplots(figsize=(6.5,4))
        for k in models:
            ax.plot(range(6), res[k][2][:, -1], "o-", label=f"{k} FFN |delta|")
        ax.set_title(f"FFN residual-update norm across layers\nfinal token of: '{s}'"); ax.set_xlabel("layer"); ax.set_ylabel("||FFN delta||"); ax.legend()
        fig.tight_layout(); fig.savefig(OUT+"/trace_ffn_delta.png",dpi=130); plt.close()
open(OUT+"/sentence_trace.md","w").write("\n".join(lines))
print("wrote sentence_trace.md +", "trace_ffn_delta.png")
print("\n".join(lines[:60]))
