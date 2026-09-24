"""Vanilla (dense MLP FFN) vs LUT47 (compression LUT FFN) interpretability comparison.
Areas: (1) attention patterns, (2) embeddings, (3) FFN-vs-LUT internals. Saves PNGs + prints stats.
Both are model_build.MinimalGPT (E=384, attn n_head=6, depth=6); only the FFN slot differs.
"""
import json, os, sys, math
import numpy as np
import torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

ROOT = "/home/astarostin/projects/ffn_fix_wt"
sys.path.insert(0, ROOT + "/experiments/ffn_replacement/tools"); sys.path.insert(0, ROOT + "/src")
from model_build import build_model, apply_rope
from nanochat.common import get_base_dir
from nanochat.tokenizer import RustBPETokenizer
from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit

OUT = ROOT + "/experiments/ffn_replacement/analysis/vanilla_vs_lut47"
LA = ROOT + "/experiments/ffn_replacement"
RUNS = {"vanilla": LA + "/runs_corrected/exp_n_0216_vanilla48k_gelu_ckpt2k_seed1",
        "LUT47":   LA + "/lut_ablation/exp_n_abl_47_B48k_light_lmfrozeng_n2_tau0p5_tv10_tph64_h8_seed1_noquant_headdrop20"}
DEV = "cuda"; V = 32768; B, T = 8, 256; NH = 6; HD = 384 // NH
torch.manual_seed(0)

def load(d):
    c = json.load(open(d + "/config.json")); m = build_model(c, V, device=DEV)
    m.load_state_dict(torch.load(d + "/checkpoint.pt", map_location=DEV), strict=False); m.eval()
    return m, c

models = {k: load(d)[0] for k, d in RUNS.items()}

# shared eval batch (same tokens for both)
tok = RustBPETokenizer.from_directory(get_base_dir() + "/tokenizer")
dl = tokenizing_distributed_data_loader_bos_bestfit(tok, B, T, split="val", device=DEV)
x, y = next(dl)

# ---- capture: attn input (pre-ln1'd? we hook block.attn input = ln1(x)), attn out, ffn out, lut read ----
def run_capture(m):
    caps = {"attn_in": [None]*6, "attn_out": [None]*6, "ffn_out": [None]*6, "lut_read": [None]*6}
    hooks = []
    for i, blk in enumerate(m.blocks):
        hooks.append(blk.attn.register_forward_pre_hook(lambda mod, inp, i=i: caps["attn_in"].__setitem__(i, inp[0].detach())))
        hooks.append(blk.attn.register_forward_hook(lambda mod, inp, out, i=i: caps["attn_out"].__setitem__(i, out.detach())))
        ffn_mod = blk.mlp if blk.ffn_type == "dense" else blk.ffn
        hooks.append(ffn_mod.register_forward_hook(lambda mod, inp, out, i=i: caps["ffn_out"].__setitem__(i, out.detach())))
        if blk.ffn_type != "dense" and hasattr(blk.ffn, "lut_light"):
            hooks.append(blk.ffn.lut_light.register_forward_hook(lambda mod, inp, out, i=i: caps["lut_read"].__setitem__(i, (out[0] if isinstance(out, tuple) else out).detach())))
    with torch.no_grad():
        logits = m(x)
    for h in hooks: h.remove()
    return caps, logits

caps = {}; logits = {}
for k, m in models.items():
    caps[k], logits[k] = run_capture(m)

# ---- recompute attention probs from captured attn input ----
@torch.no_grad()
def attn_probs(m, attn_in_i):
    out = []
    for i, blk in enumerate(m.blocks):
        xin = attn_in_i[i]                       # [B,T,C]
        q, k, v = blk.attn.qkv(xin).split(384, dim=2)
        q = q.view(B, T, NH, HD).transpose(1, 2); k = k.view(B, T, NH, HD).transpose(1, 2)
        q, k = apply_rope(q, k, m.rope.cos[:T], m.rope.sin[:T])
        logit = (q @ k.transpose(-2, -1)) / math.sqrt(HD)     # [B,NH,T,T]
        mask = torch.triu(torch.ones(T, T, device=DEV, dtype=torch.bool), 1)
        logit = logit.masked_fill(mask, float("-inf"))
        out.append(torch.softmax(logit, dim=-1))
    return out  # list of [B,NH,T,T]

P = {k: attn_probs(models[k], caps[k]["attn_in"]) for k in models}

# per (layer,head) mean attention entropy (bits), over query positions t>=1 and batch
def entropy_LH(Pl):
    E = np.zeros((6, NH))
    for L in range(6):
        p = Pl[L][:, :, 1:, :]                    # drop t=0 (only attends to itself)
        ent = -(p * (p.clamp_min(1e-12)).log2()).sum(-1)   # [B,NH,T-1]
        E[L] = ent.mean(dim=(0, 2)).cpu().numpy()
    return E
Ent = {k: entropy_LH(P[k]) for k in models}

# prev-token attention (mass on t-1) and induction score per (layer,head)
def prev_and_induction(Pl, toks):
    prev = np.zeros((6, NH)); ind = np.zeros((6, NH))
    tks = toks.cpu().numpy()  # [B,T]
    # induction target position per (b,t): 1 + last s<t with tok[s]==tok[t]
    tgt = -np.ones((B, T), dtype=int)
    for b in range(B):
        last = {}
        for t in range(T):
            tk = tks[b, t]
            if tk in last and last[tk] + 1 < t:
                tgt[b, t] = last[tk] + 1
            last[tk] = t
    tgt_t = torch.tensor(tgt, device=DEV)
    for L in range(6):
        p = Pl[L]  # [B,NH,T,T]
        pv = p[:, :, 1:, :].gather(-1, torch.arange(1, T, device=DEV).view(1,1,-1,1).expand(B,NH,T-1,1)-1).squeeze(-1)
        prev[L] = pv.mean(dim=(0,2)).cpu().numpy()
        valid = tgt_t >= 0                                   # [B,T]
        isc = np.zeros(NH)
        for h in range(NH):
            vals = []
            for b in range(B):
                vt = valid[b].nonzero(as_tuple=True)[0]
                if len(vt):
                    idx = tgt_t[b, vt]
                    vals.append(p[b, h, vt, idx].mean().item())
            isc[h] = float(np.mean(vals)) if vals else 0.0
        ind[L] = isc
    return prev, ind
Prev = {}; Ind = {}
for k in models:
    Prev[k], Ind[k] = prev_and_induction(P[k], x)

# FIG 1: entropy heatmaps vanilla|LUT|diff
fig, axs = plt.subplots(1, 3, figsize=(13, 3.6))
vmin = min(Ent["vanilla"].min(), Ent["LUT47"].min()); vmax = max(Ent["vanilla"].max(), Ent["LUT47"].max())
for ax, k in zip(axs[:2], ["vanilla", "LUT47"]):
    im = ax.imshow(Ent[k], aspect="auto", vmin=vmin, vmax=vmax, cmap="viridis")
    ax.set_title(f"{k}: attn entropy (bits)"); ax.set_xlabel("head"); ax.set_ylabel("layer"); plt.colorbar(im, ax=ax, fraction=.046)
d = Ent["LUT47"] - Ent["vanilla"]; mx = np.abs(d).max()
im = axs[2].imshow(d, aspect="auto", cmap="coolwarm", vmin=-mx, vmax=mx)
axs[2].set_title("LUT - vanilla (bits)"); axs[2].set_xlabel("head"); axs[2].set_ylabel("layer"); plt.colorbar(im, ax=axs[2], fraction=.046)
fig.tight_layout(); fig.savefig(OUT + "/attn_entropy_heatmaps.png", dpi=130); plt.close()

# FIG examples: one shared (layer,head) heatmap for both, first batch element
Lx, Hx = 2, 0
fig, axs = plt.subplots(1, 2, figsize=(9, 4.2))
for ax, k in zip(axs, ["vanilla", "LUT47"]):
    im = ax.imshow(P[k][Lx][0, Hx].cpu().numpy(), cmap="magma", aspect="auto")
    ax.set_title(f"{k}  L{Lx} H{Hx}"); ax.set_xlabel("key pos"); ax.set_ylabel("query pos"); plt.colorbar(im, ax=ax, fraction=.046)
fig.tight_layout(); fig.savefig(OUT + "/attn_examples.png", dpi=130); plt.close()

# ---- AREA 2: embeddings ----
def emb_stats(m):
    We = m.tok_emb.weight.detach().float(); Wu = m.head.weight.detach().float()
    out = {}
    for name, W in [("tok_emb", We), ("unembed", Wu)]:
        norms = W.norm(dim=1).cpu().numpy()
        s = torch.linalg.svdvals(W.cpu()).numpy()
        pr = (s.sum()**2) / (s**2).sum()                    # participation ratio (effective rank)
        e90 = int(np.searchsorted(np.cumsum(s**2)/(s**2).sum(), 0.90)) + 1
        mu = W.mean(0); iso = torch.nn.functional.cosine_similarity(W, mu.unsqueeze(0), dim=1).mean().item()
        out[name] = dict(norms=norms, sv=s, pr=float(pr), e90=e90, iso=float(iso), meanN=float(norms.mean()))
    return out
Emb = {k: emb_stats(models[k]) for k in models}

fig, axs = plt.subplots(1, 2, figsize=(11, 4))
for name, ax in zip(["tok_emb", "unembed"], axs):
    for k in models:
        ax.hist(Emb[k][name]["norms"], bins=80, alpha=.5, label=f"{k} (mean {Emb[k][name]['meanN']:.2f})", density=True)
    ax.set_title(f"{name} per-token L2 norm"); ax.set_xlabel("||row||"); ax.legend()
fig.tight_layout(); fig.savefig(OUT + "/embed_norms.png", dpi=130); plt.close()

fig, axs = plt.subplots(1, 2, figsize=(11, 4))
for name, ax in zip(["tok_emb", "unembed"], axs):
    for k in models:
        ax.plot(Emb[k][name]["sv"], label=f"{k} (eff.rank {Emb[k][name]['pr']:.0f}, 90%@{Emb[k][name]['e90']})")
    ax.set_yscale("log"); ax.set_title(f"{name} singular values"); ax.set_xlabel("index"); ax.set_ylabel("sv (log)"); ax.legend()
fig.tight_layout(); fig.savefig(OUT + "/embed_svspectrum.png", dpi=130); plt.close()

# ---- AREA 3: FFN internals ----
resid_norm = {}; ffn_dnorm = {}; ffn_rank = {}; ffn_sparsity = {}
# residual before each FFN ~ we approximate residual scale by the running x; use attn_out+prior. Simpler: use token embedding scale + cumulative. We report ||ffn delta|| and its ratio to ||ffn_out||... instead ratio to residual: reconstruct residual by re-running with hook on block outputs.
def block_resid(m):
    outs = [None]*6
    hs = [blk.register_forward_hook(lambda mod, inp, out, i=i: outs.__setitem__(i, out.detach())) for i, blk in enumerate(m.blocks)]
    with torch.no_grad(): m(x)
    for h in hs: h.remove()
    return outs
for k, m in models.items():
    ro = block_resid(m)
    rn = [r.norm(dim=-1).mean().item() for r in ro]            # mean ||residual|| after each block
    dn = []; rk = []
    for i in range(6):
        o = caps[k]["ffn_out"][i]
        if o.dim() == 2: o = o.view(B, T, 384)
        dn.append(o.norm(dim=-1).mean().item())
        flat = o.reshape(-1, 384).float()
        s = torch.linalg.svdvals(flat - flat.mean(0)).cpu().numpy()
        rk.append(float((s.sum()**2)/(s**2).sum()))
    resid_norm[k] = rn; ffn_dnorm[k] = dn; ffn_rank[k] = rk
    # sparsity
    sp = []
    if k == "vanilla":
        # hook GELU (mlp[1]) output per layer
        acts = [None]*6
        hs = [blk.mlp[1].register_forward_hook(lambda mod, inp, out, i=i: acts.__setitem__(i, out.detach())) for i, blk in enumerate(m.blocks)]
        with torch.no_grad(): m(x)
        for h in hs: h.remove()
        for a in acts:
            sp.append((a.abs() < 1e-3).float().mean().item())
    else:
        for i in range(6):
            r = caps[k]["lut_read"][i]
            sp.append((r.abs() < 1e-3).float().mean().item() if r is not None else float("nan"))
    ffn_sparsity[k] = sp

fig, axs = plt.subplots(1, 2, figsize=(11, 4))
w = 0.35; xs = np.arange(6)
for j, k in enumerate(models):
    axs[0].bar(xs + (j-0.5)*w, ffn_dnorm[k], w, label=k)
axs[0].set_title("FFN/LUT residual-update norm per layer"); axs[0].set_xlabel("layer"); axs[0].set_ylabel("mean ||FFN delta||"); axs[0].legend()
for k in models:
    axs[1].plot(xs, ffn_rank[k], "o-", label=f"{k}")
axs[1].set_title("Effective rank of FFN delta (per layer)"); axs[1].set_xlabel("layer"); axs[1].set_ylabel("participation ratio"); axs[1].legend()
fig.tight_layout(); fig.savefig(OUT + "/ffn_delta.png", dpi=130); plt.close()

fig, ax = plt.subplots(figsize=(6.5, 4))
for j, k in enumerate(models):
    ax.bar(xs + (j-0.5)*w, ffn_sparsity[k], w, label=f"{k} (frac |act|<1e-3)")
ax.set_title("FFN activation sparsity per layer\n(vanilla: GELU hidden; LUT: per-head read)"); ax.set_xlabel("layer"); ax.set_ylabel("fraction ~0"); ax.legend()
fig.tight_layout(); fig.savefig(OUT + "/ffn_sparsity.png", dpi=130); plt.close()

# ---- print compact stats for the summary ----
print("=== ENTROPY (mean over heads) per layer ===")
for k in models: print(k, [round(float(Ent[k][L].mean()),3) for L in range(6)])
print("overall mean entropy: vanilla %.3f  LUT %.3f"%(Ent["vanilla"].mean(), Ent["LUT47"].mean()))
print("=== INDUCTION top head per model (layer,head,score) ===")
for k in models:
    fl = Ind[k]; L,H = np.unravel_index(fl.argmax(), fl.shape); print(k, "top induction L%dH%d=%.3f | max prev-token %.3f"%(L,H,fl[L,H], Prev[k].max()))
print("=== EMBED effective rank (participation ratio) / isotropy(mean cos to mean) ===")
for k in models:
    for nm in ["tok_emb","unembed"]:
        e=Emb[k][nm]; print(k, nm, "effrank %.1f  90%%@%d  isotropy %.3f  meanNorm %.2f"%(e["pr"],e["e90"],e["iso"],e["meanN"]))
print("=== FFN delta norm per layer ===");
for k in models: print(k, "dnorm", [round(v,3) for v in ffn_dnorm[k]], "| residnorm", [round(v,1) for v in resid_norm[k]])
print("=== FFN delta effrank per layer ===");
for k in models: print(k, [round(v,1) for v in ffn_rank[k]])
print("=== FFN sparsity per layer ===");
for k in models: print(k, [round(v,3) for v in ffn_sparsity[k]])
print("FIGS:", sorted(os.listdir(OUT)))
