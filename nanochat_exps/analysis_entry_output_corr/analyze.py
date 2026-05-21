"""Entry <-> output alignment analysis for a tiny-LUT checkpoint.

Question: how strongly do the weight vectors stored in individual table entries
correlate with the FINAL (aggregated) output of the LUT? Are they similar or not?

Mechanism reminder: a TinyMultiHeadLut output for a given (token, head) is the SUM
over `tables_per_head` of the selected entry-vectors, each in R^{n_out}. So we ask
whether each selected entry e_t is aligned with the sum o = sum_t e_t.

Key metrics (per token, per head):
  - cos(e_t, o)             : alignment of one entry with the whole output
  - R = ||o||^2 / sum||e_t||^2 : coherence ratio. R~1 => orthogonal/distributed
                               code; R~tph => all entries aligned (redundant).
  - cbar = norm-weighted mean pairwise cos between selected entries (exact, from
           ||o||^2 = sum||e_t||^2 + sum_{s!=t} e_s.e_t).
  - dominance = max_t||e_t|| / ||o||  (does one entry carry the output?)
Random iid baseline: R ~ 1 (R/tph ~ 1/tph), E[cos(e_t,o)] ~ 1/sqrt(tph).
"""
import os
os.environ['TORCHDYNAMO_DISABLE'] = '1'   # make @torch.compile a no-op (speed/contention)
import sys, math, json
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import spiky.lutorch.tiny_multi_head_lut as tml

EXP = sys.argv[1] if len(sys.argv) > 1 else "/home/starost/spiky/nanochat_exps/exp365_bs16_weight_track"
OUT = "/home/starost/spiky/nanochat_exps/analysis_entry_output_corr"
TAG = os.path.basename(EXP.rstrip("/"))
TRAIN_PY = os.path.join(EXP, "train.py")
N_SUB = 2048   # token subsample for the heavy per-entry stats

# ---- build model + loaders by exec'ing train.py up to model creation ----------
src = open(TRAIN_PY).read()
src = src[:src.index("\ndef get_lr_scale(step):")]   # cut before optimizers/loggers/file writes
ns = {"__name__": "__analyze__", "__file__": TRAIN_PY}
print(f"exec'ing prefix of {TRAIN_PY} ({len(src)} chars) ...")
exec(compile(src, TRAIN_PY, "exec"), ns)

DEVICE = ns["DEVICE"]
model = ns["model"]
ckpt = torch.load(os.path.join(EXP, "checkpoint.pt"), map_location=DEVICE)
sd = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
missing, unexpected = model.load_state_dict(sd, strict=False)
print(f"loaded checkpoint. missing={len(missing)} unexpected={len(unexpected)}")
model.eval()

# ---- find LUT modules, hook their inputs/outputs ------------------------------
luts = {n: m for n, m in model.named_modules() if isinstance(m, tml.TinyMultiHeadLut)}
print(f"found {len(luts)} TinyMultiHeadLut modules")
captured = {}
def mk_hook(name):
    def hook(mod, inp, out):
        captured[name] = (inp[0].detach(), out.detach())
    return hook
hooks = [m.register_forward_hook(mk_hook(n)) for n, m in luts.items()]

# ---- one real batch forward ---------------------------------------------------
x, yb = next(ns["train_loader"])
with torch.no_grad():
    model(x, targets=yb)
for h in hooks:
    h.remove()
print(f"captured inputs for {len(captured)} modules; batch tokens = {x.numel()}")

# ---- per-module analysis ------------------------------------------------------
def module_type(name):
    for t in ("qkv_lut", "v_lut", "out_proj", "residual_lut"):
        if t in name:
            return t
    return name

def layer_idx(name):
    import re
    m = re.search(r"layers\.(\d+)\.", name)
    return int(m.group(1)) if m else -1

rows = []
for name, m in luts.items():
    xin, out = captured[name]
    xin = xin.reshape(-1, xin.shape[-1])
    out = out.reshape(out.shape[0] if out.dim() == 3 else -1, m.n_heads, m.n_outputs) \
            if out.dim() != 3 else out
    N = xin.shape[0]
    if N > N_SUB:
        sub = torch.randperm(N, device=xin.device)[:N_SUB]
        xin, out = xin[sub], out[sub]
    W = m.weights.detach().float()                  # [LT, table_dim, n_out]
    LT, table_dim, n_out = W.shape
    H, tph = m.n_heads, m.tables_per_head

    # bit-faithful canonical index (noise_eps=0 in these runs)
    idx = tml._soft_index_signpack(xin.float(), m.soft_anchor_a_long,
                                   m.soft_anchor_b_long, m.soft_powers, 0.0)  # [n,LT]
    tix = torch.arange(LT, device=W.device)
    sel = W[tix, idx]                                # [n, LT, n_out]
    n = sel.shape[0]
    sel = sel.view(n, H, tph, n_out)                # [n,H,tph,n_out]
    o = sel.sum(dim=2)                              # [n,H,n_out]  recomputed output

    # sanity vs hooked output (may differ slightly: bf16 argmax vs canonical sign)
    match_err = (o - out.float()).abs().max().item()

    enorm = sel.norm(dim=-1)                        # [n,H,tph]
    onorm = o.norm(dim=-1)                          # [n,H]
    sumsq = (enorm**2).sum(-1)                      # [n,H]
    sumnorm = enorm.sum(-1)                         # [n,H]
    R = (onorm**2) / sumsq.clamp_min(1e-12)        # coherence ratio
    cos_eo = F.cosine_similarity(sel, o.unsqueeze(2), dim=-1)  # [n,H,tph]
    # exact norm-weighted mean pairwise cos between selected entries
    cbar = (onorm**2 - sumsq) / (sumnorm**2 - sumsq).clamp_min(1e-12)
    dominance = enorm.max(dim=-1).values / onorm.clamp_min(1e-12)   # [n,H]

    # intrinsic (data-free): mean pairwise cos among ALL stored rows of a table
    Wn = F.normalize(W, dim=-1)                     # [LT, table_dim, n_out]
    g = Wn.sum(dim=1)                               # [LT, n_out]
    intrinsic_cbar = ((g.norm(dim=-1)**2 - table_dim) /
                      (table_dim**2 - table_dim)).mean().item()

    rows.append(dict(
        name=name, type=module_type(name), layer=layer_idx(name),
        H=H, tph=tph, n_out=n_out, table_dim=table_dim,
        match_err=match_err,
        cos_eo=cos_eo.mean().item(),
        R=R.mean().item(),
        R_over_tph=(R.mean().item() / tph),
        cbar_selected=cbar.mean().item(),
        intrinsic_cbar=intrinsic_cbar,
        dominance=dominance.mean().item(),
        rand_cos=1.0 / math.sqrt(tph),
        rand_R_over_tph=1.0 / tph,
    ))
    del sel, o, cos_eo, enorm
    torch.cuda.empty_cache()

# ---- report -------------------------------------------------------------------
rows.sort(key=lambda r: (r["type"], r["layer"]))
print("\n=== PER-MODULE ENTRY<->OUTPUT ALIGNMENT (checkpoint: %s) ===" % os.path.basename(EXP))
hdr = ("module", "tph", "n_out", "cos(e,o)", "rand_cos", "R/tph", "rand", "cbar_sel", "intrins", "domin", "merr")
print("%-26s %4s %5s %9s %9s %7s %7s %9s %8s %6s %7s" % hdr)
for r in rows:
    print("%-26s %4d %5d %9.4f %9.4f %7.4f %7.4f %9.4f %8.4f %6.3f %7.1e" % (
        f"L{r['layer']}.{r['type']}", r["tph"], r["n_out"], r["cos_eo"], r["rand_cos"],
        r["R_over_tph"], r["rand_R_over_tph"], r["cbar_selected"], r["intrinsic_cbar"],
        r["dominance"], r["match_err"]))

# aggregate by type
print("\n=== AGGREGATE BY MODULE TYPE (mean over 6 layers) ===")
print("%-14s %9s %9s %8s %9s %9s %7s" % (
    "type", "cos(e,o)", "rand_cos", "R/tph", "cbar_sel", "intrins", "domin"))
for t in ("qkv_lut", "v_lut", "out_proj", "residual_lut"):
    g = [r for r in rows if r["type"] == t]
    if not g:
        continue
    f = lambda k: sum(r[k] for r in g) / len(g)
    print("%-14s %9.4f %9.4f %8.4f %9.4f %9.4f %7.3f" % (
        t, f("cos_eo"), f("rand_cos"), f("R_over_tph"), f("cbar_selected"),
        f("intrinsic_cbar"), f("dominance")))

with open(os.path.join(OUT, f"results_{TAG}.json"), "w") as fjs:
    json.dump(rows, fjs, indent=2)

# ---- plot: cos(e,o) vs random baseline, and R/tph, per module type -----------
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
types = ["qkv_lut", "v_lut", "out_proj", "residual_lut"]
colors = dict(zip(types, ["C0", "C1", "C2", "C3"]))
for t in types:
    g = sorted([r for r in rows if r["type"] == t], key=lambda r: r["layer"])
    if not g:
        continue
    ls = [r["layer"] for r in g]
    axes[0].plot(ls, [r["cos_eo"] for r in g], "o-", color=colors[t], label=t)
    axes[0].plot(ls, [r["rand_cos"] for r in g], "--", color=colors[t], alpha=0.4)
    axes[1].plot(ls, [r["R_over_tph"] for r in g], "o-", color=colors[t], label=t)
    axes[1].plot(ls, [r["rand_R_over_tph"] for r in g], "--", color=colors[t], alpha=0.4)
axes[0].set(xlabel="layer", ylabel="cos(entry, output)",
            title="Entry-output alignment (dashed = iid-random baseline)")
axes[0].legend(); axes[0].grid(True)
axes[1].set(xlabel="layer", ylabel="R/tph = ||o||^2 / (tph * mean||e||^2)",
            title="Coherence ratio (dashed = random ~1/tph)")
axes[1].legend(); axes[1].grid(True); axes[1].set_yscale("log")
plt.tight_layout()
plt.savefig(os.path.join(OUT, f"entry_output_alignment_{TAG}.png"), dpi=120)
print(f"\nsaved plot -> entry_output_alignment_{TAG}.png")
print("=== DONE ===")
