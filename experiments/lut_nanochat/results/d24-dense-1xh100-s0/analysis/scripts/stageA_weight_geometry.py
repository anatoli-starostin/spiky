"""
[lut_nanochat] Stage A: static weight-space geometry of the trained d24 dense baseline checkpoint.
No forward passes, no data. Reads model_005568.pt (+ meta), writes a JSON of measured numbers and PNG plots.

    python results/d24-dense-1xh100-s0/analysis/scripts/stageA_weight_geometry.py [--ckpt-dir DIR] [--out DIR] [--device cuda|cpu]

Any working directory. Paths default relative to this file: the run dir is ../../ (checkpoints/ there), the analysis
dir is ../ (PNGs -> figures/, JSON -> data/). Uses the vendored nanochat (nanochat/) only to build the model on the meta
device for name/shape verification and the code's own parameter-count assertion.

Conventions (all reported numbers are measured on the stored tensors, cast to fp32/fp64):
  energy rank r_p   = smallest r with sum_{i<=r} s_i^2 >= p * sum s_i^2           (p = 0.90, 0.99)
  participation PR  = (sum s_i^2)^2 / sum s_i^4                                    (an "effective number of directions")
  entropy erank     = exp(H(p)), p_i = s_i / sum s                                 (Roy & Vetterli 2007)
  CKA (linear)      = ||Xc^T Yc||_F^2 / (||Xc^T Xc||_F ||Yc^T Yc||_F), Xc = column-centred rows
"""
import argparse, json, math, os, sys, time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))                            # .../results/<run>/analysis/scripts
ANA = os.path.dirname(HERE)                                                  # .../results/<run>/analysis
RUN = os.path.dirname(ANA)                                                   # .../results/<run>
LUT = os.path.dirname(os.path.dirname(RUN))                                                   # experiments/lut_nanochat
p = argparse.ArgumentParser()
p.add_argument("--ckpt-dir", default=os.path.join(RUN, "checkpoints"))
p.add_argument("--step", type=int, default=5568)
p.add_argument("--out", default=ANA)
p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
p.add_argument("--seed", type=int, default=0)
args = p.parse_args()
FIG, DAT = os.path.join(args.out, "figures"), os.path.join(args.out, "data")
os.makedirs(FIG, exist_ok=True); os.makedirs(DAT, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplcfg")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

T0 = time.time()
DEV = torch.device(args.device)
g = torch.Generator().manual_seed(args.seed)
R = {"timings_s": {}}
def tick(name, t):
    R["timings_s"][name] = round(time.time() - t, 1); print(f"[{time.time()-T0:7.1f}s] {name} done", flush=True)

# ---------------------------------------------------------------- 0. verify the artifact
t = time.time()
sys.path.insert(0, os.path.join(LUT, "nanochat"))
from nanochat.gpt import GPT, GPTConfig                                           # noqa: E402
meta = json.load(open(os.path.join(args.ckpt_dir, f"meta_{args.step:06d}.json")))
sd = torch.load(os.path.join(args.ckpt_dir, f"model_{args.step:06d}.pt"), map_location="cpu", mmap=True)
sd = {k.removeprefix("_orig_mod."): v for k, v in sd.items()}
cfg = GPTConfig(**meta["model_config"])
with torch.device("meta"):
    ref = GPT(cfg)
counts = ref.num_scaling_params()                                                # asserts total == sum(numel)
ref_shapes = {k: tuple(v.shape) for k, v in ref.state_dict().items()}
ck_shapes = {k: tuple(v.shape) for k, v in sd.items()}
missing = sorted(set(ref_shapes) - set(ck_shapes)); unexpected = sorted(set(ck_shapes) - set(ref_shapes))
mism = sorted(k for k in set(ref_shapes) & set(ck_shapes) if ref_shapes[k] != ck_shapes[k])
total = sum(v.numel() for v in sd.values())
V = {
    "code_total_params": counts["total"], "checkpoint_total_params": total,
    "expected_total": 1_384_122_122, "vocab_size": cfg.vocab_size, "n_embd": cfg.n_embd, "n_layer": cfg.n_layer,
    "n_head": cfg.n_head, "n_kv_head": cfg.n_kv_head, "sequence_len": cfg.sequence_len, "window_pattern": cfg.window_pattern,
    "meta_step": meta.get("step"), "meta_val_bpb": meta.get("val_bpb"),
    "keys_missing": missing, "keys_unexpected": unexpected, "shape_mismatch": mism,
}
V["ok"] = (total == counts["total"] == 1_384_122_122 and cfg.vocab_size == 32768 and cfg.n_embd == 1536
           and cfg.n_layer == 24 and not missing and not unexpected and not mism and meta.get("step") == args.step)
R["verify"] = V
print(json.dumps(V, indent=1), flush=True)
if not V["ok"]:
    json.dump(R, open(os.path.join(DAT, "stageA_results.json"), "w"), indent=1)
    sys.exit("STOP: artifact does not match the expected d24 configuration")
tick("verify", t)

# ---------------------------------------------------------------- helpers
def f32(name):  return sd[name].float()
def spectrum(W):
    """singular values (fp32 on DEV, returned as fp64 numpy, descending)"""
    s = torch.linalg.svdvals(W.to(DEV, torch.float32))
    return s.double().cpu().numpy()
def rank_stats(s):
    e = s ** 2; c = np.cumsum(e) / e.sum(); pn = s / s.sum()
    return {"full_rank": int(len(s)),
            "r90": int(np.searchsorted(c, 0.90) + 1), "r99": int(np.searchsorted(c, 0.99) + 1),
            "participation_ratio": float(e.sum() ** 2 / (e ** 2).sum()),
            "entropy_erank": float(np.exp(-(pn * np.log(pn + 1e-300)).sum())),
            "s_max": float(s[0]), "s_min": float(s[-1]), "s_median": float(np.median(s))}
def norm_stats(rows):
    n = rows.norm(dim=1).double().cpu().numpy()
    q = np.percentile(n, [1, 5, 25, 50, 75, 95, 99])
    return n, {"mean": float(n.mean()), "std": float(n.std()), "min": float(n.min()), "max": float(n.max()),
               "p1": q[0], "p5": q[1], "p25": q[2], "median": q[3], "p75": q[4], "p95": q[5], "p99": q[6]}
def spearman(a, b):
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])
def id_decile_norms(n):
    d = np.array_split(n, 10); return [float(x.mean()) for x in d]
def anisotropy(W, k=4096):
    idx = torch.randperm(W.shape[0], generator=g)[:k]
    X = W[idx].to(DEV, torch.float32)
    def mc(X):
        Xn = X / X.norm(dim=1, keepdim=True).clamp_min(1e-12)
        C = Xn @ Xn.T; m = C.shape[0]
        return float((C.sum() - C.diagonal().sum()) / (m * (m - 1)))
    return {"sample_rows": k, "mean_pairwise_cos": mc(X), "mean_pairwise_cos_centred": mc(X - X.mean(0, keepdim=True))}
def cka(X, Y):
    X = X.to(DEV, torch.float32); Y = Y.to(DEV, torch.float32)
    X = X - X.mean(0, keepdim=True); Y = Y - Y.mean(0, keepdim=True)
    xy = (X.T @ Y).double().pow(2).sum(); xx = (X.T @ X).double().pow(2).sum().sqrt(); yy = (Y.T @ Y).double().pow(2).sum().sqrt()
    return float(xy / (xx * yy))
def principal_angles_cos(A, B, k=256):
    """cosines of principal angles between the top-k right-singular subspaces of A and B (row spaces in R^d)"""
    _, _, Va = torch.linalg.svd(A.to(DEV, torch.float32), full_matrices=False)
    _, _, Vb = torch.linalg.svd(B.to(DEV, torch.float32), full_matrices=False)
    c = torch.linalg.svdvals(Va[:k] @ Vb[:k].T).double().cpu().numpy()
    return c
def per_row_cos(A, B):
    A = A.to(DEV, torch.float32); B = B.to(DEV, torch.float32)
    c = (A * B).sum(1) / (A.norm(dim=1) * B.norm(dim=1)).clamp_min(1e-12)
    return c.double().cpu().numpy()
def q(x): return {"mean": float(np.mean(x)), "p5": float(np.percentile(x, 5)), "median": float(np.median(x)), "p95": float(np.percentile(x, 95))}

# ---------------------------------------------------------------- 1. wte / lm_head
t = time.time()
V_ = cfg.vocab_size
wte, lmh = f32("transformer.wte.weight"), f32("lm_head.weight")
assert wte.shape == lmh.shape == (V_, 1536), (wte.shape, lmh.shape)
E = {}
plots = {}
for name, W in (("wte", wte), ("lm_head", lmh)):
    n, st = norm_stats(W); s = spectrum(W)
    ids = np.arange(len(n))
    E[name] = {"row_norm": st, "row_norm_vs_token_id_spearman": spearman(n, ids),
               "row_norm_mean_by_id_decile": id_decile_norms(n),
               "spectrum": rank_stats(s), "anisotropy": anisotropy(W)}
    plots[name] = (n, s)
c = per_row_cos(wte, lmh)
pa = principal_angles_cos(wte, lmh, k=256)
E["wte_vs_lm_head"] = {"per_token_cos": q(c), "cka": cka(wte, lmh),
                       "principal_angle_cos_top256": {"mean": float(pa.mean()), "min": float(pa.min()), "max": float(pa.max()),
                                                      "n_above_0.9": int((pa > 0.9).sum())}}
R["embeddings"] = E
tick("wte/lm_head", t)

# ---------------------------------------------------------------- 2. value embeddings
t = time.time()
ve_keys = sorted([k for k in sd if k.startswith("value_embeds.")], key=lambda k: int(k.split(".")[1]))
layers = [int(k.split(".")[1]) for k in ve_keys]
VE = {}
ve_spec = {}
for k, L in zip(ve_keys, layers):
    W = f32(k); n, st = norm_stats(W); s = spectrum(W)
    VE[L] = {"row_norm": st, "row_norm_vs_token_id_spearman": spearman(n, np.arange(len(n))),
             "spectrum": rank_stats(s), "anisotropy": anisotropy(W, k=2048)}
    ve_spec[L] = s
# pairwise CKA and Procrustes-aligned per-token cosine
nL = len(layers)
CKA = np.eye(nL); PROC = np.eye(nL)
tabs = {L: f32(k) for k, L in zip(ve_keys, layers)}
for i in range(nL):
    for j in range(i + 1, nL):
        A, B = tabs[layers[i]].to(DEV), tabs[layers[j]].to(DEV)
        CKA[i, j] = CKA[j, i] = cka(A, B)
        U, _, Vh = torch.linalg.svd(A.T @ B)                      # orthogonal Procrustes: A Q ~ B
        Qm = U @ Vh
        PROC[i, j] = PROC[j, i] = float(np.median(per_row_cos(A @ Qm, B)))
# stacked [V, 12*1536] spectrum via the 18432x18432 Gram (uncentred = reconstruction energy) + centred
Dtot = 1536 * nL
G = torch.zeros(Dtot, Dtot, device=DEV, dtype=torch.float32)
mu = torch.zeros(Dtot, device=DEV, dtype=torch.float64)
for i, Li in enumerate(layers):
    Ai = tabs[Li].to(DEV)
    mu[i*1536:(i+1)*1536] = Ai.double().mean(0)
    for j, Lj in enumerate(layers[i:], start=i):
        blk = Ai.T @ tabs[Lj].to(DEV)
        G[i*1536:(i+1)*1536, j*1536:(j+1)*1536] = blk
        if j != i:
            G[j*1536:(j+1)*1536, i*1536:(i+1)*1536] = blk.T
ev_unc = torch.linalg.eigvalsh(G).double().clamp_min(0).flip(0).cpu().numpy()
Gc = G.double() - V_ * torch.outer(mu, mu); del G
ev_cen = torch.linalg.eigvalsh(Gc.float()).double().clamp_min(0).flip(0).cpu().numpy(); del Gc
def retained(ev, rs): c = np.cumsum(ev) / ev.sum(); return {int(r): float(c[r - 1]) for r in rs}
RS = [8, 16, 32, 64, 128, 256, 512, 1024, 1536, 3072]
# trivial baseline: one shared table + per-layer scalar  ==  rank-1 over the layer axis of vec(V_l)
Mg = torch.zeros(nL, nL, dtype=torch.float64)
for i, Li in enumerate(layers):
    for j, Lj in enumerate(layers):
        Mg[i, j] = (tabs[Li].to(DEV).double() * tabs[Lj].to(DEV).double()).sum().cpu()
lev = torch.linalg.eigvalsh(Mg).flip(0).numpy()
# per-table individual curves (each table alone: rank-r retained of that table)
indiv = {L: retained(ve_spec[L] ** 2, [32, 64, 128, 256, 512]) for L in layers}
VE_cross = {
    "layers": layers,
    "cka_matrix": CKA.round(4).tolist(), "cka_offdiag": q(CKA[np.triu_indices(nL, 1)]),
    "procrustes_median_token_cos_matrix": PROC.round(4).tolist(),
    "procrustes_median_token_cos_offdiag": q(PROC[np.triu_indices(nL, 1)]),
    "stacked_shared_code_variance_retained_uncentred": retained(ev_unc, RS),
    "stacked_shared_code_variance_retained_centred": retained(ev_cen, RS),
    "stacked_rank_stats_uncentred": rank_stats(np.sqrt(ev_unc)),
    "one_shared_table_plus_linear_readout_retained_uncentred": float(np.cumsum(ev_unc)[1535] / ev_unc.sum()),
    "one_shared_table_plus_scalar_gate_retained": float(lev[0] / lev.sum()),
    "layer_axis_eigs_frac": (lev / lev.sum()).round(5).tolist(),
    "per_table_alone_retained": indiv,
}
# linear predictability from wte: R^2 of least-squares wte -> VE_l (centred), fp64 normal equations on CPU
Xw = wte.double(); Xw_c = Xw - Xw.mean(0, keepdim=True)
XtX = Xw_c.T @ Xw_c
Lc = torch.linalg.cholesky(XtX + 1e-8 * torch.eye(1536, dtype=torch.float64) * XtX.diagonal().mean())
r2 = {}
for L in layers:
    Y = tabs[L].double(); Yc = Y - Y.mean(0, keepdim=True)
    Wls = torch.cholesky_solve(Xw_c.T @ Yc, Lc)
    res = Yc - Xw_c @ Wls
    r2[L] = float(1 - res.pow(2).sum() / Yc.pow(2).sum())
VE_cross["r2_linear_from_wte"] = r2
del Xw, Xw_c
# product-quantisation reconstruction error (k-means per subspace, on the trained tables)
def pq_rel_err(W, gsub, K, iters=25):
    W = W.to(DEV, torch.float32); N, D = W.shape; ds = D // gsub; err = 0.0
    for s0 in range(gsub):
        X = W[:, s0*ds:(s0+1)*ds]
        C = X[torch.randperm(N, generator=g)[:K].to(DEV)].clone()
        for _ in range(iters):
            a = torch.cdist(X, C).argmin(1)
            Cn = torch.zeros_like(C).index_add_(0, a, X)
            cnt = torch.bincount(a, minlength=K).float().unsqueeze(1)
            C = torch.where(cnt > 0, Cn / cnt.clamp_min(1), C)
        a = torch.cdist(X, C).argmin(1)
        err += float((X - C[a]).pow(2).sum())
    return math.sqrt(err / float(W.pow(2).sum()))
PQ_SETTINGS = [(8, 256), (16, 256), (48, 256), (96, 256), (16, 1024), (48, 1024)]
pq_layers = [layers[0], layers[len(layers) // 2], layers[-1]]
pq = {f"g{gs}_K{K}": {L: pq_rel_err(tabs[L], gs, K) for L in pq_layers} for gs, K in PQ_SETTINGS}
VE_cross["pq_relative_frobenius_error"] = {"layers_measured": pq_layers, "settings": pq}
R["value_embeds"] = {"per_table": VE, "cross": VE_cross}
# ve_gate: Linear(12 -> n_kv_head) applied to x[..., :12]; gate = 3*sigmoid(W x)
gates = {}
for L in layers:
    Wg = f32(f"transformer.h.{L}.attn.ve_gate.weight")                      # [n_kv_head, 12]
    gates[L] = {"shape": list(Wg.shape), "weight_mean": float(Wg.mean()), "weight_absmax": float(Wg.abs().max()),
                "per_head_row_norm": [round(float(x), 4) for x in Wg.norm(dim=1)],
                "gate_at_x0_is_1.5": True}
R["ve_gate"] = gates
tick("value embeddings", t)

# ---------------------------------------------------------------- 3. per-layer MLP / attention
t = time.time()
LAY = {}
mats = ["attn.c_q", "attn.c_k", "attn.c_v", "attn.c_proj", "mlp.c_fc", "mlp.c_proj"]
for L in range(cfg.n_layer):
    LAY[L] = {}
    for m in mats:
        W = f32(f"transformer.h.{L}.{m}.weight"); s = spectrum(W); st = rank_stats(s)
        st.update({"fro_norm": float(W.norm()), "frac_r90": st["r90"] / st["full_rank"], "frac_r99": st["r99"] / st["full_rank"],
                   "frac_PR": st["participation_ratio"] / st["full_rank"]})
        LAY[L][m] = st
    up = f32(f"transformer.h.{L}.mlp.c_fc.weight")       # [6144, 1536]: row u = input weights of hidden unit u
    dn = f32(f"transformer.h.{L}.mlp.c_proj.weight")     # [1536, 6144]: column u = outgoing weights of unit u
    nin = up.norm(dim=1).numpy(); nout = dn.norm(dim=0).numpy()
    med_out = float(np.median(nout))
    LAY[L]["mlp_units"] = {"in_norm": q(nin), "out_norm": q(nout),
                           "dead_out_lt_1pct_median": int((nout < 0.01 * med_out).sum()),
                           "dead_out_lt_5pct_median": int((nout < 0.05 * med_out).sum()),
                           "out_exact_zero": int((nout == 0).sum()), "in_exact_zero": int((nin == 0).sum())}
R["layers"] = LAY
R["scalars"] = {k: [round(float(x), 5) for x in sd[k].float().flatten()] for k in
                ("resid_lambdas", "x0_lambdas", "smear_lambda", "backout_lambda", "smear_gate.weight")}
tick("per-layer matrices", t)

# ---------------------------------------------------------------- 4. overall
t = time.time()
budget = {}
fsize = os.path.getsize(os.path.join(args.ckpt_dir, f"model_{args.step:06d}.pt"))
tb = 0
nonfinite = {}
for k, v in sd.items():
    grp = ("value_embeds" if k.startswith("value_embeds") else "wte" if "wte" in k else "lm_head" if k.startswith("lm_head")
           else "attn" if ".attn." in k else "mlp" if ".mlp." in k else "scalars/smear")
    b = budget.setdefault(grp, {"params": 0, "bytes": 0, "dtypes": set()})
    b["params"] += v.numel(); b["bytes"] += v.numel() * v.element_size(); b["dtypes"].add(str(v.dtype))
    tb += v.numel() * v.element_size()
    nf = int((~torch.isfinite(v.float())).sum())
    if nf: nonfinite[k] = nf
for b in budget.values(): b["dtypes"] = sorted(b["dtypes"])
R["budget"] = {"groups": budget, "tensor_bytes": tb, "file_bytes": fsize, "container_overhead": fsize - tb}
init_check = {}
for L in range(cfg.n_layer):
    for m in ("attn.c_proj", "mlp.c_proj"):                                    # zero-initialised in init_weights
        init_check[f"h.{L}.{m}"] = float(f32(f"transformer.h.{L}.{m}.weight").norm())
R["sanity"] = {"nonfinite_tensors": nonfinite,
               "zero_init_c_proj_fro_norm": {"min": min(init_check.values()), "max": max(init_check.values())},
               "lm_head_init_std_0.001_vs_now": float(lmh.std()), "wte_init_std_0.8_vs_now": float(wte.std())}
tick("overall", t)

# ---------------------------------------------------------------- plots
def save(fig, name): fig.tight_layout(); fig.savefig(os.path.join(FIG, name), dpi=130); plt.close(fig)
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for name, (n, s) in plots.items():
    ax[0].semilogy(s / s[0], label=name)
for L in layers:
    ax[0].semilogy(ve_spec[L] / ve_spec[L][0], lw=0.8, alpha=0.6, label=f"VE L{L}" if L in (layers[0], layers[-1]) else None)
ax[0].set_xlabel("index"); ax[0].set_ylabel("s_i / s_1"); ax[0].set_title("singular values (normalised)"); ax[0].legend(fontsize=7)
for name, (n, s) in plots.items():
    c = np.cumsum(s ** 2) / (s ** 2).sum(); ax[1].plot(c, label=name)
for L in layers:
    c = np.cumsum(ve_spec[L] ** 2) / (ve_spec[L] ** 2).sum(); ax[1].plot(c, lw=0.8, alpha=0.6)
ax[1].axhline(0.9, ls=":", c="k"); ax[1].axhline(0.99, ls=":", c="k"); ax[1].set_xlabel("rank r"); ax[1].set_ylabel("energy retained")
ax[1].set_title("cumulative energy (VE tables thin lines)"); ax[1].legend(fontsize=7)
save(fig, "spectra.png")

fig, ax = plt.subplots(figsize=(7, 4.5))
cu = np.cumsum(ev_unc) / ev_unc.sum(); cc = np.cumsum(ev_cen) / ev_cen.sum()
rr = np.arange(1, len(cu) + 1)
ax.semilogx(rr, cu, label="stacked 12 tables, uncentred (reconstruction)")
ax.semilogx(rr, cc, label="stacked 12 tables, centred")
for r in (32, 64, 128, 256, 512):
    ax.plot([r], [cu[r - 1]], "o", c="C0"); ax.annotate(f"{cu[r-1]:.3f}", (r, cu[r - 1]), fontsize=7, xytext=(3, -10), textcoords="offset points")
ax.axvline(1536, ls=":", c="gray"); ax.text(1536, 0.05, " one shared table\n + linear readout", fontsize=7)
ax.axhline(VE_cross["one_shared_table_plus_scalar_gate_retained"], ls="--", c="C3", label="one shared table + per-layer scalar")
ax.set_xlabel("shared code rank r"); ax.set_ylabel("variance retained"); ax.set_ylim(0, 1.01); ax.legend(fontsize=7)
ax.set_title("value embeddings: shared rank-r code, variance retained")
save(fig, "ve_variance_retained_vs_rank.png")

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for name, (n, s) in plots.items():
    ax[0].hist(n, bins=200, alpha=0.6, label=name, density=True)
nv = tabs[layers[0]].norm(dim=1).numpy(); ax[0].hist(nv, bins=200, alpha=0.5, label=f"VE L{layers[0]}", density=True)
ax[0].set_xlabel("row L2 norm"); ax[0].set_title("row-norm distributions"); ax[0].legend(fontsize=7); ax[0].set_yscale("log")
for name, (n, s) in plots.items():
    w = 256; sm = np.convolve(n, np.ones(w) / w, mode="valid"); ax[1].plot(np.arange(len(sm)), sm / np.median(n), label=name)
sm = np.convolve(nv, np.ones(256) / 256, mode="valid"); ax[1].plot(sm / np.median(nv), label=f"VE L{layers[0]}")
ax[1].set_xlabel("token id (BPE order)"); ax[1].set_ylabel("row norm / median (256-id moving avg)"); ax[1].legend(fontsize=7)
ax[1].set_title("row norm vs token id")
save(fig, "row_norms.png")

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for m in mats:
    ax[0].plot(range(cfg.n_layer), [LAY[L][m]["frac_r90"] for L in range(cfg.n_layer)], marker=".", label=m)
    ax[1].plot(range(cfg.n_layer), [LAY[L][m]["frac_PR"] for L in range(cfg.n_layer)], marker=".", label=m)
ax[0].set_title("90%-energy rank / full rank"); ax[1].set_title("participation ratio / full rank")
for a in ax: a.set_xlabel("layer"); a.legend(fontsize=7)
save(fig, "effective_rank_vs_depth.png")

R["runtime_s"] = round(time.time() - T0, 1)
R["env"] = {"torch": torch.__version__, "device": str(DEV), "seed": args.seed}
json.dump(R, open(os.path.join(DAT, "stageA_results.json"), "w"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
print(f"DONE in {R['runtime_s']} s -> {args.out}")
