"""Embedding<->unembedder alignment: dense (exp003) vs LUT (exp070), both 16k untied.
Read-only. Tests whether the LUT reshapes the residual stream so tok_emb no longer aligns
with the learned lm_head (which would make weight-tying costlier for the LUT model)."""
import os, json
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

B = "/home/astarostin/projects/spiky/experiments/hyperplane_ffn"
OUT = f"{B}/exp071_embedding_unembedder_alignment"
os.makedirs(OUT, exist_ok=True)
MODELS = {
    "dense (exp003, 16k, bpb 1.2014)": f"{B}/exp003_untied_vanilla_baseline_nebius_astarostin/checkpoint.pt",
    "LUT (exp070, 16k, bpb 1.2310)": f"{B}/exp070_compressionmhl_A5champ_6h_64-64_nap6_g0_16k/checkpoint.pt",
}
gen = torch.Generator().manual_seed(0)


def row_cos(E, U):
    return torch.nn.functional.cosine_similarity(E, U, dim=1)


def linear_cka(X, Y):
    Xc = X - X.mean(0, keepdim=True)
    Yc = Y - Y.mean(0, keepdim=True)
    hsic = (Yc.T @ Xc).pow(2).sum()
    return (hsic / ((Xc.T @ Xc).norm() * (Yc.T @ Yc).norm())).item()


def frob_cos(E, U):
    return ((E * U).sum() / (E.norm() * U.norm())).item()


def procrustes_err(E, U):
    # unit-Frobenius-normalize each (remove global scale), then optimal rotation
    En = E / E.norm(); Un = U / U.norm()
    M = En.T @ Un                      # [d, d]
    Uu, _, Vt = torch.linalg.svd(M)
    R = Uu @ Vt
    return (En @ R - Un).norm().item()  # in [0, sqrt(2)]; lower = better aligned


res = {}
cos_arrays = {}
for name, path in MODELS.items():
    sd = torch.load(path, map_location="cpu")
    E = sd["tok_emb.weight"].float()   # [V, d]
    U = sd["head.weight"].float()      # [V, d]
    rc = row_cos(E, U)
    perm = torch.randperm(E.shape[0], generator=gen)
    rc_shuf = row_cos(E, U[perm])
    pct = np.percentile(rc.numpy(), [1, 5, 25, 50, 75, 95, 99])
    res[name] = dict(
        rc_mean=rc.mean().item(), rc_median=rc.median().item(), rc_std=rc.std().item(),
        rc_pct={p: float(v) for p, v in zip([1, 5, 25, 50, 75, 95, 99], pct)},
        cka=linear_cka(E, U), frob_cos=frob_cos(E, U),
        procrustes=procrustes_err(E, U),
        ctrl_shuffled_rc_mean=rc_shuf.mean().item(),
        Enorm=E.norm().item(), Unorm=U.norm().item(),
    )
    cos_arrays[name] = rc.numpy()

# histogram
plt.figure(figsize=(8, 4.5))
for name, arr in cos_arrays.items():
    plt.hist(arr, bins=80, alpha=0.55, density=True, label=name.split(" (")[0] + f" (mean {arr.mean():.3f})")
plt.axvline(0, color="k", lw=0.8, ls=":")
plt.xlabel("per-token cosine(tok_emb[i], lm_head[i])"); plt.ylabel("density")
plt.title("Embedding<->unembedder row-cosine alignment: dense vs LUT")
plt.legend(); plt.tight_layout()
plt.savefig(f"{OUT}/row_cosine_hist.png", dpi=120); plt.close()

for name, r in res.items():
    print(f"\n=== {name} ===")
    for k, v in r.items():
        print(f"  {k}: {v}")
with open(f"{OUT}/alignment_stats.json", "w") as f:
    json.dump(res, f, indent=2)
print("\nsaved to", OUT)
