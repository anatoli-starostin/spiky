"""[lut_nanochat] Stage A addendum: rows still at init scale (tokens whose embedding rows were never / barely updated),
and the value-embedding redundancy numbers recomputed on TRAINED rows only (rows clearly above init scale).
Init (nanochat/gpt.py init_weights): wte ~ N(0, 0.8) -> row norm ~ 0.8*sqrt(1536) = 31.4;
value_embeds ~ U(-s, s), s = sqrt(3)/sqrt(1536) -> row norm ~ 1.0; lm_head ~ N(0, 0.001) -> row norm ~ 0.039."""
import json, os, torch
ANA = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))             # .../results/<run>/analysis
CK = os.path.join(os.path.dirname(ANA), "checkpoints/model_005568.pt")
OUT = os.path.join(ANA, "data/stageA_untrained_rows.json")
sd = torch.load(CK, map_location="cpu", mmap=True)
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
res = {}
wn = sd["transformer.wte.weight"].float().norm(dim=1)
ln = sd["lm_head.weight"].float().norm(dim=1)
res["wte_rows_norm_lt_2x_init(62.8)"] = int((wn < 62.8).sum())
res["wte_rows_norm_lt_1.2x_init(37.7)"] = int((wn < 37.7).sum())
res["lm_head_rows_norm_lt_10x_init(0.39)"] = int((ln < 0.39).sum())
ve_keys = sorted([k for k in sd if k.startswith("value_embeds.")], key=lambda k: int(k.split(".")[1]))
near = None
per = {}
for k in ve_keys:
    n = sd[k].float().norm(dim=1)
    m = n < 2.0                                                    # < 2x init row norm
    per[k.split(".")[1]] = int(m.sum())
    near = m if near is None else (near & m)
res["ve_rows_norm_lt_2x_init_per_layer"] = per
res["ve_rows_lt_2x_init_in_ALL_12_tables"] = int(near.sum())
untrained_ids = torch.nonzero(near).flatten()
res["those_token_ids_min_max"] = [int(untrained_ids.min()), int(untrained_ids.max())] if len(untrained_ids) else None
res["overlap_with_wte_near_init(<37.7)"] = int((near & (wn < 37.7)).sum())
# redundancy recomputed on trained rows only
keep = ~near
X = torch.cat([sd[k].float()[keep] for k in ve_keys], dim=1).to(DEV)          # [N_trained, 12*1536]
G = X.T @ X
ev = torch.linalg.eigvalsh(G).double().clamp_min(0).flip(0).cpu()
c = torch.cumsum(ev, 0) / ev.sum()
res["trained_rows"] = int(keep.sum())
res["stacked_retained_uncentred_trained_rows"] = {r: round(float(c[r - 1]), 4) for r in (32, 64, 128, 256, 512, 1024, 1536, 3072)}
Xw = sd["transformer.wte.weight"].double()[keep]; Xw = Xw - Xw.mean(0, keepdim=True)
L = torch.linalg.cholesky(Xw.T @ Xw)
r2 = {}
for k in ve_keys:
    Y = sd[k].double()[keep]; Y = Y - Y.mean(0, keepdim=True)
    W = torch.cholesky_solve(Xw.T @ Y, L)
    r2[k.split(".")[1]] = round(float(1 - (Y - Xw @ W).pow(2).sum() / Y.pow(2).sum()), 4)
res["r2_linear_from_wte_trained_rows"] = r2
json.dump(res, open(OUT, "w"), indent=1)
print(json.dumps(res, indent=1))
