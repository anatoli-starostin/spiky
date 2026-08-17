"""E0 — Multi-Map 3D Unembedder localization feasibility (NO training).

PCA the baseline unembedder W into 3D blocks (N maps); for real validation contexts take the
baseline's top-20 predicted tokens; per map fit a 3D Gaussian to those tokens' coords and measure
bits_n = -log2(fraction of vocab inside the 2sigma ellipsoid); report Sum bits and the all-map
intersection's overlap with the top-20.

GATE (at N=11): median Sum bits >= 15  AND  median(|intersection ∩ top20|/20) >= 0.50.

Usage: e0_localization.py <exp_dir_with_checkpoint> [n_contexts] [batch_size]
Runs on CPU by default (leaves the GPU for a concurrent training run).
"""
import os, sys, json, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
NANOCHAT_ROOT = os.environ.get('NANOCHAT_ROOT', os.path.expanduser('~/projects/nanochat'))
if NANOCHAT_ROOT not in sys.path:
    sys.path.insert(0, NANOCHAT_ROOT)
import backbone
from nanochat.tokenizer import RustBPETokenizer
from nanochat.common import get_base_dir
from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit

EXP = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser(
    "~/projects/spiky/experiments/hyperplane_ffn/exp070_compressionmhl_A5champ_6h_64-64_nap6_g0_16k")
N_CTX = int(sys.argv[2]) if len(sys.argv) > 2 else 1500
BS = int(sys.argv[3]) if len(sys.argv) > 3 else 16
DEVICE = 'cpu'
NS = [4, 8, 11, 16]
TOPK = 20
torch.manual_seed(0); np.random.seed(0)

print(f"[E0] baseline = {os.path.basename(EXP)}  n_contexts={N_CTX} bs={BS} device={DEVICE}")
model, cfg, (missing, unexpected) = backbone.load_pretrained(EXP, device=DEVICE)
tied = bool(cfg.get('tie_unembedder', False))
print(f"[E0] loaded. tie={tied}  missing_keys={len(missing)} unexpected_keys={len(unexpected)}")
if missing:
    print("      missing sample:", list(missing)[:4])
if unexpected:
    print("      unexpected sample:", list(unexpected)[:4])

W = model.head.weight.detach().float().numpy()          # [V, 384]
V, D = W.shape
print(f"[E0] W shape {W.shape}  (untied head is a genuine unembedder: {not tied})")

# ---- collect real hidden states h and their top-20 predicted tokens ----
base_dir = get_base_dir()
tok = RustBPETokenizer.from_directory(os.path.join(base_dir, 'tokenizer'))
loader = tokenizing_distributed_data_loader_bos_bestfit(tok, BS, cfg['seq_len'], split='val', device=DEVICE)
Wt = torch.from_numpy(W)
hs, tops = [], []
with torch.no_grad():
    while len(hs) < N_CTX:
        x, y = next(loader)
        h = model.hidden(x).reshape(-1, D)               # [B*T, D]
        logits = h @ Wt.T                                # [B*T, V]  (bias-free head)
        t20 = torch.topk(logits, TOPK, dim=1).indices.numpy()
        hn = h.numpy()
        for i in range(hn.shape[0]):
            hs.append(hn[i]); tops.append(t20[i])
            if len(hs) >= N_CTX:
                break
print(f"[E0] collected {len(hs)} contexts (real h, real baseline top-{TOPK}).")
tops = np.array(tops)                                    # [N_CTX, 20]

# ---- PCA of W once (full), reuse leading dims for each N ----
mu_w = W.mean(0)
Wc = W - mu_w
# right singular vectors of Wc -> principal directions in the 384-space
_, S, Vt = np.linalg.svd(Wc, full_matrices=False)        # Vt [384,384]
Xfull = Wc @ Vt.T                                        # [V,384] token coords in PCA basis

CHI2_2SIGMA = 4.0                                        # Mahalanobis^2 threshold for 2-sigma (3D)


def gaussian_inside_fraction(Xn, idx):
    """Fit full-cov 3D Gaussian to Xn[idx]; return boolean mask [V] of tokens inside 2sigma."""
    pts = Xn[idx]                                        # [k,3]
    mu = pts.mean(0)
    cov = np.cov(pts.T) + 1e-4 * np.eye(3)               # regularized
    inv = np.linalg.inv(cov)
    d = Xn - mu
    m2 = np.einsum('vi,ij,vj->v', d, inv, d)
    return m2 <= CHI2_2SIGMA


results = {}
for N in NS:
    dim = 3 * N
    X = Xfull[:, :dim].copy()
    # standardize each of the 3N columns (spec: "each standardized"); STD=0 disables (sensitivity)
    if os.environ.get('STD', '1') == '1':
        X = (X - X.mean(0)) / (X.std(0) + 1e-8)
    blocks = [X[:, 3 * n:3 * n + 3] for n in range(N)]
    sumbits_list, overlap_list, inter_frac_list = [], [], []
    for c in range(len(hs)):
        top = tops[c]
        inside_all = np.ones(V, dtype=bool)
        bits_sum = 0.0
        for n in range(N):
            inside = gaussian_inside_fraction(blocks[n], top)
            frac = max(inside.mean(), 1.0 / V)
            bits_sum += -np.log2(frac)
            inside_all &= inside
        sumbits_list.append(bits_sum)
        inter_size = int(inside_all.sum())
        inter_frac_list.append(inter_size / V)
        # overlap of the all-map intersection with the true top-20
        overlap_list.append(inside_all[top].mean())      # fraction of top-20 inside intersection
    med_bits = float(np.median(sumbits_list))
    med_overlap = float(np.median(overlap_list))
    med_inter_frac = float(np.median(inter_frac_list))
    results[N] = dict(median_sum_bits=med_bits, median_top20_in_intersection=med_overlap,
                      median_intersection_frac_of_V=med_inter_frac,
                      mean_sum_bits=float(np.mean(sumbits_list)),
                      p25_sum_bits=float(np.percentile(sumbits_list, 25)),
                      p75_sum_bits=float(np.percentile(sumbits_list, 75)))
    print(f"[E0] N={N:2d} (3N={dim}): median Sum bits={med_bits:6.2f}  "
          f"median top20-in-intersection={med_overlap:.3f}  "
          f"median |∩|/V={med_inter_frac:.4g}")

# ---- GATE at N=11 ----
g = results[11]
gate_bits = g['median_sum_bits'] >= 15.0
gate_overlap = g['median_top20_in_intersection'] >= 0.50
passed = gate_bits and gate_overlap
print("\n[E0] GATE @ N=11:")
print(f"     median Sum bits = {g['median_sum_bits']:.2f}  (need >= 15)  -> {'PASS' if gate_bits else 'FAIL'}")
print(f"     median top20-in-intersection = {g['median_top20_in_intersection']:.3f}  (need >= 0.50)  -> {'PASS' if gate_overlap else 'FAIL'}")
print(f"     OVERALL E0: {'PASS' if passed else 'FAIL'}")

out = dict(baseline=os.path.basename(EXP), n_contexts=len(hs), tied=tied,
           results={str(k): v for k, v in results.items()},
           gate=dict(median_sum_bits=g['median_sum_bits'], median_top20_in_intersection=g['median_top20_in_intersection'],
                     bits_ok=bool(gate_bits), overlap_ok=bool(gate_overlap), passed=bool(passed)))
outpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'e0_results.json')
json.dump(out, open(outpath, 'w'), indent=2)
print(f"[E0] wrote {outpath}")
