"""E0-SOFT — feasibility of the SOFT-intersection MDN head (training-free diagnostic).

E0 (hard) requires a token inside ALL N 2σ ellipsoids (an N-way AND that decays ~0.85^N). The real
head intersects SOFTLY: score(v) = Σ_n log N(x_v^(n); μ_n, Σ_n). This diagnostic replaces the hard
AND with the soft log-density SUM and asks the question that actually matters: when we rank all V
tokens by the soft score, are the baseline's true top-20 recovered near the top? That is the honest,
still-training-free feasibility of the soft head with PCA-warm coordinates — it informs whether E1 is
worth running. NO head module, NO training.

Per-map Gaussians are still FIT to each context's true top-20 (same warm proxy as E0).
Reports median recall@K of the top-20 and the median soft-rank of the top-1 token.
Usage: e0_soft.py <exp_dir> [n_contexts] [batch_size]   (STD env toggles column standardization)
"""
import os, sys, json
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
N_CTX = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
BS = int(sys.argv[3]) if len(sys.argv) > 3 else 16
NS = [8, 11, 16]
TOPK = 20
KS = [20, 50, 100, 500]
torch.manual_seed(0); np.random.seed(0)

model, cfg, _ = backbone.load_pretrained(EXP, device='cpu')
W = model.head.weight.detach().float().numpy()
V, D = W.shape
print(f"[E0-soft] baseline={os.path.basename(EXP)} W={W.shape} n_ctx={N_CTX}")

base_dir = get_base_dir()
tok = RustBPETokenizer.from_directory(os.path.join(base_dir, 'tokenizer'))
loader = tokenizing_distributed_data_loader_bos_bestfit(tok, BS, cfg['seq_len'], split='val', device='cpu')
Wt = torch.from_numpy(W)
tops = []
with torch.no_grad():
    while len(tops) < N_CTX:
        x, y = next(loader)
        h = model.hidden(x).reshape(-1, D)
        t20 = torch.topk(h @ Wt.T, TOPK, dim=1).indices.numpy()
        for i in range(t20.shape[0]):
            tops.append(t20[i])
            if len(tops) >= N_CTX:
                break
tops = np.array(tops)
print(f"[E0-soft] {len(tops)} contexts")

Wc = W - W.mean(0)
_, S, Vt = np.linalg.svd(Wc, full_matrices=False)
Xfull = Wc @ Vt.T
STD = os.environ.get('STD', '1') == '1'


def soft_logscore(blocks, idx):
    """soft score over all V = Σ_n logN(x_v; μ_n fit to idx, Σ_n fit to idx)."""
    total = np.zeros(V)
    for Xn in blocks:
        pts = Xn[idx]
        mu = pts.mean(0)
        cov = np.cov(pts.T) + 1e-4 * np.eye(3)
        inv = np.linalg.inv(cov)
        sign, logdet = np.linalg.slogdet(cov)
        d = Xn - mu
        m2 = np.einsum('vi,ij,vj->v', d, inv, d)
        total += -0.5 * (m2 + logdet)     # drop const 3/2 log(2π) (rank-invariant)
    return total


for N in NS:
    dim = 3 * N
    X = Xfull[:, :dim].copy()
    if STD:
        X = (X - X.mean(0)) / (X.std(0) + 1e-8)
    blocks = [X[:, 3 * n:3 * n + 3] for n in range(N)]
    recalls = {k: [] for k in KS}
    top1_ranks = []
    for c in range(len(tops)):
        top = tops[c]
        score = soft_logscore(blocks, top)
        order = np.argsort(-score)                    # best first
        rank_of = np.empty(V, dtype=np.int64)
        rank_of[order] = np.arange(V)
        r = rank_of[top]                              # soft-ranks of the 20 true tokens
        for k in KS:
            recalls[k].append(np.mean(r < k))         # fraction of top-20 within soft top-k
        top1_ranks.append(int(r.min()))               # best-ranked of the 20
    line = " ".join(f"r@{k}={np.median(recalls[k]):.2f}" for k in KS)
    print(f"[E0-soft] N={N:2d}: {line}  median_best_rank={int(np.median(top1_ranks))}  "
          f"(top-20 soft-ranked out of V={V})")
