"""CONTROL 1 - input-resolution ceiling: full-precision LOGISTIC REGRESSION (BCE-trained linear, all 192
jointly) from the INTEGER-TICK input vector vs RAW continuous obs -> hyperplane sign labels. CPU only."""
import sys, numpy as np, torch, torch.nn.functional as F
EXP = "/home/astarostin/projects/spiky/experiments/walker2d_lut_distill"
sys.path.insert(0, EXP)
import walker_rsnn_distill_gpu_sparse_fanout as W
torch.set_num_threads(8)

W_HYP = W.Wd.reshape(-1, W.NI); B_HYP = W.Bd.reshape(-1)
C_IN, ALPHA_IN, READIN = W.C_IN, W.ALPHA_IN, W.READIN
def labels(X): return (X @ W_HYP.T + B_HYP[None] > 0).astype(np.float32)
def ticks(X):  return np.round(C_IN - ALPHA_IN * X).clip(0, READIN - 1).astype(np.float32)
def margins(X): return X @ W_HYP.T + B_HYP[None]

Xtr = W.sample_obs(8192, seed=200); Xte = W.sample_obs(8192, seed=1)
Ltr = labels(Xtr); Lte = labels(Xte); marg = margins(Xte)

def fit_logistic(Ftr, Fte, steps=3000, lr=0.05):
    Ft = torch.tensor(Ftr, dtype=torch.float32); Lt = torch.tensor(Ltr, dtype=torch.float32)
    Fv = torch.tensor(Fte, dtype=torch.float32)
    mu = Ft.mean(0); sd = Ft.std(0) + 1e-6
    Ft = (Ft - mu) / sd; Fv = (Fv - mu) / sd
    lin = torch.nn.Linear(Ft.shape[1], 192)
    opt = torch.optim.Adam(lin.parameters(), lr=lr)
    for _ in range(steps):
        opt.zero_grad(); F.binary_cross_entropy_with_logits(lin(Ft), Lt).backward(); opt.step()
    with torch.no_grad():
        pred = (lin(Fv) > 0).float().numpy()
    return (pred == Lte).mean(0), pred

acc_x, _ = fit_logistic(Xtr, Xte)
acc_t, pred_t = fit_logistic(ticks(Xtr), ticks(Xte))
print("=== CONTROL 1: logistic regression -> 192 hyperplane sign-tests (8192 held-out) ===")
print(f"continuous-x  : mean acc {100*acc_x.mean():.2f}%  >99%: {(acc_x>0.99).sum()}/192  min {100*acc_x.min():.1f}%")
print(f"integer-tick  : mean acc {100*acc_t.mean():.2f}%  >99%: {(acc_t>0.99).sum()}/192  min {100*acc_t.min():.1f}%")
print(f"-> tick-quantization cost = {100*(acc_x.mean()-acc_t.mean()):.2f} pp")
absm = np.abs(marg); correct = (pred_t == Lte)
qs = np.quantile(absm, [0, .2, .4, .6, .8, 1.0])
print("integer-tick logistic, accuracy by |w.x+b| quintile (near->far):")
for j in range(5):
    m = (absm >= qs[j]) & (absm < qs[j+1] if j < 4 else absm <= qs[j+1])
    print(f"  |margin| [{qs[j]:.2f},{qs[j+1]:.2f}): acc {100*correct[m].mean():.2f}%")
