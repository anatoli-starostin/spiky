"""Stage-1 layerwise decomposition: feedforward SPIKING HYPERPLANE DETECTORS trained (BCE) to reproduce
the teacher LUT's 192 sign-tests s_i = 1[w_i.x + b_i > 0]. Reuses the input latency encoding + LIF +
arctan surrogate from the main net; NO recurrence (input -> 192 detector neurons). Logit = soft
'fires-in-window' probability; hard eval = detector actually spikes."""
import sys, numpy as np, torch, torch.nn.functional as F
EXP = "/home/astarostin/projects/spiky/experiments/walker2d_lut_distill"
sys.path.insert(0, EXP)
import walker_rsnn_distill_gpu_sparse_fanout as W   # helpers: sample_obs, input_raster, spike, BETA, set_surr, Wd/Bd
import os
DEV, T, D, NI = W.DEV, W.T, W.D, W.NI
BETA = float(os.environ.get("RSNN_BETA", str(W.BETA)))       # neuron leak: default leaky tau=8; 1.0 = leak-free IF
HARD_RESET = BETA >= 0.999                                   # leak-free integrator uses hard reset (v->0), per spec
TAG = os.environ.get("RSNN_DET_TAG", "leaky")
spike = W.spike
print(f"[detectors] BETA={BETA:.4f} ({'leak-free IF, hard reset' if HARD_RESET else 'leaky LIF, soft reset'}) tag={TAG}", flush=True)

# --- the 192 teacher hyperplanes in normalized-obs space (from the int4 LUT) ---
W_HYP = W.Wd.reshape(-1, NI)          # (192, 17)   w_i
B_HYP = W.Bd.reshape(-1)              # (192,)      b_i
N_HYP = W_HYP.shape[0]                # 192
N_DET = N_HYP                         # one detector per hyperplane (exactly 192; report this)

def labels_of(Xn):                    # (M,17) -> (M,192) exact sign-test targets
    return ((Xn @ W_HYP.T + B_HYP[None]) > 0).astype(np.float32)

def margins_of(Xn):                   # signed distance-ish w.x+b (for boundary-sensitivity binning)
    return Xn @ W_HYP.T + B_HYP[None]

class Detectors(torch.nn.Module):
    def __init__(self, n_det, init_std=0.5):
        super().__init__()
        g = torch.Generator().manual_seed(0)
        self.Wd = torch.nn.Parameter(torch.randn(NI, n_det, generator=g) * init_std)   # SIGNED input->detector
        d0 = 1.0 + (D - 1) * torch.rand(NI, n_det, generator=g)
        self.dD = torch.nn.Parameter(torch.log((d0 - 1 + 1e-3) / (D - d0 + 1e-3)))      # DCLS delays [1,D]
        self.bias = torch.nn.Parameter(torch.zeros(n_det))                             # per-detector bias current
        self.thr = 1.0; self.sigma = 2.0; self.hard = False
        self.register_buffer("taus", torch.arange(1, D + 1).float().view(D, 1, 1))

    def kernel(self, dcont):
        if self.hard:
            k = (self.taus == dcont.round().clamp(1, D)[None]).float()
        else:
            k = torch.exp(-(self.taus - dcont[None]).pow(2) / (2 * self.sigma ** 2))
        return k / (k.sum(0, keepdim=True) + 1e-8)

    def forward(self, Ispk):
        B = Ispk.shape[0]; dev = self.Wd.device
        d = 1.0 + (D - 1) * torch.sigmoid(self.dD)
        EW = self.Wd[None] * self.kernel(d)                       # (D, NI, n_det)
        vd = torch.zeros(B, self.Wd.shape[1], device=dev)
        Ibuf = torch.zeros(B, D, NI, device=dev)
        notfired = torch.ones(B, self.Wd.shape[1], device=dev)
        fired = torch.zeros(B, self.Wd.shape[1], dtype=torch.bool, device=dev)
        for t in range(T):
            cur = torch.einsum('bdp,dpo->bo', Ibuf, EW) + self.bias
            vd = BETA * vd + cur
            s = spike(vd - self.thr); g = torch.sigmoid(W.SURR * (vd - self.thr))
            vd = vd * (1.0 - s) if HARD_RESET else vd - self.thr * s
            notfired = notfired * (1.0 - g)
            fired = fired | (s > 0.5)
            Ibuf = torch.cat([Ispk[:, t:t + 1, :], Ibuf[:, :-1, :]], 1)
        return 1.0 - notfired, fired.float()                     # soft p_fire, hard fired

@torch.no_grad()
def calibrate(det, Ispk, target=0.5):                            # global thr -> ~50% mean fire rate at init
    lo, hi = 1e-3, 50.0
    for _ in range(24):
        mid = (lo * hi) ** 0.5; det.thr = mid
        p, _ = det.forward(Ispk); r = float(p.mean())
        if r > target: lo = mid
        else: hi = mid
    det.thr = (lo * hi) ** 0.5
    return det.thr

det = Detectors(N_DET).to(DEV)
Xc = W.sample_obs(512, seed=0)
calibrate(det, W.input_raster(Xc))
opt = torch.optim.Adam(det.parameters(), lr=2e-3)
STEPS = 2048
for s in range(STEPS):
    det.sigma = max(0.5, 2.0 - 1.5 * s / (STEPS - 1))
    W.set_surr(W.SURR_START + (W.SURR_END - W.SURR_START) * (s / (STEPS - 1)))
    Xb = W.sample_obs(512, seed=1000 + s)
    lab = torch.as_tensor(labels_of(Xb), device=DEV)
    p, _ = det.forward(W.input_raster(Xb))
    loss = F.binary_cross_entropy(p.clamp(1e-5, 1 - 1e-5), lab)
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(det.parameters(), 1.0); opt.step()
    if s % 256 == 0 or s == STEPS - 1:
        print(f"step {s:4d}  bce={float(loss):.4f}  thr={det.thr:.3g}", flush=True)

# ---- hard-eval on disjoint held-out ----
det.eval(); det.hard = True
Xv = W.sample_obs(8192, seed=1); labv = labels_of(Xv); marg = margins_of(Xv)
preds = []
with torch.no_grad():
    for i in range(0, len(Xv), 1024):
        _, fired = det.forward(W.input_raster(Xv[i:i+1024]))
        preds.append(fired.cpu().numpy())
pred = np.concatenate(preds)                       # (8192, 192) hard detector fired
acc = (pred == labv).mean(0)                        # per-hyperplane accuracy
posrate = labv.mean(0)                              # per-hyperplane positive rate
majbase = np.maximum(posrate, 1 - posrate)          # majority-class baseline per hyperplane
np.savez(f"{EXP}/walker_rsnn_detectors_acc_{TAG}.npz", acc=acc, posrate=posrate)

print("\n=== SPIKING HYPERPLANE DETECTORS (feedforward, N_DET=192, hard first-spike eval, 8192 held-out) ===")
print(f"MEAN per-hyperplane accuracy: {100*acc.mean():.2f}%  (mean majority baseline {100*majbase.mean():.2f}%, "
      f"mean lift {100*(acc-majbase).mean():+.2f} pp)")
print(f">99%: {(acc>0.99).sum()}/192   >95%: {(acc>0.95).sum()}/192   >90%: {(acc>0.90).sum()}/192   "
      f">majority: {(acc>majbase+1e-9).sum()}/192")
print("accuracy histogram (bins 50-100% by 5):")
h, edges = np.histogram(acc, bins=np.linspace(0.5, 1.0, 11))
print("  " + "  ".join(f"{int(100*edges[i])}-{int(100*edges[i+1])}:{h[i]}" for i in range(len(h))))
worst = np.argsort(acc)[:6]
print("worst 6 hyperplanes (idx: acc / posrate / majbase):")
for i in worst:
    print(f"  hyp{i:3d}: acc {100*acc[i]:.1f}%  posrate {100*posrate[i]:.1f}%  maj {100*majbase[i]:.1f}%")
# boundary sensitivity: accuracy vs |w.x+b| distance to boundary (pooled over all hyperplanes)
absm = np.abs(marg); correct = (pred == labv)
qs = np.quantile(absm, [0, .2, .4, .6, .8, 1.0])
print("boundary-sensitivity: accuracy by |w.x+b| quintile (near->far):")
for j in range(5):
    m = (absm >= qs[j]) & (absm <= qs[j+1] if j == 4 else absm < qs[j+1])
    print(f"  |margin| [{qs[j]:.2f},{qs[j+1]:.2f}): acc {100*correct[m].mean():.2f}%")
print(f"\nglobal mean positive-rate {100*posrate.mean():.1f}% (min {100*posrate.min():.1f}% max {100*posrate.max():.1f}%)")
