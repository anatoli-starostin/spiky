"""Single-hyperplane unit test: ONE spiking detector learns ONE median-difficulty sign-test.
Grid: membrane tau {8,32,64,integrator} x readout {binary first-spike, graded latency, spike-count}, all SOFT reset.
Reports per cell: mean acc vs the single-hyperplane logistic ceiling, boundary-quintile acc, BCE endpoints."""
import sys, os, json, numpy as np, torch, torch.nn.functional as F
EXP = "/home/astarostin/projects/spiky/experiments/walker2d_lut_distill"
sys.path.insert(0, EXP)
import walker_rsnn_distill_gpu_sparse_fanout as W
DEV, T, D, NI = W.DEV, W.T, W.D, W.NI
spike = W.spike
C_IN, ALPHA_IN, READIN = W.C_IN, W.ALPHA_IN, W.READIN

W_HYP = W.Wd.reshape(-1, NI); B_HYP = W.Bd.reshape(-1)
# --- pick MEDIAN-difficulty hyperplane by the Stage-1 leaky baseline accuracy ---
acc_leaky = np.load(f"{EXP}/walker_rsnn_detectors_acc.npz")["acc"]   # original Stage-1 leaky (tau=8) accuracies
order = np.argsort(acc_leaky); HYP = int(order[len(order) // 2])           # median index
w_h = W_HYP[HYP]; b_h = float(B_HYP[HYP])
def label_of(X): return ((X @ w_h + b_h) > 0).astype(np.float32)
def margin_of(X): return X @ w_h + b_h
def ticks(X): return np.round(C_IN - ALPHA_IN * X).clip(0, READIN - 1).astype(np.float32)

# --- single-hyperplane full-precision logistic ceiling on the integer-tick vector ---
Xtr = W.sample_obs(8192, seed=200); Xte = W.sample_obs(8192, seed=1)
ytr = label_of(Xtr); yte = label_of(Xte); marg = margin_of(Xte)
def logistic_ceiling(Ftr, Fte):
    Ft = torch.tensor((Ftr - Ftr.mean(0)) / (Ftr.std(0) + 1e-6), dtype=torch.float32)
    Fv = torch.tensor((Fte - Ftr.mean(0)) / (Ftr.std(0) + 1e-6), dtype=torch.float32)
    yt = torch.tensor(ytr)
    lin = torch.nn.Linear(NI, 1); opt = torch.optim.Adam(lin.parameters(), lr=0.05)
    for _ in range(3000): opt.zero_grad(); F.binary_cross_entropy_with_logits(lin(Ft)[:, 0], yt).backward(); opt.step()
    with torch.no_grad(): pr = (lin(Fv)[:, 0] > 0).float().numpy()
    return (pr == yte).mean()
CEIL = float(logistic_ceiling(ticks(Xtr), ticks(Xte)))
posrate = float(yte.mean())

class Det1(torch.nn.Module):
    def __init__(self, init_std=0.5):
        super().__init__(); g = torch.Generator().manual_seed(0)
        self.Wd = torch.nn.Parameter(torch.randn(NI, 1, generator=g) * init_std)
        d0 = 1.0 + (D - 1) * torch.rand(NI, 1, generator=g)
        self.dD = torch.nn.Parameter(torch.log((d0 - 1 + 1e-3) / (D - d0 + 1e-3)))
        self.bias = torch.nn.Parameter(torch.zeros(1))
        self.a = torch.nn.Parameter(torch.tensor(1.0)); self.b = torch.nn.Parameter(torch.tensor(0.0))  # readout head
        self.thr = 1.0; self.sigma = 2.0; self.hard = False; self.beta = W.BETA
        self.register_buffer("taus", torch.arange(1, D + 1).float().view(D, 1, 1))
    def kernel(self, dc):
        k = (self.taus == dc.round().clamp(1, D)[None]).float() if self.hard else \
            torch.exp(-(self.taus - dc[None]).pow(2) / (2 * self.sigma ** 2))
        return k / (k.sum(0, keepdim=True) + 1e-8)
    def forward(self, Ispk):
        B = Ispk.shape[0]; dev = self.Wd.device
        EW = self.Wd[None] * self.kernel(1.0 + (D - 1) * torch.sigmoid(self.dD))
        vd = torch.zeros(B, 1, device=dev); Ibuf = torch.zeros(B, D, NI, device=dev)
        notf = torch.ones(B, 1, device=dev); soft_t = torch.zeros(B, 1, device=dev)
        scount = torch.zeros(B, 1, device=dev); hfirst = torch.full((B, 1), float(T), device=dev)
        hdone = torch.zeros(B, 1, dtype=torch.bool, device=dev); hcount = torch.zeros(B, 1, device=dev)
        for t in range(T):
            cur = torch.einsum('bdp,dpo->bo', Ibuf, EW) + self.bias
            vd = self.beta * vd + cur
            s = spike(vd - self.thr); g = torch.sigmoid(W.SURR * (vd - self.thr))
            vd = vd - self.thr * s                                   # SOFT reset (all cells)
            soft_t = soft_t + float(t) * g * notf; notf = notf * (1.0 - g)
            scount = scount + g; hcount = hcount + s
            nw = (s > 0.5) & (~hdone); hfirst = torch.where(nw, torch.full_like(hfirst, float(t)), hfirst); hdone = hdone | nw
            Ibuf = torch.cat([Ispk[:, t:t + 1, :], Ibuf[:, :-1, :]], 1)
        soft_t = soft_t + float(T) * notf
        return dict(p_fire=(1 - notf)[:, 0], E_t=soft_t[:, 0], scount=scount[:, 0],
                    hard_fired=hdone.float()[:, 0], hard_first=hfirst[:, 0], hard_count=hcount[:, 0])

def soft_logit(m, o, readout):
    if readout == "binary":  return None, o["p_fire"]                     # BCE on prob directly
    if readout == "latency": return m.a * (float(T) / 2 - o["E_t"]) + m.b, None
    if readout == "count":   return m.a * o["scount"] + m.b, None
def hard_pred(m, o, readout):
    if readout == "binary":  return (o["hard_fired"] > 0.5).cpu().numpy()
    if readout == "latency": return ((m.a * (float(T) / 2 - o["hard_first"]) + m.b) > 0).cpu().numpy()
    if readout == "count":   return ((m.a * o["hard_count"] + m.b) > 0).cpu().numpy()

def run_cell(tau, readout, steps=2048):
    m = Det1().to(DEV); m.beta = 1.0 if tau is None else float(np.exp(-1.0 / tau))
    opt = torch.optim.Adam(m.parameters(), lr=2e-3)
    ce0 = None; ce_last = None
    for s in range(steps):
        m.sigma = max(0.5, 2.0 - 1.5 * s / (steps - 1))
        W.set_surr(W.SURR_START + (W.SURR_END - W.SURR_START) * (s / (steps - 1)))
        Xb = W.sample_obs(512, seed=1000 + s); lab = torch.as_tensor(label_of(Xb), device=DEV)
        o = m.forward(W.input_raster(Xb)); logit, prob = soft_logit(m, o, readout)
        loss = F.binary_cross_entropy(prob.clamp(1e-5, 1 - 1e-5), lab) if prob is not None \
            else F.binary_cross_entropy_with_logits(logit, lab)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        if s == 0: ce0 = float(loss)
        ce_last = float(loss)
    m.eval(); m.hard = True; preds = []
    with torch.no_grad():
        for i in range(0, len(Xte), 2048):
            preds.append(hard_pred(m, m.forward(W.input_raster(Xte[i:i+2048])), readout))
    pred = np.concatenate(preds); acc = (pred == yte).mean()
    absm = np.abs(marg); qs = np.quantile(absm, [0, .2, .4, .6, .8, 1.0]); corr = (pred == yte)
    quint = [float(corr[(absm >= qs[j]) & (absm < qs[j+1] if j < 4 else absm <= qs[j+1])].mean()) for j in range(5)]
    return dict(acc=float(acc), quint=quint, ce0=ce0, ce_end=ce_last)

TAUS = [("8", 8), ("32", 32), ("64", 64), ("integrator", None)]
READOUTS = ["binary", "latency", "count"]
rep = [f"single-hyperplane unit test | HYP idx={HYP} (median leaky acc={100*acc_leaky[HYP]:.1f}%) "
       f"posrate={100*posrate:.1f}% | integer-tick LOGISTIC CEILING={100*CEIL:.2f}%"]
qd = np.quantile(np.abs(marg), [0, .25, .5, .75, 1.0])
rep.append(f"boundary-distance |w.x+b| quartiles: {[round(float(x),2) for x in qd]}")
results = {}
for tname, tau in TAUS:
    for ro in READOUTS:
        r = run_cell(tau, ro); results[f"{tname}/{ro}"] = r
        rep.append(f"tau={tname:10s} readout={ro:8s} | acc {100*r['acc']:.2f}% (ceil {100*CEIL:.1f}) | "
                   f"quint(near->far) {[round(100*q,1) for q in r['quint']]} | BCE {r['ce0']:.3f}->{r['ce_end']:.3f}")
        print(rep[-1], flush=True)
open(f"{EXP}/walker_rsnn_report_single_hyperplane.txt", "w").write("\n".join(rep))
json.dump(dict(hyp_idx=HYP, median_leaky_acc=float(acc_leaky[HYP]), posrate=posrate, logistic_ceiling=CEIL,
               cells=results), open(f"{EXP}/walker_rsnn_result_single_hyperplane.json", "w"), indent=2)
print("\nwrote walker_rsnn_report_single_hyperplane.txt + walker_rsnn_result_single_hyperplane.json")
