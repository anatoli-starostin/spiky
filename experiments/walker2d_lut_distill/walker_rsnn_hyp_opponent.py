"""Dale-legal opponent-pair WTA microcircuit for ONE hyperplane (#84, ceiling 98.97%).
P,N = excitatory principal cells (+softplus input weights, LEARNABLE DCLS delays [1,64]); race on first-spike.
Inhibitory interneurons (-softplus weights) mediate competition. PROJECT INVARIANT (b379f628):
  - E edges (input->P/N, and E->I excitatory) : learnable DCLS delays [1,64].
  - I edges (I->targets)                      : FIXED delay = 1, delay NOT trained (weight still learnable).
Near-integrator (BETA=1), soft reset. Comparator readout on P vs N first-spike. Arms: single/pair_noinh/shared/cross."""
import sys, json, numpy as np, torch, torch.nn.functional as F
EXP = "/home/astarostin/projects/spiky/experiments/walker2d_lut_distill"
sys.path.insert(0, EXP)
import walker_rsnn_distill_gpu_sparse_fanout as W
DEV, T, D, NI = W.DEV, W.T, W.D, W.NI
spike = W.spike; C_IN, ALPHA_IN, READIN = W.C_IN, W.ALPHA_IN, W.READIN
BETA = 1.0

W_HYP = W.Wd.reshape(-1, NI); B_HYP = W.Bd.reshape(-1); HYP = 84
w_h = W_HYP[HYP]; b_h = float(B_HYP[HYP])
def label_of(X): return ((X @ w_h + b_h) > 0).astype(np.float32)
def margin_of(X): return X @ w_h + b_h
Xte = W.sample_obs(8192, seed=1); yte = label_of(Xte); marg = margin_of(Xte); CEIL = 0.9897

def splus_inv(y): return torch.log(torch.expm1(torch.as_tensor(y).clamp_min(1e-9)))
def dpar(shape, g):
    d0 = 1.0 + (D - 1) * torch.rand(shape, generator=g)
    return torch.nn.Parameter(torch.log((d0 - 1 + 1e-3) / (D - d0 + 1e-3)))
def wpar(shape, g, std=0.5): return torch.nn.Parameter(splus_inv((torch.randn(shape, generator=g) * std).abs()))

class Circuit(torch.nn.Module):
    def __init__(self, arm):
        super().__init__(); self.arm = arm
        gP = torch.Generator().manual_seed(0); gN = torch.Generator().manual_seed(1); gI = torch.Generator().manual_seed(2)
        self.tP = wpar((NI, 1), gP); self.dP = dpar((NI, 1), gP); self.biasP = torch.nn.Parameter(torch.zeros(1))
        self.tN = wpar((NI, 1), gN); self.dN = dpar((NI, 1), gN); self.biasN = torch.nn.Parameter(torch.zeros(1))
        self.a = torch.nn.Parameter(torch.tensor(0.3))
        if arm == "pair_shared":                              # E->I: learnable DCLS ; I->tgt: fixed delay 1
            self.wP_I = wpar((1, 1), gI); self.dP_I = dpar((1, 1), gI)
            self.wN_I = wpar((1, 1), gI); self.dN_I = dpar((1, 1), gI)
            self.wI_P = wpar((1, 1), gI); self.wI_N = wpar((1, 1), gI)
        if arm == "pair_cross":
            self.wP_IP = wpar((1, 1), gI); self.dP_IP = dpar((1, 1), gI)
            self.wN_IN = wpar((1, 1), gI); self.dN_IN = dpar((1, 1), gI)
            self.wIP_N = wpar((1, 1), gI); self.wIN_P = wpar((1, 1), gI)
        self.thr = 1.0; self.sigma = 2.0; self.hard = False
        self.register_buffer("taus", torch.arange(1, D + 1).float().view(D, 1, 1))
    def kernel(self, dc):
        k = (self.taus == dc.round().clamp(1, D)[None]).float() if self.hard else \
            torch.exp(-(self.taus - dc[None]).pow(2) / (2 * self.sigma ** 2))
        return k / (k.sum(0, keepdim=True) + 1e-8)
    def ew(self, w, draw): return F.softplus(w)[None] * self.kernel(1 + (D - 1) * torch.sigmoid(draw))   # (D,pre,post)
    def forward(self, Ispk):
        B = Ispk.shape[0]; dev = self.tP.device
        EWP = self.ew(self.tP, self.dP); EWN = self.ew(self.tN, self.dN)
        EW_PI = EW_NI = EW_PIP = EW_NIN = None
        if self.arm == "pair_shared": EW_PI = self.ew(self.wP_I, self.dP_I); EW_NI = self.ew(self.wN_I, self.dN_I)
        if self.arm == "pair_cross":  EW_PIP = self.ew(self.wP_IP, self.dP_IP); EW_NIN = self.ew(self.wN_IN, self.dN_IN)
        z1 = lambda: torch.zeros(B, 1, device=dev)
        vP = z1(); vN = z1(); vI = z1(); vIP = z1(); vIN = z1()
        Ibuf = torch.zeros(B, D, NI, device=dev); Pbuf = torch.zeros(B, D, 1, device=dev); Nbuf = torch.zeros(B, D, 1, device=dev)
        pI = z1(); pIP = z1(); pIN = z1()
        stP = z1(); nfP = torch.ones(B, 1, device=dev); stN = z1(); nfN = torch.ones(B, 1, device=dev)
        cP = z1(); cN = z1(); cI = z1()
        hP = torch.full((B, 1), float(T), device=dev); hdP = torch.zeros(B, 1, dtype=torch.bool, device=dev)
        hN = torch.full((B, 1), float(T), device=dev); hdN = torch.zeros(B, 1, dtype=torch.bool, device=dev)
        for t in range(T):
            curP = torch.einsum('bdp,dpo->bo', Ibuf, EWP) + self.biasP
            curN = torch.einsum('bdp,dpo->bo', Ibuf, EWN) + self.biasN
            if self.arm == "pair_shared":                     # I->P,N : FIXED delay 1 (prev I spike), learnable weight
                curP = curP - F.softplus(self.wI_P)[0] * pI; curN = curN - F.softplus(self.wI_N)[0] * pI
            elif self.arm == "pair_cross":
                curN = curN - F.softplus(self.wIP_N)[0] * pIP; curP = curP - F.softplus(self.wIN_P)[0] * pIN
            vP = BETA * vP + curP; sP = spike(vP - self.thr); gP = torch.sigmoid(W.SURR * (vP - self.thr)); vP = vP - self.thr * sP
            vN = BETA * vN + curN; sN = spike(vN - self.thr); gN = torch.sigmoid(W.SURR * (vN - self.thr)); vN = vN - self.thr * sN
            if self.arm == "pair_shared":                     # E->I : learnable DCLS (delayed P,N spike buffers)
                curI = torch.einsum('bdp,dpo->bo', Pbuf, EW_PI) + torch.einsum('bdp,dpo->bo', Nbuf, EW_NI)
                vI = BETA * vI + curI; sI = spike(vI - self.thr); vI = vI - self.thr * sI; cI = cI + sI; pI = sI
            elif self.arm == "pair_cross":
                vIP = BETA * vIP + torch.einsum('bdp,dpo->bo', Pbuf, EW_PIP); sIP = spike(vIP - self.thr); vIP = vIP - self.thr * sIP; pIP = sIP
                vIN = BETA * vIN + torch.einsum('bdp,dpo->bo', Nbuf, EW_NIN); sIN = spike(vIN - self.thr); vIN = vIN - self.thr * sIN; pIN = sIN
                cI = cI + 0.5 * (sIP + sIN)
            stP = stP + float(t) * gP * nfP; nfP = nfP * (1 - gP)
            stN = stN + float(t) * gN * nfN; nfN = nfN * (1 - gN)
            cP = cP + sP; cN = cN + sN
            nwP = (sP > 0.5) & (~hdP); hP = torch.where(nwP, torch.full_like(hP, float(t)), hP); hdP = hdP | nwP
            nwN = (sN > 0.5) & (~hdN); hN = torch.where(nwN, torch.full_like(hN, float(t)), hN); hdN = hdN | nwN
            Ibuf = torch.cat([Ispk[:, t:t + 1, :], Ibuf[:, :-1, :]], 1)
            Pbuf = torch.cat([sP[:, None, :], Pbuf[:, :-1, :]], 1); Nbuf = torch.cat([sN[:, None, :], Nbuf[:, :-1, :]], 1)
        stP = stP + float(T) * nfP; stN = stN + float(T) * nfN
        return dict(EtP=stP[:, 0], EtN=stN[:, 0], hP=hP[:, 0], hN=hN[:, 0],
                    rP=(cP.mean() / T).item(), rN=(cN.mean() / T).item(), rI=(cI.mean() / T).item())

def run(arm, steps=2048):
    m = Circuit(arm).to(DEV); opt = torch.optim.Adam(m.parameters(), lr=2e-3); ce0 = ce = None
    for s in range(steps):
        m.sigma = max(0.5, 2.0 - 1.5 * s / (steps - 1))
        W.set_surr(W.SURR_START + (W.SURR_END - W.SURR_START) * (s / (steps - 1)))
        Xb = W.sample_obs(512, seed=1000 + s); lab = torch.as_tensor(label_of(Xb), device=DEV)
        o = m.forward(W.input_raster(Xb))
        d = (float(T) / 2 - o["EtP"]) if arm == "single" else (o["EtN"] - o["EtP"])
        loss = F.binary_cross_entropy_with_logits(m.a * d, lab)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        if s == 0: ce0 = float(loss)
        ce = float(loss)
    m.eval(); m.hard = True; preds = []; rP = rN = rI = 0.0
    with torch.no_grad():
        for i in range(0, len(Xte), 2048):
            o = m.forward(W.input_raster(Xte[i:i+2048]))
            pr = (o["hP"] < float(T)).cpu().numpy() if arm == "single" else (o["hP"] < o["hN"]).cpu().numpy()
            preds.append(pr); rP, rN, rI = o["rP"], o["rN"], o["rI"]
    pred = np.concatenate(preds); acc = (pred == yte).mean()
    absm = np.abs(marg); qs = np.quantile(absm, [0, .2, .4, .6, .8, 1.0]); corr = (pred == yte)
    quint = [float(corr[(absm >= qs[j]) & (absm < qs[j+1] if j < 4 else absm <= qs[j+1])].mean()) for j in range(5)]
    return dict(acc=float(acc), quint=quint, ce0=ce0, ce=ce, rP=rP, rN=rN, rI=rI)

ARMS = ["single", "pair_noinh", "pair_shared", "pair_cross"]
rep = [f"opponent-pair WTA | HYP #{HYP} ceiling {100*CEIL:.2f}% | BETA={BETA} near-integrator, soft reset, Dale-legal;"
       f" E edges learnable-DCLS, I edges FIXED delay 1"]
res = {}
for arm in ARMS:
    r = run(arm); res[arm] = r
    rep.append(f"{arm:12s} | acc {100*r['acc']:.2f}% | quint(near->far) {[round(100*q,1) for q in r['quint']]} | "
               f"rates P/N/I {r['rP']:.3f}/{r['rN']:.3f}/{r['rI']:.3f} | BCE {r['ce0']:.3f}->{r['ce']:.3f}")
    print(rep[-1], flush=True)
open(f"{EXP}/walker_rsnn_report_single_hyperplane_opponent.txt", "w").write("\n".join(rep))
json.dump(dict(hyp=HYP, ceiling=CEIL, arms=res), open(f"{EXP}/walker_rsnn_result_single_hyperplane_opponent.json", "w"), indent=2)
print("wrote report + json (single_hyperplane_opponent)")
