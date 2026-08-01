"""Companded-OUTPUT variant of walker_rsnn_distill_gpu_phased.py (keeps phased 64/32/32 T=128 schedule,
GPU, random per-step batches, delays [1,64]). Originals untouched; nothing committed.

ONLY change vs the phased run: the linear first-spike output code is replaced by a per-dim NON-LINEAR
probit (Gaussian-CDF) COMPANDED code. Per action dim d: standardize z=(a-mu_d)/sigma_d (mu/sigma = mean/std
of the teacher action dist over a 16384 independent sample, seed=7); the first-spike tick maps through the
Gaussian quantile warp so tick-resolution is FINE near each dim's centre (dense mass) and COARSE toward the
tails (uniform tick occupancy = information-optimal for ~Gaussian data). Read-out window/length unchanged
(32 ticks). Orientation kept as the linear run: EARLY spike -> HIGH action, non-fire -> LOW action (so a
silent output saturates low, not high). Decode: u=(T-t)/READOUT_SPAN in [0,1], z=Phi^{-1}(u), a=mu+sigma*z,
via torch.erfinv (fully differentiable). u clamped to [U_EPS,1-U_EPS] (U_EPS=1e-4 -> |z|<=3.72, which covers
the full observed per-dim action range with 0% tail clip). Loss stays action-space MSE (isolates the code).

--- original phased header ---
Phased-schedule variant of walker_rsnn_distill_gpu.py (GPU port + random per-step batches kept).
Original walker_rsnn_distill.py untouched; nothing committed.

NEW: explicit three-phase tick schedule replacing the flat T=32 rollout:
  READ-IN  [0,64)   : 64 ticks. Inputs are latency-coded to SPAN this window (C_IN=32, ALPHA_IN=6.0,
                      double the baseline 16/3.0 so x in [-5,5] fills [~0,64)); input spikes clipped to [0,63].
  HIDDEN   [64,96)  : 32 ticks. Pure recurrent settling/computation (no output readout).
  READ-OUT [96,128) : 32 ticks. The 6 first-spike outputs are decoded from spikes in THIS window only.
  Total T = 64 + 32 + 32 = 128.
Output ground-truth shifted right into the read-out window: decode uses C_OUT = READOUT_START + READOUT//2
= 112 (= baseline's 16 + 96 read-in+hidden offset), ALPHA_OUT=3.0 unchanged (same per-tick resolution and
32-tick span as baseline). Predicted E[t] (soft) and hard first-spike both accumulate only in [96,128) and
decode through the same C_OUT, so predicted and target live in the same (action) frame — equivalent to adding
+96 to the target spike-time encoding.
Learnable DCLS delays widened to [1,64] (D=64, was 16) on the E connections; delay buffers sized to 64.
Inhibitory delay fixed at 1 (unchanged, still representable).
Otherwise baseline: H128/32, Adam 2e-3, clip 1.0, init-std selection logic, batch 512 random per step, 1000 steps.
"""
import os, sys, json, time
import numpy as np
import torch
import torch.nn.functional as F

torch.set_num_threads(max(1, os.cpu_count() - 1))
DEV = "cuda" if torch.cuda.is_available() else "cpu"
SMOKE = os.environ.get("RSNN_SMOKE", "0") == "1"

# ------------------------------------------------------------------ sizes / hyperparams
NI, NEX, NINH, NO = 17, 256, 64, 6                        # BIGGER hidden: H_ex 128->256, H_inh 32->64
if SMOKE:
    NEX, NINH = 32, 8
NH = NEX + NINH

# ---- three-phase tick schedule ----
if SMOKE:
    READIN, HIDDEN, READOUT = 4, 2, 2
    D = 8
else:
    READIN  = int(os.environ.get("RSNN_READIN", "64"))     # input presentation / encoding
    HIDDEN  = int(os.environ.get("RSNN_HIDDEN", "32"))     # recurrent settling
    READOUT = int(os.environ.get("RSNN_READOUT", "32"))    # output first-spike decode window
    D = 64                                                 # max learnable delay (ticks 1..D) on E connections
T = READIN + HIDDEN + READOUT                              # total rollout
READOUT_START = READIN + HIDDEN                            # readout gated to [READOUT_START, T)
READOUT_SPAN = READOUT
TAU = 8.0
BETA = float(np.exp(-1.0 / TAU))
SURR = 5.0                               # arctan surrogate scale (LIVE value; annealed during training)
SURR_START, SURR_END = 5.0, 30.0         # anneal surrogate/gate slope 5->30 over training to close soft/hard gap
def set_surr(v):                          # rebind module-global SURR (read by SpikeFn.backward + the soft gate g)
    global SURR
    SURR = float(v)
# Input latency spans the READ-IN window [0,READIN): t_i = C_IN - ALPHA_IN*x_i, clipped to [0,READIN-1].
C_IN   = READIN // 2
ALPHA_IN = 1.0 if SMOKE else 6.0                           # baseline was 3.0 over a 32-tick window; 6.0 spans 64
# Output decode maps the action range into the READ-OUT window [READOUT_START,T) at baseline per-tick
# resolution: a = (C_OUT - t_first)/ALPHA_OUT, C_OUT at the window centre (= baseline 16 shifted right by 96).
C_OUT  = READOUT_START + READOUT_SPAN // 2
ALPHA_OUT = 1.0 if SMOKE else 3.0
RATE_TGT = 0.10
L1_W, DELAY_W, RATE_W = 1e-7, 1e-4, 3e-2
PRUNE_THR = 1e-3
BATCH = 512
TRAIN_SEED_BASE = 1000                                     # per-step train seeds BASE+s -> never == val seed 1
# ---- FIXED random sparse outgoing fan-out (structural sparsity chosen once at init, then frozen) ----
SPARSE_SEED = 1234                                         # deterministic mask seed (logged)
FANOUT_RECUR = 32                                          # random outgoing RECURRENT targets per hidden neuron
#   (H->output stays FULLY connected = +NO, so real out-degree per hidden neuron = FANOUT_RECUR + NO)
EVO = os.path.dirname(os.path.abspath(__file__))

# ------------------------------------------------------------------ oracle (int4 LUT) + obs
Q = np.load(f"{EVO}/walker2d_lut_actor_int4.npz", allow_pickle=True)
wq = Q["w_q"].astype(np.int64); bq = Q["b_q"].astype(np.int64)
ws = Q["w_scale"].astype(np.float64); bs = Q["b_scale"].astype(np.float64)
wtq = Q["weights_q"].astype(np.int64)[:, :, :6]; wsc = Q["weights_scale"].astype(np.float64)
NT = wq.shape[0]; pow2 = np.array([32, 16, 8, 4, 2, 1])
Wd = wq * ws[:, None, None]; Bd = bq * bs[:, None]; TABd = wtq * wsc[:, None, None]
STATS = json.load(open(f"{EVO}/walker_dataset_stats.json"))
OM = np.asarray(STATS["obs_mean"], np.float64); OSD = np.asarray(STATS["obs_std"], np.float64)


def oracle_actions(Xn):
    addr = ((np.einsum('mi,tki->mtk', Xn, Wd) + Bd[None]) > 0).astype(int) @ pow2
    return np.stack([sum(TABd[t, addr[m, t]] for t in range(NT)) for m in range(Xn.shape[0])])


def sample_obs(M, seed):
    rng = np.random.default_rng(seed)
    raw = OM[None] + OSD[None] * rng.standard_normal((M, NI))
    return (raw - OM[None]) / (OSD[None] + 1e-6)


# ---- per-dim OUTPUT companding (probit / Gaussian-CDF warp) ----
# mu/sigma = mean/std of the teacher action distribution over a large independent sample (seed=7).
_YPOOL = oracle_actions(sample_obs(16384, seed=7))
MU_np = _YPOOL.mean(0); SIGMA_np = _YPOOL.std(0)
U_EPS = 1e-4                                                  # clamp u away from 0/1 -> |z| <= Z_MAX
Z_MAX = float(np.sqrt(2.0) * torch.erfinv(torch.tensor(1.0 - 2.0 * U_EPS)).item())   # ~3.72
MU = torch.tensor(MU_np, dtype=torch.float32, device=DEV)
SIGMA = torch.tensor(SIGMA_np, dtype=torch.float32, device=DEV)


def input_raster(Xn):
    """Latency code into the READ-IN window: input i emits one spike at t_i = C_IN - ALPHA_IN*x_i, clipped
    to [0, READIN-1] so all input activity lands in the 64-tick read-in phase; raster length is full T."""
    x = torch.as_tensor(Xn, dtype=torch.float32)
    t = torch.round(C_IN - ALPHA_IN * x).clamp(0, READIN - 1).long()     # (M,17) -> read-in window
    R = torch.zeros(x.shape[0], T, NI)
    R.scatter_(1, t.unsqueeze(1), 1.0)
    return R.to(DEV)


# ------------------------------------------------------------------ surrogate spike
class SpikeFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, v):
        ctx.save_for_backward(v)
        return (v > 0).float()

    @staticmethod
    def backward(ctx, g):
        (v,) = ctx.saved_tensors
        return g / (1.0 + (SURR * v).pow(2))

spike = SpikeFn.apply


def splus_inv(y):
    return torch.log(torch.expm1(y.clamp_min(1e-9)))


# ------------------------------------------------------------------ model
class RSNN(torch.nn.Module):
    def __init__(self, init_std):
        super().__init__()
        g = torch.Generator().manual_seed(0)

        def wpar(shape):
            w0 = (torch.randn(shape, generator=g) * init_std).abs()
            return torch.nn.Parameter(splus_inv(w0))

        def dpar(shape):                                   # delay raw so d_cont = 1+(D-1)*sigmoid ~ U[1,D] at init
            d0 = 1.0 + (D - 1) * torch.rand(shape, generator=g)
            return torch.nn.Parameter(torch.log((d0 - 1 + 1e-3) / (D - d0 + 1e-3)))

        self.tIH, self.dIH = wpar((NI, NH)),  dpar((NI, NH))
        self.tXH, self.dXH = wpar((NEX, NH)), dpar((NEX, NH))
        self.tXO, self.dXO = wpar((NEX, NO)), dpar((NEX, NO))
        self.tNH = wpar((NINH, NH))
        self.tNO = wpar((NINH, NO))
        mXH = torch.ones(NEX, NH); mXH[torch.arange(NEX), torch.arange(NEX)] = 0.0
        mNH = torch.ones(NINH, NH); mNH[torch.arange(NINH), NEX + torch.arange(NINH)] = 0.0
        self.register_buffer("mXH", mXH); self.register_buffer("mNH", mNH)
        # FIXED random outgoing-recurrent mask, chosen once from SPARSE_SEED and frozen. Each hidden neuron
        # keeps FANOUT_RECUR distinct random recurrent targets (excluding self); all other outgoing recurrent
        # synapses are 0 in every forward -> 0 gradient -> frozen at 0 (weights AND their DCLS delays inert).
        # H->output (tXO/tNO) is NOT masked (stays full). Input->hidden (tIH) is NOT masked.
        gm = torch.Generator().manual_seed(SPARSE_SEED)
        K = min(FANOUT_RECUR, NH - 1)

        def _sparse_mask(n_src, self_col):
            M = torch.zeros(n_src, NH)
            for r in range(n_src):
                sc = self_col(r)
                cand = torch.cat([torch.arange(0, sc), torch.arange(sc + 1, NH)])   # exclude self
                pick = cand[torch.randperm(cand.numel(), generator=gm)[:K]]
                M[r, pick] = 1.0
            return M

        self.register_buffer("smXH", _sparse_mask(NEX, lambda e: e))               # H_ex -> H (E->E, E->I)
        self.register_buffer("smNH", _sparse_mask(NINH, lambda i: NEX + i))        # H_inh -> H (I->E, I->I)
        self.thr_h = 1.0; self.thr_o = 1.0
        self.sigma = 2.0
        self.hard = False
        self.register_buffer("taus", torch.arange(1, D + 1).float().view(D, 1, 1))

    def kernel(self, dcont):                               # (pre,post)->(D,pre,post) Gaussian over delay axis
        if self.hard:
            k = (self.taus == dcont.round().clamp(1, D)[None]).float()
        else:
            k = torch.exp(-(self.taus - dcont[None]).pow(2) / (2 * self.sigma ** 2))
        return k / (k.sum(0, keepdim=True) + 1e-8)

    def ew(self, theta, draw, mask=None, neg=False):
        w = F.softplus(theta)
        if neg: w = -w
        if mask is not None: w = w * mask
        d = 1.0 + (D - 1) * torch.sigmoid(draw)            # delay in [1, D]
        return w[None] * self.kernel(d), w

    def forward(self, Ispk, ret_act=False, soft=True):
        dev = self.tIH.device
        B = Ispk.shape[0]
        EW_IH, wIH = self.ew(self.tIH, self.dIH)                       # input->H : unmasked
        EW_XH, wXH = self.ew(self.tXH, self.dXH, self.mXH * self.smXH)  # H_ex->H : fixed sparse fan-out
        EW_XO, wXO = self.ew(self.tXO, self.dXO)                       # H_ex->readout : FULL (unmasked)
        wNH = -F.softplus(self.tNH) * self.mNH * self.smNH             # H_inh->H : fixed sparse fan-out, delay 1
        wNO = -F.softplus(self.tNO)                                    # H_inh->readout : FULL (unmasked)
        vh = torch.zeros(B, NH, device=dev); vo = torch.zeros(B, NO, device=dev)
        Ibuf = torch.zeros(B, D, NI, device=dev); Hbuf = torch.zeros(B, D, NEX, device=dev)
        prev_h = torch.zeros(B, NH, device=dev)
        o_first = torch.full((B, NO), float(T), device=dev); o_done = torch.zeros(B, NO, dtype=torch.bool, device=dev)
        h_spk_sum = torch.zeros(B, NH, device=dev); o_spk_sum = torch.zeros(B, NO, device=dev)
        soft_t = torch.zeros(B, NO, device=dev); notfired = torch.ones(B, NO, device=dev)
        for t in range(T):
            cur_h = (torch.einsum('bdp,dpo->bo', Ibuf, EW_IH)
                     + torch.einsum('bdp,dpo->bo', Hbuf, EW_XH)
                     + prev_h[:, NEX:] @ wNH)
            cur_o = (torch.einsum('bdp,dpo->bo', Hbuf, EW_XO)
                     + prev_h[:, NEX:] @ wNO)
            vh = BETA * vh + cur_h
            s_h = spike(vh - self.thr_h); vh = vh - self.thr_h * s_h
            vo = BETA * vo + cur_o
            s_o = spike(vo - self.thr_o)
            g = torch.sigmoid(SURR * (vo - self.thr_o))
            vo = vo - self.thr_o * s_o
            if t >= READOUT_START:                         # readout gated to [READOUT_START, T)
                soft_t = soft_t + float(t) * g * notfired
                notfired = notfired * (1.0 - g)
                newly = (s_o > 0.5) & (~o_done)
                o_first = torch.where(newly, torch.full_like(o_first, float(t)), o_first)
                o_done = o_done | newly
            h_spk_sum = h_spk_sum + s_h; o_spk_sum = o_spk_sum + s_o
            Ibuf = torch.cat([Ispk[:, t:t + 1, :], Ibuf[:, :-1, :]], 1)
            Hbuf = torch.cat([s_h[:, None, :NEX], Hbuf[:, :-1, :]], 1)
            prev_h = s_h
        soft_t = soft_t + float(T) * notfired              # never-fired mass -> max latency T=128
        act = self.decode(soft_t if soft else o_first)
        if ret_act:
            return act, dict(h_rate=(h_spk_sum.mean() / T).item(),
                             hex_rate=(h_spk_sum[:, :NEX].mean() / T).item(),
                             hinh_rate=(h_spk_sum[:, NEX:].mean() / T).item(),
                             o_rate=(o_spk_sum.mean() / T).item(),
                             o_nonfire=int((o_spk_sum < 0.5).sum().item()),
                             h_rate_t=h_spk_sum.mean() / T, o_rate_t=o_spk_sum.mean() / T,
                             wsum=(wIH.abs().sum() + wXH.abs().sum() + wXO.abs().sum()
                                   + wNH.abs().sum() + wNO.abs().sum()),
                             dmean=(0.5 * (1 + (D - 1) * torch.sigmoid(self.dIH)).mean()
                                    + 0.5 * (1 + (D - 1) * torch.sigmoid(self.dXH)).mean()))
        return act

    def decode(self, o_first):                             # companded probit decode (per-dim), differentiable
        # early spike (small t) -> u->1 -> high action; non-fire (t=T) -> u->0 -> low action (saturates low)
        u = ((T - o_first) / READOUT_SPAN).clamp(U_EPS, 1.0 - U_EPS)
        z = torch.erfinv(2.0 * u - 1.0) * (2.0 ** 0.5)     # Phi^{-1}(u)
        return MU + SIGMA * z

    @torch.no_grad()
    def calibrate(self, Ispk, target=RATE_TGT):
        for attr, tgt in (("thr_h", target), ("thr_o", target)):
            lo, hi = 1e-4, 50.0
            for _ in range(22):
                mid = (lo * hi) ** 0.5
                setattr(self, attr, mid)
                _, a = self.forward(Ispk, ret_act=True)
                r = a["h_rate"] if attr == "thr_h" else a["o_rate"]
                if r > tgt: lo = mid
                else: hi = mid
            setattr(self, attr, (lo * hi) ** 0.5)
        _, a = self.forward(Ispk, ret_act=True)
        return a["h_rate"], a["o_rate"]


def loss_fn(model, Ispk, target):
    act, a = model.forward(Ispk, ret_act=True)
    mse = F.mse_loss(act, target)
    l1 = L1_W * a["wsum"]
    dpen = DELAY_W * a["dmean"]
    rate = RATE_W * ((a["h_rate_t"] - RATE_TGT) ** 2 + (a["o_rate_t"] - RATE_TGT) ** 2)
    return mse + l1 + dpen + rate, mse.item(), a


# ------------------------------------------------------------------ train / eval
def evaluate(model, Xn, target_np, hard=True):
    was = model.hard; model.hard = hard
    Ispk = input_raster(Xn)
    with torch.no_grad():
        act, a = model.forward(Ispk, ret_act=True, soft=False)
    model.hard = was
    dec = act.cpu().numpy(); err = np.abs(dec - target_np)
    arange = target_np.max() - target_np.min(); tick = 1.0 / ALPHA_OUT   # 0.333 action = linear one-tick
    # tail clip: fraction of targets whose standardized z falls outside the representable +-Z_MAX
    z_t = (target_np - MU_np[None]) / SIGMA_np[None]
    clip_lo = float((z_t < -Z_MAX).mean()); clip_hi = float((z_t > Z_MAX).mean())
    return dict(mean=err.mean(), med=np.median(err), mx=err.max(), arange=arange,
                w1=(err <= tick + 1e-9).mean(), w2=(err <= 2 * tick + 1e-9).mean(),        # 0.333 / 0.667 action
                w2pct=(err <= 0.02 * arange + 1e-9).mean(), w5pct=(err <= 0.05 * arange + 1e-9).mean(),
                clip_lo=clip_lo, clip_hi=clip_hi, a=a, dec=dec, err=err)


def ordering_stats(dec, tgt):
    ra = dec.argsort(1).argsort(1).astype(np.float64)
    rb = tgt.argsort(1).argsort(1).astype(np.float64)
    ra -= ra.mean(1, keepdims=True); rb -= rb.mean(1, keepdims=True)
    num = (ra * rb).sum(1)
    den = np.sqrt((ra ** 2).sum(1) * (rb ** 2).sum(1)) + 1e-12
    rho = float((num / den).mean())
    top1 = float((dec.argmax(1) == tgt.argmax(1)).mean())
    return rho, top1


def train_run(init_std, steps, Xtr, Ytr, log, resample=False):
    model = RSNN(init_std).to(DEV)
    Ispk_cal = input_raster(Xtr[:64])
    hr, orr = model.calibrate(Ispk_cal)
    log(f"  std={init_std:g}: calibrated thr_h={model.thr_h:.4g} thr_o={model.thr_o:.4g} "
        f"init rates hidden={hr:.3f} out={orr:.3f}")
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    Ispk = input_raster(Xtr); target = torch.as_tensor(oracle_actions(Xtr), dtype=torch.float32).to(DEV)
    l0 = None; hist = []
    for s in range(steps):
        model.sigma = max(0.5, 2.0 - 1.5 * s / max(1, steps - 1))
        if resample:
            set_surr(SURR_START + (SURR_END - SURR_START) * (s / max(1, steps - 1)))  # anneal surrogate/gate slope
            Xb = sample_obs(BATCH, seed=TRAIN_SEED_BASE + s)
            Ispk = input_raster(Xb); target = torch.as_tensor(oracle_actions(Xb), dtype=torch.float32).to(DEV)
        opt.zero_grad()
        loss, mse, a = loss_fn(model, Ispk, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if l0 is None: l0 = mse
        if s % max(1, steps // 10) == 0 or s == steps - 1:
            hist.append((s, mse, a["h_rate"], a["o_rate"], a["o_nonfire"]))
            log(f"    step {s:4d}  mse={mse:.4f}  surr={SURR:.1f}  hid_rate={a['h_rate']:.3f} out_rate={a['o_rate']:.3f} "
                f"o_nonfire={a['o_nonfire']}")
    return model, l0, hist


def main():
    rep = []
    def log(m): rep.append(m); print(m, flush=True)
    if DEV == "cuda":
        torch.cuda.reset_peak_memory_stats()

    t0 = time.time()
    Xtr = sample_obs(512, seed=0); Ytr = oracle_actions(Xtr)
    Xval = sample_obs(512, seed=1); Yval = oracle_actions(Xval)
    log("=== RSNN distill Walker2d int4 LUT (BIG H256/64, PINNED std=0.1 + SURR anneal + FIXED SPARSE fan-out) ===")
    _rec_dense = NEX * (NH - 1) + NINH * (NH - 1)
    _rec_sparse = (NEX + NINH) * min(FANOUT_RECUR, NH - 1)
    log(f"fixed sparse fan-out: seed={SPARSE_SEED}, {FANOUT_RECUR} random recurrent targets/neuron (+{NO} full to output) "
        f"-> real out-degree {FANOUT_RECUR + NO}; recurrent synapses {_rec_sparse}/{_rec_dense} "
        f"({100*_rec_sparse/_rec_dense:.1f}% of dense) + H->out {(NEX+NINH)*NO} full")
    log(f"output companding: probit per-dim, U_EPS={U_EPS} (|z|<= {Z_MAX:.3f}); "
        f"MU={np.round(MU_np,3).tolist()} SIGMA={np.round(SIGMA_np,3).tolist()}")
    log(f"device={DEV} | arch I={NI} H_ex={NEX} H_inh={NINH} O={NO} | phases read-in[0,{READIN}) "
        f"hidden[{READIN},{READOUT_START}) read-out[{READOUT_START},{T}) | T={T} D(delay)={D} tau={TAU} | "
        f"C_IN={C_IN} ALPHA_IN={ALPHA_IN} C_OUT={C_OUT} ALPHA_OUT={ALPHA_OUT} | "
        f"action range={Ytr.max()-Ytr.min():.3f} one-tick={1/ALPHA_OUT:.3f}")

    # FIX 1: init-std sweep SKIPPED; PINNED to 0.1 (the sweep mis-selected 0.01 at this width -> soft/hard gap).
    chosen = 0.1; sweep = []
    log(f"\n-- init-std sweep SKIPPED; PINNED std = {chosen:g} (fix for the H256/64 mis-selection) --")
    log(f"-- FIX 2: surrogate/gate slope SURR annealed {SURR_START:g} -> {SURR_END:g} over training (close soft/hard gap) --")
    log(f"\nCHOSEN init std = {chosen:g}")

    full_steps = 8 if SMOKE else int(os.environ.get("RSNN_STEPS", "2048"))   # fast ranker budget for the sweep
    tag = f"ro{READOUT}_steps{full_steps}"                # READOUT via RSNN_READOUT env; companded span auto-fills
    log(f"\n-- full training: std={chosen:g}, {full_steps} steps, RANDOM per-step batches (size {BATCH}) --")
    model, l0, hist = train_run(chosen, full_steps, Xtr, Ytr, log, resample=True)

    with torch.no_grad():
        groups = [(model.tIH, False), (model.tXH, False), (model.tXO, False),
                  (model.tNH, True), (model.tNO, True)]
        total = 0; pruned = 0
        for th, neg in groups:
            w = F.softplus(th); w = -w if neg else w
            total += w.numel(); pruned += int((w.abs() < PRUNE_THR).sum().item())
    spars = pruned / total

    ev_tr = evaluate(model, Xtr, Ytr, hard=True)
    ev = evaluate(model, Xval, Yval, hard=True)
    a = ev["a"]
    rho, top1 = ordering_stats(ev["dec"], Yval)
    gpu_mem = (torch.cuda.max_memory_allocated() / 1e9) if DEV == "cuda" else 0.0
    log("\n=== FINAL (integer/snapped delays) ===")
    log(f"device={DEV}  peak_gpu_mem={gpu_mem:.2f}GB")
    log(f"init std needed for activity+learning: {chosen:g}")
    log(f"loss curve (mse): start {l0:.4f} -> end {hist[-1][1]:.4f}")
    log(f"held-out val distill err vs LUT: mean {ev['mean']:.4f} ({100*ev['mean']/ev['arange']:.2f}% of range) "
        f"median {ev['med']:.4f} ({100*ev['med']/ev['arange']:.2f}%) max {ev['mx']:.4f}")
    log(f"train distill err: mean {ev_tr['mean']:.4f} ({100*ev_tr['mean']/ev_tr['arange']:.2f}%)")
    log(f"within 0.333 action (=lin 1-tick): {ev['w1']:.3f}  within 0.667 (=lin 2-tick): {ev['w2']:.3f}")
    log(f"within 2% of range: {ev['w2pct']:.3f}  within 5% of range: {ev['w5pct']:.3f}  (fixed action-threshold bins)")
    log(f"companding tail clip (targets beyond +-{Z_MAX:.2f} sigma): low {100*ev['clip_lo']:.3f}%  high {100*ev['clip_hi']:.3f}%")
    log(f"ordering: Spearman rho={rho:.3f}  top-1 argmax match={100*top1:.1f}% (chance 16.7%)")
    log(f"firing rates: H_ex={a['hex_rate']:.3f} H_inh={a['hinh_rate']:.3f} O={a['o_rate']:.3f} spk/neuron/tick")
    log(f"O non-firing (of {512*NO}): {a['o_nonfire']}")
    log(f"prunable weights (|w|<{PRUNE_THR}): {pruned}/{total} -> sparsity {100*spars:.1f}%")
    ex_lines = []
    for i in (0, 1, 2):
        ex_lines.append(f"  obs#{i}: net={np.round(ev['dec'][i],3).tolist()}  LUT={np.round(Yval[i],3).tolist()}")
    log("example decoded net action vs LUT (held-out):")
    for e in ex_lines: log(e)
    log(f"\n(elapsed {time.time()-t0:.0f}s, device={DEV}, peak_gpu_mem={gpu_mem:.2f}GB, threads={torch.get_num_threads()})")

    open(f"{EVO}/walker_rsnn_report_gpu_{tag}.txt", "w").write("\n".join(rep))
    json.dump(dict(device=DEV, peak_gpu_mem_gb=gpu_mem, chosen_std=chosen, sweep=sweep,
                   val=ev['mean'], val_pct=100*ev['mean']/ev['arange'], w1=ev['w1'], w2=ev['w2'],
                   w2pct=ev['w2pct'], w5pct=ev['w5pct'], clip_lo=ev['clip_lo'], clip_hi=ev['clip_hi'], spars=spars,
                   o_nonfire=a['o_nonfire'], rho=rho, top1=top1,
                   rates=dict(hex=a['hex_rate'], hinh=a['hinh_rate'], o=a['o_rate']),
                   companding=dict(U_EPS=U_EPS, Z_MAX=Z_MAX, MU=MU_np.tolist(), SIGMA=SIGMA_np.tolist()),
                   phases=dict(readin=READIN, hidden=HIDDEN, readout=READOUT, T=T, D=D,
                               C_IN=C_IN, ALPHA_IN=ALPHA_IN, C_OUT=C_OUT, ALPHA_OUT=ALPHA_OUT),
                   l0=l0, lend=hist[-1][1], steps=full_steps, elapsed=time.time()-t0),
              open(f"{EVO}/walker_rsnn_result_gpu_{tag}.json", "w"), indent=2)
    log(f"wrote walker_rsnn_report_gpu_{tag}.txt + walker_rsnn_result_gpu_{tag}.json")


if __name__ == "__main__":
    main()
