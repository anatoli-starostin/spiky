"""Prototype recurrent E/I spiking net distilling the Walker2d int4 LUT actor (oracle only, NO warm-start).
PyTorch surrogate-gradient BPTT. Open-loop I/O-consistency vs the LUT only (NO mujoco/env).

Architecture (Dale's law, recurrent):
  I=17 inputs (no incoming) -> H_ex=256 excitatory + H_inh=64 inhibitory (all-to-all recurrent among the
  320 hidden, NO self-loops) -> O=6 outputs (no outgoing).
  Excitatory outgoing (from I and H_ex): +softplus(theta) weights AND trainable per-synapse integer delays
  [1,16] trained DCLS-style (continuous position, Gaussian over the delay axis, width annealed, snapped at end).
  Inhibitory outgoing (from H_inh): -softplus(theta) weights, FIXED delay = 1.
Neurons: single-tau LIF (tau~=8), soft reset (subtract threshold), arctan surrogate. Rollout T=32, BPTT.
Input coding: latency  t_i = c_in - alpha_in*x_i  (obs normalized (obs-mean)/(std+1e-6), spikes injected).
Output coding: first-spike time  action = (c_out - t_first)/alpha_out ; non-firing -> t_first = T.
Loss: MSE(decoded, oracle) + L1(weights) + delay penalty + firing-rate penalty (moderate target).
"""
import os, sys, json, time
import numpy as np
import torch
import torch.nn.functional as F

torch.set_num_threads(max(1, os.cpu_count() - 1))
DEV = "cpu"
SMOKE = os.environ.get("RSNN_SMOKE", "0") == "1"

# ------------------------------------------------------------------ sizes / hyperparams
NI, NEX, NINH, NO = 17, 128, 32, 6
if SMOKE:
    NEX, NINH = 32, 8
NH = NEX + NINH
T = 8 if SMOKE else int(os.environ.get("RSNN_T", "32"))          # rollout length
# Optional dedicated settling phase: ticks [0,READOUT_START) are pure compute for the recurrent hidden
# dynamics; the output first-spike readout is gated to [READOUT_START, T]. Default 0 == original T=32 net.
READOUT_START = 0 if SMOKE else int(os.environ.get("RSNN_READOUT_START", "0"))
READOUT_SPAN = T - READOUT_START
D = 16                                   # max delay (ticks 1..16)
TAU = 8.0
BETA = float(np.exp(-1.0 / TAU))
SURR = 5.0                               # arctan surrogate scale
# Inputs are latency-coded into the FIRST 32-tick window (independent of T so the settling phase sees them).
C_IN, ALPHA_IN = ((T // 2) if SMOKE else 16), 3.0                # t = c_in - alpha_in*x -> x in [-5,5]
# Output decode maps the action range into the readout window [READOUT_START, T] at the SAME per-tick
# resolution (alpha_out=3, a 32-tick span): a = (c_out - t_first)/alpha_out, c_out at the window centre.
C_OUT, ALPHA_OUT = ((T // 2) if SMOKE else (READOUT_START + READOUT_SPAN // 2)), 3.0
RATE_TGT = 0.10                          # moderate per-neuron-per-tick firing target
# Regularizer weights are deliberately << the task-MSE gradient so Adam follows the distillation
# target first (the softplus reparam + recurrent contraction make the task gradient ~1e-7/param; an
# L1 at 2e-4 swamped it 100x and the net just pruned itself without learning). Gentle nudges only.
L1_W, DELAY_W, RATE_W = 1e-7, 1e-4, 3e-2
PRUNE_THR = 1e-3
# Data files (teacher LUT + obs stats) are resolved relative to this script so the committed bundle
# is self-contained and runs after a fresh clone on any machine.
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


def oracle_actions(Xn):                                   # Xn: (M,17) normalized obs -> (M,6) action means
    addr = ((np.einsum('mi,tki->mtk', Xn, Wd) + Bd[None]) > 0).astype(int) @ pow2
    return np.stack([sum(TABd[t, addr[m, t]] for t in range(NT)) for m in range(Xn.shape[0])])


def sample_obs(M, seed):
    rng = np.random.default_rng(seed)
    raw = OM[None] + OSD[None] * rng.standard_normal((M, NI))
    return (raw - OM[None]) / (OSD[None] + 1e-6)          # ~ N(0,1); SAME normalization used everywhere


def input_raster(Xn):
    """Latency code: input neuron i emits ONE spike at tick t_i = c_in - alpha_in*x_i (clipped to [0,T-1])."""
    x = torch.as_tensor(Xn, dtype=torch.float32)
    t = torch.round(C_IN - ALPHA_IN * x).clamp(0, T - 1).long()         # (M,17)
    R = torch.zeros(x.shape[0], T, NI)
    R.scatter_(1, t.unsqueeze(1), 1.0)                                   # one spike per input neuron
    return R                                                            # (M,T,NI)


# ------------------------------------------------------------------ surrogate spike
class SpikeFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, v):                                   # v = membrane - threshold
        ctx.save_for_backward(v)
        return (v > 0).float()

    @staticmethod
    def backward(ctx, g):
        (v,) = ctx.saved_tensors
        return g / (1.0 + (SURR * v).pow(2))               # arctan-derivative surrogate

spike = SpikeFn.apply


def splus_inv(y):
    return torch.log(torch.expm1(y.clamp_min(1e-9)))


# ------------------------------------------------------------------ model
class RSNN(torch.nn.Module):
    def __init__(self, init_std):
        super().__init__()
        g = torch.Generator().manual_seed(0)

        def wpar(shape):                                   # weight raw param so softplus(theta) ~ |N(0,std)|
            w0 = (torch.randn(shape, generator=g) * init_std).abs()
            return torch.nn.Parameter(splus_inv(w0))

        def dpar(shape):                                   # delay raw so d_cont = 1+15*sigmoid ~ U[1,16] at init
            d0 = 1.0 + 15.0 * torch.rand(shape, generator=g)
            return torch.nn.Parameter(torch.log((d0 - 1 + 1e-3) / (16 - d0 + 1e-3)))

        # excitatory (positive, delayed) : I->H, Hex->H, Hex->O
        self.tIH, self.dIH = wpar((NI, NH)),  dpar((NI, NH))
        self.tXH, self.dXH = wpar((NEX, NH)), dpar((NEX, NH))
        self.tXO, self.dXO = wpar((NEX, NO)), dpar((NEX, NO))
        # inhibitory (negative, fixed delay 1) : Hinh->H, Hinh->O
        self.tNH = wpar((NINH, NH))
        self.tNO = wpar((NINH, NO))
        # no-self-loop masks (hidden order: 0..NEX-1 = H_ex, NEX..NH-1 = H_inh)
        mXH = torch.ones(NEX, NH); mXH[torch.arange(NEX), torch.arange(NEX)] = 0.0
        mNH = torch.ones(NINH, NH); mNH[torch.arange(NINH), NEX + torch.arange(NINH)] = 0.0
        self.register_buffer("mXH", mXH); self.register_buffer("mNH", mNH)
        self.thr_h = 1.0; self.thr_o = 1.0                 # calibrated at init
        self.sigma = 2.0                                   # DCLS kernel width (annealed)
        self.hard = False                                  # snap delays to integer ticks
        self.register_buffer("taus", torch.arange(1, D + 1).float().view(D, 1, 1))

    def kernel(self, dcont):                               # (pre,post)->(D,pre,post) Gaussian over delay axis
        if self.hard:
            k = (self.taus == dcont.round().clamp(1, D)[None]).float()
        else:
            k = torch.exp(-(self.taus - dcont[None]).pow(2) / (2 * self.sigma ** 2))
        return k / (k.sum(0, keepdim=True) + 1e-8)

    def ew(self, theta, draw, mask=None, neg=False):       # per-delay effective weight tensor (D,pre,post)
        w = F.softplus(theta)
        if neg: w = -w
        if mask is not None: w = w * mask
        d = 1.0 + 15.0 * torch.sigmoid(draw)
        return w[None] * self.kernel(d), w

    def forward(self, Ispk, ret_act=False, soft=True):
        B = Ispk.shape[0]
        EW_IH, wIH = self.ew(self.tIH, self.dIH)
        EW_XH, wXH = self.ew(self.tXH, self.dXH, self.mXH)
        EW_XO, wXO = self.ew(self.tXO, self.dXO)
        wNH = -F.softplus(self.tNH) * self.mNH             # inhibitory, fixed delay 1
        wNO = -F.softplus(self.tNO)
        vh = torch.zeros(B, NH); vo = torch.zeros(B, NO)
        Ibuf = torch.zeros(B, D, NI); Hbuf = torch.zeros(B, D, NEX)
        prev_h = torch.zeros(B, NH)
        o_first = torch.full((B, NO), float(T)); o_done = torch.zeros(B, NO, dtype=torch.bool)
        h_spk_sum = torch.zeros(B, NH); o_spk_sum = torch.zeros(B, NO)
        soft_t = torch.zeros(B, NO); notfired = torch.ones(B, NO)   # differentiable expected first-spike time
        for t in range(T):
            cur_h = (torch.einsum('bdp,dpo->bo', Ibuf, EW_IH)
                     + torch.einsum('bdp,dpo->bo', Hbuf, EW_XH)
                     + prev_h[:, NEX:] @ wNH)              # inhibitory delay-1
            cur_o = (torch.einsum('bdp,dpo->bo', Hbuf, EW_XO)
                     + prev_h[:, NEX:] @ wNO)
            vh = BETA * vh + cur_h
            s_h = spike(vh - self.thr_h); vh = vh - self.thr_h * s_h
            vo = BETA * vo + cur_o
            s_o = spike(vo - self.thr_o)
            g = torch.sigmoid(SURR * (vo - self.thr_o))    # smooth output-fire gate (differentiable)
            vo = vo - self.thr_o * s_o
            if t >= READOUT_START:                         # readout gated to [READOUT_START, T]
                # differentiable first-spike time: E[t] = sum_t t * g_t * prod_{t'<t}(1-g_t') + T*(never)
                soft_t = soft_t + float(t) * g * notfired
                notfired = notfired * (1.0 - g)
                # hard first-spike time in the window (used for the real spiking readout at eval)
                newly = (s_o > 0.5) & (~o_done)
                o_first = torch.where(newly, torch.full_like(o_first, float(t)), o_first)
                o_done = o_done | newly
            h_spk_sum = h_spk_sum + s_h; o_spk_sum = o_spk_sum + s_o
            # push spikes into delay buffers for next tick
            Ibuf = torch.cat([Ispk[:, t:t + 1, :], Ibuf[:, :-1, :]], 1)
            Hbuf = torch.cat([s_h[:, None, :NEX], Hbuf[:, :-1, :]], 1)
            prev_h = s_h
        soft_t = soft_t + float(T) * notfired              # leftover mass -> max latency T
        act = self.decode(soft_t if soft else o_first)     # train on soft time, eval on hard first-spike
        if ret_act:
            return act, dict(h_rate=(h_spk_sum.mean() / T).item(),
                             hex_rate=(h_spk_sum[:, :NEX].mean() / T).item(),
                             hinh_rate=(h_spk_sum[:, NEX:].mean() / T).item(),
                             o_rate=(o_spk_sum.mean() / T).item(),
                             o_nonfire=int((o_spk_sum < 0.5).sum().item()),
                             h_rate_t=h_spk_sum.mean() / T, o_rate_t=o_spk_sum.mean() / T,
                             wsum=(wIH.abs().sum() + wXH.abs().sum() + wXO.abs().sum()
                                   + wNH.abs().sum() + wNO.abs().sum()),
                             dmean=(0.5 * (1 + 15 * torch.sigmoid(self.dIH)).mean()
                                    + 0.5 * (1 + 15 * torch.sigmoid(self.dXH)).mean()))
        return act

    def decode(self, o_first):
        return (C_OUT - o_first) / ALPHA_OUT

    # ---- threshold calibration by bisection on firing rate ----
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
        act, a = model.forward(Ispk, ret_act=True, soft=False)   # real spiking readout = hard first-spike
    model.hard = was
    dec = act.numpy(); err = np.abs(dec - target_np)
    arange = target_np.max() - target_np.min(); tick = 1.0 / ALPHA_OUT
    return dict(mean=err.mean(), med=np.median(err), mx=err.max(), arange=arange,
                w1=(err <= tick + 1e-9).mean(), w2=(err <= 2 * tick + 1e-9).mean(),
                a=a, dec=dec, err=err)


def train_run(init_std, steps, Xtr, Ytr, log):
    model = RSNN(init_std)
    Ispk_cal = input_raster(Xtr[:64])
    hr, orr = model.calibrate(Ispk_cal)
    log(f"  std={init_std:g}: calibrated thr_h={model.thr_h:.4g} thr_o={model.thr_o:.4g} "
        f"init rates hidden={hr:.3f} out={orr:.3f}")
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    Ispk = input_raster(Xtr); target = torch.as_tensor(oracle_actions(Xtr), dtype=torch.float32)
    l0 = None; hist = []
    for s in range(steps):
        model.sigma = max(0.5, 2.0 - 1.5 * s / max(1, steps - 1))   # anneal DCLS width
        opt.zero_grad()
        loss, mse, a = loss_fn(model, Ispk, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if l0 is None: l0 = mse
        if s % max(1, steps // 10) == 0 or s == steps - 1:
            hist.append((s, mse, a["h_rate"], a["o_rate"], a["o_nonfire"]))
            log(f"    step {s:4d}  mse={mse:.4f}  hid_rate={a['h_rate']:.3f} out_rate={a['o_rate']:.3f} "
                f"o_nonfire={a['o_nonfire']}")
    return model, l0, hist


def main():
    rep = []
    def log(m): rep.append(m); print(m, flush=True)

    t0 = time.time()
    Xtr = sample_obs(512, seed=0); Ytr = oracle_actions(Xtr)
    Xval = sample_obs(512, seed=1); Yval = oracle_actions(Xval)
    log("=== RSNN distillation of the Walker2d int4 LUT actor (open-loop, no env) ===")
    log(f"arch I={NI} H_ex={NEX} H_inh={NINH} O={NO} | T={T} readout=[{READOUT_START},{T}] c_out={C_OUT} "
        f"tau={TAU} D={D} | action range={Ytr.max()-Ytr.min():.3f} one-tick={1/ALPHA_OUT:.3f}")

    # --- init-std sweep: smallest std that is active AND makes training progress ---
    stds = [1e-3] if SMOKE else [1e-3, 1e-2, 1e-1, 3e-1, 1.0, 3.0]
    probe_steps = 5 if SMOKE else 30
    chosen = None; sweep = []
    log("\n-- init-std sweep (probe %d steps, pick smallest std with >10%% loss drop & live activity) --" % probe_steps)
    for std in stds:
        m, l0, hist = train_run(std, probe_steps, Xtr, Ytr, log)
        lf = hist[-1][1]; drop = (l0 - lf) / l0 if l0 else 0.0
        hr = hist[-1][2]
        alive = 0.02 < hr < 0.5
        ok = alive and drop > 0.10
        sweep.append((std, l0, lf, drop, hr, alive, ok))
        log(f"  -> std={std:g}: loss {l0:.4f}->{lf:.4f} (drop {100*drop:.1f}%) hid_rate={hr:.3f} "
            f"alive={alive} progressing={ok}")
        if ok: chosen = std; break
    if chosen is None:
        chosen = sweep[-1][0]
        log(f"  none passed cleanly; proceeding with largest tried std={chosen:g}")
    log(f"\nCHOSEN init std = {chosen:g}")

    # --- full training with chosen std ---
    full_steps = 8 if SMOKE else 300
    log(f"\n-- full training: std={chosen:g}, {full_steps} steps --")
    model, l0, hist = train_run(chosen, full_steps, Xtr, Ytr, log)

    # --- pruning (soft-set weights near 0 to exactly 0) ---
    with torch.no_grad():
        groups = [(model.tIH, False), (model.tXH, False), (model.tXO, False),
                  (model.tNH, True), (model.tNO, True)]
        total = 0; pruned = 0
        for th, neg in groups:
            w = F.softplus(th); w = -w if neg else w
            total += w.numel(); pruned += int((w.abs() < PRUNE_THR).sum().item())
    spars = pruned / total

    # --- final eval (hard integer delays) on train + held-out val ---
    ev_tr = evaluate(model, Xtr, Ytr, hard=True)
    ev = evaluate(model, Xval, Yval, hard=True)
    a = ev["a"]
    log("\n=== FINAL (integer/snapped delays) ===")
    log(f"init std needed for activity+learning: {chosen:g}")
    log(f"loss curve (mse): start {l0:.4f} -> end {hist[-1][1]:.4f}")
    log(f"held-out val distill err vs LUT: mean {ev['mean']:.4f} ({100*ev['mean']/ev['arange']:.2f}% of range) "
        f"median {ev['med']:.4f} ({100*ev['med']/ev['arange']:.2f}%) max {ev['mx']:.4f}")
    log(f"train distill err: mean {ev_tr['mean']:.4f} ({100*ev_tr['mean']/ev_tr['arange']:.2f}%)")
    log(f"fraction of dims within 1 output tick: {ev['w1']:.3f}  within 2 ticks: {ev['w2']:.3f}")
    log(f"firing rates: H_ex={a['hex_rate']:.3f} H_inh={a['hinh_rate']:.3f} O={a['o_rate']:.3f} spk/neuron/tick")
    log(f"O non-firing (of {512*NO}): {a['o_nonfire']}")
    log(f"prunable weights (|w|<{PRUNE_THR}): {pruned}/{total} -> sparsity {100*spars:.1f}%")
    ex_lines = []
    for i in (0, 1, 2):
        ex_lines.append(f"  obs#{i}: net={np.round(ev['dec'][i],3).tolist()}  LUT={np.round(Yval[i],3).tolist()}")
    log("example decoded net action vs LUT (held-out):")
    for e in ex_lines: log(e)
    log(f"\n(elapsed {time.time()-t0:.0f}s, threads={torch.get_num_threads()})")

    open(f"{EVO}/walker_rsnn_report.txt", "w").write("\n".join(rep))
    json.dump(dict(chosen_std=chosen, sweep=sweep, val=ev['mean'], val_pct=100*ev['mean']/ev['arange'],
                   w1=ev['w1'], w2=ev['w2'], spars=spars, o_nonfire=a['o_nonfire'],
                   rates=dict(hex=a['hex_rate'], hinh=a['hinh_rate'], o=a['o_rate']),
                   l0=l0, lend=hist[-1][1], elapsed=time.time()-t0),
              open(f"{EVO}/walker_rsnn_result.json", "w"), indent=2)
    log("wrote walker_rsnn_report.txt + walker_rsnn_result.json")


if __name__ == "__main__":
    main()
