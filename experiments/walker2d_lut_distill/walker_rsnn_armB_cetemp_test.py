"""Test the proposed one-line fix for Arm B: temper the CE logits (-E[t]/CE_TEMP) so init isn't
confidently-wrong. Quick 512-step train, compare accuracy vs the collapsed baseline (~14.3% chance)."""
import sys, torch, numpy as np, torch.nn.functional as F
sys.path.insert(0, "/home/astarostin/projects/spiky/experiments/walker2d_lut_distill")
import walker_rsnn_distill_gpu_disc_B as W
DEV = W.DEV

def run(CE_TEMP, steps=512):
    m = W.RSNN(0.1).to(DEV)
    m.calibrate(W.input_raster(W.sample_obs(512, seed=0)[:64]))
    opt = torch.optim.Adam(m.parameters(), lr=2e-3)
    ce_hist = []
    for s in range(steps):
        m.sigma = max(0.5, 2.0 - 1.5 * s / (steps - 1))
        W.set_surr(W.SURR_START + (W.SURR_END - W.SURR_START) * (s / (steps - 1)))
        Xb = W.sample_obs(512, seed=1000 + s)
        Ispk = W.input_raster(Xb)
        labels = torch.as_tensor(W.to_levels(W.oracle_actions(Xb)), dtype=torch.long).to(DEV)
        act, a = m.forward(Ispk, ret_act=True)
        logits = (-act / CE_TEMP).reshape(-1, W.N_JOINTS, W.K_LEVELS)
        ce = F.cross_entropy(logits.reshape(-1, W.K_LEVELS), labels.reshape(-1))
        loss = ce + W.L1_W * a["wsum"] + W.DELAY_W * a["dmean"] \
            + W.RATE_W * ((a["h_rate_t"] - W.RATE_TGT) ** 2 + (a["o_rate_t"] - W.RATE_TGT) ** 2)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        if s % 64 == 0 or s == steps - 1: ce_hist.append((s, round(float(ce), 3)))
    Xval = W.sample_obs(512, seed=1); Yval = W.oracle_actions(Xval)
    ev = W.evaluate(m, Xval, Yval)
    return ev["mean_acc"], ev["pred"], ce_hist

for T in [1.0, 10.0, 30.0]:
    acc, pred, ceh = run(T)
    hist = np.bincount(pred.reshape(-1), minlength=7).tolist()
    print(f"CE_TEMP={T:5.1f} | 512-step mean-acc {100*acc:.2f}% (chance 14.3) | pred-hist {hist} | CE {ceh[0][1]}->{ceh[-1][1]}", flush=True)
