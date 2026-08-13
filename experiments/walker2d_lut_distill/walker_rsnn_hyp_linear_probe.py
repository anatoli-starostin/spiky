"""Linear (ridge) probe on the FROZEN hidden state of the best-config RSNN distiller.
Diagnoses representation-limit vs decode-limit: if a linear map from frozen hidden features -> teacher
6-dim LUT outputs beats the spiking readout's ~9% MAE, the info is present (decode-limited); if it lands
near ~9%, it's representation/capacity-limited."""
import sys, numpy as np, torch
EXP = "/home/astarostin/projects/spiky/experiments/walker2d_lut_distill"
sys.path.insert(0, EXP)
import walker_rsnn_probe_model as W

torch.manual_seed(0)
# 1. train best-config net (companded 6-out, sparse 32+6, pinned std 0.1, SURR anneal, 2048 steps), then freeze
Xtr = W.sample_obs(512, seed=0); Ytr = W.oracle_actions(Xtr)
model, l0, hist = W.train_run(0.1, 2048, Xtr, Ytr, lambda m: None, resample=True)
model.eval()
print(f"trained: CE/MSE start {l0:.3f} -> end {hist[-1][1]:.3f}", flush=True)

def capture(Xn, chunk=1024):
    lat, cnt, dec = [], [], []
    model.hard = True
    for i in range(0, len(Xn), chunk):
        Ispk = W.input_raster(Xn[i:i+chunk])
        with torch.no_grad():
            act, a = model.forward(Ispk, ret_act=True, soft=False)
        lat.append(a["h_first"].cpu().numpy()); cnt.append(a["h_count"].cpu().numpy())
        dec.append(act.cpu().numpy())
    return np.concatenate(lat), np.concatenate(cnt), np.concatenate(dec)

# 2. capture on a big train split + a disjoint held-out split
Xp = W.sample_obs(8192, seed=100); Yp = W.oracle_actions(Xp)
Xe = W.sample_obs(4096, seed=101); Ye = W.oracle_actions(Xe)
lat_p, cnt_p, _ = capture(Xp)
lat_e, cnt_e, dec_e = capture(Xe)               # dec_e = the spiking readout's own prediction on held-out
arange = float(Ye.max() - Ye.min())

def ridge_eval(Xtr, Ytr, Xte, Yte, lams=(1e-2, 1e-1, 1, 10, 100, 1e3)):
    mu = Xtr.mean(0); sd = Xtr.std(0) + 1e-6
    A = np.concatenate([(Xtr - mu) / sd, np.ones((len(Xtr), 1))], 1)
    Bx = np.concatenate([(Xte - mu) / sd, np.ones((len(Xte), 1))], 1)
    XtX = A.T @ A; XtY = A.T @ Ytr; I = np.eye(A.shape[1])
    best = None
    for lam in lams:
        Wm = np.linalg.solve(XtX + lam * I, XtY)
        pr = Bx @ Wm
        relL2 = float(np.linalg.norm(pr - Yte) / np.linalg.norm(Yte))
        if best is None or relL2 < best[0]:
            best = (relL2, float(np.mean((pr - Yte) ** 2)), float(np.mean(np.abs(pr - Yte))), lam)
    return best  # relL2, MSE, MAE, best-lam

print("\n=== LINEAR (ridge) PROBE: frozen hidden features -> teacher 6-dim LUT outputs (action space) ===")
print(f"{'feature':22s} {'relL2':>8s} {'MSE':>9s} {'MAE':>9s} {'MAE%range':>10s} {'lam':>7s}")
feats = {"(a) first-spike latency": (lat_p, lat_e), "(b) spike-count": (cnt_p, cnt_e),
         "(a+b) latency+count": (np.concatenate([lat_p, cnt_p], 1), np.concatenate([lat_e, cnt_e], 1))}
for name, (Xt, Xv) in feats.items():
    relL2, mse, mae, lam = ridge_eval(Xt, Yp, Xv, Ye)
    print(f"{name:22s} {relL2:8.4f} {mse:9.4f} {mae:9.4f} {100*mae/arange:9.2f}% {lam:7g}", flush=True)

# spiking readout baseline on the SAME held-out split
r_mae = float(np.mean(np.abs(dec_e - Ye))); r_mse = float(np.mean((dec_e - Ye) ** 2))
r_rel = float(np.linalg.norm(dec_e - Ye) / np.linalg.norm(Ye))
print(f"{'READOUT (spiking)':22s} {r_rel:8.4f} {r_mse:9.4f} {r_mae:9.4f} {100*r_mae/arange:9.2f}% {'--':>7s}")
print(f"\n(action range={arange:.3f}; readout held-out MAE {100*r_mae/arange:.2f}% of range)")
