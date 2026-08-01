"""Audit Arm B (disc_B): group wiring, gradient flow to the 42 output params, label/sign, E[t] spread."""
import sys, torch, numpy as np, torch.nn.functional as F
sys.path.insert(0, "/home/astarostin/projects/spiky/experiments/walker2d_lut_distill")
import walker_rsnn_distill_gpu_disc_B as W

DEV = W.DEV
m = W.RSNN(0.1).to(DEV)
Xtr = W.sample_obs(512, seed=0)
m.calibrate(W.input_raster(Xtr[:64]))

Xb = W.sample_obs(512, seed=1000)
Ispk = W.input_raster(Xb)
labels = torch.as_tensor(W.to_levels(W.oracle_actions(Xb)), dtype=torch.long).to(DEV)
opt = torch.optim.Adam(m.parameters(), lr=2e-3)

# ---- (2) GRADIENT FLOW: one step, check grads on the 42 output params ----
opt.zero_grad()
loss, ce, a = W.loss_fn(m, Ispk, labels)
loss.backward()
def gn(p): return None if p.grad is None else round(float(p.grad.norm()), 6)
print("CE at init:", round(ce, 4), " (ln7 =", round(float(np.log(7)), 4), ")")
print("GRAD NORMS -> tXO (H_ex->out):", gn(m.tXO), " tNO (H_inh->out):", gn(m.tNO),
      " dXO (out delays):", gn(m.dXO), " | tXH (recurrent):", gn(m.tXH))
print("output param shapes: tXO", tuple(m.tXO.shape), " tNO", tuple(m.tNO.shape), " (want (*,42))")

# ---- confirm output synapses are NOT caught by the recurrent sparse mask ----
print("sparse masks apply only to recurrent? smXH shape", tuple(m.smXH.shape),
      " smNH", tuple(m.smNH.shape), " (NH-wide, not NO=42)")

# ---- (4) E[t] SPREAD: soft first-spike times, per 7-group ----
with torch.no_grad():
    soft = m.forward(Ispk)                      # (B, 42) soft E[t]
of = soft.reshape(-1, W.N_JOINTS, W.K_LEVELS)   # (B,6,7)
print("within-group E[t] std (mean over B,joints):", round(float(of.std(-1).mean()), 4),
      " | overall E[t] range:", round(float(soft.min()), 2), "-", round(float(soft.max()), 2))
for b in (0, 1):
    for j in (0, 3):
        print(f"  obs{b} joint{j} 7xE[t]:", np.round(of[b, j].cpu().numpy(), 2).tolist())

# ---- (4b) HARD eval: how many groups have NO neuron firing in the read-out window (-> argmin=0)? ----
was = m.hard; m.hard = True
with torch.no_grad():
    hard = m.forward(Ispk, soft=False)          # (B,42) hard o_first
m.hard = was
oh = hard.reshape(-1, W.N_JOINTS, W.K_LEVELS).cpu().numpy()
nofire = (oh >= float(W.T)).all(-1).mean()
pred = oh.argmin(-1)
print(f"HARD @init: frac groups all-nonfired-in-window (->argmin defaults to level0): {100*nofire:.1f}%")
print("HARD @init pred-level histogram:", np.bincount(pred.reshape(-1), minlength=7).tolist())

# ---- (3) LABEL/SIGN sanity: does making a neuron fire earlier raise its predicted prob? ----
logits = (-of)                                   # (B,6,7) — earlier E[t] (smaller) -> higher logit
print("argmax(logits)==argmin(E[t]) per group?", bool((logits.argmax(-1) == of.argmin(-1)).all()))
