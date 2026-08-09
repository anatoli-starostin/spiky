"""Tune the RACE layer in isolation: does a two-synapse LIF actually detect spike order?

Reports, per parameter set: how often r+ / r- get the order right, how often exactly one of
each pair fires (the condition a cell needs), and the median |t_a - t_b| among the errors --
which is the detector's DEAD ZONE, the resolution a real spiking comparator gives up.
"""
import numpy as np
import torch

from real_snn import RealSNN

Z = np.load("../distill_exp19_100k.npz") if False else np.load(
    __file__.rsplit("/", 1)[0] + "/../distill_exp19_100k.npz")
dev = "cuda" if torch.cuda.is_available() else "cpu"
X = torch.tensor(Z["x_norm"][:512], dtype=torch.float32, device=dev)
W = torch.tensor(Z["weights"], device=dev)
aa = torch.tensor(Z["anchor_a"], device=dev).long()
bb = torch.tensor(Z["anchor_b"], device=dev).long()
TAU = float(Z["tau"])


def probe(**kw):
    net = RealSNN(W, aa, bb, TAU, **kw).to(dev)
    with torch.no_grad():
        t = net.enc_c - net.enc_m * X
        ta, tb = t[:, aa], t[:, bb]
        want_p = (tb > ta).float().reshape(X.shape[0], -1)      # r+ should fire
        inj_e, inj_i = net.input_current(X)
        B = X.shape[0]
        Ir = torch.zeros(B, net.nR, device=dev)
        Ii = torch.zeros_like(Ir)
        Vr = torch.zeros_like(Ir)
        fr = torch.zeros_like(Ir)
        for n in range(net.n_steps):
            Vr = net.a_m_r * Vr + Ir - Ii
            Ir = net.a_s_r * Ir + inj_e[:, :, n]
            Ii = net.a_i_r * Ii + inj_i[:, :, n]
            s = (Vr > net.theta_race).float() * (1 - fr)
            fr = torch.clamp(fr + s, max=1)
        gp, gm = fr[:, :192], fr[:, 192:]
        accp = float((gp == want_p).float().mean())
        accm = float((gm == (1 - want_p)).float().mean())
        one = float(((gp + gm) == 1).float().mean())
        gap = (ta - tb).abs().reshape(B, -1)
        wrong = gp != want_p
        med = float(gap[wrong].median()) if int(wrong.sum()) else 0.0
        p95 = float(gap[wrong].quantile(0.95)) if int(wrong.sum()) else 0.0
        # per-cell: a cell needs all 6 of its races correct
        okc = ((gp == want_p) & (gm == (1 - want_p))).reshape(B, 32, 6).all(-1)
        return accp, accm, one, float(fr.sum(-1).mean()), med, p95, float(okc.float().mean())


CONFIGS = [
    ("baseline (d_exc=0.03, veto 6)", dict()),
    ("d_exc=0", dict(race_lat=0.0)),
    ("d_exc=0, veto 3", dict(race_lat=0.0, veto=3.0)),
    ("d_exc=0, veto 12", dict(race_lat=0.0, veto=12.0)),
    ("d_exc=0, veto 30", dict(race_lat=0.0, veto=30.0)),
    ("fast tau (.008/.016), dt 1/128", dict(race_lat=0.0, tau_s_race=0.008,
                                            tau_m_race=0.016, dt=1 / 128, n_steps=320)),
    ("fast tau, veto 12, dt 1/128", dict(race_lat=0.0, tau_s_race=0.008, tau_m_race=0.016,
                                         veto=12.0, dt=1 / 128, n_steps=320)),
    ("fast tau, veto 30, dt 1/128", dict(race_lat=0.0, tau_s_race=0.008, tau_m_race=0.016,
                                         veto=30.0, dt=1 / 128, n_steps=320)),
    ("v.fast (.004/.008), veto 30, dt 1/256", dict(race_lat=0.0, tau_s_race=0.004,
                                                   tau_m_race=0.008, veto=30.0,
                                                   dt=1 / 256, n_steps=640)),
]

print(f"{'config':40s} {'acc+':>6s} {'acc-':>6s} {'1of2':>6s} {'fired':>7s} "
      f"{'dead med':>9s} {'dead p95':>9s} {'cells ok':>9s}")
for name, kw in CONFIGS:
    r = probe(**kw)
    print(f"{name:40s} {r[0]:6.4f} {r[1]:6.4f} {r[2]:6.4f} {r[3]:7.1f} "
          f"{r[4]:9.4f} {r[5]:9.4f} {r[6]:9.4f}")
print("\n'cells ok' = fraction of the 32 cells per sample whose all 6 races are correct;")
print("that is the ceiling on how many of the 32 output synapses can fire correctly.")
