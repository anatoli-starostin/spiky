"""Where does the real SNN's error actually come from? Decompose it before training.

Three numbers per configuration, all before any learning:

  ORACLE   feed the output layer the CORRECT 32 cell spikes, all at one common time. This
           is the output neuron + affine decode + dt quantisation ALONE, with a perfect
           front end. It is the floor the full network could reach if the race and cell
           layers were flawless.
  FULL     the actual simulated network, front end included.
  the gap between them is what the spiking RACE/CELL layers cost.
"""
import numpy as np
import torch

from real_snn import RealSNN

HERE = __file__.rsplit("/", 1)[0]
Z = np.load(HERE + "/../distill_exp19_100k.npz")
dev = "cuda" if torch.cuda.is_available() else "cpu"
X = torch.tensor(Z["x_norm"][:1024], dtype=torch.float32, device=dev)
A = torch.tensor(Z["y_action_mean"][:1024], dtype=torch.float32, device=dev)
W = torch.tensor(Z["weights"], device=dev)
aa = torch.tensor(Z["anchor_a"], device=dev).long()
bb = torch.tensor(Z["anchor_b"], device=dev).long()
TAU = float(Z["tau"])
ASTD = 1.1309


def true_cells(net, x):
    """The address the teacher would select, as a one-hot over the 2048 cells."""
    t = net.enc_c - net.enc_m * x
    ta, tb = t[:, aa], t[:, bb]
    bits = (tb > ta).long()
    pw = (1 << torch.arange(5, -1, -1, device=x.device)).view(1, 1, -1)
    sel = (bits * pw).sum(-1)                                   # [B, 32]
    oh = torch.zeros(x.shape[0], 32 * 64, device=x.device)
    idx = sel + torch.arange(32, device=x.device).view(1, -1) * 64
    oh.scatter_(1, idx, 1.0)
    return oh


@torch.no_grad()
def oracle(net, x, a_ref, fire_step):
    """Output layer only: all 32 correct cells spike simultaneously at step `fire_step`."""
    B = x.shape[0]
    oh = true_cells(net, x)
    Io = torch.zeros(B, net.O, device=dev)
    Vo = torch.zeros_like(Io)
    fired = torch.zeros_like(Io)
    t_out = torch.zeros_like(Io)
    pk = torch.full((B, net.O), -1e30, device=dev)
    for pas in (0, 1):
        Io.zero_(); Vo.zero_(); fired.zero_(); t_out.zero_()
        for n in range(net.n_steps):
            Vo = net.a_m_o * Vo + Io
            s_c = oh if n == fire_step else torch.zeros_like(oh)
            Io = net.a_s_o * Io + net.out_scale * (s_c @ net.w_out)
            if pas == 0:
                pk = torch.maximum(pk, Vo)
            else:
                s = (Vo > net.theta_out.view(1, -1)).float() * (1 - fired)
                t_out = t_out + s * (n * net.dt)
                fired = torch.clamp(fired + s, max=1.0)
        if pas == 0:
            net.theta_out.copy_(net.theta_frac * pk.quantile(0.002, dim=0))
    t_out = t_out + (1 - fired) * (net.n_steps * net.dt)
    a = torch.zeros_like(t_out)
    feat = t_out
    if net.decode_mode != "affine":
        net.t_ref.copy_(t_out.min(0).values - 1e-3)
        xf = torch.exp(-(t_out - net.t_ref) / net.tau_m_o).clamp(1e-6, 1 - 1e-6)
        feat = torch.log(net.theta_out.view(1, -1) / (xf * (1 - xf)))
    for o in range(net.O):
        to, ao = feat[:, o].double(), a_ref[:, o].double()
        tm, am = to.mean(), ao.mean()
        al = ((to - tm) * (ao - am)).sum() / ((to - tm) ** 2).sum().clamp_min(1e-30)
        a[:, o] = (al * to + (am - al * tm)).float()
    e = (a - a_ref).abs()
    return float(e.mean()) / ASTD, float(fired.mean()), float(t_out.max() - t_out.min())


CONFIGS = [
    ("affine k4 tf0.8", dict(k_out=4.0, theta_frac=0.8, decode="affine")),
    ("affine k4 tf0.95", dict(k_out=4.0, theta_frac=0.95, decode="affine")),
    ("corrected k4 tf0.8", dict(k_out=4.0, theta_frac=0.8)),
    ("k4 tf0.95", dict(k_out=4.0, theta_frac=0.95)),
    ("k4 tf0.99", dict(k_out=4.0, theta_frac=0.99)),
    ("k8 tf0.95", dict(k_out=8.0, theta_frac=0.95, n_steps=430)),
    ("k16 tf0.95", dict(k_out=16.0, theta_frac=0.95, n_steps=700)),
    ("k8 tf0.95 dt1/256", dict(k_out=8.0, theta_frac=0.95, dt=1 / 256, n_steps=860,
                               tau_s_race=0.004, tau_m_race=0.008)),
    ("mlp k4 tf0.95", dict(k_out=4.0, theta_frac=0.95, decode="mlp")),
    ("mlp k8 tf0.95", dict(k_out=8.0, theta_frac=0.95, decode="mlp", n_steps=430)),
    ("mlp k8 tf0.99", dict(k_out=8.0, theta_frac=0.99, decode="mlp", n_steps=430)),
    ("mlp k8 tf0.95 dt1/256", dict(k_out=8.0, theta_frac=0.95, decode="mlp", dt=1 / 256,
                                   n_steps=860, tau_s_race=0.004, tau_m_race=0.008)),
]

print(f"{'config':22s} {'ORACLE err':>11s} {'fired':>7s} {'t range':>9s} "
      f"{'FULL err':>9s} {'cells':>7s} {'ofired':>7s}")
for name, kw in CONFIGS:
    kw = dict(dict(dt=1 / 128, n_steps=330, tau_s_race=0.008, tau_m_race=0.016,
                   race_lat=0.0, gate_open=1.05), **kw)
    net = RealSNN(W, aa, bb, TAU, **kw).to(dev)
    orc, fr, rng = oracle(net, X, A, int(1.06 / net.dt))
    cal = net.calibrate(X[:512], A[:512])
    with torch.no_grad():
        a, info = net(X)
    full = float((a - A).abs().mean()) / ASTD
    print(f"{name:22s} {orc:11.4f} {fr:7.3f} {rng:9.4f} {full:9.4f} "
          f"{float(info['cell_spikes'].mean()):7.1f} {float(info['fired_o'].mean()):7.3f}")
print("\nORACLE = perfect front end; FULL = the real network. The difference is the price")
print("of simulating the race and cell layers as genuine spiking neurons.")
