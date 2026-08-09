"""Train the fully-simulated SNN (real_snn.py) to approximate the exp19 LUT teacher.

BPTT through the simulated dynamics with a fast-sigmoid surrogate for the spike
nonlinearity. Target is the ACTION MEAN, trained through the output neuron -- not the
prelog sum, which the earlier sweep established fails on any non-exact neuron.

Initialised from the analytic construction wherever it maps: output synapse weights
exp(w/tau), the anchor wiring, tau_m tied to the teacher's tau, and the decode fitted by
least squares at init. Training starts near a sensible solution, not from noise.

  python train_real_snn.py --steps 400                    # one config
  python train_real_snn.py --sweep                        # the config matrix
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch

from real_snn import RealSNN

HERE = os.path.dirname(os.path.abspath(__file__))
NPZ = os.path.join(HERE, "..", "distill_exp19_100k.npz")
OUT = os.path.join(HERE, "results")
MILESTONES = [1_000, 2_000, 5_000, 10_000, 20_000, 50_000, 100_000, 200_000]


def load(dev, n_train=20000, n_val=4000, seed=0):
    Z = np.load(NPZ)
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(Z["x_norm"].shape[0], generator=g).numpy()
    tr, va = perm[:n_train], perm[n_train:n_train + n_val]
    T = lambda i, k: torch.tensor(Z[k][i], dtype=torch.float32, device=dev)
    meta = dict(weights=torch.tensor(Z["weights"], device=dev),
                anchor_a=torch.tensor(Z["anchor_a"], device=dev).long(),
                anchor_b=torch.tensor(Z["anchor_b"], device=dev).long(),
                tau=float(Z["tau"]))
    return (T(tr, "x_norm"), T(tr, "y_action_mean"), T(va, "x_norm"), T(va, "y_action_mean"),
            meta, float(torch.tensor(Z["y_action_mean"]).std()))


@torch.no_grad()
def evaluate(net, X, A, chunk=256):
    tot, mx, cnt, fired, cells = 0.0, 0.0, 0, 0.0, 0.0
    for i in range(0, X.shape[0], chunk):
        a, info = net(X[i:i + chunk])
        e = (a - A[i:i + chunk]).abs()
        tot += float(e.sum()); mx = max(mx, float(e.max())); cnt += e.numel()
        fired += float(info["fired_o"].sum())
        cells += float(info["cell_spikes"].sum())
    n = X.shape[0]
    return dict(mean=tot / cnt, max=mx, fired=fired / (n * net.O), cells=cells / n)


def run(cfg, data, args, dev):
    Xtr, Atr, Xva, Ava, meta, astd = data
    torch.manual_seed(cfg["seed"])
    net = RealSNN(meta["weights"], meta["anchor_a"], meta["anchor_b"], meta["tau"],
                  dt=cfg["dt"], n_steps=cfg["n_steps"], k_out=cfg["k_out"],
                  tau_s_race=cfg["tau_s_race"], tau_m_race=cfg["tau_m_race"],
                  veto=cfg["veto"], race_lat=0.0, gate_open=cfg["gate_open"],
                  tau_m_cell=cfg["tau_m_cell"],
                  decode=cfg["decode"], theta_frac=cfg["theta_frac"],
                  train_race=cfg["train_race"], train_cell=cfg["train_cell"],
                  train_out=cfg["train_out"]).to(dev)
    cal = net.calibrate(Xtr[:512], Atr[:512])
    base_tr = evaluate(net, Xtr[:2000], Atr[:2000])
    base_va = evaluate(net, Xva, Ava)

    params = [p for p in net.parameters() if p.requires_grad]
    n_par = sum(p.numel() for p in params)
    opt = torch.optim.Adam(params, lr=cfg["lr"])
    best = dict(val=base_va["mean"], max=base_va["max"], pairs=0,
                state={k: v.detach().clone() for k, v in net.state_dict().items()})
    curve, seen, mi, t0, diverged = [], 0, 0, time.time(), False
    n = Xtr.shape[0]
    for step in range(args.steps):
        idx = torch.randint(0, n, (cfg["batch"],), device=dev)
        a, _ = net(Xtr[idx])
        loss = ((a - Atr[idx]) ** 2).mean()
        if not math.isfinite(float(loss.detach())):
            diverged = True
            break
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        seen += cfg["batch"]
        if (step + 1) % args.eval_every == 0 or step == args.steps - 1:
            m = evaluate(net, Xva, Ava)
            if m["mean"] < best["val"]:
                best = dict(val=m["mean"], max=m["max"], pairs=seen,
                            state={k: v.detach().clone()
                                   for k, v in net.state_dict().items()})
            while mi < len(MILESTONES) and seen >= MILESTONES[mi]:
                curve.append(dict(pairs=MILESTONES[mi], val=m["mean"],
                                  best=best["val"]))
                mi += 1
            print(f"      step {step+1:4d}  seen {seen:7,d}  loss {float(loss):.4f}  "
                  f"val {m['mean']:.4f} (norm {m['mean']/astd:.4f})  best "
                  f"{best['val']/astd:.4f}", flush=True)
    fin = evaluate(net, Xva, Ava)
    net.load_state_dict(best["state"])
    bt, bv = evaluate(net, Xtr[:2000], Atr[:2000]), evaluate(net, Xva, Ava)
    return dict(cfg=cfg, n_trainable=n_par, action_std=astd, calib=cal,
                wall_s=round(time.time() - t0, 1), diverged=diverged,
                base_train=base_tr, base_val=base_va, final_val=fin,
                best_train=bt, best_val=bv, best_pairs=best["pairs"],
                base_val_norm=base_va["mean"] / astd, best_val_norm=bv["mean"] / astd,
                final_val_norm=fin["mean"] / astd, curve=curve)


def base_cfg(**kw):
    c = dict(dt=1 / 128, n_steps=330, k_out=4.0, tau_s_race=0.008, tau_m_race=0.016,
             veto=6.0, gate_open=1.05, tau_m_cell=15.0, batch=64, lr=3e-3, seed=0,
             decode="mlp", theta_frac=0.95,
             train_race=True, train_cell=False, train_out=False)
    c.update(kw)
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--n-train", type=int, default=20000)
    ap.add_argument("--n-val", type=int, default=4000)
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--tag", default="real_snn")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    data = load(dev, a.n_train, a.n_val)
    os.makedirs(OUT, exist_ok=True)
    print(f"device {dev}  train {tuple(data[0].shape)}  val {tuple(data[2].shape)}  "
          f"action std {data[5]:.4f}")

    if a.sweep:
        jobs = [
            ("BASE dt 1/128, mlp decode", base_cfg()),
            ("frozen front end (decode only)", base_cfg(train_race=False)),
            ("decode affine", base_cfg(decode="affine")),
            ("decode corrected", base_cfg(decode="corrected")),
            ("dt 1/64  (165 sim steps)", base_cfg(dt=1 / 64, n_steps=165,
                                                  tau_s_race=0.016, tau_m_race=0.032)),
            ("dt 1/256 (860 sim steps)", base_cfg(dt=1 / 256, n_steps=860,
                                                  tau_s_race=0.004, tau_m_race=0.008)),
            ("k_out 8", base_cfg(k_out=8.0, n_steps=430)),
            ("veto 20", base_cfg(veto=20.0)),
            ("cell tau_m 50", base_cfg(tau_m_cell=50.0)),
            ("+ train output weights", base_cfg(train_out=True)),
            ("+ train out & cell", base_cfg(train_out=True, train_cell=True)),
            ("dt 1/256 + train out", base_cfg(dt=1 / 256, n_steps=860, tau_s_race=0.004,
                                              tau_m_race=0.008, train_out=True)),
        ]
    else:
        jobs = [("BASE", base_cfg())]

    res = []
    for i, (name, cfg) in enumerate(jobs, 1):
        print(f"\n[{i}/{len(jobs)}] {name}", flush=True)
        r = run(cfg, data, a, dev)
        r["name"] = name
        res.append(r)
        print(f"    params {r['n_trainable']:,}  base {r['base_val_norm']:.4f} -> best "
              f"{r['best_val_norm']:.4f} (max {r['best_val']['max']:.3f}, @"
              f"{r['best_pairs']:,} pairs)  cells/sample {r['best_val']['cells']:.1f}  "
              f"fired {r['best_val']['fired']:.3f}  {r['wall_s']}s", flush=True)
        json.dump(res, open(os.path.join(OUT, f"{a.tag}.json"), "w"), indent=1)
    print(f"\nwrote {os.path.join(OUT, a.tag)}.json")


if __name__ == "__main__":
    main()
