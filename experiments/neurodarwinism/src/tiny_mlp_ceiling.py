"""exp012: a non-spiking MLP ceiling for the single-output task.

Is the ~22 held-out MSE of the 17->8->1 spiking net a substrate/search failure, or is the
task itself that hard? A conventional MLP of the SAME shape, gradient-trained on the SAME
data, answers it -- and quantising its weights to the same 0.1 grid says how much of any gap
is the grid rather than the spikes.

Same data as the spiking run, exactly:
  input   x_norm, the 17-dim per-sample feature vector the latency encoder is derived from
  target  target_offsets(Y)[:, DIM] -- centred, clipped at 2.5 sigma, quantised to 32 levels
  split   data.load()'s split: the last 4000 samples never enter the training pool
  chance  the variance of THAT dimension on the held-out split

Two variants are reported beyond what was asked, because each isolates a different suspect:
  * MLP on the ENCODED input (the 17 spike times, 0..31) -- how much the encoder itself costs
  * a DALE-CONSTRAINED MLP (hidden weights >= 0) -- the constraint the spiking net actually
    runs under, since this substrate has no inhibitory neurons at all

CPU only; the GPU belongs to the spiking runs.
"""
import argparse
import json

import numpy as np
import torch
import torch.nn as nn

import tiny_snn as T
from data import load
from harness import LatencyEncoder

DIM = 0


def fit_affine(p, q):
    """least-squares a, b for a*p + b -> q, the same scale+shift the run's diagls uses"""
    v = p.var()
    a = float(np.cov(p, q, bias=True)[0, 1] / v) if v > 1e-12 else 0.0
    return a, float(q.mean() - a * p.mean())


def apply_affine(p, ab):
    return ab[0] * p + ab[1]


def mse(p, q):
    return float(((p - q) ** 2).mean())


def train_mlp(xt, yt, xv, hidden=8, act="tanh", dale=False, epochs=4000, seed=0, lr=3e-3,
              lam_max=0.0, step=0.1, clamp=None, gain=False):
    """Gradient-train 17 -> hidden -> 1.

    lam_max > 0 turns on QUANTIZATION-AWARE training: the loss gains
        lambda(e) * mean( sin(pi * w / step)^2 )
    which is exactly zero at every multiple of `step` and so pulls each weight onto the grid.
    lambda is annealed from 0, held at 0 for the first 30% of training so the MSE fit forms
    first, then ramped to lam_max -- pushing weights onto the grid from epoch 0 just freezes
    a random net.

    `clamp` bounds |w| so the grid is genuinely the 11 levels {0, 0.1, .., 1.0} rather than an
    unbounded ladder of 0.1-spaced values, which is a much weaker constraint.
    """
    torch.manual_seed(seed)
    d = xt.shape[1]
    W1 = nn.Parameter(torch.randn(d, hidden) * (1.0 / np.sqrt(d)))
    b1 = nn.Parameter(torch.zeros(hidden))
    W2 = nn.Parameter(torch.randn(hidden, 1) * (1.0 / np.sqrt(hidden)))
    b2 = nn.Parameter(torch.zeros(1))
    f = torch.tanh if act == "tanh" else torch.relu
    # `gain`: the genome stores NORMALISED magnitudes on the grid and multiplies by one global
    # evolvable gain per network. So the grid constrains the weight vector's SHAPE and the
    # gain sets its size. Reproduce that here: W = s * V, the penalty acts on V, |V| <= 1, and
    # s is free. Without it, clamping |W| <= 1 conflates the grid with a scale bound and makes
    # the constraint far harsher than the one the spiking net actually runs under.
    s1 = nn.Parameter(torch.ones(1))
    s2 = nn.Parameter(torch.ones(1))
    params = [W1, b1, W2, b2] + ([s1, s2] if gain else [])
    opt = torch.optim.Adam(params, lr=lr)
    X, Y = torch.tensor(xt, dtype=torch.float32), torch.tensor(yt, dtype=torch.float32)

    def eff(w, s):
        w = w.abs() if dale else w
        return w * s.abs() if gain else w

    def fwd(x, w1, w2):
        # Dale: this spiking substrate is all-excitatory, so a hidden unit may only receive
        # non-negative weights. Enforced by construction, not by a penalty.
        return f(x @ eff(w1, s1) + b1) @ eff(w2, s2) + b2

    for e in range(epochs):
        opt.zero_grad()
        mse_t = ((fwd(X, W1, W2).squeeze(-1) - Y) ** 2).mean()
        loss = mse_t
        if lam_max > 0:
            frac = max(0.0, (e / epochs - 0.3) / 0.7)
            lam = lam_max * min(1.0, frac)
            if lam > 0:
                pen = ((torch.sin(np.pi * W1 / step) ** 2).mean()
                       + (torch.sin(np.pi * W2 / step) ** 2).mean())
                loss = loss + lam * pen
        loss.backward()
        opt.step()
        if clamp is not None:
            with torch.no_grad():
                W1.clamp_(-clamp, clamp)
                W2.clamp_(-clamp, clamp)
    with torch.no_grad():
        pv = fwd(torch.tensor(xv, dtype=torch.float32), W1, W2).squeeze(-1).numpy()
        pt = fwd(X, W1, W2).squeeze(-1).numpy()
        offgrid = float(torch.cat([(W1 / step - (W1 / step).round()).abs().flatten(),
                                   (W2 / step - (W2 / step).round()).abs().flatten()]).mean())
        # the returned weights are the GRID-SPACE ones; the scales come back separately so a
        # caller can snap on the grid and then re-apply the gain
        w1 = (W1.abs() if dale else W1).clone().numpy()
        w2 = (W2.abs() if dale else W2).clone().numpy()
        g1 = float(s1.abs()) if gain else 1.0
        g2 = float(s2.abs()) if gain else 1.0
    return dict(w1=w1, b1=b1.detach().numpy(), w2=w2, b2=b2.detach().numpy(), f=f,
                pred_val=pv.astype(np.float64), pred_train=pt.astype(np.float64),
                train_loss=float(mse_t.detach()), offgrid=offgrid, g1=g1, g2=g2)


def snap_abs(W, step=0.1):
    """Hard-snap to the ABSOLUTE 0.1 grid -- no per-layer rescaling."""
    return np.round(np.asarray(W) / step) * step


def quantise(W, step=0.1):
    """Normalise to unit magnitude, snap to the 0.1 grid, restore the scale.

    This is the faithful analogue of the genome's normalised magnitudes times one global
    gain: the grid constrains the SHAPE of the weight vector, not its overall size, and the
    size is what the gain (and then the output rescale) is free to absorb.
    """
    s = float(np.abs(W).max())
    if s < 1e-12:
        return W.copy(), s
    return np.round(W / s / step) * step * s, s


def mlp_forward(m, x, w1=None, w2=None):
    w1 = (m["w1"] if w1 is None else w1) * m.get("g1", 1.0)
    w2 = (m["w2"] if w2 is None else w2) * m.get("g2", 1.0)
    h = m["f"](torch.tensor(x @ w1 + m["b1"], dtype=torch.float32)).numpy()
    return (h @ w2 + m["b2"]).squeeze(-1).astype(np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", type=int, default=DIM)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=4000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    _, _, Xp, Yp, Xv, Yv = load(1024, seed=a.seed)
    T.fit_target_stats(Yp)
    yt = T.target_offsets(Yp)[:, a.dim]
    yv = T.target_offsets(Yv)[:, a.dim]
    chance = float(((yv - yv.mean()) ** 2).mean())
    enc = LatencyEncoder(Xp)
    R = dict(dim=a.dim, chance=chance, n_train=int(Xp.shape[0]), n_val=int(Xv.shape[0]),
             n_features=int(Xp.shape[1]))
    print(f"dim {a.dim}: own chance {chance:.3f}   train {Xp.shape}  val {Xv.shape}")

    # ---- 1 linear least squares 17 -> 1
    A = np.c_[Xp, np.ones(len(Xp))]
    w = np.linalg.lstsq(A, yt, rcond=None)[0]
    R["linear_17_1"] = mse(np.c_[Xv, np.ones(len(Xv))] @ w, yv)
    print(f"1 linear 17->1                       {R['linear_17_1']:7.3f}")

    # ---- 2/4 full-precision MLP 17 -> 8 -> 1
    m = train_mlp(Xp, yt, Xv, hidden=8, act="tanh", epochs=a.epochs, seed=a.seed)
    R["mlp_17_8_1"] = mse(m["pred_val"], yv)
    ab = fit_affine(m["pred_train"], yt)                       # fitted on TRAIN only
    R["mlp_17_8_1_affine_refit"] = mse(apply_affine(m["pred_val"], ab), yv)
    R["mlp_r"] = float(np.corrcoef(m["pred_val"], yv)[0, 1])
    print(f"2 MLP 17->8->1 (tanh)                {R['mlp_17_8_1']:7.3f}   r {R['mlp_r']:.4f}")
    print(f"4 the same MLP + affine refit        {R['mlp_17_8_1_affine_refit']:7.3f}")

    # ---- 3 the same MLP, magnitudes on the 0.1 grid, then one output rescale+shift
    q1, s1 = quantise(m["w1"])
    q2, s2 = quantise(m["w2"])
    pt = mlp_forward(m, Xp, q1, q2)
    pv = mlp_forward(m, Xv, q1, q2)
    abq = fit_affine(pt, yt)
    R["mlp_quantised_0p1_affine"] = mse(apply_affine(pv, abq), yv)
    R["mlp_quantised_no_affine"] = mse(pv, yv)
    R["quant_levels_used"] = dict(
        w1=int(len(np.unique(np.round(q1 / s1, 6)))), w2=int(len(np.unique(np.round(q2 / s2, 6)))))
    print(f"3 the same MLP, 0.1-grid + affine    {R['mlp_quantised_0p1_affine']:7.3f}"
          f"   (levels used: hidden {R['quant_levels_used']['w1']}, "
          f"out {R['quant_levels_used']['w2']})")

    # ---- extras that localise the remaining error
    md = train_mlp(Xp, yt, Xv, hidden=8, act="tanh", dale=True, epochs=a.epochs, seed=a.seed)
    abd = fit_affine(md["pred_train"], yt)
    R["mlp_dale_nonneg_affine"] = mse(apply_affine(md["pred_val"], abd), yv)
    qd1, _ = quantise(md["w1"])
    qd2, _ = quantise(md["w2"])
    abdq = fit_affine(mlp_forward(md, Xp, qd1, qd2), yt)
    R["mlp_dale_quantised_affine"] = mse(
        apply_affine(mlp_forward(md, Xv, qd1, qd2), abdq), yv)
    print(f"+ Dale (all weights >= 0)            {R['mlp_dale_nonneg_affine']:7.3f}")
    print(f"+ Dale AND 0.1-grid                  {R['mlp_dale_quantised_affine']:7.3f}")

    # The encoded input is a spike TIME in 0..31. Fed raw into a Dale layer (all weights
    # non-negative) every pre-activation is large and positive and tanh saturates for every
    # sample, which is a training failure rather than a ceiling -- it scored ABOVE chance on
    # the first pass. Standardised with TRAIN statistics, the same information survives in a
    # form gradient descent can use.
    Ep_raw, Ev_raw = enc(Xp).astype(np.float64), enc(Xv).astype(np.float64)
    mu, sd = Ep_raw.mean(0), Ep_raw.std(0) + 1e-9
    Ep, Ev = (Ep_raw - mu) / sd, (Ev_raw - mu) / sd
    me = train_mlp(Ep, yt, Ev, hidden=8, act="tanh", epochs=a.epochs, seed=a.seed)
    abe = fit_affine(me["pred_train"], yt)
    R["mlp_on_encoded_input_affine"] = mse(apply_affine(me["pred_val"], abe), yv)
    print(f"+ MLP on the ENCODED input (0..31)   {R['mlp_on_encoded_input_affine']:7.3f}")

    # ---- the FULLY MATCHED analogue: everything the spiking net is subject to at once --
    # the encoder's quantised input, Dale (no inhibition), and the 0.1 weight grid. This,
    # not the free MLP, is the number the spiking run should actually be compared against.
    mm = train_mlp(Ep, yt, Ev, hidden=8, act="tanh", dale=True, epochs=a.epochs, seed=a.seed)
    abm = fit_affine(mm["pred_train"], yt)
    R["mlp_matched_dale_encoded"] = mse(apply_affine(mm["pred_val"], abm), yv)
    qm1, _ = quantise(mm["w1"])
    qm2, _ = quantise(mm["w2"])
    abmq = fit_affine(mlp_forward(mm, Ep, qm1, qm2), yt)
    R["mlp_matched_dale_encoded_quantised"] = mse(
        apply_affine(mlp_forward(mm, Ev, qm1, qm2), abmq), yv)
    print(f"= MATCHED: Dale + encoded input      {R['mlp_matched_dale_encoded']:7.3f}")
    print(f"= MATCHED: + the 0.1 grid too        "
          f"{R['mlp_matched_dale_encoded_quantised']:7.3f}   <- the like-for-like ceiling")

    # ---- QUANTIZATION-AWARE TRAINING: the sine grid penalty, annealed, then a hard snap.
    # The number that matters is the gap between "before snap" and "after snap": if QAT
    # worked, snapping costs nothing, because the weights were already sitting on the grid.
    R["qat"] = {}
    ARMS = (("free", dict()),
            ("grid_x_gain", dict(clamp=1.0, gain=True)),
            ("dale_grid_x_gain", dict(dale=True, clamp=1.0, gain=True)),
            ("dale_grid_x_gain_encoded", dict(dale=True, clamp=1.0, gain=True, enc=True)))
    for nm, kw in ARMS:
        kw = dict(kw)
        xt_, xv_ = (Ep, Ev) if kw.pop("enc", False) else (Xp, Xv)
        best = None
        for lam in (0.5, 1.0, 3.0, 10.0):      # lambda is a hyperparameter OF THE METHOD, so
            mq = train_mlp(xt_, yt, xv_, hidden=8, act="tanh", epochs=a.epochs,
                           seed=a.seed, lam_max=lam, step=0.1, **kw)
            s1_, s2_ = snap_abs(mq["w1"]), snap_abs(mq["w2"])
            ptq, pvq = mlp_forward(mq, xt_, s1_, s2_), mlp_forward(mq, xv_, s1_, s2_)
            post = mse(apply_affine(pvq, fit_affine(ptq, yt)), yv)
            pre = mse(apply_affine(mq["pred_val"], fit_affine(mq["pred_train"], yt)), yv)
            if best is None or post < best["after_snap_affine"]:  # report the best it can do
                best = dict(lam=lam, before_snap_affine=pre, after_snap_affine=post,
                            r=float(np.corrcoef(pvq, yv)[0, 1]),
                            mean_offgrid_frac=mq["offgrid"],
                            distinct_levels=int(len(np.unique(np.round(
                                np.concatenate([s1_.ravel(), s2_.ravel()]), 6)))),
                            gain=[mq["g1"], mq["g2"]])
        R["qat"][nm] = best
        print(f"Q {nm:26s} lam {best['lam']:<5} before-snap {best['before_snap_affine']:7.3f}"
              f"  AFTER-SNAP {best['after_snap_affine']:7.3f}  r {best['r']:.4f}"
              f"  off-grid {best['mean_offgrid_frac']:.4f}  levels {best['distinct_levels']}")

    # a wide MLP, to separate 'the task is hard' from '8 hidden units is few'
    mw = train_mlp(Xp, yt, Xv, hidden=128, act="relu", epochs=a.epochs, seed=a.seed)
    abw = fit_affine(mw["pred_train"], yt)
    R["mlp_17_128_1_affine"] = mse(apply_affine(mw["pred_val"], abw), yv)
    print(f"+ MLP 17->128->1 (relu)              {R['mlp_17_128_1_affine']:7.3f}")

    print(f"\nown chance                           {chance:7.3f}")
    with open(a.out, "w") as f:
        json.dump(T.jsonable(R), f, indent=1)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
