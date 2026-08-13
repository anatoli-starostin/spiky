"""exp012: can BACKPROP find an Izhikevich output-stage operating point the sweep missed?

The hand sweep capped at R2 0.586 (analysis/handcraft_output_izhikevich.json). The failure was
structural -- Izhikevich is leaky below -53.49 and anti-leaky only above it, and above it is
the 2-tick self-committing upstroke -- so the question is whether a trainable encoding can
work around it.

IMPORTANT ABOUT WHAT THIS IS. The engine has no gradients, so this is a PyTorch replica of the
kernel's own update, term for term:

    if V >= 30:  V = c;  U += d
    twice:       V += 0.5 * ((0.04*V + 5.0)*V + 140.0 - U + I)
    U += a*(b*V - U)

Spikes use a fast-sigmoid surrogate on the backward pass only; the forward pass is the exact
hard threshold, so the reported numbers are the real discrete dynamics. Any operating point
this finds still has to be re-verified on the real engine before it counts.

Trainable: a per-table amplitude and time offset for the input encoding, plus the output
affine. The Izhikevich parameters themselves are FIXED at the standard values -- tuning them
would just be re-deriving the LIF.
"""
import argparse
import json

import numpy as np
import torch
import torch.nn as nn

TPH, TAU, CLAMP = 32, 0.09036568, 60.0
IZH = dict(cf2=0.04, cf1=5.0, cf0=140.0, a=0.02, b=0.2, c=-65.0, d=8.0, th=30.0)


class Spike(torch.autograd.Function):
    """Hard threshold forward, fast-sigmoid surrogate backward."""

    @staticmethod
    def forward(ctx, v):
        ctx.save_for_backward(v)
        return (v >= 0).float()

    @staticmethod
    def backward(ctx, g):
        (v,) = ctx.saved_tensors
        return g / (1.0 + 10.0 * v.abs()) ** 2


class IzhOutput(nn.Module):
    """32 value-encoded inputs -> one standard Izhikevich cell -> soft first-spike time."""

    def __init__(self, n_in=32, n_ticks=64, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.n_ticks = n_ticks
        self.amp = nn.Parameter(torch.full((n_in,), 3.0) +
                                0.1 * torch.randn(n_in, generator=g))
        self.tshift = nn.Parameter(torch.zeros(n_in))
        self.tscale = nn.Parameter(torch.tensor(8.0))
        self.sharp = nn.Parameter(torch.tensor(2.0))
        self.out_a = nn.Parameter(torch.tensor(0.1))
        self.out_b = nn.Parameter(torch.tensor(0.0))

    def forward(self, w):
        """w [B, 32] selected cell values -> predicted output [B]."""
        B, n = w.shape
        dev = w.device
        # value -> arrival time, trainable scale/offset; bigger value fires earlier
        t_arr = -self.tscale * (w / TAU) + self.tshift[None, :]
        t_arr = t_arr - t_arr.min(dim=1, keepdim=True).values.detach() + 2.0
        ticks = torch.arange(self.n_ticks, device=dev, dtype=w.dtype)
        # soft one-hot over ticks so the arrival time is differentiable
        k = torch.exp(-((ticks[None, None, :] - t_arr[:, :, None]) ** 2)
                      / (2.0 * torch.nn.functional.softplus(self.sharp) ** 2))
        k = k / (k.sum(-1, keepdim=True) + 1e-9)
        I = torch.einsum("bnt,n->bt", k, torch.nn.functional.softplus(self.amp))

        V = torch.full((B,), IZH["c"], device=dev, dtype=w.dtype)
        U = torch.full((B,), IZH["b"] * IZH["c"], device=dev, dtype=w.dtype)
        alive = torch.ones(B, device=dev, dtype=w.dtype)
        Tsoft = torch.zeros(B, device=dev, dtype=w.dtype)
        psum = torch.zeros(B, device=dev, dtype=w.dtype)
        for t in range(self.n_ticks):
            s = Spike.apply(V - IZH["th"])                 # detect (the kernel's order)
            p = s * alive
            Tsoft = Tsoft + p * t
            psum = psum + p
            alive = alive * (1.0 - s)
            V = torch.where(s > 0, torch.full_like(V, IZH["c"]), V)
            U = U + s * IZH["d"]
            for _ in range(2):
                V = V + 0.5 * ((IZH["cf2"] * V + IZH["cf1"]) * V + IZH["cf0"] - U + I[:, t])
            U = U + IZH["a"] * (IZH["b"] * V - U)
        T = Tsoft + (1.0 - psum) * float(self.n_ticks)     # never fired -> the last tick
        return self.out_a * T + self.out_b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dims", default="0")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--ntrain", type=int, default=20000)
    ap.add_argument("--ticks", type=int, default=64)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    import os
    Z = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data",
                             "distill_exp19_100k.npz"))
    x = Z["x_norm"].astype(np.float64)
    A_, B_ = Z["anchor_a"], Z["anchor_b"]
    W = Z["weights"].astype(np.float64)
    d = x[:, A_] - x[:, B_]
    idx = ((d > 0) * (2 ** np.arange(5, -1, -1))).sum(-1)
    w_sel = W.reshape(32 * 64, 6)[idx + (np.arange(32) * 64)[None, :]]
    y = Z["y_action_mean_f64"]
    ntr = len(x) - 4000
    dev = "cuda"
    R = {"note": "PyTorch replica of the kernel update; surrogate gradient; "
                 "standard Izhikevich params FIXED", "dims": {}}

    for o in [int(v) for v in a.dims.split(",")]:
        Xtr = torch.tensor(w_sel[:a.ntrain, :, o], dtype=torch.float32, device=dev)
        Ytr = torch.tensor(y[:a.ntrain, o], dtype=torch.float32, device=dev)
        Xv = torch.tensor(w_sel[ntr:, :, o], dtype=torch.float32, device=dev)
        Yv = torch.tensor(y[ntr:, o], dtype=torch.float32, device=dev)
        net = IzhOutput(n_ticks=a.ticks).to(dev)
        opt = torch.optim.Adam(net.parameters(), lr=0.02)
        nb = 256
        for ep in range(a.epochs):
            perm = torch.randperm(len(Xtr), device=dev)
            tot = 0.0
            for i in range(0, len(Xtr), nb):
                j = perm[i:i + nb]
                opt.zero_grad()
                loss = ((net(Xtr[j]) - Ytr[j]) ** 2).mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
                tot += float(loss) * len(j)
            if ep % 5 == 0 or ep == a.epochs - 1:
                with torch.no_grad():
                    pv = torch.cat([net(Xv[i:i + 512]) for i in range(0, len(Xv), 512)])
                    mse = float(((pv - Yv) ** 2).mean())
                    var = float(Yv.var())
                print(f"  dim {o} epoch {ep:3d}  train {tot / len(Xtr):.4f}  "
                      f"held-out MSE {mse:.4f}  R2 {1 - mse / var:.4f}", flush=True)
        with torch.no_grad():
            pv = torch.cat([net(Xv[i:i + 512]) for i in range(0, len(Xv), 512)])
            mse = float(((pv - Yv) ** 2).mean())
            var = float(Yv.var())
            mx = float((pv - Yv).abs().max())
        R["dims"][str(o)] = dict(mse=mse, max_err=mx, target_var=var, r2=1 - mse / var,
                                 epochs=a.epochs, ntrain=a.ntrain, ticks=a.ticks)
        print(f"  -> dim {o} FINAL held-out MSE {mse:.6f}  max|err| {mx:.4f}  "
              f"R2 {1 - mse / var:.4f}\n")
    if a.out:
        json.dump(R, open(a.out, "w"), indent=1)
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
