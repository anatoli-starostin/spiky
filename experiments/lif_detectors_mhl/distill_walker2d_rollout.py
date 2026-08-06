"""Distill the frozen int4 Walker2d LUT actor into RolloutLIFGroupsMHL (genuine time-stepped LIF net).

Same protocol as distill_walker2d_bucket.py plus grad-clip 1.0 (standard for surrogate-gradient BPTT). This
is the best lean variant: hard R2 ~0.502 at the default M=8 groups x N=14 neurons (~4.6k params), beating the
plain bucket table (0.418). Slower per step (sequential K=32 unroll).

Run: PYTHONPATH=<spiky>/src python experiments/lif_detectors_mhl/distill_walker2d_rollout.py [--steps 6000]
"""
import os, time, argparse
import numpy as np
import torch
import torch.nn.functional as F

from rollout_lif_groups_mhl import RolloutLIFGroupsMHL

HERE = os.path.dirname(os.path.abspath(__file__))


def load_oracle():
    Q = np.load(os.path.join(HERE, "walker2d_lut_actor_int4.npz"), allow_pickle=True)
    Wd = torch.tensor(Q["w_q"].astype(np.float64) * Q["w_scale"].astype(np.float64)[:, None, None], dtype=torch.float32)
    Bd = torch.tensor(Q["b_q"].astype(np.float64) * Q["b_scale"].astype(np.float64)[:, None], dtype=torch.float32)
    TW = torch.tensor(Q["weights_q"].astype(np.float64) * Q["weights_scale"].astype(np.float64)[:, None, None], dtype=torch.float32)[:, :, :6]
    NT, NAP, N = Wd.shape
    pow2 = (1 << torch.arange(NAP - 1, -1, -1)).long()

    @torch.no_grad()
    def oracle(x):
        a = torch.einsum('bi,tki->btk', x, Wd) + Bd
        addr = ((a > 0).long() * pow2).sum(-1)
        return TW[torch.arange(NT).unsqueeze(0), addr].sum(1)   # (B, 6)
    return oracle, N


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", type=int, default=8)
    ap.add_argument("--neurons", type=int, default=14)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--rollout-steps", type=int, default=32)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--save", type=str, default="")
    a = ap.parse_args()
    torch.manual_seed(0); torch.set_num_threads(max(1, os.cpu_count() - 1))
    oracle, N = load_oracle()
    s = RolloutLIFGroupsMHL(in_dim=N, out_dim=6, groups=a.groups, neurons_per_group=a.neurons, steps=a.rollout_steps)
    print(f"RolloutLIFGroupsMHL params: {s.param_count()} (M={a.groups} N={a.neurons} K={a.rollout_steps} P={s.P})", flush=True)
    opt = torch.optim.Adam(s.parameters(), lr=3e-3)
    gen = torch.Generator().manual_seed(1); t0 = time.time()
    for step in range(a.steps):
        x = torch.randn(a.batch, N, generator=gen)
        loss = F.mse_loss(s(x, mode="st"), oracle(x))
        opt.zero_grad(); loss.backward()
        if a.clip > 0:
            torch.nn.utils.clip_grad_norm_(s.parameters(), a.clip)
        opt.step()
        if step % 1000 == 0 or step == a.steps - 1:
            print(f"step {step:5d} MSE {loss.item():.4f} ({time.time()-t0:.0f}s)", flush=True)
    xe = torch.randn(4096, N, generator=torch.Generator().manual_seed(7))
    ye = oracle(xe); ovar = ye.var(0).mean().item(); arange = float(ye.max() - ye.min())
    with torch.no_grad():
        mse = F.mse_loss(s(xe, mode="hard"), ye).item()
    r2 = 1 - mse / ovar
    print(f"FINAL hard R2 {r2:.4f} MSE {mse:.4f} RMSE {mse**0.5:.4f} = {100*mse**0.5/arange:.2f}% of range | {time.time()-t0:.0f}s", flush=True)
    if a.save:
        torch.save({"state_dict": s.state_dict(), "metrics": dict(hard_r2=r2, hard_mse=mse)}, a.save)


if __name__ == "__main__":
    main()
