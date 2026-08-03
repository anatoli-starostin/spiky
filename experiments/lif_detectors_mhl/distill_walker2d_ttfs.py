"""Distill the real Walker2d int4 LUT actor into PureLIFDetectorsMHL (TTFS variant). Same recipe as
distill_walker2d.py (constant LR, mode='st' train / 'hard' eval), swapping in the TTFS module.

Run: PYTHONPATH=<spiky>/src python experiments/lif_detectors_mhl/distill_walker2d_ttfs.py [--steps 6000] [--save PATH]
"""
import os, time, argparse
import numpy as np
import torch
import torch.nn.functional as F

from spiky.lutorch.pure_lif_detectors_mhl import PureLIFDetectorsMHL

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
        return TW[torch.arange(NT).unsqueeze(0), addr].sum(1).unsqueeze(1)

    return oracle, dict(input_dim=N, n_heads=1, n_outputs=6, n_anchor_pairs=NAP, tables_per_head=NT), TW


def mean_abs_logit(student, x):
    with torch.no_grad():
        _, soft, _ = student._spike_bits(x)
        logit = torch.log(soft / (1 - soft))          # == s_soft/temp_bit (pre-sigmoid bit logit)
    return logit.abs().mean().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--save", type=str, default=os.path.join(HERE, "pure_lif_ttfs_student.pt"))
    a = ap.parse_args()
    torch.manual_seed(0); torch.set_num_threads(max(1, os.cpu_count() - 1))
    oracle, cfg, TW = load_oracle()
    student = PureLIFDetectorsMHL(table_init=TW, **cfg)
    tot = sum(p.numel() for p in student.parameters())
    print("PureLIFDetectorsMHL total params:", tot, flush=True)
    opt = torch.optim.Adam(student.parameters(), lr=3e-3)             # constant LR
    gen = torch.Generator().manual_seed(1); t0 = time.time(); curve = []
    xprobe = torch.randn(512, cfg["input_dim"], generator=torch.Generator().manual_seed(5))
    logit0 = mean_abs_logit(student, xprobe)
    for s in range(a.steps):
        eps = 2.0 + (0.3 - 2.0) * s / max(1, a.steps - 1)             # accepted but unused by TTFS
        x = torch.randn(a.batch, cfg["input_dim"], generator=gen)
        loss = F.mse_loss(student(x, eps=eps, mode="st"), oracle(x))
        opt.zero_grad(); loss.backward(); opt.step()
        if s % 500 == 0 or s == a.steps - 1:
            curve.append((s, round(loss.item(), 4)))
            print(f"step {s:5d} MSE {loss.item():.4f} ({time.time()-t0:.0f}s)", flush=True)
    xe = torch.randn(4096, cfg["input_dim"], generator=torch.Generator().manual_seed(7))
    ye = oracle(xe); ovar = ye.var(0).mean().item(); arange = float(ye.max() - ye.min())
    with torch.no_grad():
        hard = F.mse_loss(student(xe, eps=0.15, mode="hard"), ye).item()
    r2 = 1 - hard / ovar; rmse = hard ** 0.5
    tau = student.tau.detach(); Tc = student.T_cross.detach(); tb = student.temp_bit.detach()
    print(f"FINAL hard MSE {hard:.4f} R2 {r2:.4f} RMSE {rmse:.4f} = {100*rmse/arange:.2f}% of range", flush=True)
    print(f"per-LUT tau [{tau.min():.3f},{tau.max():.3f}] med {tau.median():.3f} | T_cross [{Tc.min():.3f},{Tc.max():.3f}] "
          f"med {Tc.median():.3f} | temp_bit [{tb.min():.3f},{tb.max():.3f}] med {tb.median():.3f}", flush=True)
    print(f"mean|per-bit logit| {logit0:.3f}->{mean_abs_logit(student, xprobe):.3f} | {time.time()-t0:.0f}s", flush=True)
    metrics = dict(hard_mse=hard, hard_r2=r2, hard_rmse=rmse, rmse_pct_range=100 * rmse / arange,
                   total_params=tot, steps=a.steps, batch=a.batch, lr_schedule="constant", model="ttfs", curve=curve)
    torch.save({"state_dict": student.state_dict(), "config": cfg, "metrics": metrics}, a.save)
    print("saved checkpoint ->", a.save, flush=True)


if __name__ == "__main__":
    main()
