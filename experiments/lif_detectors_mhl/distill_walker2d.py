"""Repro: distill the real trained Walker2d int4 LUT actor into LIFDetectorsMHL (output-action MSE).

Produces the numbers in the README (hard-inference R2 ~0.61). This is the thin experiment wrapper; the
reviewable core is the module spiky.lutorch.lif_detectors_mhl.LIFDetectorsMHL + its tests.

Run (CPU is fine):
    PYTHONPATH=<spiky>/src python experiments/lif_detectors_mhl/distill_walker2d.py [--steps 6000]

The teacher (walker2d_lut_actor_int4.npz) + obs stats sit beside this script so it is self-contained.
"""
import os, json, time, argparse
import numpy as np
import torch
import torch.nn.functional as F

from spiky.lutorch.lif_detectors_mhl import LIFDetectorsMHL

HERE = os.path.dirname(os.path.abspath(__file__))


def load_oracle():
    Q = np.load(os.path.join(HERE, "walker2d_lut_actor_int4.npz"), allow_pickle=True)
    Wd = torch.tensor(Q["w_q"].astype(np.float64) * Q["w_scale"].astype(np.float64)[:, None, None], dtype=torch.float32)  # (32,6,17)
    Bd = torch.tensor(Q["b_q"].astype(np.float64) * Q["b_scale"].astype(np.float64)[:, None], dtype=torch.float32)          # (32,6)
    TW = torch.tensor(Q["weights_q"].astype(np.float64) * Q["weights_scale"].astype(np.float64)[:, None, None], dtype=torch.float32)[:, :, :6]  # (32,64,6)
    NT, NAP, N = Wd.shape
    pow2 = (1 << torch.arange(NAP - 1, -1, -1)).long()

    @torch.no_grad()
    def oracle(x):                                       # (B,N) -> (B,1,6)
        a = torch.einsum('bi,tki->btk', x, Wd) + Bd
        addr = ((a > 0).long() * pow2).sum(-1)
        rows = TW[torch.arange(NT).unsqueeze(0), addr].sum(1)
        return rows.unsqueeze(1)

    return oracle, dict(input_dim=N, n_heads=1, n_outputs=6, n_anchor_pairs=NAP, tables_per_head=NT), TW


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=256)
    a = ap.parse_args()
    torch.manual_seed(0); torch.set_num_threads(max(1, os.cpu_count() - 1))
    oracle, cfg, TW = load_oracle()
    student = LIFDetectorsMHL(table_init=TW, **cfg)      # warm-start tables from oracle dequant weights
    opt = torch.optim.Adam(student.parameters(), lr=3e-3)
    gen = torch.Generator().manual_seed(1); t0 = time.time()
    for s in range(a.steps):
        eps = 2.0 + (0.3 - 2.0) * s / max(1, a.steps - 1)
        x = torch.randn(a.batch, cfg["input_dim"], generator=gen)
        loss = F.mse_loss(student(x, eps=eps, mode="st"), oracle(x))
        opt.zero_grad(); loss.backward(); opt.step()
        if s % 500 == 0:
            print(f"step {s:5d}  MSE {loss.item():.4f}  ({time.time()-t0:.0f}s)", flush=True)

    xe = torch.randn(4096, cfg["input_dim"], generator=torch.Generator().manual_seed(7))
    ye = oracle(xe); ovar = ye.var(0).mean().item()
    with torch.no_grad():
        hard = F.mse_loss(student(xe, eps=0.15, mode="hard"), ye).item()
    print(f"HARD/argmax held-out MSE {hard:.4f}  R2 {1-hard/ovar:.4f}  "
          f"RMSE {hard**0.5:.4f} = {100*hard**0.5/float(ye.max()-ye.min()):.2f}% of range")


if __name__ == "__main__":
    main()
