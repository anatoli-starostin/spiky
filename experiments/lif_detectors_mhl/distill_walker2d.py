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

from lif_detectors_mhl import LIFDetectorsMHL

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


def _mean_abs_logit(student, oracle, cfg, x):
    with torch.no_grad():
        V = student.detector_membrane(student.latency(x), 0.3).view(
            x.shape[0], student.n_tables, student.n_anchor_pairs)
        s = (V - student.theta.view(1, student.n_tables, 1)) / student.temp_bit
    return s.abs().mean().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--save", type=str, default=os.path.join(HERE, "lif_detectors_mhl_student.pt"),
                    help="path to save the trained student checkpoint")
    a = ap.parse_args()
    torch.manual_seed(0); torch.set_num_threads(max(1, os.cpu_count() - 1))
    oracle, cfg, TW = load_oracle()
    student = LIFDetectorsMHL(table_init=TW, **cfg)      # warm-start tables from oracle dequant weights
    opt = torch.optim.Adam(student.parameters(), lr=3e-3)          # constant LR (cosine anneal was tried and dropped — worse)
    gen = torch.Generator().manual_seed(1); t0 = time.time(); curve = []
    xprobe = torch.randn(512, cfg["input_dim"], generator=torch.Generator().manual_seed(5))
    temp0, logit0 = float(student.temp_bit), _mean_abs_logit(student, oracle, cfg, xprobe)
    for s in range(a.steps):
        eps = 2.0 + (0.3 - 2.0) * s / max(1, a.steps - 1)
        x = torch.randn(a.batch, cfg["input_dim"], generator=gen)
        loss = F.mse_loss(student(x, eps=eps, mode="st"), oracle(x))
        opt.zero_grad(); loss.backward(); opt.step()
        if s % 500 == 0 or s == a.steps - 1:
            curve.append((s, round(loss.item(), 4)))
            print(f"step {s:5d}  MSE {loss.item():.4f}  ({time.time()-t0:.0f}s)", flush=True)

    xe = torch.randn(4096, cfg["input_dim"], generator=torch.Generator().manual_seed(7))
    ye = oracle(xe); ovar = ye.var(0).mean().item(); arange = float(ye.max() - ye.min())
    with torch.no_grad():
        hard = F.mse_loss(student(xe, eps=0.15, mode="hard"), ye).item()
    r2 = 1 - hard / ovar; rmse = hard ** 0.5
    temp_end, logit_end = float(student.temp_bit), _mean_abs_logit(student, oracle, cfg, xprobe)
    print(f"HARD/argmax held-out MSE {hard:.4f}  R2 {r2:.4f}  RMSE {rmse:.4f} = {100*rmse/arange:.2f}% of range", flush=True)
    print(f"temp_bit {temp0:.3f}->{temp_end:.3f} | mean|logit| {logit0:.3f}->{logit_end:.3f} | {time.time()-t0:.0f}s", flush=True)

    metrics = dict(hard_mse=hard, hard_r2=r2, hard_rmse=rmse, rmse_pct_range=100 * rmse / arange,
                   temp_bit=temp_end, mean_abs_logit=logit_end, steps=a.steps, batch=a.batch,
                   lr_schedule="constant", curve=curve)
    torch.save({"state_dict": student.state_dict(), "config": cfg, "metrics": metrics,
                "latency": dict(t_window=student.t_window, c=student.latency_c, alpha=student.latency_alpha)},
               a.save)
    print(f"saved checkpoint -> {a.save}", flush=True)


if __name__ == "__main__":
    main()
