"""Distill the real Walker2d int4 LUT actor into BucketLIFDetectorsMHL (M=16 time buckets). Same recipe
and eval protocol as distill_walker2d_ttfs.py (constant LR 3e-3, train-mode ST forward / eval-mode hard forward, same seeds
and batch/step counts) so the hard-R2 is apples-to-apples with the TTFS and bit variants.

The bucket table has 16 rows (one per time bucket), which does NOT match the oracle's 64-row (2**6) LUT, so
unlike the TTFS harness there is no table warm-start — the 16-row table is learned from scratch. Teacher is
still the frozen int4 LUT oracle.

Run: PYTHONPATH=<spiky>/src python experiments/lif_detectors_mhl/distill_walker2d_bucket.py [--steps 6000] [--save PATH]
"""
import os, time, argparse
import numpy as np
import torch
import torch.nn.functional as F

from spiky.lutorch.lif_multi_head_lut import LIFMultiHeadLUT

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

    # student cfg: bucket variant takes no n_anchor_pairs / no table_init (16 buckets != 64 LUT rows)
    return oracle, dict(input_dim=N, n_heads=1, n_outputs=6, tables_per_head=NT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--buckets", type=int, default=16)
    ap.add_argument("--tables", type=int, default=0)   # 0 => use the oracle's table count (32); overrides the STUDENT only
    ap.add_argument("--save", type=str, default=os.path.join(HERE, "bucket_lif_ttfs_student.pt"))
    a = ap.parse_args()
    torch.manual_seed(0); torch.set_num_threads(max(1, os.cpu_count() - 1))
    oracle, cfg = load_oracle()
    if a.tables > 0:
        cfg["tables_per_head"] = a.tables              # oracle is a closure over its own 32 tables -> unaffected
    student = LIFMultiHeadLUT(n_buckets=a.buckets, n_det=1, **cfg)
    norm = torch.nn.LayerNorm(cfg["input_dim"])                       # standardize input for the fixed latency code
    tot = sum(p.numel() for p in student.parameters())
    print(f"LIFMultiHeadLUT (n_det=1) total params: {tot}  (n_buckets={a.buckets})", flush=True)
    opt = torch.optim.Adam(list(student.parameters()) + list(norm.parameters()), lr=3e-3)   # constant LR
    gen = torch.Generator().manual_seed(1); t0 = time.time(); curve = []
    student.train(); norm.train()                                    # training -> straight-through forward
    for s in range(a.steps):
        x = torch.randn(a.batch, cfg["input_dim"], generator=gen)
        loss = F.mse_loss(student(norm(x)), oracle(x))                # straight-through forward
        opt.zero_grad(); loss.backward(); opt.step()
        if s % 500 == 0 or s == a.steps - 1:
            curve.append((s, round(loss.item(), 4)))
            print(f"step {s:5d} MSE {loss.item():.4f} ({time.time()-t0:.0f}s)", flush=True)
    xe = torch.randn(4096, cfg["input_dim"], generator=torch.Generator().manual_seed(7))
    ye = oracle(xe); ovar = ye.var(0).mean().item(); arange = float(ye.max() - ye.min())
    student.eval(); norm.eval()                                       # eval -> efficient hard forward (no soft math)
    with torch.no_grad():
        hard = F.mse_loss(student(norm(xe)), ye).item()
    r2 = 1 - hard / ovar; rmse = hard ** 0.5
    tau = student.tau.detach(); Tc = student.T_cross.detach(); Tb = student.T_bkt.detach()
    print(f"FINAL hard MSE {hard:.4f} R2 {r2:.4f} RMSE {rmse:.4f} = {100*rmse/arange:.2f}% of range", flush=True)
    print(f"per-LUT tau [{tau.min():.3f},{tau.max():.3f}] med {tau.median():.3f} | T_cross [{Tc.min():.3f},{Tc.max():.3f}] "
          f"med {Tc.median():.3f} | T_bkt [{Tb.min():.3f},{Tb.max():.3f}] med {Tb.median():.3f}", flush=True)
    # bucket occupancy over the eval set (how many of the 16 buckets are actually used, per table on average)
    with torch.no_grad():
        addr = student.address(norm(xe))                                    # (4096, n_tables)
        used = torch.stack([torch.bincount(addr[:, t], minlength=a.buckets).gt(0).sum() for t in range(cfg["tables_per_head"])])
    print(f"bucket occupancy: mean {used.float().mean():.1f}/{a.buckets} used per table (min {int(used.min())}, max {int(used.max())}) | {time.time()-t0:.0f}s", flush=True)
    metrics = dict(hard_mse=hard, hard_r2=r2, hard_rmse=rmse, rmse_pct_range=100 * rmse / arange,
                   total_params=tot, steps=a.steps, batch=a.batch, n_buckets=a.buckets,
                   lr_schedule="constant", model="bucket", curve=curve)
    torch.save({"state_dict": student.state_dict(), "config": {**cfg, "n_buckets": a.buckets}, "metrics": metrics}, a.save)
    print("saved checkpoint ->", a.save, flush=True)


if __name__ == "__main__":
    main()
