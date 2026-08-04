"""Distill the frozen int4 Walker2d LUT actor into ProductBucketLIFMHL (mixed-radix product of detectors).

Same protocol as distill_walker2d_bucket.py plus grad-clip 1.0. Each head has N_det M-way bucket detectors
whose digits form a mixed-radix index into an M**N_det cell table (hard gather + rank-1 tensor-product soft
backward). See RESULTS_product_bucket.md for the sweep. Best config found: M=2, N_det=6, heads=32 -> R2 0.523.

Run: PYTHONPATH=<spiky>/src python experiments/lif_detectors_mhl/distill_walker2d_product.py [--heads 32] [--n-det 6] [--buckets 2]
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
        return TW[torch.arange(NT).unsqueeze(0), addr].sum(1)   # (B, 6)
    return oracle, N


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--heads", type=int, default=32)
    ap.add_argument("--n-det", type=int, default=6)
    ap.add_argument("--buckets", type=int, default=2)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--save", type=str, default="")
    a = ap.parse_args()
    torch.manual_seed(0); torch.set_num_threads(max(1, os.cpu_count() - 1))
    oracle, N = load_oracle()
    # product config = LIFMultiHeadLUT with a single output head, `heads` tables summed within it, n_det/table.
    s = LIFMultiHeadLUT(input_dim=N, n_heads=1, n_outputs=6, tables_per_head=a.heads, n_det=a.n_det, n_buckets=a.buckets)
    norm = torch.nn.LayerNorm(N)                              # standardize input for the fixed latency code
    fwd = lambda xx: s(norm(xx)).sum(dim=1)                   # (B,1,out) -> (B,out): ST forward; product sums the heads
    print(f"LIFMultiHeadLUT (product config) params: {s.param_count()} (heads={a.heads} n_det={a.n_det} M={a.buckets} "
          f"cells/head={s.cells})", flush=True)
    opt = torch.optim.Adam(list(s.parameters()) + list(norm.parameters()), lr=3e-3)
    gen = torch.Generator().manual_seed(1); t0 = time.time()
    for step in range(a.steps):
        x = torch.randn(a.batch, N, generator=gen)
        loss = F.mse_loss(fwd(x), oracle(x))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(s.parameters(), 1.0)
        opt.step()
        if step % 1000 == 0 or step == a.steps - 1:
            print(f"step {step:5d} MSE {loss.item():.4f} ({time.time()-t0:.0f}s)", flush=True)
    xe = torch.randn(4096, N, generator=torch.Generator().manual_seed(7))
    ye = oracle(xe); ovar = ye.var(0).mean().item(); arange = float(ye.max() - ye.min())
    with torch.no_grad():
        mse = F.mse_loss(s.eval_forward(norm(xe)).sum(dim=1), ye).item()   # efficient hard eval (no soft math)
    r2 = 1 - mse / ovar
    print(f"FINAL hard R2 {r2:.4f} MSE {mse:.4f} RMSE {mse**0.5:.4f} = {100*mse**0.5/arange:.2f}% of range | {time.time()-t0:.0f}s", flush=True)
    if a.save:
        torch.save({"state_dict": s.state_dict(), "metrics": dict(hard_r2=r2, hard_mse=mse)}, a.save)


if __name__ == "__main__":
    main()
