"""Measured per-token decode cost of one attention-replacement layer (recurrent / KV-cache form), CPU.

For each mixer and context length t, the state is filled as if t tokens had been processed (random
content: cost does not depend on values), then `reps` decode steps are timed. One step = everything
the mixer does for one new token: projections, address, write, read. MLP and embeddings are excluded
(identical for all mixers). Batch B sequences decoded in parallel; B=1 shows latency incl. framework
overhead, larger B shows throughput.

Writes runs/bench_decode/results.csv and results.md.
"""
import csv
import os
import platform
import time

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
D, H, DH, DK = 128, 4, 32, 32


class Softmax:
    """KV cache; step cost grows with t."""
    def __init__(self, B, t, tmax):
        self.Wqkv = torch.randn(D, 3 * D) / D ** 0.5
        self.K = torch.randn(B, H, tmax, DH)
        self.V = torch.randn(B, H, tmax, DH)
        self.t = t

    def step(self, x):
        B = x.shape[0]
        q, k, v = (x @ self.Wqkv).view(B, 3, H, DH).unbind(1)
        self.K[:, :, self.t] = k
        self.V[:, :, self.t] = v
        self.t += 1
        s = torch.einsum("bhd,bhtd->bht", q, self.K[:, :, :self.t]) / DH ** 0.5
        return torch.einsum("bht,bhtd->bhd", s.softmax(-1), self.V[:, :, :self.t])

    def state_bytes(self):
        return 2 * self.K[:, :, :self.t].numel() * 4


class Linear:
    """Linear attention with feature dim F per head: state F x DH plus normaliser."""
    def __init__(self, B, feat):
        self.F = feat
        self.Wqk = torch.randn(D, 2 * H * feat) / D ** 0.5
        self.Wv = torch.randn(D, D) / D ** 0.5
        self.S = torch.zeros(B, H, feat, DH)
        self.z = torch.zeros(B, H, feat)

    def step(self, x):
        B = x.shape[0]
        q, k = (x @ self.Wqk).view(B, 2, H, self.F).unbind(1)
        q, k = F.elu(q) + 1, F.elu(k) + 1
        v = (x @ self.Wv).view(B, H, DH)
        self.S += k[..., None] * v[:, :, None, :]
        self.z += k
        return torch.einsum("bhf,bhfd->bhd", q, self.S) / (q * self.z).sum(-1, keepdim=True)

    def state_bytes(self):
        return (self.S.numel() + self.z.numel()) * 4


class Drawers:
    """Tied drawer memory, tph tables x 2**nap cells per head; hard addressing (inference form)."""
    def __init__(self, B, tph, nap):
        self.tph, self.nap, self.K = tph, nap, 2 ** nap
        self.Wqk = torch.randn(D, H * DK) / D ** 0.5
        self.Wv = torch.randn(D, D) / D ** 0.5
        pairs = torch.combinations(torch.arange(DK), 2)
        idx = torch.stack([torch.randperm(len(pairs))[:nap] for _ in range(H * tph)]).view(H, tph * nap)
        self.a, self.b = pairs[idx, 0], pairs[idx, 1]  # [H, tph*nap]
        self.pow2 = 2 ** torch.arange(nap - 1, -1, -1)
        cells = H * tph * self.K
        self.S = torch.zeros(B, cells, DH)       # all heads' tables flattened: row = (h, t, c)
        self.N = torch.zeros(B, cells)
        self.base_h = (torch.arange(H) * tph * self.K)[:, None]           # [H,1]
        self.base_t = (torch.arange(tph) * self.K)[None, :]               # [1,tph]

    def step(self, x):
        B = x.shape[0]
        z = (x @ self.Wqk).view(B, H, DK)
        u = (z.gather(-1, self.a.expand(B, -1, -1)) - z.gather(-1, self.b.expand(B, -1, -1)))
        c = ((u.view(B, H, self.tph, self.nap) > 0).long() * self.pow2).sum(-1)    # [B,H,tph]
        rows = (c + self.base_h + self.base_t).view(B, H * self.tph)                # flat row ids
        v = (x @ self.Wv).view(B, H, 1, DH).expand(B, H, self.tph, DH).reshape(B, H * self.tph, DH)
        self.S.scatter_add_(1, rows[..., None].expand(-1, -1, DH), v)              # write tph rows/head
        self.N.scatter_add_(1, rows, torch.ones(B, H * self.tph))
        got = self.S.gather(1, rows[..., None].expand(-1, -1, DH)).view(B, H, self.tph, DH)  # read
        cnt = self.N.gather(1, rows).view(B, H, self.tph)
        return got.sum(2) / cnt.sum(2, keepdim=True)

    def state_bytes(self):
        return (self.S.numel() + self.N.numel()) * 4


class DrawersTopK(Drawers):
    """Bounded soft read/write: per table, the k most likely cells (product of bit sigmoids), weighted."""
    def __init__(self, B, tph, nap, k=4, tau=1.0):
        super().__init__(B, tph, nap)
        self.k, self.tau = k, tau
        order = torch.arange(nap - 1, -1, -1)
        self.cell_bits = ((torch.arange(self.K)[:, None] >> order[None, :]) & 1).float()   # [K,nap]

    def step(self, x):
        B = x.shape[0]
        z = (x @ self.Wqk).view(B, H, DK)
        z = z / (z.abs().mean(-1, keepdim=True) + 1e-6)
        u = (z.gather(-1, self.a.expand(B, -1, -1)) - z.gather(-1, self.b.expand(B, -1, -1))).view(B, H, self.tph, self.nap)
        lp1, lp0 = F.logsigmoid(u / self.tau), F.logsigmoid(-u / self.tau)
        logp = lp1 @ self.cell_bits.T + lp0 @ (1 - self.cell_bits).T                  # [B,H,tph,K]
        vals, idx = logp.topk(self.k, -1)
        w = (vals - vals.max(-1, keepdim=True).values).exp()
        w = (w / w.sum(-1, keepdim=True)).view(B, H * self.tph * self.k)
        rows = (idx + self.base_h[..., None] + self.base_t[..., None]).view(B, H * self.tph * self.k)
        v = (x @ self.Wv).view(B, H, 1, DH).expand(B, H, self.tph * self.k, DH).reshape(B, -1, DH)
        self.S.scatter_add_(1, rows[..., None].expand(-1, -1, DH), w[..., None] * v)
        self.N.scatter_add_(1, rows, w)
        got = (self.S.gather(1, rows[..., None].expand(-1, -1, DH)) * w[..., None]).view(B, H, -1, DH)
        cnt = (self.N.gather(1, rows) * w).view(B, H, -1)
        return got.sum(2) / cnt.sum(2, keepdim=True)


class DrawersFlip(Drawers):
    """Cheap bounded soft read/write: cells reached by flipping subsets of the b least-certain bits."""
    def __init__(self, B, tph, nap, b=2, tau=1.0):
        super().__init__(B, tph, nap)
        self.nb, self.tau = b, tau
        subs = torch.arange(2 ** b)
        self.sel = ((subs[:, None] >> torch.arange(b)[None, :]) & 1)              # [2^b, b]

    def step(self, x):
        B = x.shape[0]
        z = (x @ self.Wqk).view(B, H, DK)
        z = z / (z.abs().mean(-1, keepdim=True) + 1e-6)
        u = (z.gather(-1, self.a.expand(B, -1, -1)) - z.gather(-1, self.b.expand(B, -1, -1))).view(B, H, self.tph, self.nap)
        c = ((u > 0).long() * self.pow2).sum(-1)
        mb, jb = (-u.abs()).topk(self.nb, -1)
        flip = (self.sel[None, None, None] * self.pow2[jb][..., None, :]).sum(-1)      # [B,H,tph,2^b]
        cells = c[..., None] ^ flip
        w = torch.exp((self.sel.float()[None, None, None] * mb[..., None, :]).sum(-1) / self.tau)
        w = (w / w.sum(-1, keepdim=True)).view(B, -1)
        rows = (cells + self.base_h[..., None] + self.base_t[..., None]).view(B, -1)
        k = 2 ** self.nb
        v = (x @ self.Wv).view(B, H, 1, DH).expand(B, H, self.tph * k, DH).reshape(B, -1, DH)
        self.S.scatter_add_(1, rows[..., None].expand(-1, -1, DH), w[..., None] * v)
        self.N.scatter_add_(1, rows, w)
        got = (self.S.gather(1, rows[..., None].expand(-1, -1, DH)) * w[..., None]).view(B, H, -1, DH)
        cnt = (self.N.gather(1, rows) * w).view(B, H, -1)
        return got.sum(2) / cnt.sum(2, keepdim=True)


def bench(make, B, reps):
    m = make()
    x = torch.randn(B, D)
    for _ in range(5):  # warm-up
        m.step(x)
    t0 = time.perf_counter()
    for _ in range(reps):
        m.step(x)
    return (time.perf_counter() - t0) / reps, m.state_bytes()


def main():
    torch.manual_seed(0)
    torch.set_grad_enabled(False)
    out = os.path.join(HERE, "runs", "bench_decode")
    os.makedirs(out, exist_ok=True)
    rows = []
    lengths = [256, 1024, 4096, 16384]
    for B in (1, 64):
        reps = 400 if B == 1 else 100
        for t in lengths:
            cfgs = {
                "softmax (KV cache)": lambda: Softmax(B, t, t + reps + 10),
                "linear F=256": lambda: Linear(B, 256),
                "linear F=32": lambda: Linear(B, 32),
                "drawers 16x4 (F=256)": lambda: Drawers(B, 16, 4),
                "drawers 4x6 (F=256)": lambda: Drawers(B, 4, 6),
            }
            for name, make in cfgs.items():
                sec, nbytes = bench(make, B, reps)
                rows.append(dict(batch=B, context=t, mixer=name, us_per_step=round(sec * 1e6, 1),
                                 us_per_token=round(sec * 1e6 / B, 2), state_kb_per_seq=round(nbytes / B / 1024, 1)))
                print(rows[-1], flush=True)
    with open(os.path.join(out, "results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    lines = [f"# Decode benchmark — {time.strftime('%Y-%m-%d %H:%M')}",
             f"CPU: {platform.processor() or platform.machine()}, torch {torch.__version__}, threads {torch.get_num_threads()}",
             "", "| batch | context | mixer | µs per step | µs per token | state KB / sequence |", "|---|---|---|---|---|---|"]
    lines += [f"| {r['batch']} | {r['context']} | {r['mixer']} | {r['us_per_step']} | {r['us_per_token']} | {r['state_kb_per_seq']} |"
              for r in rows]
    open(os.path.join(out, "results.md"), "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
