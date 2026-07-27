"""t08 — does a cheap per-table approximation survive the head?

A FastMHL/HyperplaneMHL head output is the SUM of tables_per_head=256 table rows.
Per-table fidelity numbers (t06/t07) are only interesting if they survive that sum,
because what the NEXT stage reads is the ordering of the summed output.

Here: approximate ALL 256 tables of one real trained layer with each candidate family,
map back to the value domain, and measure the head output's relative error and
pairwise-order agreement against the exact head.
"""
import torch
from table_io import CKPT, PREFIX

DEV = "cuda:0"
SPAN = 120
torch.manual_seed(0)


def quantise(W, span=SPAN):
    lo = W.amin(dim=(1, 2), keepdim=True)
    hi = W.amax(dim=(1, 2), keepdim=True)
    lat = torch.round((W - lo) / (hi - lo) * span)
    return lat, lo, hi


def dequantise(lat, lo, hi, span=SPAN):
    return lat / span * (hi - lo) + lo


def act_bits(K, NAP, device):
    bits = torch.tensor([[(k >> (NAP - 1 - i)) & 1 for i in range(NAP)] for k in range(K)],
                        device=device, dtype=torch.float32)
    act = torch.zeros(K, 2 * NAP, device=device)
    for i in range(NAP):
        act[:, 2 * i] = bits[:, i]
        act[:, 2 * i + 1] = 1 - bits[:, i]
    return act


def fit_bitlines_batch(lat, act, iters=1500, chunk=32, tau=0.5, lr=0.4):
    """lat: [Tn, K, D] targets. Returns predictions of the same shape."""
    Tn, K, D = lat.shape
    L = act.shape[1]
    out = torch.empty_like(lat)
    for s in range(0, Tn, chunk):
        t = lat[s:s + chunk].float()
        n = t.shape[0]
        dl = (torch.rand(n, D, L, device=DEV) * SPAN).requires_grad_()
        opt = torch.optim.Adam([dl], lr=lr)
        mask = (1 - act).view(1, K, 1, L) * 1e4
        for it in range(iters):
            s_ = dl.unsqueeze(1) + mask                      # [n, K, D, L]
            soft = -tau * torch.logsumexp(-s_ / tau, dim=-1)
            loss = (soft - t).abs().mean()
            opt.zero_grad(); loss.backward(); opt.step()
            if it == iters * 2 // 3:
                for g in opt.param_groups:
                    g["lr"] = lr * 0.2
        with torch.no_grad():
            dli = dl.round().clamp(0, 255)
            out[s:s + chunk] = (dli.unsqueeze(1) + mask).min(dim=-1).values
    return out


def sparse_override(lat, q):
    t = lat.float()
    default = torch.quantile(t, q, dim=1, keepdim=True)       # [Tn, 1, D]
    use = t <= default
    pred = torch.where(use, t, default.expand_as(t))
    return pred, int(use.sum().item()) + lat.shape[0] * lat.shape[2]


def head_metrics(Wex, Wap, n_draws=256, n_pairs=40000):
    """Sample a random row per table, sum -> head output; compare exact vs approx."""
    Tn, K, D = Wex.shape
    g = torch.Generator(device=DEV).manual_seed(0)
    idx = torch.randint(0, K, (n_draws, Tn), device=DEV, generator=g)
    ar = torch.arange(Tn, device=DEV)
    ex = torch.stack([Wex[ar, idx[i], :].sum(0) for i in range(n_draws)])   # [n_draws, D]
    ap = torch.stack([Wap[ar, idx[i], :].sum(0) for i in range(n_draws)])
    rel = (ap - ex).std().item() / ex.std().item()
    gi = torch.randint(0, D, (n_pairs,), device=DEV, generator=g)
    gj = torch.randint(0, D, (n_pairs,), device=DEV, generator=g)
    m = gi != gj
    gi, gj = gi[m], gj[m]
    se = torch.sign(ex[:, gi] - ex[:, gj])
    sa = torch.sign(ap[:, gi] - ap[:, gj])
    agree = (se == sa).float().mean().item()
    return rel, agree


if __name__ == "__main__":
    sd = torch.load(CKPT, map_location="cpu", weights_only=False)
    W = sd[f"{PREFIX}.weights"].float().to(DEV)              # [256, 64, 384]
    Tn, K, D = W.shape
    NAP = 6
    print(f"one real trained head: {Tn} tables x {K} rows x {D} outputs "
          f"(the head output is the SUM of the {Tn} selected rows)")
    lat, lo, hi = quantise(W)
    act = act_bits(K, NAP, DEV)

    # reference 1: exact quantisation to the 8-bit latency grid (what a spiking net stores)
    Wq = dequantise(lat, lo, hi)
    rel, agree = head_metrics(W, Wq)
    print(f"  latency quantisation only (span {SPAN}):        "
          f"head rel-err={rel*100:5.2f}%  head pair-order={agree*100:5.1f}%   syn={Tn*K*D}")

    for q in [0.75, 0.5, 0.25]:
        p, syn = sparse_override(lat, q)
        rel, agree = head_metrics(W, dequantise(p, lo, hi))
        print(f"  sparse override q={q:<5}                        "
              f"head rel-err={rel*100:5.2f}%  head pair-order={agree*100:5.1f}%   syn={syn}")

    pred = fit_bitlines_batch(lat, act)
    rel, agree = head_metrics(W, dequantise(pred, lo, hi))
    print(f"  fitted bit lines (2*NAP={2*NAP} per output)        "
          f"head rel-err={rel*100:5.2f}%  head pair-order={agree*100:5.1f}%   syn={Tn*2*NAP*D}")

    print("\n  --- how much latency resolution does one stage actually need? ---")
    for span in [255, 120, 60, 30, 15, 7, 3, 1]:
        l2, lo2, hi2 = quantise(W, span)
        rel, agree = head_metrics(W, dequantise(l2, lo2, hi2, span))
        bits = torch.log2(torch.tensor(float(span + 1))).item()
        print(f"  span={span:>4} ticks ({bits:4.1f} bit/entry)                 "
              f"head rel-err={rel*100:5.2f}%  head pair-order={agree*100:5.1f}%   "
              f"table bits={Tn*K*D*bits/8/1024:7.0f} KiB")

    # control: replace every table by its column mean (row ignored)
    cm = lat.float().mean(dim=1, keepdim=True).expand_as(lat)
    rel, agree = head_metrics(W, dequantise(cm, lo, hi))
    print(f"  column constant (row ignored)                  "
          f"head rel-err={rel*100:5.2f}%  head pair-order={agree*100:5.1f}%   syn={Tn*D}")
