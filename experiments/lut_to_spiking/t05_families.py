"""t05 — how cheaply can a REAL trained table be reproduced, numerically?

Everything a leak-free IF population can compute about output timing has the form

    t_out(d) = min over active source events s of ( t_s + delay(d, s) )         (min-plus)

(threshold-1 neurons; threshold-k generalises 'min' to the k-th order statistic).
So the design choice is: WHICH intermediate event basis do we build, and how dense is
the min-plus matrix from it to the outputs?  Three families, all measured on the same
real trained table:

  (a) basis = rows        : K one-hot row events   -> K x D delays   (exact)
  (c) basis = min-plus bus: r shared events        -> r x (K + D) delays
  (b) basis = bit lines   : 2*NAP comparator events-> 2*NAP x D delays
  (0) control             : ignore the row entirely (column constant)

Metrics: exact-tick match, mean |dt|, and PAIRWISE ORDER ACCURACY -- the fraction of
output pairs (d, d') whose ordering is preserved, which is what a downstream LUT stage
actually reads.
"""
import torch, math
from table_io import load_table, bits_of

DEV = "cuda:0"
torch.manual_seed(0)


def metrics(pred, tgt, n_pairs=20000):
    """pred, tgt: [K, D] integer-ish latencies."""
    pred = pred.float(); tgt = tgt.float()
    exact = (pred.round() == tgt.round()).float().mean().item()
    mad = (pred - tgt).abs().mean().item()
    K, D = tgt.shape
    g = torch.Generator(device=pred.device).manual_seed(0)
    i = torch.randint(0, D, (n_pairs,), device=pred.device, generator=g)
    j = torch.randint(0, D, (n_pairs,), device=pred.device, generator=g)
    m = i != j
    i, j = i[m], j[m]
    st = torch.sign(tgt[:, i] - tgt[:, j])
    sp = torch.sign(pred[:, i] - pred[:, j])
    valid = st != 0
    agree = ((sp == st) & valid).sum().item() / max(valid.sum().item(), 1)
    return dict(exact=exact, mad=mad, pair_order=agree)


def report(name, pred, tgt, neurons, synapses, spikes):
    m = metrics(pred, tgt)
    print(f"  {name:<34} neurons={neurons:>6}  synapses={synapses:>7}  spikes/inf={spikes:>5}  "
          f"exact={m['exact']*100:5.1f}%  MAD={m['mad']:6.2f}tk  pair-order={m['pair_order']*100:5.1f}%")
    return m


def fit_minplus_bus(tgt, r, iters=4000, lr=0.15, tau=0.5):
    """out[k,d] = min_j ( rr[k,j] + cc[d,j] ), fitted with a soft-min surrogate."""
    K, D = tgt.shape
    span = tgt.max().item()
    rr = (torch.rand(K, r, device=DEV) * span * 0.5).requires_grad_()
    cc = (torch.rand(D, r, device=DEV) * span * 0.5).requires_grad_()
    opt = torch.optim.Adam([rr, cc], lr=lr)
    t = tgt.float()
    for it in range(iters):
        s = rr.unsqueeze(1) + cc.unsqueeze(0)              # [K, D, r]
        soft = -tau * torch.logsumexp(-s / tau, dim=-1)    # smooth min
        loss = (soft - t).abs().mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if it == iters // 2:
            for g in opt.param_groups:
                g["lr"] = lr * 0.2
    with torch.no_grad():
        rr_i = rr.round().clamp(0, 255)
        cc_i = cc.round().clamp(0, 255)
        pred = (rr_i.unsqueeze(1) + cc_i.unsqueeze(0)).min(dim=-1).values
    return pred


def fit_bitlines(tgt, bits, iters=4000, lr=0.4, tau=0.5, extra_lines=0):
    """out[k,d] = min over the NAP ACTIVE bit lines of delay[d, line].

    bits: [K, NAP] in {0,1}. Line 2i is 'bit i == 1', line 2i+1 is 'bit i == 0';
    exactly one of each pair is active per row, so exactly NAP lines are active.
    `extra_lines` adds unconditional (always-active) lines = a per-output floor.
    """
    K, NAP = bits.shape
    D = tgt.shape[1]
    L = 2 * NAP + extra_lines
    act = torch.zeros(K, L, device=DEV)
    for i in range(NAP):
        act[:, 2 * i] = bits[:, i].float()
        act[:, 2 * i + 1] = 1.0 - bits[:, i].float()
    if extra_lines:
        act[:, 2 * NAP:] = 1.0
    span = tgt.max().item()
    dl = (torch.rand(D, L, device=DEV) * span).requires_grad_()
    opt = torch.optim.Adam([dl], lr=lr)
    t = tgt.float()
    BIG = 1e4
    for it in range(iters):
        s = dl.unsqueeze(0) + (1 - act).unsqueeze(1) * BIG   # [K, D, L], inactive -> +inf
        soft = -tau * torch.logsumexp(-s / tau, dim=-1)
        loss = (soft - t).abs().mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if it == iters // 2:
            for g in opt.param_groups:
                g["lr"] = lr * 0.2
    with torch.no_grad():
        dli = dl.round().clamp(0, 255)
        s = dli.unsqueeze(0) + (1 - act).unsqueeze(1) * BIG
        pred = s.min(dim=-1).values
    return pred, dli, act


if __name__ == "__main__":
    for D in [64, 384]:
        t = load_table(0, d_out=D, span=120)
        K, NAP = t["K"], t["NAP"]
        tgt = t["lat"].to(DEV)
        bits = torch.tensor([bits_of(k, NAP) for k in range(K)], device=DEV)
        print(f"\n===== real trained table: K={K} rows, D_out={D}, NAP={NAP}, "
              f"latency span 0..{t['span']} ticks =====")

        # (0) control: ignore the row
        pred0 = tgt.float().mean(0, keepdim=True).expand(K, D).round()
        report("(0) column constant (row ignored)", pred0, tgt, D, 0, D)

        # (a) exact brute force: one row neuron per row, one delay synapse per (row, out)
        report("(a) brute force: rows x outputs", tgt, tgt,
               2 * NAP + K + D, 4 * NAP + K * NAP + K * D, NAP + 1 + D)

        # (c) min-plus bus of rank r
        for r in [1, 2, 4, 8, 16]:
            pred = fit_minplus_bus(tgt, r)
            report(f"(c) min-plus bus, r={r:<2}", pred, tgt,
                   2 * NAP + K + r + D, 4 * NAP + K * NAP + r * (K + D), NAP + 1 + r + D)

        # (b) direct fit from the bit lines (no row neurons at all)
        for extra in [0, 2]:
            pred, dl, act = fit_bitlines(tgt, bits, extra_lines=extra)
            report(f"(b) bit-line min-plus (+{extra} const)", pred, tgt,
                   2 * NAP + D, 4 * NAP + (2 * NAP + extra) * D, NAP + D)
