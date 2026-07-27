"""t06 — cost/fidelity frontier for reproducing ONE real trained table.

Adds to t05:
  * a fixed pair-order metric (ties reported separately, they are a real failure mode),
  * factored addressing (g groups of NAP/g bits, combined by min or max),
  * sparse override (row synapses only for the entries below a per-output default),
  * a RANDOM table control of the same shape -> is the trained table compressible at all?
  * a cost/fidelity figure.
"""
import torch, json
from table_io import load_table, bits_of

DEV = "cuda:0"
torch.manual_seed(0)
BIG = 1e4


def metrics(pred, tgt, n_pairs=40000):
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
    nv = max(valid.sum().item(), 1)
    agree = ((sp == st) & valid).sum().item() / nv
    tie = ((sp == 0) & valid).sum().item() / nv
    return dict(exact=exact, mad=mad, pair=agree, tie=tie)


def show(name, pred, tgt, syn, neurons=None):
    m = metrics(pred, tgt)
    print(f"  {name:<38} syn={syn:>7}  exact={m['exact']*100:5.1f}%  MAD={m['mad']:6.2f}  "
          f"pair-order={m['pair']*100:5.1f}%  (pred-ties {m['tie']*100:4.1f}%)")
    return dict(name=name, syn=syn, **m)


def softmin_fit(build, params, tgt, iters=3000, lr=0.3, tau=0.5):
    opt = torch.optim.Adam(params, lr=lr)
    t = tgt.float()
    for it in range(iters):
        s = build()                                  # [K, D, L] candidate arrival times
        soft = -tau * torch.logsumexp(-s / tau, dim=-1)
        loss = (soft - t).abs().mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if it == iters * 2 // 3:
            for g in opt.param_groups:
                g["lr"] = lr * 0.2
    with torch.no_grad():
        return build().min(dim=-1).values


def fam_minplus_bus(tgt, r):
    K, D = tgt.shape
    span = tgt.max().item()
    rr = (torch.rand(K, r, device=DEV) * span * .5).requires_grad_()
    cc = (torch.rand(D, r, device=DEV) * span * .5).requires_grad_()
    build = lambda: rr.unsqueeze(1) + cc.unsqueeze(0)
    softmin_fit(build, [rr, cc], tgt)
    with torch.no_grad():
        return (rr.round().clamp(0, 255).unsqueeze(1) + cc.round().clamp(0, 255).unsqueeze(0)).min(-1).values


def fam_lines(tgt, act, iters=3000):
    """out[k,d] = min over ACTIVE lines of delay[d, line];  act: [K, L] in {0,1}."""
    K, L = act.shape
    D = tgt.shape[1]
    span = tgt.max().item()
    dl = (torch.rand(D, L, device=DEV) * span).requires_grad_()
    build = lambda: dl.unsqueeze(0) + (1 - act).unsqueeze(1) * BIG
    softmin_fit(build, [dl], tgt, iters=iters)
    with torch.no_grad():
        dli = dl.round().clamp(0, 255)
        return (dli.unsqueeze(0) + (1 - act).unsqueeze(1) * BIG).min(-1).values


def act_bits(bits):
    K, NAP = bits.shape
    act = torch.zeros(K, 2 * NAP, device=DEV)
    for i in range(NAP):
        act[:, 2 * i] = bits[:, i].float()
        act[:, 2 * i + 1] = 1 - bits[:, i].float()
    return act


def act_groups(bits, g):
    """Factored addressing: split NAP bits into g groups, one-hot per group."""
    K, NAP = bits.shape
    per = NAP // g
    cols = []
    for gi in range(g):
        sub = bits[:, gi * per:(gi + 1) * per]
        idx = torch.zeros(K, dtype=torch.long, device=DEV)
        for b in range(per):
            idx = idx * 2 + sub[:, b].long()
        oh = torch.zeros(K, 2 ** per, device=DEV)
        oh[torch.arange(K, device=DEV), idx] = 1.0
        cols.append(oh)
    return torch.cat(cols, dim=1)


def fam_sparse_override(tgt, q):
    """Per-output default (a constant line) + row synapses only where the target is
    below the default -> min semantics can realise exactly those entries."""
    K, D = tgt.shape
    t = tgt.float()
    default = torch.quantile(t, q, dim=0, keepdim=True)          # [1, D]
    use = t <= default
    pred = torch.where(use, t, default.expand(K, D))
    syn = int(use.sum().item()) + D
    return pred, syn


if __name__ == "__main__":
    out = []
    D = 384
    t = load_table(0, d_out=D, span=120)
    K, NAP = t["K"], t["NAP"]
    tgt = t["lat"].to(DEV)
    bits = torch.tensor([bits_of(k, NAP) for k in range(K)], device=DEV)
    print(f"===== real trained table (exp011 lut1 #0): K={K}, D={D}, NAP={NAP} =====")

    out.append(show("(0) column constant", tgt.float().mean(0, True).expand(K, D).round(), tgt, D))
    out.append(show("(a) brute force rows x outputs", tgt, tgt, K * D))
    for r in [2, 4, 8, 16, 32]:
        out.append(show(f"(c1) min-plus bus r={r}", fam_minplus_bus(tgt, r), tgt, r * (K + D)))
    out.append(show("(b) bit lines (2*NAP)", fam_lines(tgt, act_bits(bits)), tgt, 2 * NAP * D))
    for g in [3, 2, 1]:
        act = act_groups(bits, g)
        out.append(show(f"(c2) factored addressing g={g} ({act.shape[1]} lines)",
                        fam_lines(tgt, act), tgt, act.shape[1] * D))
    for q in [0.1, 0.25, 0.5, 0.75]:
        p, syn = fam_sparse_override(tgt, q)
        out.append(show(f"(d) sparse override q={q}", p, tgt, syn))

    print("\n===== control: a RANDOM table of the same shape (is the trained one special?) =====")
    rnd = torch.randint(0, 121, (K, D), device=DEV)
    show("(c1) min-plus bus r=8  [random]", fam_minplus_bus(rnd, 8), rnd, 8 * (K + D))
    show("(b) bit lines          [random]", fam_lines(rnd, act_bits(bits)), rnd, 2 * NAP * D)
    show("(c2) factored g=2      [random]", fam_lines(rnd, act_groups(bits, 2)), rnd, act_groups(bits, 2).shape[1] * D)

    json.dump(out, open("t06_results.json", "w"), indent=1)
    print("\nwrote t06_results.json")
