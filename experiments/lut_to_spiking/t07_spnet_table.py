"""t07 — EXECUTE both constructions on a real trained table inside SPNet.

(a) brute force  : 2*NAP comparators -> K row coincidence neurons -> D outputs,
                   table value carried by the row->output synaptic DELAY.
(b) fitted lines : 2*NAP comparators -> alignment stage -> D outputs directly,
                   delays fitted (min-plus over the active bit lines). No row neurons.

Both are run for all K=64 rows in ONE batched process_ticks call and checked against
the table.  Cost is reported as neurons / synapses / spikes per inference.
"""
import torch, json
from spiky.spnet.spnet import NeuronMeta
from snn_harness import Net
from table_io import load_table, bits_of

DEV = "cuda:0"
THETA = 120.0
DRIVE = 240.0
IF = NeuronMeta(neuron_type=0, cf_2=0.0, cf_1=0.0, cf_0=0.0,
                a=0.0, b=0.0, c=0.0, d=0.0, spike_threshold=THETA)


def make_inputs(K, NAP, early=0, late=6, jitter=True):
    """[K, 2*NAP] input tick per anchor neuron so that row k's bit pattern is realised.
    neuron 2i = a_i, neuron 2i+1 = b_i;  bit_i = 1[x_a > x_b] (b earlier)."""
    x = torch.zeros(K, 2 * NAP)
    for k in range(K):
        for i, bit in enumerate(bits_of(k, NAP)):
            off = i if jitter else 0
            ta, tb = (late + off, early + off) if bit else (early + off, late + off)
            x[k, 2 * i], x[k, 2 * i + 1] = ta, tb
    return x


def comparators(net, NAP, base=0, dly=1):
    """Add 2*NAP comparator neurons after the 2*NAP input neurons. Returns their idx."""
    lines = []
    for i in range(NAP):
        a, b = base + 2 * i, base + 2 * i + 1
        c1 = net.n_neurons_used; net.n_neurons_used += 1     # bit_i == 1  (b earlier)
        c0 = net.n_neurons_used; net.n_neurons_used += 1     # bit_i == 0  (a earlier)
        net.connect(b, c1, THETA, dly); net.connect(a, c1, -2 * THETA, dly)
        net.connect(a, c0, THETA, dly); net.connect(b, c0, -2 * THETA, dly)
        lines += [c1, c0]
    return lines


class Builder(Net):
    def __init__(self, n, **kw):
        super().__init__(n, neuron_meta=IF, device=DEV, **kw)
        self.n_neurons_used = 0

    def alloc(self, k=1):
        i = self.n_neurons_used
        self.n_neurons_used += k
        return list(range(i, i + k))


def run_bruteforce(D=384, table_idx=0, span=120):
    t = load_table(table_idx, d_out=D, span=span)
    K, NAP, lat = t["K"], t["NAP"], t["lat"]
    n_total = 2 * NAP + 2 * NAP + K + D
    g = Builder(n_total)
    inp = g.alloc(2 * NAP)
    lines = comparators(g, NAP, base=0)
    rows = g.alloc(K)
    outs = g.alloc(D)
    # row k = coincidence over its NAP active lines
    for k in range(K):
        for i, bit in enumerate(bits_of(k, NAP)):
            g.connect(lines[2 * i + (0 if bit else 1)], rows[k], THETA / NAP, 1)
    # value fan-out: delay carries the table entry
    for k in range(K):
        for d in range(D):
            g.connect(rows[k], outs[d], THETA, int(lat[k, d]))
    g.build()

    x = make_inputs(K, NAP)
    st = torch.full((K, n_total), -1.0)
    st[:, :2 * NAP] = x
    n_ticks = int(lat.max()) + 40
    first, raster, _ = g.run(st, n_ticks=n_ticks, amp=DRIVE)
    first = first.cpu()
    t_row = first[:, rows[0]:rows[0] + K]
    sel = torch.arange(K).unsqueeze(1)
    t_row_sel = torch.gather(t_row, 1, sel).squeeze(1)             # the row that fired
    got = first[:, outs[0]:outs[0] + D] - t_row_sel.unsqueeze(1) - 1
    ok_row = int(((t_row >= 0).sum(1) == 1).sum())
    match = (got == lat).float().mean().item()
    spikes = (raster > 0).sum().item() / K
    print(f"(a) brute force  K={K} D={D}: exactly-one-row-fired {ok_row}/{K};  "
          f"output latency match {match*100:.2f}%")
    print(f"    cost: neurons={n_total}  synapses={g.n_synapses()}  spikes/inference={spikes:.1f}")
    return dict(family="bruteforce", D=D, neurons=n_total, synapses=g.n_synapses(),
                spikes=spikes, match=match)


def run_fitted(D=384, table_idx=0, span=120, iters=3000):
    """Fitted bit-line min-plus, with a real alignment stage so the lines are
    canonically timed (which the numeric model assumes)."""
    from t06_costfidelity import fam_lines, act_bits, metrics
    t = load_table(table_idx, d_out=D, span=span)
    K, NAP, lat = t["K"], t["NAP"], t["lat"]
    bits = torch.tensor([bits_of(k, NAP) for k in range(K)], device=DEV)
    act = act_bits(bits)
    pred_num = fam_lines(lat.to(DEV).float(), act, iters=iters)
    # recover the fitted delays by refitting deterministically: fam_lines returns preds
    # only, so refit here and keep the delays.
    L = 2 * NAP
    span_t = float(lat.max())
    dl = (torch.rand(D, L, device=DEV) * span_t).requires_grad_()
    opt = torch.optim.Adam([dl], lr=0.3)
    tgt = lat.to(DEV).float()
    for it in range(iters):
        s = dl.unsqueeze(0) + (1 - act).unsqueeze(1) * 1e4
        soft = -0.5 * torch.logsumexp(-s / 0.5, dim=-1)
        loss = (soft - tgt).abs().mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if it == iters * 2 // 3:
            for gp in opt.param_groups:
                gp["lr"] = 0.06
    dli = dl.detach().round().clamp(0, 250).cpu()
    pred = (dli.unsqueeze(0).to(DEV) + (1 - act).unsqueeze(1) * 1e4).min(-1).values.cpu()

    n_total = 2 * NAP + 2 * NAP + 1 + 2 * NAP + D     # inputs, comparators, ref, aligned, outs
    g = Builder(n_total)
    inp = g.alloc(2 * NAP)
    lines = comparators(g, NAP, base=0)
    ref = g.alloc(1)[0]
    for j in inp:                                     # reference = last input arrival
        g.connect(j, ref, THETA / (2 * NAP), 1)
    al = g.alloc(2 * NAP)
    for i, l in enumerate(lines):                     # aligned line = line AND ref(+2)
        g.connect(l, al[i], THETA / 2, 1)
        g.connect(ref, al[i], THETA / 2, 3)
    outs = g.alloc(D)
    for d in range(D):
        for i in range(2 * NAP):
            g.connect(al[i], outs[d], THETA, int(dli[d, i]))
        # one-shot: a strong self-veto after the first spike (no leak -> permanent),
        # otherwise every later active line makes the output fire again.
        g.connect(outs[d], outs[d], -100 * THETA, 1)
    g.build()

    x = make_inputs(K, NAP)
    st = torch.full((K, n_total), -1.0)
    st[:, :2 * NAP] = x
    n_ticks = int(dli.max()) + 40
    first, raster, _ = g.run(st, n_ticks=n_ticks, amp=DRIVE)
    first = first.cpu()
    t0 = first[:, al[0]].clone()
    for i in range(2 * NAP):                          # all aligned lines share one t0
        m = first[:, al[i]] >= 0
        t0[m] = first[m, al[i]]
    got = first[:, outs[0]:outs[0] + D] - t0.unsqueeze(1) - 1
    agree_sim_vs_num = (got == pred.long()).float().mean().item()
    m = metrics(got.float().to(DEV), lat.to(DEV))
    print(f"(b) fitted lines K={K} D={D}: simulated == numeric model on {agree_sim_vs_num*100:.1f}% of entries")
    print(f"    fidelity vs table: exact={m['exact']*100:.1f}%  MAD={m['mad']:.2f}tk  "
          f"pair-order={m['pair']*100:.1f}%")
    print(f"    cost: neurons={n_total}  synapses={g.n_synapses()}  "
          f"spikes/inference={(raster>0).sum().item()/K:.1f}")
    return dict(family="fitted", D=D, neurons=n_total, synapses=g.n_synapses(),
                spikes=(raster > 0).sum().item() / K, **m,
                sim_vs_num=agree_sim_vs_num)


if __name__ == "__main__":
    res = []
    res.append(run_bruteforce(D=64))
    res.append(run_bruteforce(D=384))
    res.append(run_fitted(D=64))
    res.append(run_fitted(D=384))
    json.dump(res, open("t07_results.json", "w"), indent=1)
