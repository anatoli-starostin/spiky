"""t11 — steps 3+4: build the spiking analogue of ONE REAL table of
exp025 layers[3].out_proj and validate it on the REAL input distribution.

Uses the capture from t10 (8192 real validation tokens, the actual attention-output
vectors that feed out_proj).  Construction = the validated leak-free IF recipe:
  comparator (1 neuron / 2 synapses) per anchor pair and its complement,
  row = threshold-NAP coincidence neuron,
  table value carried on the row->output synaptic delay.
"""
import torch, json
from spiky.spnet.spnet import NeuronMeta
from snn_harness import Net

DEV = "cuda:0"
THETA = 120.0
DRIVE = 240.0
IF = NeuronMeta(neuron_type=0, cf_2=0.0, cf_1=0.0, cf_0=0.0,
                a=0.0, b=0.0, c=0.0, d=0.0, spike_threshold=THETA)
from paths import out
CAP = out("real_capture_layer3.pt")


class Builder(Net):
    def __init__(self, n, **kw):
        super().__init__(n, neuron_meta=IF, device=DEV, **kw)
        self.used = 0

    def alloc(self, k=1):
        i = self.used; self.used += k
        return list(range(i, i + k))


def row_index(X, a, b, powers):
    """X: [N, in_dim]; a,b: [NAP]; returns [N] row index, MSB-first (FastMHL convention)."""
    bits = (X[:, a] - X[:, b] > 0).long()
    return (bits * powers.view(1, -1)).sum(-1)


def encode_global(X, R):
    """Global affine latency encoding of the whole 384-dim input vector into [0, R] ticks."""
    lo = torch.quantile(X.flatten().float(), 0.001)
    hi = torch.quantile(X.flatten().float(), 0.999)
    t = ((X - lo) / (hi - lo)).clamp(0, 1) * R
    return t.round().long()


def encode_rank(X):
    """Per-sample rank coding of the whole vector (order-exact by construction)."""
    return torch.argsort(torch.argsort(X, dim=-1), dim=-1)


def main(table_idx=0, n_eval=256, span=120, R_list=(15, 31, 63, 127, 255)):
    cap = torch.load(CAP, map_location="cpu", weights_only=False)
    X, W, A, B, P = cap["X"], cap["weights"], cap["anchor_a"], cap["anchor_b"], cap["powers"]
    n_tables, K, D = W.shape
    NAP = A.shape[1]
    print(f"real capture: X={tuple(X.shape)}  out_proj: {n_tables} tables x K={K} x D={D}, NAP={NAP}")
    print(f"(reconstructed model val bpb {cap['bpb']:.4f}; layer {cap['layer']})")

    # ---------- row coverage on real data ----------
    rows_all = torch.stack([row_index(X, A[t], B[t], P) for t in range(n_tables)])  # [T, N]
    cov = torch.stack([torch.bincount(rows_all[t], minlength=K).gt(0).sum() for t in range(n_tables)])
    print(f"\n[row coverage over {X.shape[0]} real tokens]  K={K} rows per table")
    print(f"  rows actually used: mean {cov.float().mean():.1f}/{K} "
          f"(min {cov.min()}, max {cov.max()}, median {cov.median()})")
    r0 = rows_all[table_idx]
    cnt = torch.bincount(r0, minlength=K)
    used = int(cnt.gt(0).sum())
    top = torch.sort(cnt, descending=True).values
    print(f"  table #{table_idx}: {used}/{K} rows used; "
          f"top-1 row covers {top[0].item()/len(r0)*100:.1f}% of tokens, "
          f"top-8 {top[:8].sum().item()/len(r0)*100:.1f}%, "
          f"top-{used//2 or 1} {top[:max(used//2,1)].sum().item()/len(r0)*100:.1f}%")

    # ---------- how much timing resolution does the REAL input need? ----------
    a, b = A[table_idx], B[table_idx]
    exact_rows = r0
    print(f"\n[latency encoding fidelity on real inputs, table #{table_idx}]")
    enc = {}
    for R in R_list:
        Xt = encode_global(X, R).float()
        rr = row_index(Xt, a, b, P)
        agree = (rr == exact_rows).float().mean().item()
        enc[R] = agree
        print(f"  global affine, {R+1:>4} ticks: row reproduced on {agree*100:6.2f}% of real tokens")
    Xr = encode_rank(X).float()
    rr = row_index(Xr, a, b, P)
    print(f"  per-sample rank code (384 ticks): row reproduced on "
          f"{(rr == exact_rows).float().mean().item()*100:6.2f}% of real tokens")

    # ---------- build the spiking circuit for this real table ----------
    R = 127
    Xt = encode_global(X, R)
    lat_lo, lat_hi = W[table_idx].min().item(), W[table_idx].max().item()
    lat = torch.round((W[table_idx] - lat_lo) / (lat_hi - lat_lo) * span).long()   # [K, D]

    anch = sorted(set(a.tolist() + b.tolist()))
    idx_of = {c: i for i, c in enumerate(anch)}
    g = Builder(len(anch) + 2 * NAP + K + D)
    inp = g.alloc(len(anch))
    lines = []
    for i in range(NAP):
        ia, ib = idx_of[int(a[i])], idx_of[int(b[i])]
        c1, c0 = g.alloc(1)[0], g.alloc(1)[0]
        # bit_i = 1[x_a > x_b] with STRICT '>' : a tie must give bit = 0.
        # c1 ("b strictly earlier"): symmetric delays -> a tie vetoes it.
        # c0 ("a earlier OR equal"): the veto arrives one tick late, so a tie fires it.
        g.connect(ib, c1, THETA, 1); g.connect(ia, c1, -2 * THETA, 1)
        g.connect(ia, c0, THETA, 1); g.connect(ib, c0, -2 * THETA, 2)
        lines += [c1, c0]
    rows = g.alloc(K)
    outs = g.alloc(D)
    for k in range(K):
        for i in range(NAP):
            bit = (k >> (NAP - 1 - i)) & 1
            # 1.001 margin: THETA/NAP is not exact in fp32 for NAP=7 and 7 * (THETA/7)
            # lands just under the threshold, so the row would never fire.
            g.connect(lines[2 * i + (0 if bit else 1)], rows[k], THETA / NAP * 1.001, 1)
    for k in range(K):
        for d in range(D):
            g.connect(rows[k], outs[d], THETA, int(lat[k, d]))
    g.build()
    n_neurons, n_syn = g.used, g.n_synapses()
    print(f"\n[spiking circuit for real table #{table_idx}]  neurons={n_neurons} "
          f"(inputs {len(anch)}, comparators {2*NAP}, rows {K}, outputs {D})  synapses={n_syn}")

    # ---------- run the REAL inputs through it ----------
    sub = torch.randperm(X.shape[0])[:n_eval]
    results = {}
    for enc_name, Xenc, win in [("global affine 128 ticks", Xt, R),
                                ("per-sample rank code", encode_rank(X), X.shape[1] - 1)]:
        st = torch.full((n_eval, n_neurons), -1.0)
        st[:, :len(anch)] = Xenc[sub][:, anch].float()
        n_ticks = int(win + span + 12)
        first, raster, _ = g.run(st, n_ticks=n_ticks, amp=DRIVE)
        first = first.cpu()
        t_rows = first[:, rows[0]:rows[0] + K]
        fired = (t_rows >= 0)
        one_row = (fired.sum(1) == 1)
        sim_row = fired.float().argmax(1)
        tgt_row = exact_rows[sub]                       # the row the REAL model selects
        row_ok = (sim_row == tgt_row) & one_row
        t_row_sel = torch.gather(t_rows, 1, sim_row.unsqueeze(1)).squeeze(1)
        got = first[:, outs[0]:outs[0] + D] - t_row_sel.unsqueeze(1) - 1
        want = lat[tgt_row]
        ex = (got == want).float().mean().item()
        gi = torch.randint(0, D, (40000,)); gj = torch.randint(0, D, (40000,))
        mm = gi != gj; gi, gj = gi[mm], gj[mm]
        sg = torch.sign((got[:, gi] - got[:, gj]).float())
        sw = torch.sign((want[:, gi] - want[:, gj]).float())
        valid = sw != 0
        pair = ((sg == sw) & valid).sum().item() / max(int(valid.sum()), 1)
        spikes = (raster > 0).sum().item() / n_eval
        print(f"\n  --- encoding: {enc_name} ---")
        print(f"  exactly one row neuron fired:               {int(one_row.sum())}/{n_eval}")
        print(f"  spiking row == the row the model selects:   {row_ok.float().mean().item()*100:6.2f}%")
        print(f"  output latency exact match vs the table:    {ex*100:6.2f}%")
        print(f"  output pair-order agreement:                {pair*100:6.2f}%")
        print(f"  spikes per inference: {spikes:.1f}")
        results[enc_name] = dict(row=row_ok.float().mean().item(), exact=ex,
                                 pair=pair, spikes=spikes)
    exact = results["per-sample rank code"]["exact"]
    pair = results["per-sample rank code"]["pair"]
    spikes = results["per-sample rank code"]["spikes"]

    # ---------- demand-driven variant: only wire the rows that occur ----------
    used_rows = torch.bincount(rows_all[table_idx], minlength=K).gt(0)
    n_used = int(used_rows.sum())
    syn_full = K * D + K * NAP + 4 * NAP
    syn_dd = n_used * D + n_used * NAP + 4 * NAP
    print(f"\n[demand-driven] rows seen on real data: {n_used}/{K} "
          f"-> synapses {syn_dd:,} vs {syn_full:,} ({syn_full/max(syn_dd,1):.2f}x fewer)")
    # what fraction of a FRESH sample would hit an unseen row?
    half = X.shape[0] // 2
    seen = torch.bincount(rows_all[table_idx][:half], minlength=K).gt(0)
    hit = seen[rows_all[table_idx][half:]].float().mean().item()
    print(f"  built from the first half of the data, the second half hits a built row "
          f"{hit*100:.2f}% of the time (the rest would produce NO output spike)")

    json.dump(dict(table=table_idx, K=K, D=D, NAP=NAP, neurons=n_neurons, synapses=n_syn,
                   spikes=spikes, exact=exact, pair=pair, rows_used=n_used,
                   enc_fidelity={str(k): v for k, v in enc.items()},
                   coverage_mean=float(cov.float().mean())),
              open("t11_results.json", "w"), indent=1)


if __name__ == "__main__":
    main()
