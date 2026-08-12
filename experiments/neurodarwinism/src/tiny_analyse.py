"""exp012: dissect the best network evolution found. Read-only -- no training, no mutation.

Takes the asexual pool-512 run's final EWMA leader out of its checkpoint and reports what it
actually is: topology by class, which neurons are on a live input->output path, the recurrent
structure and whether it is being used to accumulate delay into the readout window, the weight
and delay distributions, what each output neuron reads, and four ablations that say which
parts of the solution are load-bearing.

    python tiny_analyse.py --ckpt <dir>/ck_s0.npz --out <dir>/analysis_leader.json
"""
import argparse
import json
from collections import deque

import numpy as np

import tiny_snn as T
from data import load
from harness import LatencyEncoder
from tiny_evolve import load_ckpt

# node indexing for the graph views: inputs 0..16, exc 17..24, inh 25..26, out 27..32
N_IN, N_EXC, N_INH, N_OUT = T.N_IN, T.N_EXC, T.N_INH, T.N_OUT
OFF_EXC, OFF_INH, OFF_OUT = N_IN, N_IN + N_EXC, N_IN + N_EXC + N_INH
N_NODE = OFF_OUT + N_OUT


def node_name(n):
    if n < N_IN:
        return f"in{n:02d}"
    if n < OFF_INH:
        return f"E{n - OFF_EXC}"
    if n < OFF_OUT:
        return f"I{n - OFF_INH}"
    return f"OUT{n - OFF_OUT}"


def edges(g):
    """-> list of (src_node, tgt_node, weight, delay) in the flat node indexing."""
    out = []
    r, c = np.nonzero(g["mask"])
    for i, j in zip(r, c):
        s = i if i < N_IN else (OFF_EXC + (i - N_IN) if i < N_IN + N_EXC
                                else OFF_INH + (i - N_IN - N_EXC))
        t = (OFF_EXC + j if j < N_EXC else
             OFF_INH + (j - N_EXC) if j < N_EXC + N_INH else
             OFF_OUT + (j - N_EXC - N_INH))
        out.append((int(s), int(t), float(g["weight"][i, j]), int(g["delay"][i, j])))
    return out


def edge_class(s, t):
    sc = "in" if s < N_IN else ("exc" if s < OFF_INH else "inh")
    tc = "exc" if t < OFF_INH else ("inh" if t < OFF_OUT else "out")
    return f"{sc}->{tc}"


def reachability(E):
    """Which nodes are downstream of an input, and which can still reach an output."""
    fwd = {n: [] for n in range(N_NODE)}
    bwd = {n: [] for n in range(N_NODE)}
    for s, t, _w, _d in E:
        fwd[s].append(t)
        bwd[t].append(s)

    def bfs(seed, adj):
        seen, q = set(seed), deque(seed)
        while q:
            n = q.popleft()
            for m in adj[n]:
                if m not in seen:
                    seen.add(m)
                    q.append(m)
        return seen

    from_in = bfs(list(range(N_IN)), fwd)
    to_out = bfs(list(range(OFF_OUT, N_NODE)), bwd)
    return fwd, bwd, from_in, to_out


def simple_cycles(E):
    """All simple cycles among the 10 hidden nodes. The graph is tiny; brute DFS is fine."""
    hid = list(range(OFF_EXC, OFF_OUT))
    adj = {n: [] for n in hid}
    dly = {}
    for s, t, _w, d in E:
        if s in adj and t in adj:
            adj[s].append(t)
            dly[(s, t)] = d
    found = []
    for start in hid:
        stack = [(start, [start], 0)]
        while stack:
            n, path, acc = stack.pop()
            for m in adj[n]:
                if m == start:
                    found.append((list(path), acc + dly[(n, m)]))
                elif m not in path and len(path) < len(hid):
                    stack.append((m, path + [m], acc + dly[(n, m)]))
    # dedupe rotations
    seen, uniq = set(), []
    for p, d in found:
        k = tuple(sorted(p)) + (len(p), d)
        if k not in seen:
            seen.add(k)
            uniq.append(dict(cycle=[node_name(x) for x in p], length=len(p), delay_sum=d))
    return sorted(uniq, key=lambda x: (x["length"], x["delay_sum"]))


def paths_to_outputs(E, max_hops=4, cap=200000):
    """Enumerate input->output paths up to max_hops, recording the summed delay.

    A path's delay sum is what decides WHEN an output can fire: the spike lands at
    `input_tick + sum(delays)`, and the input tick is in [0, 31]. So a path is USABLE if
    `sum(delays) + t` falls in [64, 96) for some t in that range, i.e. sum in [33, 96).
    (Whether the path actually carries a spike also depends on integration; this is the
    structural constraint, not a simulation.)
    """
    fwd = {n: [] for n in range(N_NODE)}
    for s, t, w, d in E:
        fwd[s].append((t, d, w))
    per_out = {o: [] for o in range(N_OUT)}
    n = 0
    for src in range(N_IN):
        stack = [(src, 0, 0, (src,))]
        while stack and n < cap:
            node, acc, hops, path = stack.pop()
            for t, d, w in fwd[node]:
                if t >= OFF_OUT:
                    per_out[t - OFF_OUT].append((acc + d, path + (t,)))
                    n += 1
                elif hops + 1 < max_hops and t not in path:
                    stack.append((t, acc + d, hops + 1, path + (t,)))
    return per_out


def summarise_paths(per_out):
    rows = []
    for o, lst in per_out.items():
        if not lst:
            rows.append(dict(out=o, n_paths=0))
            continue
        s = np.array([x[0] for x in lst])
        # usable: some input tick in [0,31] puts input_tick + sum inside [64,96)
        usable = ((s + 31 >= 64) & (s <= 95)).mean()
        hops = np.array([len(x[1]) - 1 for x in lst])
        best = min(lst, key=lambda x: abs(x[0] - 79.5))
        rows.append(dict(out=o, n_paths=len(lst), delay_sum_min=int(s.min()),
                         delay_sum_med=float(np.median(s)), delay_sum_max=int(s.max()),
                         frac_usable=float(usable),
                         hops_min=int(hops.min()), hops_med=float(np.median(hops)),
                         frac_multihop=float((hops > 2).mean()),
                         example_path=" -> ".join(node_name(x) for x in best[1]),
                         example_delay_sum=int(best[0])))
    return rows


def ablate(g, kind, frac=0.5):
    h = {k: v.copy() for k, v in g.items()}
    if kind == "no_inhibition":
        h["weight"][T.ROW_INH] = 0.0
        h["mask"][T.ROW_INH] = False
    elif kind == "no_recurrence":                 # keep only input->hidden and hidden->output
        h["mask"][N_IN:, T.COL_EXC] = False
        h["mask"][N_IN:, T.COL_INH] = False
    elif kind.startswith("ko_inh"):
        k = int(kind[-1])
        row = N_IN + N_EXC + k
        h["mask"][row, :] = False
        h["mask"][:, N_EXC + k] = False
    elif kind == "prune50":
        w = np.abs(h["weight"])[h["mask"]]
        if w.size:
            thr = np.median(w)
            h["mask"] &= ~(np.abs(h["weight"]) <= thr)
    return h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    pool, ewma, age, rnd, hist, best, _ = load_ckpt(a.ckpt)
    finite = np.where(np.isfinite(ewma))[0]
    lead = int(finite[np.argmin(ewma[finite])])
    g = pool[lead]

    _, _, Xp, Yp, Xv, Yv = load(a.batch, seed=a.seed)
    T.fit_target_stats(Yp)
    enc = LatencyEncoder(Xp)
    tgt = T.target_offsets(Yv)

    R = {}
    E = edges(g)
    R["source"] = dict(ckpt=a.ckpt, round=rnd, leader_index=lead, pool=len(pool))

    # ---------------------------------------------------------------- TOPOLOGY
    cls = {}
    for s, t, w, d in E:
        cls[edge_class(s, t)] = cls.get(edge_class(s, t), 0) + 1
    R["topology"] = dict(n_synapses=len(E), n_legal_cells=int(T.LEGAL.sum()),
                         frac_of_legal_used=len(E) / float(T.LEGAL.sum()),
                         by_class=cls)

    fwd, bwd, from_in, to_out = reachability(E)
    live = sorted(from_in & to_out)
    R["topology"]["fan"] = {node_name(n): dict(out=len(fwd[n]), inp=len(bwd[n]))
                            for n in range(N_NODE)}
    R["topology"]["live_nodes"] = [node_name(n) for n in live]
    R["topology"]["dead_nodes"] = [node_name(n) for n in range(N_NODE) if n not in set(live)]
    R["topology"]["live_hidden"] = [node_name(n) for n in live if N_IN <= n < OFF_OUT]
    R["topology"]["inh_used"] = {node_name(OFF_INH + k):
                                 dict(out=len(fwd[OFF_INH + k]), inp=len(bwd[OFF_INH + k]),
                                      live=(OFF_INH + k) in set(live)) for k in range(N_INH)}

    # ---------------------------------------------------------------- RECURRENCE
    rec = [(node_name(s), node_name(t), round(w, 2), d) for s, t, w, d in E
           if OFF_EXC <= s < OFF_OUT and OFF_EXC <= t < OFF_OUT]
    R["recurrence"] = dict(n_hidden_to_hidden=len(rec), edges=rec,
                           self_loops=[e for e in rec if e[0] == e[1]],
                           cycles=simple_cycles(E))
    per_out = paths_to_outputs(E)
    R["paths"] = summarise_paths(per_out)

    # ---------------------------------------------------------------- WEIGHTS
    w_all = g["weight"][g["mask"]]
    w_exc = g["weight"][:N_IN + N_EXC][g["mask"][:N_IN + N_EXC]]
    w_inh = g["weight"][N_IN + N_EXC:][g["mask"][N_IN + N_EXC:]]
    R["weights"] = dict(
        n=len(w_all), n_excitatory=len(w_exc), n_inhibitory=len(w_inh),
        exc=dict(mean=float(np.mean(w_exc)), median=float(np.median(w_exc)),
                 max=float(np.max(w_exc)), min=float(np.min(w_exc))) if len(w_exc) else None,
        inh=dict(mean=float(np.mean(w_inh)), median=float(np.median(w_inh)),
                 max=float(np.max(np.abs(w_inh))),
                 min=float(np.min(np.abs(w_inh)))) if len(w_inh) else None,
        inh_to_exc_magnitude_ratio=(float(np.mean(np.abs(w_inh)) / np.mean(np.abs(w_exc)))
                                    if len(w_inh) and len(w_exc) else None),
        n_below_1pct_of_max=int((np.abs(w_all) < 0.01 * np.abs(w_all).max()).sum()),
        n_exactly_zero=int((w_all == 0).sum()),
        percentiles={str(p): float(np.percentile(np.abs(w_all), p))
                     for p in (5, 25, 50, 75, 95)})

    # ---------------------------------------------------------------- DELAYS
    dl = {}
    for name in ("in->exc", "in->inh", "exc->exc", "exc->inh", "inh->exc", "inh->inh",
                 "exc->out", "inh->out"):
        v = [d for s, t, w, d in E if edge_class(s, t) == name]
        if v:
            dl[name] = dict(n=len(v), mean=float(np.mean(v)), median=float(np.median(v)),
                            min=int(min(v)), max=int(max(v)))
    R["delays"] = dict(by_class=dl,
                       overall=dict(mean=float(np.mean([d for *_x, d in E])),
                                    median=float(np.median([d for *_x, d in E])),
                                    min=int(min(d for *_x, d in E)),
                                    max=int(max(d for *_x, d in E))))

    # ---------------------------------------------------------------- FUNCTION
    H = T.build([g], device=a.device)
    s = T.score(H, Xv, Yv, enc)
    first = s["first"][:, 0, :]
    base_mse = float(s["mse"][0])
    per_dim = []
    for d in range(N_OUT):
        p, t = first[:, d], tgt[:, d]
        r = float(np.corrcoef(p, t)[0, 1]) if p.std() > 1e-9 else 0.0
        vals, cnt = np.unique(p, return_counts=True)
        per_dim.append(dict(
            out=d, r=r, mse=float(((p - t) ** 2).mean()),
            offset_mean=float(p.mean()), offset_sd=float(p.std()),
            offset_min=int(p.min()), offset_max=int(p.max()),
            n_distinct_offsets=int(len(vals)),
            frac_silent=float((p >= T.READOUT_WINDOW).mean()),
            top_offsets=[[int(v), int(c)] for v, c in
                         sorted(zip(vals, cnt), key=lambda x: -x[1])[:5]]))
    R["function"] = dict(heldout_mse=base_mse, tau=float(s["tau"][0]),
                         mean_abs_r=float(np.mean([abs(x["r"]) for x in per_dim])),
                         per_output=per_dim)

    # permutation sensitivity: shuffle one input dim, measure |delta offset| per output
    rngp = np.random.default_rng(0)
    sens = np.zeros((N_IN, N_OUT))
    for j in range(N_IN):
        Xs = Xv.copy()
        Xs[:, j] = Xs[rngp.permutation(len(Xs)), j]
        fs = T.score(H, Xs, Yv, enc)["first"][:, 0, :]
        sens[j] = np.abs(fs - first).mean(0)
    R["function"]["sensitivity_mean_abs_delta_offset"] = sens.tolist()
    R["function"]["top_drivers_per_output"] = [
        [[int(j), float(sens[j, o])] for j in np.argsort(-sens[:, o])[:4]]
        for o in range(N_OUT)]
    R["function"]["input_total_influence"] = [
        [int(j), float(sens[j].mean())] for j in np.argsort(-sens.mean(1))]

    # ---------------------------------------------------------------- ABLATIONS
    abl = []
    for kind in ("no_inhibition", "no_recurrence", "ko_inh0", "ko_inh1", "prune50"):
        h = ablate(g, kind)
        n = int(h["mask"].sum())
        if n == 0:
            abl.append(dict(ablation=kind, n_synapses=0, heldout_mse=None))
            continue
        Ha = T.build([h], device=a.device)
        sa = T.score(Ha, Xv, Yv, enc)
        abl.append(dict(ablation=kind, n_synapses=n,
                        heldout_mse=float(sa["mse"][0]),
                        delta=float(sa["mse"][0] - base_mse),
                        tau=float(sa["tau"][0]),
                        silent=float(sa["silent"][0])))
    R["ablations"] = dict(baseline_mse=base_mse,
                          constant_baseline=T.constant_baseline(Yv), results=abl)

    print(json.dumps(T.jsonable({k: v for k, v in R.items()
                                 if k in ("source", "topology", "recurrence", "weights",
                                          "delays", "ablations")}), indent=1)[:6000])
    print("\nPATHS")
    for r in R["paths"]:
        print("  ", r)
    print("\nPER-OUTPUT")
    for r in R["function"]["per_output"]:
        print("  ", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})
    print("\nTOP DRIVERS")
    for o, d in enumerate(R["function"]["top_drivers_per_output"]):
        print(f"   OUT{o}: " + ", ".join(f"in{j:02d} ({v:.2f})" for j, v in d))

    if a.out:
        with open(a.out, "w") as f:
            json.dump(T.jsonable(R), f, indent=1)
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
