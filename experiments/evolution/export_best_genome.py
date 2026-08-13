"""export_best_genome.py — export the BEST evolved genome (neuroevolution result)
to the browser inspector's JSON contract (graph_evolved.json + activity_evolved.json),
matching the schema of spnet_export_clean.py so inspector.html?v=evolved renders it.

Unlike the hand-constructed CLEAN demo, this net was DISCOVERED by the DB-backed
neuroevolution search (best score ~0.88, ~1.5M genomes evaluated). Topology is arbitrary:
6 latency-coded inputs (i0-5) -> N hidden (h*) -> 4 output neurons (o0-3) whose
first-spike ORDER must match the LUT oracle's row ranking.

graph_evolved.json carries every neuron's full Izhikevich NeuronMeta (cf_2/cf_1/cf_0/
a/b/c/d/spike_threshold) so the inspector's node panel shows real values.
activity_evolved.json is a BUNDLE of 8 precomputed input variants (a combobox in the
inspector): each carries the input vector, its LUT ground-truth output + ranking, the
evolved net's first-spike ordering (and whether it matches), plus the full spike +
membrane-voltage trace for the animation/raster.
"""
import json
import os
from collections import deque

import torch

import neuroevo_lut as N
from neuroevo_lut import (build_population, IN_LABELS, OUT_LABELS, D, Dout, N_TICKS,
                          A_LAT, B_LAT, IN_THR, OUT_THR, leakfree, type_meta, m,
                          lut_bits, bits_to_row)
from evo_config import fixed_eval_set, EVAL_VERSION
from spiky.spnet.spnet import NeuronDataType

BEST = "/home/astarostin/projects/evo-run/best_genome.json"
OUT_DIR = "/home/astarostin/projects/spiky/experiments/evolution/visualiser/public"
COL_W, ROW_H = 190.0, 46.0
N_VARIANTS = 8
META_FIELDS = ["cf_2", "cf_1", "cf_0", "a", "b", "c", "d", "spike_threshold"]


def stim_tick(v):
    return min(max(int(round(A_LAT - B_LAT * v)), 1), N_TICKS - 1)


def run_input(sp, GID, x, record_voltage=False):
    """Stimulate the 6 inputs at latency ticks, run N_TICKS."""
    S = torch.zeros((1, N_TICKS, D), dtype=torch.int32)
    Vv = torch.zeros((1, N_TICKS, D), dtype=torch.float32)
    for i in range(D):
        S[0, stim_tick(x[i]), i] = GID["i%d" % i]
        Vv[0, stim_tick(x[i]), i] = 50.0
    sp.process_ticks(n_ticks_to_process=N_TICKS, batch_size=1, n_input_ticks=N_TICKS,
                     input_values=Vv, do_train=False, sparse_input=S,
                     do_record_voltage=record_voltage, do_reset_context=True)


def output_first_spike(sp, GID):
    Oids = torch.tensor([GID["o%d" % d] for d in range(Dout)], dtype=torch.int32)
    spk = sp.export_neuron_data(Oids, 1, NeuronDataType.Spike, 0, N_TICKS - 1).view(Dout, N_TICKS)
    first = []
    for d in range(Dout):
        nz = torch.nonzero(spk[d] > 0.5).flatten()
        first.append(int(nz[0].item()) if nz.numel() else N_TICKS + 1)
    return tuple(sorted(range(Dout), key=lambda d: (first[d], d))), first


def ground_truth(x):
    """LUT oracle for input x: selected row, its stored 4-vector, and the target ranking."""
    row = bits_to_row(lut_bits(m, x))
    out_values = [round(float(v), 4) for v in m["V"][row]]
    order = tuple(sorted(range(Dout), key=lambda d: -m["V"][row][d]))
    return row, out_values, order


def node_meta(g, n):
    if n[0] == "i":
        return leakfree(IN_THR)
    if n[0] == "o":
        return leakfree(OUT_THR)
    return type_meta(g, n)


def bfs_columns(nodes, edges):
    adj = {}
    for s, t in edges:
        adj.setdefault(s, []).append(t)
    dist = {n: 0 for n in nodes if n[0] == "i"}
    dq = deque(dist)
    while dq:
        u = dq.popleft()
        for v in adj.get(u, []):
            if v not in dist and v[0] != "i":
                dist[v] = dist[u] + 1
                dq.append(v)
    hid = [n for n in nodes if n[0] == "h"]
    max_reached = max([dist[h] for h in hid if h in dist] or [1])
    col = {}
    for n in nodes:
        col[n] = 0 if n[0] == "i" else (max_reached + 1 if n[0] == "o" else dist.get(n, 1))
    return col


def collect_activity(sp, GID, nodes, x):
    """Run x with voltage recording; return the full variant record."""
    run_input(sp, GID, x, record_voltage=True)
    no, first = output_first_spike(sp, GID)
    row, out_values, oracle = ground_truth(x)
    all_ids = torch.tensor([GID[n] for n in nodes], dtype=torch.int32)
    spk = sp.export_neuron_data(all_ids, 1, NeuronDataType.Spike, 0, N_TICKS - 1).view(len(nodes), N_TICKS)
    vol = sp.export_neuron_data(all_ids, 1, NeuronDataType.Voltage, 0, N_TICKS - 1).view(len(nodes), N_TICKS)
    spikes, voltages = [], []
    for ni, n in enumerate(nodes):
        voltages.append({"neuron_id": GID[n], "trace": [round(float(vol[ni, t]), 4) for t in range(N_TICKS)]})
        for t in range(N_TICKS):
            if float(spk[ni, t]) > 0.5:
                spikes.append({"tick": t, "neuron_id": GID[n]})
    correct = tuple(no) == oracle
    order_str = " > ".join("o%d" % d for d in oracle)
    return {
        "label": "truth %s · net %s" % (order_str, "✓" if correct else "✗"),
        "input": [round(v, 4) for v in x], "stim_ticks": [stim_tick(v) for v in x],
        "row": row, "out_values": out_values, "oracle_order": list(oracle),
        "net_order": list(no), "output_first_spike": first, "correct": correct,
        "t0": 0, "t1": N_TICKS - 1, "dt": 1, "spikes": spikes, "voltages": voltages,
    }


def main():
    best = json.load(open(BEST))
    g = best["best_genome"]
    xs, tos = fixed_eval_set()

    packed = build_population([g], device="cpu")
    sp = packed["spnet"]
    nodes = IN_LABELS + OUT_LABELS + list(g["hid"].keys())
    GID = {name: packed["gid"](0, i) for i, name in enumerate(nodes)}
    gid2name = {v: k for k, v in GID.items()}

    # ---- pick 8 inputs with DISTINCT ground-truth orderings, preferring for each
    #      ordering an input the net solves exactly (honest but representative) ----
    oracle_of = {}                       # idx -> ground-truth ordering
    correct, wrong = [], []
    for idx, x in enumerate(xs):
        run_input(sp, GID, x)            # no voltage — cheap correctness probe
        no, _ = output_first_spike(sp, GID)
        _, _, oracle = ground_truth(x)
        oracle_of[idx] = oracle
        (correct if tuple(no) == oracle else wrong).append(idx)
    n_strict = len(correct)

    def spread(idxs, k):                 # up to k inputs, distinct orderings first, then fill
        seen, out = set(), []
        for idx in idxs:
            if oracle_of[idx] not in seen:
                seen.add(oracle_of[idx])
                out.append(idx)
            if len(out) >= k:
                return out
        for idx in idxs:
            if idx not in out:
                out.append(idx)
            if len(out) >= k:
                break
        return out

    n_ok_target = min(5, len(correct))   # correct-majority mix, remainder = instructive misses
    chosen = spread(correct, n_ok_target)
    chosen += spread([i for i in wrong if i not in chosen], N_VARIANTS - len(chosen))
    if len(chosen) < N_VARIANTS:         # few misses -> top up with more correct
        chosen += [i for i in spread(correct, len(correct)) if i not in chosen][:N_VARIANTS - len(chosen)]
    chosen = chosen[:N_VARIANTS]

    variants = [collect_activity(sp, GID, nodes, xs[i]) for i in chosen]
    variants.sort(key=lambda v: (not v["correct"], v["input"]))   # a correct one first (default view)

    # ---- graph (input-independent): full NeuronMeta per neuron ----
    nsyn = sp.n_synapses()
    buf = {k: torch.zeros([nsyn], dtype=(torch.float32 if k == "weights" else torch.int32))
           for k in ["source_ids", "synapse_metas", "weights", "delays", "target_ids"]}
    sp.export_synapses(sp.get_all_neuron_ids(), buf["source_ids"], buf["synapse_metas"],
                       buf["weights"], buf["delays"], buf["target_ids"], forward_or_backward=True)
    edges, synapses_json = [], []
    for i in range(nsyn):
        s, t = int(buf["source_ids"][i]), int(buf["target_ids"][i])
        if s not in gid2name or t not in gid2name:
            continue
        edges.append((gid2name[s], gid2name[t]))
        synapses_json.append({"id": i, "source": s, "target": t,
                              "weight": round(float(buf["weights"][i]), 6), "delay": int(buf["delays"][i]),
                              "synapse_meta_index": int(buf["synapse_metas"][i]),
                              "learning_rate": 0.0, "min_weight": -1000.0, "max_weight": 1000.0})

    col = bfs_columns(nodes, edges)
    members = {}
    for n in nodes:
        members.setdefault(col[n], []).append(n)
    coord = {}
    for c, ms in members.items():
        for i, n in enumerate(ms):
            coord[n] = (c * COL_W, (i - (len(ms) - 1) / 2.0) * ROW_H)

    ntype = {"i": "input", "o": "output", "h": "hidden"}
    neurons_json = []
    for n in nodes:
        nm = node_meta(g, n)
        xx, yy = coord[n]
        rec = {"id": GID[n], "label": n, "type": ntype[n[0]], "col": col[n], "x": xx, "y": yy, "z": 0.0}
        rec.update({f: float(getattr(nm, f)) for f in META_FIELDS})
        neurons_json.append(rec)

    graph = {"neurons": neurons_json, "synapses": synapses_json,
             "layers": ["input", "hidden", "output"],
             "note": ("EVOLVED spiking net (neuroevolution result, best score %.4f, depth %s, "
                      "%d hidden / %d synapses). Latency-coded LUT: the 4 output neurons' first-spike "
                      "ORDER must match the LUT oracle's row ranking. Pick an input from the combobox."
                      % (best["best_score"], best.get("best_depth"), len(g["hid"]), len(synapses_json)))}
    os.makedirs(OUT_DIR, exist_ok=True)
    json.dump(graph, open(os.path.join(OUT_DIR, "graph_evolved.json"), "w"))

    activity = {"t0": 0, "t1": N_TICKS - 1, "dt": 1, "variants": variants}
    json.dump(activity, open(os.path.join(OUT_DIR, "activity_evolved.json"), "w"))

    n_ok = sum(v["correct"] for v in variants)
    print("=" * 68)
    print("EVOLVED EXPORT -> %s" % OUT_DIR)
    print("  best_score=%.4f depth=%s | %d neurons (%d hidden) / %d synapses | eval=%s"
          % (best["best_score"], best.get("best_depth"), len(neurons_json), len(g["hid"]),
             len(synapses_json), EVAL_VERSION))
    print("  %d input variants (%d solved exactly):" % (len(variants), n_ok))
    for i, v in enumerate(variants):
        print("   %d) x=%s  truth=%s vals=%s  net=%s  %s"
              % (i + 1, v["input"], tuple(v["oracle_order"]), v["out_values"],
                 tuple(v["net_order"]), "✓" if v["correct"] else "✗"))
    print("=" * 68)


if __name__ == "__main__":
    main()
