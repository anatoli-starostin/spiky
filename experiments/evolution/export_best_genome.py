"""export_best_genome.py — export the BEST evolved genome (neuroevolution result)
to the browser inspector's JSON contract (graph_evolved.json + activity_evolved.json),
matching the schema of spnet_export_clean.py so inspector.html?v=evolved renders it.

Unlike the hand-constructed CLEAN demo, this net was DISCOVERED by the DB-backed
neuroevolution search (best score ~0.88, ~1.5M genomes evaluated). Topology is arbitrary:
6 latency-coded inputs (i0-5) -> N hidden (h*) -> 4 output neurons (o0-3) whose
first-spike ORDER must match the LUT oracle's row ranking.

We build the single genome into a native SpikingNet, pick a demo input the net solves
correctly, record spikes + membrane voltages over N_TICKS, lay the neurons out in
BFS columns (input -> hidden depth -> output), and dump the two JSON files.
"""
import json
import os
import sys
from collections import deque

import torch

import neuroevo_lut as N
from neuroevo_lut import build_population, IN_LABELS, OUT_LABELS, D, Dout, N_TICKS, A_LAT, B_LAT, IN_THR, OUT_THR
from evo_config import fixed_eval_set, EVAL_VERSION
from spiky.spnet.spnet import NeuronDataType

BEST = "/home/astarostin/projects/evo-run/best_genome.json"
OUT_DIR = "/home/astarostin/projects/spiky/experiments/evolution/visualiser/public"
COL_W, ROW_H = 190.0, 46.0


def stim_tick(v):
    return min(max(int(round(A_LAT - B_LAT * v)), 1), N_TICKS - 1)


def run_input(sp, GID, x, record_voltage=False):
    """Stimulate the 6 inputs at latency ticks, run N_TICKS, return the Spike raster
    (and Voltage if requested) as tensors of shape (n_nodes, N_TICKS)."""
    S = torch.zeros((1, N_TICKS, D), dtype=torch.int32)
    Vv = torch.zeros((1, N_TICKS, D), dtype=torch.float32)
    for i in range(D):
        S[0, stim_tick(x[i]), i] = GID["i%d" % i]
        Vv[0, stim_tick(x[i]), i] = 50.0
    sp.process_ticks(n_ticks_to_process=N_TICKS, batch_size=1, n_input_ticks=N_TICKS,
                     input_values=Vv, do_train=False, sparse_input=S,
                     do_record_voltage=record_voltage, do_reset_context=True)


def net_order(sp, GID):
    Oids = torch.tensor([GID["o%d" % d] for d in range(Dout)], dtype=torch.int32)
    spk = sp.export_neuron_data(Oids, 1, NeuronDataType.Spike, 0, N_TICKS - 1).view(Dout, N_TICKS)
    first = []
    for d in range(Dout):
        nz = torch.nonzero(spk[d] > 0.5).flatten()
        first.append(int(nz[0].item()) if nz.numel() else N_TICKS + 1)
    return tuple(sorted(range(Dout), key=lambda d: (first[d], d))), first


def bfs_columns(nodes, edges):
    """Column per node: inputs=0, hidden=BFS distance from inputs (>=1), outputs rightmost."""
    adj = {}
    for s, t, *_ in edges:
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
        if n[0] == "i":
            col[n] = 0
        elif n[0] == "o":
            col[n] = max_reached + 1
        else:
            col[n] = dist.get(n, 1)  # unreached hidden -> col 1
    return col


def main():
    best = json.load(open(BEST))
    g = best["best_genome"]
    xs, tos = fixed_eval_set()

    packed = build_population([g], device="cpu")
    sp = packed["spnet"]
    nodes = IN_LABELS + OUT_LABELS + list(g["hid"].keys())
    GID = {name: packed["gid"](0, i) for i, name in enumerate(nodes)}
    gid2name = {v: k for k, v in GID.items()}

    # pick a demo input the net solves EXACTLY (strict order match); else best ordering match
    def concord(a, b):
        return sum(1 for i in range(Dout) for j in range(i + 1, Dout)
                   if (a.index(i) < a.index(j)) == (b.index(i) < b.index(j)))
    demo_i, demo_conc, demo_strict = 0, -1, False
    for idx, x in enumerate(xs):
        run_input(sp, GID, x)
        no, _ = net_order(sp, GID)
        strict = tuple(no) == tuple(tos[idx])
        c = concord(list(no), list(tos[idx]))
        if strict:
            demo_i, demo_strict = idx, True
            break
        if c > demo_conc:
            demo_i, demo_conc = idx, c
    x = xs[demo_i]
    true_order = tuple(tos[demo_i])

    # final run for the chosen input WITH voltage recording
    run_input(sp, GID, x, record_voltage=True)
    no, first = net_order(sp, GID)

    all_ids = torch.tensor([GID[n] for n in nodes], dtype=torch.int32)
    spk = sp.export_neuron_data(all_ids, 1, NeuronDataType.Spike, 0, N_TICKS - 1).view(len(nodes), N_TICKS)
    vol = sp.export_neuron_data(all_ids, 1, NeuronDataType.Voltage, 0, N_TICKS - 1).view(len(nodes), N_TICKS)

    # ---- synapses (named nodes only) ----
    nsyn = sp.n_synapses()
    buf = {k: torch.zeros([nsyn], dtype=(torch.float32 if k == "weights" else torch.int32))
           for k in ["source_ids", "synapse_metas", "weights", "delays", "target_ids"]}
    sp.export_synapses(sp.get_all_neuron_ids(), buf["source_ids"], buf["synapse_metas"],
                       buf["weights"], buf["delays"], buf["target_ids"], forward_or_backward=True)
    edges = []
    synapses_json = []
    for i in range(nsyn):
        s, t = int(buf["source_ids"][i]), int(buf["target_ids"][i])
        if s not in gid2name or t not in gid2name:
            continue
        edges.append((gid2name[s], gid2name[t]))
        synapses_json.append({"id": i, "source": s, "target": t,
                              "weight": round(float(buf["weights"][i]), 6), "delay": int(buf["delays"][i]),
                              "synapse_meta_index": int(buf["synapse_metas"][i]),
                              "learning_rate": 0.0, "min_weight": -1000.0, "max_weight": 1000.0})

    # ---- layout: BFS columns, spread rows within each column ----
    col = bfs_columns(nodes, edges)
    members = {}
    for n in nodes:
        members.setdefault(col[n], []).append(n)
    coord = {}
    for c, ms in members.items():
        for i, n in enumerate(ms):
            coord[n] = (c * COL_W, (i - (len(ms) - 1) / 2.0) * ROW_H)

    def ntype(n):
        return {"i": "input", "o": "output", "h": "hidden"}[n[0]]

    def nthr(n):
        if n[0] == "i":
            return IN_THR
        if n[0] == "o":
            return OUT_THR
        return float(g["types"][g["hid"][n]]["thr"])

    neurons_json = []
    for n in nodes:
        xx, yy = coord[n]
        rec = {"id": GID[n], "label": n, "type": ntype(n), "col": col[n],
               "x": xx, "y": yy, "z": 0.0, "spike_threshold": nthr(n)}
        if n[0] == "h":
            ty = g["types"][g["hid"][n]]
            rec.update({"leak": round(float(ty["leak"]), 4), "delay": round(float(ty["d"]), 3)})
        neurons_json.append(rec)

    graph = {"neurons": neurons_json, "synapses": synapses_json,
             "layers": ["input", "hidden", "output"],
             "note": ("EVOLVED spiking net (neuroevolution result, best score %.4f, depth %s, "
                      "%d hidden / %d synapses). Latency-coded LUT: the 4 output neurons' first-spike "
                      "ORDER must match the LUT oracle's row ranking."
                      % (best["best_score"], best.get("best_depth"), len(g["hid"]), len(synapses_json)))}
    os.makedirs(OUT_DIR, exist_ok=True)
    json.dump(graph, open(os.path.join(OUT_DIR, "graph_evolved.json"), "w"))

    # ---- activity ----
    spikes, voltages = [], []
    for ni, n in enumerate(nodes):
        voltages.append({"neuron_id": GID[n], "trace": [round(float(vol[ni, t]), 4) for t in range(N_TICKS)]})
        for t in range(N_TICKS):
            if float(spk[ni, t]) > 0.5:
                spikes.append({"tick": t, "neuron_id": GID[n]})
    activity = {"t0": 0, "t1": N_TICKS - 1, "dt": 1, "input": [round(v, 4) for v in x],
                "stim_ticks": [stim_tick(v) for v in x], "spikes": spikes, "voltages": voltages,
                "net_order": list(no), "oracle_order": list(true_order),
                "output_first_spike": first, "correct": tuple(no) == true_order}
    json.dump(activity, open(os.path.join(OUT_DIR, "activity_evolved.json"), "w"))

    print("=" * 64)
    print("EVOLVED EXPORT -> %s" % OUT_DIR)
    print("  best_score=%.4f depth=%s | %d neurons (%d hidden) / %d synapses"
          % (best["best_score"], best.get("best_depth"), len(neurons_json), len(g["hid"]), len(synapses_json)))
    print("  demo input idx=%d strict=%s  x=%s" % (demo_i, demo_strict, [round(v, 3) for v in x]))
    print("  net_order=%s  oracle_order=%s  CORRECT=%s" % (no, true_order, tuple(no) == true_order))
    print("  output first-spike ticks=%s  | %d spikes over %d ticks | eval=%s"
          % (first, len(spikes), N_TICKS, EVAL_VERSION))
    print("=" * 64)


if __name__ == "__main__":
    main()
