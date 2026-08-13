"""spnet_export_latency.py — export the LATENCY-INPUT variant to the browser
inspector as SEPARATE files (graph_latency.json + activity_latency.json), so it
does not touch the coincidence export (graph.json/activity.json).

Reuses evolution/native_encoding_latency.py verbatim (imported) — no logic change:
its already-built palette SpikingNet, gid map, node list and run_input() are used
directly. Adds the decoder layer (per-input latch u_i) and the GATE node to the
graph, records one demo input's spikes+voltages, and writes both JSONs.
"""
import json
import os
import torch

import native_encoding_latency as L                    # builds the latency net on import
from lutmodel import lut_bits, bits_to_row

OUT_DIR = "/home/astarostin/projects/lut2spiking_toy/visualiser/public"
spnet, gid, local, nodes = L.spnet, L.gid, L.local, L.nodes
K, D, Dout, N_TICKS = L.K, L.D, L.Dout, L.N_TICKS
neuron_metas, node_meta = L.neuron_metas, L.node_meta

DEMO_INPUT = [-1, -1, -1, 1, -1, 1]                     # representative (bits [1,1,1] -> row 7)


def layer_of(n):
    if n == "START":
        return "clock"
    if n == "GATE":
        return "gate"
    return {"x": "input", "u": "decode", "H": "detect", "C": "compl", "r": "rows", "o": "output"}[n[0]]


COLS = {"clock": 0, "gate": 1, "input": 2, "decode": 3, "detect": 4, "compl": 5, "rows": 6, "output": 7}
layer_members = {}
for n in nodes:
    layer_members.setdefault(layer_of(n), []).append(n)
coord = {}
for lyr, members in layer_members.items():
    for i, n in enumerate(members):
        coord[n] = (COLS[lyr] * 170.0, (i - (len(members) - 1) / 2.0) * 44.0)

GID = {n: gid(local[n]) for n in nodes}
gid2name = {v: k for k, v in GID.items()}

# ---- graph_latency.json ----
nm_fields = ["cf_2", "cf_1", "cf_0", "a", "b", "c", "d", "spike_threshold"]
neurons_json = []
for n in nodes:
    nmeta = neuron_metas[node_meta[local[n]]]
    x, y = coord[n]
    neurons_json.append({
        "id": GID[n], "label": n, "type": layer_of(n), "col": COLS[layer_of(n)],
        "x": x, "y": y, "z": 0.0,
        **{f: float(getattr(nmeta, f)) for f in nm_fields},
    })

nsyn = spnet.n_synapses()
exp = {kk: torch.zeros([nsyn], dtype=(torch.float32 if kk == "weights" else torch.int32))
       for kk in ["source_ids", "synapse_metas", "weights", "delays", "target_ids"]}
spnet.export_synapses(spnet.get_all_neuron_ids(), exp["source_ids"], exp["synapse_metas"],
                      exp["weights"], exp["delays"], exp["target_ids"], forward_or_backward=True)
synapses_json = []
for i in range(nsyn):
    s, t = int(exp["source_ids"][i]), int(exp["target_ids"][i])
    if s not in gid2name or t not in gid2name:
        continue
    synapses_json.append({
        "id": i, "source": s, "target": t,
        "weight": round(float(exp["weights"][i]), 6), "delay": int(exp["delays"][i]),
        "synapse_meta_index": int(exp["synapse_metas"][i]),
        "learning_rate": 0.0, "min_weight": -1000.0, "max_weight": 1000.0,
    })

graph = {"neurons": neurons_json, "synapses": synapses_json, "layers": list(COLS.keys()),
         "note": "LATENCY-input re-encoded seed-0 toy LUT (time-to-first-spike inputs + "
                 "delay-line decoder u_i + GATE; clock-comb latency output)"}
os.makedirs(OUT_DIR, exist_ok=True)
with open(os.path.join(OUT_DIR, "graph_latency.json"), "w") as f:
    json.dump(graph, f)

# ---- activity_latency.json: one demo input ----
L.run_input(DEMO_INPUT)                                 # records spikes + voltages
all_ids = torch.tensor([GID[n] for n in nodes], dtype=torch.int32)
spk = spnet.export_neuron_data(all_ids, 1, L.NeuronDataType.Spike, 0, N_TICKS - 1).view(len(nodes), N_TICKS)
vol = spnet.export_neuron_data(all_ids, 1, L.NeuronDataType.Voltage, 0, N_TICKS - 1).view(len(nodes), N_TICKS)
spikes, voltages = [], []
for ni, n in enumerate(nodes):
    voltages.append({"neuron_id": GID[n], "trace": [round(float(vol[ni, t]), 4) for t in range(N_TICKS)]})
    for t in range(N_TICKS):
        if float(spk[ni, t]) > 0.5:
            spikes.append({"tick": t, "neuron_id": GID[n]})
stim_ticks = [L.value_to_tin(v) for v in DEMO_INPUT]
activity = {"t0": 0, "t1": N_TICKS - 1, "dt": 1, "input": DEMO_INPUT, "stim_ticks": stim_ticks,
            "spikes": spikes, "voltages": voltages}
with open(os.path.join(OUT_DIR, "activity_latency.json"), "w") as f:
    json.dump(activity, f)

# ---- report numbers ----
bits = lut_bits(L.m, DEMO_INPUT)
row = bits_to_row(bits)


def fired(name):
    ni = nodes.index(name)
    return bool((spk[ni] > 0.5).any().item())


drows = [a for a in range(1 << K) if fired("r%d" % a)]
t_out = []
for j in range(Dout):
    ni = nodes.index("o%d" % j)
    nz = torch.nonzero(spk[ni] > 0.5).flatten()
    t_out.append(int(nz[0].item()) if nz.numel() else None)
order = sorted(range(Dout), key=lambda j: (t_out[j] if t_out[j] is not None else 1e9))
print("=" * 60)
print("LATENCY EXPORT written to %s" % OUT_DIR)
print("  graph_latency.json : %d neurons, %d synapses" % (len(neurons_json), len(synapses_json)))
print("  activity_latency.json : %d spikes over %d ticks" % (len(spikes), N_TICKS))
print("  demo x=%s  stim_ticks=%s (+1=%d,-1=%d)" % (DEMO_INPUT, stim_ticks, L.TIN_HI, L.TIN_LO))
print("  oracle bits=%s row=%d ; decoded row(s)=%s" % (bits, row, drows))
print("  output spike ticks o0..o3=%s -> order %s" % (t_out, order))
print("=" * 60)
