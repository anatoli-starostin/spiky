"""spnet_export_clean.py — export the CLEAN leaky-detector variant to SEPARATE
browser files (graph_clean.json + activity_clean.json) in visualiser/public/,
without touching the other exports.

Reuses evolution/native_encoding_clean.py (imported) — its calibrated leaky-detector
SpikingNet, gid map, node list, stim() and value_to_stim(). Small topology (no tap
bank): START -> 6 inputs -> 3 LEAKY detectors -> complement -> rows -> comb output.
"""
import json
import os
import torch

import native_encoding_clean as C
from lutmodel import lut_bits, bits_to_row

OUT_DIR = "/home/astarostin/projects/lut2spiking_toy/visualiser/public"
spnet, gid, local, nodes = C.spnet, C.gid, C.local, C.nodes
K, D, Dout, N_TICKS = C.K, C.D, C.Dout, C.N_TICKS
neuron_metas, node_meta = C.neuron_metas, C.node_meta

DEMO_INPUT = [-0.88, 0.87, -0.72, 0.77, -0.91, 0.3]


def layer_of(n):
    if n == "START":
        return "clock"
    return {"x": "input", "H": "detect", "C": "compl", "r": "rows", "o": "output"}[n[0]]


COLS = {"clock": 0, "input": 1, "detect": 2, "compl": 3, "rows": 4, "output": 5}
layer_members = {}
for n in nodes:
    layer_members.setdefault(layer_of(n), []).append(n)
coord = {}
for lyr, members in layer_members.items():
    for i, n in enumerate(members):
        coord[n] = (COLS[lyr] * 180.0, (i - (len(members) - 1) / 2.0) * 46.0)

GID = {n: gid(local[n]) for n in nodes}
gid2name = {v: k for k, v in GID.items()}

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
         "note": "CLEAN latency LUT: real-valued time-to-first-spike inputs read directly by "
                 "LEAKY-integrator detectors (no tap bank); comb latency output"}
os.makedirs(OUT_DIR, exist_ok=True)
with open(os.path.join(OUT_DIR, "graph_clean.json"), "w") as f:
    json.dump(graph, f)

# ---- activity_clean.json ----
C.stim(spnet, gid, DEMO_INPUT)
all_ids = torch.tensor([GID[n] for n in nodes], dtype=torch.int32)
spk = spnet.export_neuron_data(all_ids, 1, C.NeuronDataType.Spike, 0, N_TICKS - 1).view(len(nodes), N_TICKS)
vol = spnet.export_neuron_data(all_ids, 1, C.NeuronDataType.Voltage, 0, N_TICKS - 1).view(len(nodes), N_TICKS)
spikes, voltages = [], []
for ni, n in enumerate(nodes):
    voltages.append({"neuron_id": GID[n], "trace": [round(float(vol[ni, t]), 4) for t in range(N_TICKS)]})
    for t in range(N_TICKS):
        if float(spk[ni, t]) > 0.5:
            spikes.append({"tick": t, "neuron_id": GID[n]})
stim_ticks = [C.value_to_stim(v) for v in DEMO_INPUT]
activity = {"t0": 0, "t1": N_TICKS - 1, "dt": 1, "input": DEMO_INPUT, "stim_ticks": stim_ticks,
            "spikes": spikes, "voltages": voltages}
with open(os.path.join(OUT_DIR, "activity_clean.json"), "w") as f:
    json.dump(activity, f)

bits = lut_bits(C.m, DEMO_INPUT)
row = bits_to_row(bits)
t_out = []
for d in range(Dout):
    ni = nodes.index("o%d" % d)
    nz = torch.nonzero(spk[ni] > 0.5).flatten()
    t_out.append(int(nz[0].item()) if nz.numel() else None)
print("=" * 60)
print("CLEAN EXPORT written to %s" % OUT_DIR)
print("  graph_clean.json : %d neurons, %d synapses" % (len(neurons_json), len(synapses_json)))
print("  activity_clean.json : %d spikes over %d ticks" % (len(spikes), N_TICKS))
print("  demo x=%s  stim ticks=%s" % (DEMO_INPUT, stim_ticks))
print("  oracle bits=%s row=%d ; output spike ticks=%s" % (bits, row, t_out))
print("=" * 60)
