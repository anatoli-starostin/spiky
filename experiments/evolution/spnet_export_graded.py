"""spnet_export_graded.py — export the GRADED real-valued variant to SEPARATE
browser files (graph_graded.json + activity_graded.json) in visualiser/public/,
without touching the coincidence or latency exports.

Reuses evolution/native_encoding_graded.py verbatim (imported) — its built palette
SpikingNet, gid map, node list, run_input() and value_to_stim(). Lays the large
multi-tap decoder out as a readable grid: inputs -> gate bank -> a 6x L tap grid
(thermometer) -> detectors -> complement -> rows -> outputs.
"""
import json
import os
import torch

import native_encoding_graded as G
from lutmodel import lut_bits, bits_to_row

OUT_DIR = "/home/astarostin/projects/lut2spiking_toy/visualiser/public"
spnet, gid, local, nodes = G.spnet, G.gid, G.local, G.nodes
K, D, Dout, L, N_TICKS = G.K, G.D, G.Dout, G.L, G.N_TICKS
neuron_metas, node_meta = G.neuron_metas, G.node_meta

DEMO_INPUT = [-0.88, 0.87, -0.72, 0.77, -0.91, 0.3]     # robust non-trivial row (row 3)


def layer_of(n):
    if n == "START":
        return "clock"
    return {"x": "input", "g": "gate", "u": "decode", "H": "detect",
            "C": "compl", "r": "rows", "o": "output"}[n[0]]


COLS = {"clock": 0, "input": 1, "gate": 2, "decode": 3, "detect": 4, "compl": 5, "rows": 6, "output": 7}
COLX = {"clock": 0, "input": 150, "gate": 300, "detect": 300 + 40 * (L + 2),
        "compl": 300 + 40 * (L + 2) + 150, "rows": 300 + 40 * (L + 2) + 300,
        "output": 300 + 40 * (L + 2) + 450}


def coord(n):
    lyr = layer_of(n)
    if n == "START":
        return (COLX["clock"], 0.0)
    if lyr == "input":
        i = int(n[1:])
        return (COLX["input"], (i - (D - 1) / 2.0) * 60.0)
    if lyr == "gate":
        j = int(n[1:])
        return (COLX["gate"], (j - (L - 1) / 2.0) * 18.0)
    if lyr == "decode":                                 # u{i}_{j}: 6 x L grid
        i, j = (int(v) for v in n[1:].split("_"))
        return (COLX["gate"] + 40 + j * 40.0, (i - (D - 1) / 2.0) * 66.0)
    if lyr == "detect":
        return (COLX["detect"], (int(n[1:]) - (K - 1) / 2.0) * 70.0)
    if lyr == "compl":
        return (COLX["compl"], (int(n[1:]) - (K - 1) / 2.0) * 70.0)
    if lyr == "rows":
        a = int(n[1:])
        return (COLX["rows"], (a - ((1 << K) - 1) / 2.0) * 34.0)
    d = int(n[1:])
    return (COLX["output"], (d - (Dout - 1) / 2.0) * 44.0)


GID = {n: gid(local[n]) for n in nodes}
gid2name = {v: k for k, v in GID.items()}

# ---- graph_graded.json ----
nm_fields = ["cf_2", "cf_1", "cf_0", "a", "b", "c", "d", "spike_threshold"]
neurons_json = []
for n in nodes:
    nmeta = neuron_metas[node_meta[local[n]]]
    x, y = coord(n)
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
         "note": "GRADED real-valued latency LUT: multi-tap thermometer decoder "
                 "(inputs -> gate bank -> 6xL tap grid -> detectors -> rows -> comb output)"}
os.makedirs(OUT_DIR, exist_ok=True)
with open(os.path.join(OUT_DIR, "graph_graded.json"), "w") as f:
    json.dump(graph, f)

# ---- activity_graded.json ----
G.run_input(DEMO_INPUT, record=True)
all_ids = torch.tensor([GID[n] for n in nodes], dtype=torch.int32)
spk = spnet.export_neuron_data(all_ids, 1, G.NeuronDataType.Spike, 0, N_TICKS - 1).view(len(nodes), N_TICKS)
vol = spnet.export_neuron_data(all_ids, 1, G.NeuronDataType.Voltage, 0, N_TICKS - 1).view(len(nodes), N_TICKS)
spikes, voltages = [], []
for ni, n in enumerate(nodes):
    voltages.append({"neuron_id": GID[n], "trace": [round(float(vol[ni, t]), 4) for t in range(N_TICKS)]})
    for t in range(N_TICKS):
        if float(spk[ni, t]) > 0.5:
            spikes.append({"tick": t, "neuron_id": GID[n]})
stim_ticks = [G.value_to_stim(v) for v in DEMO_INPUT]
activity = {"t0": 0, "t1": N_TICKS - 1, "dt": 1, "input": DEMO_INPUT, "stim_ticks": stim_ticks,
            "spikes": spikes, "voltages": voltages}
with open(os.path.join(OUT_DIR, "activity_graded.json"), "w") as f:
    json.dump(activity, f)

# ---- report ----
bits = lut_bits(G.m, DEMO_INPUT)
row = bits_to_row(bits)


def fired(name):
    ni = nodes.index(name)
    return bool((spk[ni] > 0.5).any().item())


taps_fired = [sum(1 for j in range(L) if fired("u%d_%d" % (i, j))) for i in range(D)]
drows = [a for a in range(1 << K) if fired("r%d" % a)]
t_out = []
for d in range(Dout):
    ni = nodes.index("o%d" % d)
    nz = torch.nonzero(spk[ni] > 0.5).flatten()
    t_out.append(int(nz[0].item()) if nz.numel() else None)
order = sorted(range(Dout), key=lambda d: (t_out[d] if t_out[d] is not None else 1e9))
print("=" * 64)
print("GRADED EXPORT written to %s" % OUT_DIR)
print("  graph_graded.json : %d neurons, %d synapses" % (len(neurons_json), len(synapses_json)))
print("  activity_graded.json : %d spikes over %d ticks" % (len(spikes), N_TICKS))
print("  demo x=%s" % DEMO_INPUT)
print("  stim ticks (distinct latencies)=%s" % stim_ticks)
print("  thermometer taps fired per input=%s (of %d)" % (taps_fired, L))
print("  oracle bits=%s row=%d ; decoded row(s)=%s" % (bits, row, drows))
print("  output spike ticks o0..o3=%s -> order %s" % (t_out, order))
print("=" * 64)
