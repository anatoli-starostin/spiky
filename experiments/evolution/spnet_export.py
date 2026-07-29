"""spnet_export.py — export a live spiky SpikingNet (CPU) to JSON for the browser
inspector. Demo: builds the re-encoded toy-LUT net (native_encoding scheme),
runs one input over a window, writes graph.json + activity.json into the
visualiser public dir. Real values pulled from export_synapses + the metas.

Input is a RANDOM binary vector x in {+1,-1}^D. The RNG seed defaults to a
time-derived value and is PRINTED so any export is reproducible: re-run with
`SEED=<n> python spnet_export.py` to regenerate the exact same input.
"""
import json
import os
import random
import time
import torch
import spiky_cuda  # noqa
from lutmodel import build_model
from spnet_harness import build_packed, leakfree_meta
from spiky.spnet.spnet import NeuronDataType

OUT_DIR = "/home/astarostin/projects/lut2spiking_toy/visualiser/public"

# ---- build the re-encoded LUT net (same as native_encoding) with node tracking ----
m = build_model(0)
W, b, V, K, D, Dout = m["W"], m["b"], m["V"], m["K"], m["D"], m["Dout"]
sumW = [sum(W[k]) for k in range(K)]
det_theta = [(sumW[k] - b[k]) / 2.0 for k in range(K)]
COMP_THR, ROW_THR, IO_THR, LOW_THR, THETA_BIG = 5.0, 2.5, 6.0, 0.5, 50.0
CLK_W, INH_W, CLK_C_DELAY = 10.0, -20.0, 8
COMB_DELTA, COMB_BASE, COMB_M = 0.5, 16, 26   # spiking-output clock comb
N_TICKS = 46

nodes = ["START"] + ["x%d" % i for i in range(D)] + ["H%d" % k for k in range(K)] \
        + ["C%d" % k for k in range(K)] + ["r%d" % a for a in range(1 << K)] \
        + ["o%d" % j for j in range(Dout)]
local = {n: i for i, n in enumerate(nodes)}


def layer_of(n):
    if n == "START":
        return "clock"
    return {"x": "input", "H": "detect", "C": "compl", "r": "rows", "o": "output"}[n[0]]


COLS = {"clock": 0, "input": 1, "detect": 2, "compl": 3, "rows": 4, "output": 5}
# layered layout coordinates
layer_members = {}
for n in nodes:
    layer_members.setdefault(layer_of(n), []).append(n)
coord = {}
for lyr, members in layer_members.items():
    for i, n in enumerate(members):
        coord[n] = (COLS[lyr] * 180.0, (i - (len(members) - 1) / 2.0) * 46.0)

thr = {"START": LOW_THR}
for i in range(D):
    thr["x%d" % i] = LOW_THR
for k in range(K):
    thr["H%d" % k] = THETA_BIG
for k in range(K):
    thr["C%d" % k] = COMP_THR
for a in range(1 << K):
    thr["r%d" % a] = ROW_THR
for j in range(Dout):
    thr["o%d" % j] = IO_THR
uthr = sorted(set(thr.values()))
neuron_metas = [leakfree_meta(t) for t in uthr]
node_meta = [uthr.index(thr[n]) for n in nodes]

syn = []
for k in range(K):
    for i in range(D):
        syn.append((local["x%d" % i], local["H%d" % k], W[k][i], 1))
    syn.append((local["START"], local["H%d" % k], THETA_BIG - det_theta[k], 1))
    syn.append((local["START"], local["C%d" % k], CLK_W, CLK_C_DELAY))
    syn.append((local["H%d" % k], local["C%d" % k], INH_W, 1))
for a in range(1 << K):
    for k in range(K):
        bit = (a >> (K - 1 - k)) & 1
        syn.append((local[("H%d" % k) if bit else ("C%d" % k)], local["r%d" % a], 1.0, 1))
    for j in range(Dout):
        syn.append((local["r%d" % a], local["o%d" % j], V[a][j], 1))
for j in range(Dout):
    for i in range(COMB_M):
        syn.append((local["START"], local["o%d" % j], COMB_DELTA, COMB_BASE + i))

packed = build_packed([{"n_nodes": len(nodes), "node_meta": node_meta, "synapses": syn}], neuron_metas)
spnet, gid = packed["spnet"], packed["gid"]
GID = {n: gid(0, local[n]) for n in nodes}
gid2name = {v: k for k, v in GID.items()}

# ---- graph.json ----
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

# real synapse data via export_synapses
nsyn = spnet.n_synapses()
exp = {kk: torch.zeros([nsyn], dtype=(torch.float32 if kk == "weights" else torch.int32))
       for kk in ["source_ids", "synapse_metas", "weights", "delays", "target_ids"]}
spnet.export_synapses(spnet.get_all_neuron_ids(), exp["source_ids"], exp["synapse_metas"],
                      exp["weights"], exp["delays"], exp["target_ids"], forward_or_backward=True)
syn_metas = packed.get("syn_metas_list")   # may be None; fall back to per-synapse export only
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

graph = {"neurons": neurons_json, "synapses": synapses_json,
         "layers": list(COLS.keys()),
         "note": "re-encoded seed-0 toy LUT on real spnet (leak-free neurons)"}
os.makedirs(OUT_DIR, exist_ok=True)
with open(os.path.join(OUT_DIR, "graph.json"), "w") as f:
    json.dump(graph, f)

# ---- activity.json: run one RANDOM input, record spikes + voltages ----
SEED = int(os.environ.get("SEED", time.time_ns() % (2 ** 31)))
_rng = random.Random(SEED)
DEMO_INPUT = [_rng.choice([1, -1]) for _ in range(D)]   # random binary input in {+1,-1}^D
print("random input seed=%d  ->  x=%s" % (SEED, DEMO_INPUT))
ev = [GID["START"]] + [GID["x%d" % i] for i in range(D) if DEMO_INPUT[i] > 0]
S = torch.zeros((1, N_TICKS, len(ev)), dtype=torch.int32)
Vv = torch.zeros((1, N_TICKS, len(ev)), dtype=torch.float32)
for j, nid in enumerate(ev):
    S[0, 1, j] = nid
    Vv[0, 1, j] = 50.0
spnet.process_ticks(n_ticks_to_process=N_TICKS, batch_size=1, n_input_ticks=N_TICKS,
                    input_values=Vv, do_train=False, sparse_input=S,
                    do_record_voltage=True, do_reset_context=True)
all_ids = torch.tensor([GID[n] for n in nodes], dtype=torch.int32)
spk = spnet.export_neuron_data(all_ids, 1, NeuronDataType.Spike, 0, N_TICKS - 1).view(len(nodes), N_TICKS)
vol = spnet.export_neuron_data(all_ids, 1, NeuronDataType.Voltage, 0, N_TICKS - 1).view(len(nodes), N_TICKS)
spikes = []
voltages = []
for ni, n in enumerate(nodes):
    trace = [round(float(vol[ni, t]), 4) for t in range(N_TICKS)]
    voltages.append({"neuron_id": GID[n], "trace": trace})
    for t in range(N_TICKS):
        if float(spk[ni, t]) > 0.5:
            spikes.append({"tick": t, "neuron_id": GID[n]})
activity = {"t0": 0, "t1": N_TICKS - 1, "dt": 1, "seed": SEED,
            "input": DEMO_INPUT, "spikes": spikes, "voltages": voltages}
with open(os.path.join(OUT_DIR, "activity.json"), "w") as f:
    json.dump(activity, f)

# ---- diagnostics: detector bits + selected row (which fired?) ----
def _fired(name):
    ni = nodes.index(name)
    return bool((spk[ni] > 0.5).any().item())


det_bits = [1 if _fired("H%d" % k) else 0 for k in range(K)]        # H_k fires => bit k = 1
selected_rows = [a for a in range(1 << K) if _fired("r%d" % a)]
out_fired = [j for j in range(Dout) if _fired("o%d" % j)]
addr = sum(bit << (K - 1 - k) for k, bit in enumerate(det_bits))
print("wrote graph.json (%d neurons, %d synapses) and activity.json (%d spikes, %d ticks, seed=%d) to %s"
      % (len(neurons_json), len(synapses_json), len(spikes), N_TICKS, SEED, OUT_DIR))
print("detector bits (H0..H%d): %s  -> row address a=%d" % (K - 1, det_bits, addr))
print("selected row(s) that fired: %s   expected from bits: r%d" % (selected_rows, addr))
print("outputs that fired: %s   LUT row V[%d]=%s" % (out_fired, addr, [round(float(v), 3) for v in V[addr]]))
