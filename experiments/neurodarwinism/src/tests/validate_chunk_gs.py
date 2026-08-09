"""Run ChunkOfConnectionsValidator on the crashing gs=128 chunk, and on the gs=2 control.

Then still call add_connections, to see whether a validator-clean chunk crashes anyway.
"""
# tests live one level below src/; make the sibling modules importable.
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import argparse

import numpy as np
import torch
from spiky.spnet.spnet import NeuronMeta, SpikingNet, SynapseMeta
from spiky.util.chunk_of_connections import ChunkOfConnections, ChunkOfConnectionsValidator
from spiky.util.synapse_growth import SynapseGrowthEngine

ap = argparse.ArgumentParser()
ap.add_argument("--gs", type=int, default=128)
ap.add_argument("--metas", type=int, default=6)
ap.add_argument("--fanout", type=int, default=12)
ap.add_argument("--n", type=int, default=64)
ap.add_argument("--meta-gs", type=int, default=8)
a = ap.parse_args()

metas = [SynapseMeta(learning_rate=0.1, min_delay=d + 1, max_delay=d + 1,
                     initial_weight=0.0, min_weight=0.0, max_weight=45.0,
                     initial_noise_level=0.0, weight_decay=0.9, weight_scaling_cf=0.0,
                     _forward_group_size=a.meta_gs, _backward_group_size=a.meta_gs)
         for d in range(a.metas)]
sp = SpikingNet(synapse_metas=metas,
                neuron_metas=[NeuronMeta(neuron_type=0, a=0.02, d=8.0)],
                neuron_counts=[a.n], initial_synapse_capacity=1 << 20,
                summation_dtype=torch.float32)
sp.to_device("cuda")
ids = sp.get_neuron_ids_by_meta(0).cpu().numpy()

tri, w = [], []
for s in range(a.n):
    for j in range(a.fanout):
        t = (s + 1 + j) % a.n
        if t != s:
            tri.append([j % a.metas, int(ids[s]), int(ids[t])])
            w.append(1.0 + (s * a.fanout + j) % 40)
tri = np.array(tri, np.int32)
w = np.array(w, np.float32)

ge = SynapseGrowthEngine(device="cuda", synapse_group_size=a.gs,
                         max_groups_in_buffer=max(4096, 8 * (len(tri) + a.n)))
ge.register_neuron_type(max_synapses=8 * a.n, growth_command_list=[])
nid = torch.tensor(ids, dtype=torch.int32)
ge.add_neurons(neuron_type_index=0, identifiers=nid,
               coordinates=torch.stack([torch.arange(nid.numel()).float(),
                                        torch.zeros(nid.numel()),
                                        torch.zeros(nid.numel())], dim=1))
tri_t = torch.tensor(tri, dtype=torch.int32, device="cuda")
w_t = torch.tensor(w, dtype=torch.float32, device="cuda")

chunk = ge._grow_explicit(tri_t, 1, weights=w_t)
conn = chunk.get_connections()
print(f"gs={a.gs} metas={a.metas} synapses={len(tri)} "
      f"blocks={conn.numel() // (4 + 2 * a.gs)} "
      f"weights={chunk.get_weights().numel()}")

ok, errors = ChunkOfConnectionsValidator(chunk).validate_all()
print(f"VALIDATOR valid={ok} n_errors={len(errors)}")
for e in errors:
    print(f"    ERROR: {e}")

print("now calling add_connections ...")
sp.add_connections(chunk, 1)
chunk.recycle()
sp.compile(shuffle_synapses_random_seed=None)
torch.cuda.synchronize()
print("BUILD-OK")
