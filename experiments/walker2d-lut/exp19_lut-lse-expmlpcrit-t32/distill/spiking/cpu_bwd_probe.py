"""Does the bwd<fan-in LTP overflow reproduce on the CPU backend?

The .proto codegen emits both a CUDA kernel and an `_on_cpu_wrapper` host variant from the
same source, and the runtime branches on device == -1 throughout, so the same logic should
run on CPU. If it faults there too, the bug is cheaply debuggable with ASAN/valgrind and no
GPU. Small net, but fan-in deliberately above the backward group size.
"""
import argparse

import numpy as np
import torch
from spiky.spnet.spnet import NeuronMeta, SpikingNet, SynapseMeta
from spiky.util.chunk_of_connections import ChunkOfConnections
from spiky.util.synapse_growth import SynapseGrowthEngine

ap = argparse.ArgumentParser()
ap.add_argument("--bwd", type=int, default=8)
ap.add_argument("--fwd", type=int, default=8)
ap.add_argument("--gs", type=int, default=2, help="engine synapse_group_size")
ap.add_argument("--n", type=int, default=64)
ap.add_argument("--fanout", type=int, default=32)
ap.add_argument("--metas", type=int, default=1)
ap.add_argument("--ticks", type=int, default=64)
ap.add_argument("--device", default="cpu")
a = ap.parse_args()

dev = a.device
metas = [SynapseMeta(learning_rate=0.1, min_delay=d + 1, max_delay=d + 1,
                     initial_weight=6.0, min_weight=0.0, max_weight=45.0,
                     initial_noise_level=0.0, weight_decay=0.9, weight_scaling_cf=0.0,
                     _forward_group_size=a.fwd, _backward_group_size=a.bwd)
         for d in range(a.metas)]
sp = SpikingNet(synapse_metas=metas,
                neuron_metas=[NeuronMeta(neuron_type=0, a=0.02, d=8.0)],
                neuron_counts=[a.n], initial_synapse_capacity=1 << 18,
                summation_dtype=torch.float32)
sp.to_device(dev)
ids = sp.get_neuron_ids_by_meta(0).cpu().numpy()

tri = np.array([[j % a.metas, int(ids[s]), int(ids[(s + 1 + j) % a.n])]
                for s in range(a.n) for j in range(a.fanout)
                if (s + 1 + j) % a.n != s], np.int32)
w = np.full(len(tri), 6.0, np.float32)
_, cnt = np.unique(tri[:, 2], return_counts=True)
print(f"device={dev} bwd={a.bwd} fwd={a.fwd} engine_gs={a.gs} metas={a.metas}")
print(f"synapses={len(tri)} incoming per target: max={cnt.max()} mean={cnt.mean():.1f} "
      f"(backward group size {a.bwd} -> {int(np.ceil(cnt.max() / a.bwd))} chained groups)")

ge = SynapseGrowthEngine(device=dev, synapse_group_size=a.gs,
                         max_groups_in_buffer=max(4096, 8 * (len(tri) + a.n)))
ge.register_neuron_type(max_synapses=8 * a.n, growth_command_list=[])
nid = torch.tensor(ids, dtype=torch.int32)
ge.add_neurons(neuron_type_index=0, identifiers=nid,
               coordinates=torch.stack([torch.arange(nid.numel()).float(),
                                        torch.zeros(nid.numel()),
                                        torch.zeros(nid.numel())], dim=1))
tri_t = torch.tensor(tri, dtype=torch.int32, device=dev)
w_t = torch.tensor(w, dtype=torch.float32, device=dev)
conn = ge._grow_explicit(tri_t, 1).get_connections()
import es_harness
chunk = ChunkOfConnections(conn, a.gs,
                           weights=es_harness.group_aligned_weights(conn, tri_t, w_t, a.gs))
sp.add_connections(chunk, 1)
chunk.recycle()
sp.compile(shuffle_synapses_random_seed=None)
print("BUILD-OK")

B = 1
spk = torch.randint(a.n, [B, a.ticks, 1], dtype=torch.int32, device=dev)
val = torch.ones_like(spk, dtype=torch.float32) * 20.0
sp.process_ticks(n_ticks_to_process=a.ticks, batch_size=B, n_input_ticks=a.ticks,
                 input_values=val, do_train=True, sparse_input=spk,
                 do_record_voltage=False, do_reset_context=True)
if dev == "cuda":
    torch.cuda.synchronize()
print("STDP-OK -- no fault")
