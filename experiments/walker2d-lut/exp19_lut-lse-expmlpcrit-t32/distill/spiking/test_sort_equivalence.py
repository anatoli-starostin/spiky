"""Does the atomicCAS change the RESULT of the chain sort, or only who performs it?

The CPU path is untouched by the fix (the claim lives inside #ifdef ATOMIC and the host path
is single-threaded), so the CPU-emitted connections buffer is a fixed reference for what the
sort is SUPPOSED to produce. If the GPU buffer is bit-identical to it, the CAS has not
perturbed ordering — it only made exactly one thread do the sorting.

Compares the raw connections buffer element-for-element across a range of shapes, including
multi-meta chains (which are the ones the sort actually reorders).
"""
import numpy as np
import torch
from spiky.spnet.spnet import NeuronMeta, SpikingNet, SynapseMeta
from spiky.util.synapse_growth import SynapseGrowthEngine


def emit(device, n_neurons, n_metas, fanout, engine_gs, seed=1):
    metas = [SynapseMeta(learning_rate=0.1, min_delay=d + 1, max_delay=d + 1,
                         initial_weight=6.0, min_weight=0.0, max_weight=45.0,
                         initial_noise_level=0.0, weight_decay=0.9, weight_scaling_cf=0.0,
                         _forward_group_size=8, _backward_group_size=32)
             for d in range(n_metas)]
    sp = SpikingNet(synapse_metas=metas,
                    neuron_metas=[NeuronMeta(neuron_type=0, a=0.02, d=8.0)],
                    neuron_counts=[n_neurons], initial_synapse_capacity=1 << 20,
                    summation_dtype=torch.float32)
    sp.to_device(device)
    ids = sp.get_neuron_ids_by_meta(0).cpu().numpy()
    tri = np.array([[j % n_metas, int(ids[s]), int(ids[(s + 1 + j) % n_neurons])]
                    for s in range(n_neurons) for j in range(fanout)
                    if (s + 1 + j) % n_neurons != s], np.int32)
    w = np.array([1.0 + (i % 40) for i in range(len(tri))], np.float32)

    ge = SynapseGrowthEngine(device=device, synapse_group_size=engine_gs,
                             max_groups_in_buffer=max(4096, 8 * (len(tri) + n_neurons)))
    ge.register_neuron_type(max_synapses=8 * n_neurons, growth_command_list=[])
    nid = torch.tensor(ids, dtype=torch.int32)
    ge.add_neurons(neuron_type_index=0, identifiers=nid,
                   coordinates=torch.stack([torch.arange(nid.numel()).float(),
                                            torch.zeros(nid.numel()),
                                            torch.zeros(nid.numel())], dim=1))
    tri_t = torch.tensor(tri, dtype=torch.int32, device=device)
    w_t = torch.tensor(w, dtype=torch.float32, device=device)
    chunk = ge._grow_explicit(tri_t, seed, weights=w_t)
    return (chunk.get_connections().cpu().numpy().copy(),
            chunk.get_weights().cpu().numpy().copy())


CASES = [   # neurons, metas, fanout, engine_gs
    (64, 1, 8, 2), (64, 4, 8, 2), (64, 20, 20, 2),
    (64, 4, 8, 128), (64, 20, 20, 128), (256, 20, 20, 128),
]
allsame = True
for (n, m, f, gs) in CASES:
    c_cpu, w_cpu = emit("cpu", n, m, f, gs)
    c_gpu, w_gpu = emit("cuda", n, m, f, gs)
    same_c = c_cpu.shape == c_gpu.shape and bool((c_cpu == c_gpu).all())
    same_w = w_cpu.shape == w_gpu.shape and np.allclose(w_cpu, w_gpu, atol=0, rtol=0)
    allsame &= same_c and same_w
    diff = int((c_cpu != c_gpu).sum()) if c_cpu.shape == c_gpu.shape else -1
    print(f"  neurons {n:3d} metas {m:2d} fanout {f:2d} gs {gs:3d}: "
          f"conn identical {same_c} (differing ints {diff}), weights identical {same_w}")
print("VERDICT:", "GPU SORT == CPU SORT, BIT-IDENTICAL" if allsame else "ORDERING DIFFERS")
