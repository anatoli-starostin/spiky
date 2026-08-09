"""Shrink the engine-group-size-128 build crash to the smallest topology that still fails.

Config held fixed (the spnet.ipynb-derived one): engine synapse_group_size=128, excitatory
metas fwd/bwd=8, inhibitory metas fwd/bwd=128. Vary only the topology and count crashes.
One fresh process per cell -- an illegal address poisons the context.
"""
import argparse
import subprocess
import sys

import numpy as np


def attempt(n_neurons, n_meta, fanout, engine_gs, exc_gs, inh_gs):
    import torch
    from spiky.spnet.spnet import SpikingNet, SynapseMeta, NeuronMeta
    from spiky.util.synapse_growth import SynapseGrowthEngine
    from spiky.util.chunk_of_connections import ChunkOfConnections

    metas = [SynapseMeta(learning_rate=0.1, min_delay=d + 1, max_delay=d + 1,
                         initial_weight=0.0, min_weight=0.0, max_weight=45.0,
                         initial_noise_level=0.0, weight_decay=0.9, weight_scaling_cf=0.0,
                         _forward_group_size=exc_gs, _backward_group_size=exc_gs)
             for d in range(n_meta)]
    sp = SpikingNet(synapse_metas=metas,
                    neuron_metas=[NeuronMeta(neuron_type=0, a=0.02, d=8.0)],
                    neuron_counts=[n_neurons], initial_synapse_capacity=1 << 20,
                    summation_dtype=torch.float32)
    sp.to_device("cuda")
    ids = sp.get_neuron_ids_by_meta(0).cpu().numpy()

    # every source spans all metas: synapse j of source s goes to meta j % n_meta
    tri = []
    for si in range(n_neurons):
        for j in range(fanout):
            ti = (si + 1 + j) % n_neurons
            if ti != si:
                tri.append([j % n_meta, int(ids[si]), int(ids[ti])])
    tri = np.array(tri, np.int32)
    w = np.full(len(tri), 6.0, np.float32)

    ge = SynapseGrowthEngine(device="cuda", synapse_group_size=engine_gs,
                             max_groups_in_buffer=max(4096, 8 * (len(tri) + n_neurons)))
    ge.register_neuron_type(max_synapses=8 * n_neurons, growth_command_list=[])
    t = torch.tensor(ids, dtype=torch.int32)
    ge.add_neurons(neuron_type_index=0, identifiers=t,
                   coordinates=torch.stack([torch.arange(t.numel()).float(),
                                            torch.zeros(t.numel()),
                                            torch.zeros(t.numel())], 1))
    tri_t = torch.tensor(tri, dtype=torch.int32, device="cuda")
    w_t = torch.tensor(w, device="cuda")
    from es_harness import group_aligned_weights
    conn = ge._grow_explicit(tri_t, 1).get_connections()
    chunk = ChunkOfConnections(conn, engine_gs,
                               weights=group_aligned_weights(conn, tri_t, w_t, engine_gs))
    sp.add_connections(chunk, 1)
    chunk.recycle()
    sp.compile(shuffle_synapses_random_seed=None)
    torch.cuda.synchronize()
    print(f"BUILD-OK synapses={len(tri)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--child", action="store_true")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--metas", type=int, default=20)
    ap.add_argument("--fanout", type=int, default=20)
    ap.add_argument("--engine-gs", type=int, default=128)
    ap.add_argument("--exc-gs", type=int, default=8)
    ap.add_argument("--inh-gs", type=int, default=128)
    ap.add_argument("--tries", type=int, default=3)
    a = ap.parse_args()
    if a.child:
        attempt(a.n, a.metas, a.fanout, a.engine_gs, a.exc_gs, a.inh_gs)
        sys.exit(0)

    # S sublists per source = min(metas, fanout). 64 neurons keeps every (src,tgt) unique,
    # so duplicates cannot be confounding the result.
    CASES = [(64, 2, 2)]        # the minimized 100% crasher -- confirm with --tries 10
    for (n, m, f) in CASES:
        crashes = 0
        detail = ""
        nsyn = "?"
        for _ in range(a.tries):
            r = subprocess.run(
                [sys.executable, __file__, "--child", "--n", str(n), "--metas", str(m),
                 "--fanout", str(f), "--engine-gs", str(a.engine_gs),
                 "--exc-gs", str(a.exc_gs), "--inh-gs", str(a.inh_gs)],
                capture_output=True, text=True, timeout=180)
            txt = r.stdout + r.stderr
            if "BUILD-OK" in txt:
                nsyn = txt.split("synapses=")[1].split()[0]
            else:
                crashes += 1
                e = [l.strip() for l in txt.splitlines() if "Error" in l]
                if e and not detail:
                    detail = e[-1][-60:]
        print(f"  neurons {n:3d} metas {m:2d} fanout {f:2d} (syn~{n*f:5d}) -> "
              f"CRASH {crashes}/{a.tries}  {detail}", flush=True)
