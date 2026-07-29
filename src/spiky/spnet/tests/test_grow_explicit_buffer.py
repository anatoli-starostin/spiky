#!/usr/bin/env python3
"""Regression test for _grow_explicit connections-buffer under-sizing.

With a SINGLE shared SynapseMeta and many distinct (source) sublists, the old
sizing  ceil(n_triples/gs) + (n_metas-1)*(n_sources-1)  collapses to ceil(n/gs)
(second term = 0), which is far fewer groups than the one-per-(meta,source)-sublist
the native grow actually writes -> buffer overflow -> segfault. This is exactly
the per-edge-weights (#78) single-meta path. Here we grow 16 sources -> 16 targets
(16 sublists, group_size 8 -> old formula asks for only 2 groups) with distinct
per-edge weights, and assert: (a) no crash, (b) each edge carries its own weight.
"""
import torch

from spiky.util.synapse_growth import SynapseGrowthEngine
from spiky.spnet.spnet import SpikingNet, SynapseMeta, NeuronMeta


def test_grow_explicit_buffer(device='cpu', summation_dtype=torch.float32, seed=1):
    if device != 'cpu' and str(device) != 'cpu':
        return None
    NS = 16                          # 16 sources, each -> its own target = 16 sublists
    N = 2 * NS
    GS = 8                           # old formula: ceil(16/8)=2 groups << 16 needed
    edges = [(i, NS + i) for i in range(NS)]
    weights = [round(0.5 + 0.37 * i - (7.0 if i % 3 == 0 else 0.0), 3) for i in range(NS)]  # distinct, mixed sign
    metas = [SynapseMeta(learning_rate=0.0, min_delay=1, max_delay=1, min_weight=-50.0,
                         max_weight=50.0, initial_weight=0.0, _forward_group_size=8, _backward_group_size=8)]
    spnet = SpikingNet(synapse_metas=metas, neuron_metas=[NeuronMeta(neuron_type=0)],
                       neuron_counts=[N], summation_dtype=summation_dtype)
    spnet.to_device(device)
    ids = spnet.get_neuron_ids_by_meta(0)
    ge = SynapseGrowthEngine(device=device, synapse_group_size=GS, max_groups_in_buffer=512)
    ge.register_neuron_type(max_synapses=64, growth_command_list=[])
    coords = torch.stack([torch.arange(N).float(), torch.zeros(N), torch.zeros(N)], dim=1)
    ge.add_neurons(neuron_type_index=0, identifiers=ids, coordinates=coords)
    triples = torch.tensor([[0, int(ids[s]), int(ids[t])] for (s, t) in edges], dtype=torch.int32, device=device)
    wt = torch.tensor(weights, dtype=torch.float32, device=device)

    # This call SEGFAULTED before the fix (undersized connections buffer).
    chunk = ge._grow_explicit(triples, seed, weights=wt)
    spnet.add_connections(chunk, seed)
    spnet.compile(shuffle_synapses_random_seed=None)
    spnet.to_device(device)

    n = spnet.n_synapses()
    ok = (n == NS)
    if not ok:
        print(f"❌ expected {NS} synapses, got {n}")
    exp = {k: torch.zeros([n], dtype=(torch.float32 if k == 'weights' else torch.int32), device=device)
           for k in ['source_ids', 'synapse_metas', 'weights', 'delays', 'target_ids']}
    spnet.export_synapses(spnet.get_all_neuron_ids(), exp['source_ids'], exp['synapse_metas'],
                          exp['weights'], exp['delays'], exp['target_ids'], forward_or_backward=True)
    got = {(int(exp['source_ids'][i]), int(exp['target_ids'][i])): float(exp['weights'][i]) for i in range(n)}
    for (s, t), w in zip(edges, weights):
        g = got.get((int(ids[s]), int(ids[t])))
        if g is None or abs(g - w) > 1e-5:
            print(f"❌ edge {s}->{t}: expected {w}, got {g}")
            ok = False
    if ok:
        print(f"✅ grew {NS} sublists under 1 shared meta with distinct per-edge weights — "
              f"no crash, all weights correct")
        print("🎉 grow_explicit buffer-sizing regression test passed!")
    return ok


def main():
    return 0 if test_grow_explicit_buffer('cpu', torch.float32, 1) else 1


if __name__ == "__main__":
    exit(main())
