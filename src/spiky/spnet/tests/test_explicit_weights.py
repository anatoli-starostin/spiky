#!/usr/bin/env python3
"""Test explicit per-edge weights in _grow_explicit.

Grows a small net whose synapses ALL share a SINGLE SynapseMeta (initial_weight=0)
but pass several DISTINCT explicit per-edge weights via the new `weights=` argument,
then asserts each edge actually carries its distinct weight (via export_synapses).
Also checks backward compatibility: omitting `weights` falls back to the meta's
initial_weight. Runs on the CPU build.
"""
import torch

from spiky.util.synapse_growth import SynapseGrowthEngine
from spiky.spnet.spnet import SpikingNet, SynapseMeta, NeuronMeta


def _build(device, summation_dtype, edges, weights, seed):
    N = 6
    # ONE shared synapse meta; wide [min,max] so the explicit weights are not clipped.
    metas = [SynapseMeta(learning_rate=0.0, min_delay=1, max_delay=1,
                         min_weight=-10.0, max_weight=10.0, initial_weight=0.0,
                         _forward_group_size=8, _backward_group_size=8)]
    nmetas = [NeuronMeta(neuron_type=0)]
    spnet = SpikingNet(synapse_metas=metas, neuron_metas=nmetas, neuron_counts=[N],
                       summation_dtype=summation_dtype)
    spnet.to_device(device)
    ids = spnet.get_neuron_ids_by_meta(0)
    ge = SynapseGrowthEngine(device=device, synapse_group_size=8, max_groups_in_buffer=256)
    ge.register_neuron_type(max_synapses=64, growth_command_list=[])
    coords = torch.stack([torch.arange(N).float(), torch.zeros(N), torch.zeros(N)], dim=1)
    ge.add_neurons(neuron_type_index=0, identifiers=ids, coordinates=coords)
    triples = torch.tensor([[0, int(ids[s]), int(ids[t])] for (s, t) in edges],
                           dtype=torch.int32, device=device)
    wt = None if weights is None else torch.tensor(weights, dtype=torch.float32, device=device)
    chunk = ge._grow_explicit(triples, seed, weights=wt)
    spnet.add_connections(chunk, seed)
    spnet.compile(shuffle_synapses_random_seed=None)
    spnet.to_device(device)
    n = spnet.n_synapses()
    exp = {k: torch.zeros([n], dtype=(torch.float32 if k == 'weights' else torch.int32), device=device)
           for k in ['source_ids', 'synapse_metas', 'weights', 'delays', 'target_ids']}
    spnet.export_synapses(spnet.get_all_neuron_ids(), exp['source_ids'], exp['synapse_metas'],
                          exp['weights'], exp['delays'], exp['target_ids'], forward_or_backward=True)
    got = {(int(exp['source_ids'][i]), int(exp['target_ids'][i])): float(exp['weights'][i])
           for i in range(n)}
    return ids, got


def test_explicit_per_edge_weights(device='cpu', summation_dtype=torch.float32, seed=1):
    if device != 'cpu' and str(device) != 'cpu':
        return None  # CPU test
    edges = [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5)]
    weights = [1.5, 2.5, -3.0, 4.25, 0.75]           # distinct, incl. negative, one shared meta
    ids, got = _build(device, summation_dtype, edges, weights, seed)
    ok = True
    for (s, t), w in zip(edges, weights):
        g = got.get((int(ids[s]), int(ids[t])))
        if g is None or abs(g - w) > 1e-5:
            print(f"❌ edge {s}->{t}: expected {w}, got {g}")
            ok = False
        else:
            print(f"✅ edge {s}->{t}: weight {g:.4f} (== explicit {w})")
    if len(set(round(v, 4) for v in got.values())) < len(weights):
        print("❌ weights are not distinct — per-edge weights not applied")
        ok = False

    # backward compat: no weights -> every edge seeded from meta.initial_weight (0.0)
    _, got0 = _build(device, summation_dtype, edges, None, seed)
    if any(abs(v - 0.0) > 1e-6 for v in got0.values()):
        print(f"❌ backward-compat: expected all initial_weight=0.0, got {sorted(got0.values())}")
        ok = False
    else:
        print("✅ backward compat: omitting weights -> all edges use meta.initial_weight (0.0)")

    print("🎉 explicit per-edge weights test passed!" if ok else "test FAILED")
    return ok


def main():
    return 0 if test_explicit_per_edge_weights('cpu', torch.float32, 1) else 1


if __name__ == "__main__":
    exit(main())
