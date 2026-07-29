#!/usr/bin/env python3
"""Regression test for the initialize_neurons STDP-table heap-buffer-overflow.

initialize_neurons allocated nm_to_ltp/nm_to_ltd (per-NEURON-meta STDP table
offsets) sized by n_synapse_metas but wrote them at the neuron-meta index
i in [0, n_neuron_metas). When n_neuron_metas > n_synapse_metas this wrote past
the buffer -> heap corruption that surfaced later as a segfault. Here we build a
SpikingNet with MORE neuron metas than synapse metas (5 vs 2 — the exact crashing
shape), grow explicit connections, run a few ticks, and assert it all completes
with the expected spike. Before the fix this construction corrupted the heap and
the run crashed; after the fix it runs cleanly.
"""
import torch

from spiky.util.synapse_growth import SynapseGrowthEngine
from spiky.spnet.spnet import SpikingNet, SynapseMeta, NeuronMeta, NeuronDataType


def _lf(thr, ltp, ltd):
    # leak-free integrator (v'=I) with DISTINCT stdp params per meta so the
    # nm_to_ltp/nm_to_ltd loop writes distinct slots for every neuron-meta index.
    return NeuronMeta(neuron_type=0, cf_2=0.0, cf_1=0.0, cf_0=0.0, a=0.0, b=0.0, c=0.0, d=0.0,
                      spike_threshold=float(thr), ltp_max=float(ltp), ltd_max=float(ltd))


def test_initialize_neurons_stdp_oob(device='cpu', summation_dtype=torch.float32, seed=1):
    if device != 'cpu' and str(device) != 'cpu':
        return None
    NM = 5                                   # neuron metas
    NS = 2                                    # synapse metas  -> NM(5) > NS(2): the OOB shape
    per = 4                                   # neurons per meta
    neuron_metas = [_lf(0.5, 1.0 + 0.1 * i, 1.2 + 0.1 * i) for i in range(NM)]
    synapse_metas = [SynapseMeta(learning_rate=0.0, min_delay=1, max_delay=1, min_weight=-10.0,
                                 max_weight=10.0, initial_weight=5.0, _forward_group_size=4, _backward_group_size=4),
                     SynapseMeta(learning_rate=0.0, min_delay=2, max_delay=2, min_weight=-10.0,
                                 max_weight=10.0, initial_weight=5.0, _forward_group_size=4, _backward_group_size=4)]
    spnet = SpikingNet(synapse_metas=synapse_metas, neuron_metas=neuron_metas,
                       neuron_counts=[per] * NM, summation_dtype=summation_dtype)
    spnet.to_device(device)
    ids_by = [spnet.get_neuron_ids_by_meta(i) for i in range(NM)]

    # one explicit edge across metas: neuron 0 of meta0 -> neuron 0 of meta1, delay 1, weight 5
    src, tgt = int(ids_by[0][0]), int(ids_by[1][0])
    ge = SynapseGrowthEngine(device=device, synapse_group_size=4, max_groups_in_buffer=256)
    for i in range(NM):
        ge.register_neuron_type(max_synapses=16, growth_command_list=[])
    for i in range(NM):
        ids = ids_by[i]
        ge.add_neurons(neuron_type_index=i, identifiers=ids,
                       coordinates=torch.stack([torch.arange(len(ids)).float(), torch.zeros(len(ids)),
                                                torch.full((len(ids),), float(i))], dim=1))
    triples = torch.tensor([[0, src, tgt]], dtype=torch.int32, device=device)
    chunk = ge._grow_explicit(triples, seed)
    spnet.add_connections(chunk, seed)
    spnet.compile(shuffle_synapses_random_seed=None)
    spnet.to_device(device)

    # drive the source at tick 1; the target (leak-free, thr 0.5, weight 5) should fire
    N_TICKS = 8
    S = torch.zeros((1, N_TICKS, 1), dtype=torch.int32, device=device)
    V = torch.zeros((1, N_TICKS, 1), dtype=torch.float32, device=device)
    S[0, 1, 0] = src
    V[0, 1, 0] = 50.0
    spnet.process_ticks(n_ticks_to_process=N_TICKS, batch_size=1, n_input_ticks=N_TICKS,
                        input_values=V, do_train=False, sparse_input=S, do_reset_context=True)
    out = spnet.export_neuron_data(torch.tensor([tgt], dtype=torch.int32, device=device),
                                   1, NeuronDataType.Spike, 0, N_TICKS - 1).view(N_TICKS)
    fired = bool((out > 0.5).any().item())
    ok = (spnet.n_synapses() == 1) and fired
    if ok:
        print("✅ n_neuron_metas(5) > n_synapse_metas(2): construct + grow + run OK, target fired — no OOB")
        print("🎉 initialize_neurons STDP-table OOB regression test passed!")
    else:
        print(f"❌ n_synapses={spnet.n_synapses()} target_fired={fired}")
    return ok


def main():
    return 0 if test_initialize_neurons_stdp_oob('cpu', torch.float32, 1) else 1


if __name__ == "__main__":
    exit(main())
