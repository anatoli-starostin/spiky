"""spnet_harness.py — population-parallel fitness on the REAL spiky spnet (CPU).

Packs candidate genomes as DISJOINT sub-networks (neuron-id blocks) into ONE
SpikingNet and runs them in a single process_ticks. Explicit wiring via
_grow_explicit; per-synapse (weight,delay) encoded through a palette of
SynapseMetas. Per-node neuron metas allow per-node thresholds and swapping the
neuron model (stock Izhikevich vs leak-free v'=I via cf_2=cf_1=cf_0=a=b=c=d=0).

genome = dict(
  n_nodes      = int,
  node_meta    = [neuron-meta index per node],       # indexes `neuron_metas`
  synapses     = [(src_local, tgt_local, weight, delay_int), ...],
)

NOTE on _grow_explicit group sizing: it allocates
  ceil(n_triples/group_size) + (n_metas-1)*(n_sources-1)
groups; with a single shared synapse-meta this under-counts and silently drops
synapses. We therefore give every candidate its OWN synapse-meta objects (same
weights, distinct indices) so n_metas is large enough — behaviourally identical,
groups guaranteed. group_size must be even (>=2).
"""
import torch
import spiky_cuda  # noqa: F401
from spiky.util.synapse_growth import SynapseGrowthEngine
from spiky.spnet.spnet import SpikingNet, SynapseMeta, NeuronMeta, NeuronDataType

WQ = 4


def leakfree_meta(threshold):
    """v' = I exactly: cf_2=cf_1=cf_0=0, a=b=c=d=0 -> u stays 0; fire at threshold, reset to c=0."""
    return NeuronMeta(neuron_type=0, cf_2=0.0, cf_1=0.0, cf_0=0.0,
                      a=0.0, b=0.0, c=0.0, d=0.0, spike_threshold=float(threshold))


def stock_meta(_threshold=30.0):
    return NeuronMeta(neuron_type=0)  # a=.02 b=.2 c=-65 d=8 cf=.04/5/140 thr=30


def build_packed(genomes, neuron_metas):
    """neuron_metas: list of NeuronMeta; genome['node_meta'][local] indexes it."""
    n_cand = len(genomes)
    n_nodes = max(g["n_nodes"] for g in genomes)

    # ---- neuron metas: one spnet neuron-meta (= one id block) per provided meta.
    # node global id = meta_block_start[node_meta] + slot, packed per (meta, cand).
    # Simplest correct layout: give EACH node its own contiguous slot in a single
    # neuron-meta only if metas identical. Instead we allocate per neuron-meta a
    # block of n_cand*n_nodes and use node_meta to pick which block + offset.
    # To keep node identity simple we instead assign every node to its meta's block
    # with a global running slot; map (cand,local) -> (meta_idx, slot).
    counts = [0] * len(neuron_metas)
    slot_of = {}  # (cand, local) -> (meta_idx, slot_within_meta)
    for c, g in enumerate(genomes):
        for local in range(g["n_nodes"]):
            mi = g["node_meta"][local]
            slot_of[(c, local)] = (mi, counts[mi])
            counts[mi] += 1
    counts = [max(1, x) for x in counts]

    # ---- synapse-meta palette: (weight,delay) -> meta (deduped; the manager
    # dedups identical metas anyway). The construction's ~40 distinct weights give
    # the _grow_explicit group formula plenty of headroom.
    syn_metas = []
    pal = {}
    def syn_meta_idx(w, d):
        key = (round(float(w), WQ), int(d))
        if key not in pal:
            wv = key[0]
            pal[key] = len(syn_metas)
            syn_metas.append(SynapseMeta(
                learning_rate=0.0, min_delay=int(d), max_delay=int(d),
                min_weight=min(wv, 0.0), max_weight=max(wv, 0.0),
                initial_weight=wv, weight_decay=0.9,
                _forward_group_size=2, _backward_group_size=2))
        return pal[key]

    plan = []  # (syn_meta_idx, (cand,src), (cand,tgt))
    for c, g in enumerate(genomes):
        for (s, t, w, d) in g["synapses"]:
            plan.append((syn_meta_idx(w, d), (c, s), (c, t)))

    spnet = SpikingNet(synapse_metas=syn_metas, neuron_metas=neuron_metas,
                       neuron_counts=counts,
                       initial_synapse_capacity=max(1024, 2 * len(plan)))
    spnet.to_device('cpu')
    ids_by_meta = [spnet.get_neuron_ids_by_meta(i) for i in range(len(neuron_metas))]

    def gid(c, local):
        mi, slot = slot_of[(c, local)]
        return int(ids_by_meta[mi][slot].item())

    triples = [[mi, gid(*src), gid(*tgt)] for (mi, src, tgt) in plan]
    triples_t = torch.tensor(triples, dtype=torch.int32)

    total_neurons = sum(counts)
    ge = SynapseGrowthEngine(device='cpu', synapse_group_size=2,
                             max_groups_in_buffer=max(4096, 8 * (len(plan) + total_neurons)))
    for i in range(len(neuron_metas)):
        ge.register_neuron_type(max_synapses=8 * n_nodes, growth_command_list=[])
    for i in range(len(neuron_metas)):
        ids = ids_by_meta[i]
        coords = torch.stack([torch.arange(len(ids)).float(),
                              torch.zeros(len(ids)), torch.full((len(ids),), float(i))], dim=1)
        ge.add_neurons(neuron_type_index=i, identifiers=ids, coordinates=coords)

    chunk = ge._grow_explicit(triples_t, 1)
    spnet.add_connections(chunk, 1)
    chunk.recycle()
    spnet.compile(shuffle_synapses_random_seed=1)
    spnet.to_device('cpu')
    return {"spnet": spnet, "gid": gid, "n_nodes": n_nodes, "n_cand": n_cand,
            "n_synapses": spnet.n_synapses(), "n_syn_metas": len(syn_metas)}


def run(packed, input_events_per_cand, output_locals, n_ticks):
    spnet = packed["spnet"]; gid = packed["gid"]; n_cand = packed["n_cand"]
    per_tick = [[] for _ in range(n_ticks)]
    for c, evs in enumerate(input_events_per_cand):
        for (local, tick, cur) in evs:
            if 0 <= tick < n_ticks:
                per_tick[tick].append((gid(c, local), float(cur)))
    K = max(1, max(len(pt) for pt in per_tick))
    S = torch.zeros((1, n_ticks, K), dtype=torch.int32)
    V = torch.zeros((1, n_ticks, K), dtype=torch.float32)
    for tk in range(n_ticks):
        for j, (nid, cur) in enumerate(per_tick[tk]):
            S[0, tk, j] = nid; V[0, tk, j] = cur
    spnet.process_ticks(n_ticks_to_process=n_ticks, batch_size=1, n_input_ticks=n_ticks,
                        input_values=V, do_train=False, sparse_input=S,
                        do_record_voltage=False, do_reset_context=True)
    out_ids = torch.tensor([gid(c, ol) for c in range(n_cand) for ol in output_locals],
                           dtype=torch.int32)
    raster = spnet.export_neuron_data(out_ids, 1, NeuronDataType.Spike, 0, n_ticks - 1)
    raster = raster.view(n_cand, len(output_locals), n_ticks)
    res = []
    for c in range(n_cand):
        d = {}
        for oi, ol in enumerate(output_locals):
            nz = torch.nonzero(raster[c, oi] > 0.5).flatten()
            d[ol] = int(nz[0].item()) if nz.numel() else None
        res.append(d)
    return res
