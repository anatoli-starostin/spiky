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
from spiky.util.chunk_of_connections import ChunkOfConnections
from spiky.spnet.spnet import SpikingNet, SynapseMeta, NeuronMeta, NeuronDataType


def _group_aligned_weights_vec(conn, triples, weights, gs):
    """Weights aligned to the grown connection groups, bound to each edge by POSITION.

    `conn` is the 1-D int32 buffer of blocks [src, meta, n_tgt, shift, (meta_i, tgt_i) x gs].
    `_grow_explicit` sorts the input triples by (source, then meta) and the native grow
    emits the body-pair targets in exactly that sorted order; every real (target != 0)
    body slot, scanned block-major then slot-minor, corresponds 1:1 to a sorted triple.
    So we replay that same sort on the weights and drop them onto the real slots in order.

    This binds a weight to its edge by POSITION — never by a lossy (src, tgt) key and never
    by re-deriving a group's source via carry-forward. The old key/carry approach mis-assigned
    ~38% of weights (some zeroed, some stealing another edge's weight) and made a genome's
    effective weights depend on its packmates; position binding is exact and packing-invariant,
    and it also handles parallel (src, tgt) edges (each occupies its own slot)."""
    # replay _grow_explicit's internal sort: by source (col 1), then by meta (col 0), both stable
    i1 = torch.argsort(triples[:, 1], stable=True)
    i2 = torch.argsort(triples[i1][:, 0], stable=True)
    ws = weights[i1][i2].flatten().to(torch.float32)
    block = 4 + 2 * gs
    n_blocks = conn.numel() // block
    b = conn.view(n_blocks, block)
    tgt = torch.stack([b[:, 4 + 2 * j + 1] for j in range(gs)], dim=1).reshape(-1)  # block-major, slot-minor
    wbuf = torch.zeros(n_blocks * gs, dtype=torch.float32, device=conn.device)
    mask = tgt != 0
    n_slots = int(mask.sum().item())
    assert n_slots == ws.numel(), \
        "grown body slots (%d) != number of edges (%d); position binding broken" % (n_slots, ws.numel())
    wbuf[mask] = ws
    return wbuf

WQ = 4


def leakfree_meta(threshold):
    """v' = I exactly: cf_2=cf_1=cf_0=0, a=b=c=d=0 -> u stays 0; fire at threshold, reset to c=0."""
    return NeuronMeta(neuron_type=0, cf_2=0.0, cf_1=0.0, cf_0=0.0,
                      a=0.0, b=0.0, c=0.0, d=0.0, spike_threshold=float(threshold))


def stock_meta(_threshold=30.0):
    return NeuronMeta(neuron_type=0)  # a=.02 b=.2 c=-65 d=8 cf=.04/5/140 thr=30


def _build_packed_impl(genomes, neuron_metas, device='cpu', fast=True):
    """neuron_metas: list of NeuronMeta; genome['node_meta'][local] indexes it.
    device: 'cpu' (default, unchanged) or 'cuda[:i]' to build the packed net on GPU.
    fast=True vectorizes the per-edge weight alignment (default); fast=False uses the
    growth engine's reference Python weight path. Both produce an identical packed net."""
    n_cand = len(genomes)
    n_nodes = max(g["n_nodes"] for g in genomes)
    n_metas = len(neuron_metas)
    n_edges = sum(len(g["synapses"]) for g in genomes)

    # ---- FIXED universal DELAY palette: one SynapseMeta per delay d in 1..255, all
    # other fields constant across every meta and every net (never mutated). The per-edge
    # WEIGHT is NOT folded into the meta (initial_weight=0); it is applied per-synapse via
    # the explicit weights= buffer below (-> SynapseInfo.weight). So distinct weights no
    # longer create metas: a synapse of delay d references meta index (d-1), and the whole
    # table is a constant 255 metas shared across all packed genomes (register dedups it).
    N_DELAYS = 255
    syn_metas = [SynapseMeta(
        learning_rate=0.0, min_delay=d, max_delay=d,
        min_weight=-1.0e4, max_weight=1.0e4, initial_weight=0.0,
        initial_noise_level=0.0, weight_decay=0.9, weight_scaling_cf=0.0,
        _forward_group_size=2, _backward_group_size=2) for d in range(1, N_DELAYS + 1)]

    # node global id = ids_by_meta[node_meta][slot]; slot = running count within that meta
    # over nodes in (cand-major, local) order. SAME layout in both the vectorized and the
    # reference path, so the packed nets are identical.
    if fast:
        nm_t = torch.tensor([m for g in genomes for m in g["node_meta"]], dtype=torch.long)
        counts = [max(1, int(x)) for x in torch.bincount(nm_t, minlength=n_metas).tolist()]
    else:
        counts = [0] * n_metas
        slot_of = {}  # (cand, local) -> (meta_idx, slot_within_meta)
        for c, g in enumerate(genomes):
            for local in range(g["n_nodes"]):
                mi = g["node_meta"][local]
                slot_of[(c, local)] = (mi, counts[mi])
                counts[mi] += 1
        counts = [max(1, x) for x in counts]

    spnet = SpikingNet(synapse_metas=syn_metas, neuron_metas=neuron_metas,
                       neuron_counts=counts,
                       initial_synapse_capacity=max(1024, 2 * n_edges))
    spnet.to_device(device)
    ids_by_meta = [spnet.get_neuron_ids_by_meta(i) for i in range(n_metas)]  # tensors (add_neurons)
    ids_cpu = [t.cpu().tolist() for t in ids_by_meta]                        # sync-free gid lookups

    if fast:
        # ---- VECTORIZED global-id table + per-edge triples/weights (no per-edge Python) ----
        node_base = [0]
        for g in genomes:
            node_base.append(node_base[-1] + g["n_nodes"])
        gid_arr = torch.empty(node_base[-1], dtype=torch.long)  # flat node idx -> global id
        for m in range(n_metas):
            mask = nm_t == m
            cnt = int(mask.sum())
            if cnt:
                gid_arr[mask] = torch.tensor(ids_cpu[m][:cnt], dtype=torch.long)
        gid_list = gid_arr.tolist()

        def gid(c, local):
            return gid_list[node_base[c] + local]

        edges = [(c, s, t, w, d) for c, g in enumerate(genomes) for (s, t, w, d) in g["synapses"]]
        if edges:
            ec, es, et, ew, ed = zip(*edges)
            base_t = torch.tensor(node_base, dtype=torch.long)
            ec_t = torch.tensor(ec, dtype=torch.long)
            srcf = base_t[ec_t] + torch.tensor(es, dtype=torch.long)   # flat node idx of source
            tgtf = base_t[ec_t] + torch.tensor(et, dtype=torch.long)   # flat node idx of target
            meta = torch.tensor(ed, dtype=torch.long).clamp(1, N_DELAYS) - 1
            triples_t = torch.stack([meta, gid_arr[srcf], gid_arr[tgtf]], dim=1).to(torch.int32).to(device)
            weights_t = torch.tensor(ew, dtype=torch.float32, device=device)
        else:
            triples_t = torch.zeros((0, 3), dtype=torch.int32, device=device)
            weights_t = torch.zeros((0,), dtype=torch.float32, device=device)
    else:
        def gid(c, local):
            mi, slot = slot_of[(c, local)]
            return ids_cpu[mi][slot]

        plan = []  # (meta_idx, (cand,src), (cand,tgt), weight)
        for c, g in enumerate(genomes):
            for (s, t, w, d) in g["synapses"]:
                dd = min(max(int(d), 1), N_DELAYS)          # clamp delay to [1,255]
                plan.append((dd - 1, (c, s), (c, t), float(w)))
        triples = [[mi, gid(*src), gid(*tgt)] for (mi, src, tgt, w) in plan]
        triples_t = torch.tensor(triples, dtype=torch.int32, device=device)
        weights_t = torch.tensor([w for (mi, src, tgt, w) in plan], dtype=torch.float32, device=device)

    total_neurons = sum(counts)
    ge = SynapseGrowthEngine(device=device, synapse_group_size=2,
                             max_groups_in_buffer=max(4096, 8 * (n_edges + total_neurons)))
    for i in range(len(neuron_metas)):
        ge.register_neuron_type(max_synapses=8 * n_nodes, growth_command_list=[])
    for i in range(len(neuron_metas)):
        ids = ids_by_meta[i]
        coords = torch.stack([torch.arange(len(ids)).float(),
                              torch.zeros(len(ids)), torch.full((len(ids),), float(i))], dim=1)
        ge.add_neurons(neuron_type_index=i, identifiers=ids, coordinates=coords)

    # Grow WITHOUT the engine's (src,tgt)-keyed weights path, then attach each edge's
    # weight by POSITION in the grow's sorted order (see _group_aligned_weights_vec).
    # fast/ref now share this one correct path; `fast` is kept only for call compatibility.
    chunk = ge._grow_explicit(triples_t, 1)
    conn = chunk.get_connections()
    wbuf = _group_aligned_weights_vec(conn, triples_t, weights_t, ge._synapse_group_size)
    chunk = ChunkOfConnections(conn, ge._synapse_group_size, weights=wbuf)
    spnet.add_connections(chunk, 1)
    chunk.recycle()
    # Shuffle OFF: keep synapse order stable/deterministic (no reshuffle at compile).
    spnet.compile(shuffle_synapses_random_seed=None)
    spnet.to_device(device)
    return {"spnet": spnet, "gid": gid, "n_nodes": n_nodes, "n_cand": n_cand,
            "n_synapses": spnet.n_synapses(), "n_syn_metas": len(syn_metas)}


def build_packed(genomes, neuron_metas, device='cpu'):
    """Pack genomes into one SpikingNet (vectorized per-edge weight alignment)."""
    return _build_packed_impl(genomes, neuron_metas, device=device, fast=True)


def build_packed_ref(genomes, neuron_metas, device='cpu'):
    """Reference build (growth engine's Python weight path) — equivalence baseline."""
    return _build_packed_impl(genomes, neuron_metas, device=device, fast=False)


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
