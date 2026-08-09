"""Does the GPU chain sort still produce correctly META-ORDERED chains after the atomicCAS?

Comparing raw connections buffers CPU-vs-GPU is the wrong test: block ALLOCATION order is
backend-dependent by design (the GPU grows blocks in parallel), so the buffers differ even
for a single meta where no sorting happens at all. What the sort actually promises is
ChunkOfConnections spec rule 7 -- "Groups in each list are sorted by the synapse_meta_index
header field" -- so check that invariant directly on GPU-built chunks, plus rule 5 (one list
per source) and edge-set equality against the input.
"""
import numpy as np
import torch
from spiky.spnet.spnet import NeuronMeta, SpikingNet, SynapseMeta
from spiky.util.chunk_of_connections import ChunkOfConnectionsValidator
from spiky.util.synapse_growth import SynapseGrowthEngine


def build(device, n_neurons, n_metas, fanout, engine_gs, seed=1):
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
    sp.add_connections(chunk, seed)
    sp.compile(shuffle_synapses_random_seed=None)
    if device == "cuda":
        torch.cuda.synchronize()
    return sp, chunk, tri, w, ids


def meta_order_ok(conn, gs):
    """Rule 7: walking each chain from its root, synapse_meta_index must be non-decreasing."""
    block = 4 + 2 * gs
    buf = np.asarray(conn.cpu().tolist(), np.int64).reshape(-1, block)
    n = buf.shape[0]
    bad = chains = 0
    for b in range(n):
        if buf[b, 0] <= 0:
            continue
        chains += 1
        cur, prev_meta, steps = b, -1, 0
        while True:
            if buf[cur, 2] > 0:                      # sublist head carries the count
                if buf[cur, 1] < prev_meta:
                    bad += 1
                    break
                prev_meta = buf[cur, 1]
            s = int(buf[cur, 3])
            if s == 0 or steps > n:
                break
            cur = (cur * block + s) // block
            if not 0 <= cur < n:
                bad += 1
                break
            steps += 1
    return chains, bad


CASES = [(64, 1, 8, 2), (64, 4, 8, 2), (64, 20, 20, 2),
         (64, 4, 8, 128), (64, 20, 20, 128), (256, 20, 20, 128)]
allok = True
for (n, m, f, gs) in CASES:
    for dev in ("cuda", "cpu"):
        sp, chunk, tri, w, ids = build(dev, n, m, f, gs)
        conn = chunk.get_connections()
        chains, bad = meta_order_ok(conn, gs)
        ok, errs = ChunkOfConnectionsValidator(chunk).validate_all()
        # rule 9 (targets sorted within group) is expected to fail: do_sort_by_target_id=False
        errs = [e for e in errs if "Target neuron IDs not sorted" not in e]
        # edge set must survive intact
        all_ids = torch.tensor(ids, dtype=torch.int32, device=dev)
        cnt = sp.count_synapses(all_ids, True)
        b = [torch.zeros(cnt, dtype=t, device=dev) for t in
             (torch.int32, torch.int32, torch.float32, torch.int32, torch.int32)]
        sp.export_synapses(all_ids, b[0], b[1], b[2], b[3], b[4], True)
        es, _, ew, _, et = (x.cpu().numpy() for x in b)
        got = {(int(s), int(t)): float(v) for s, t, v in zip(es, et, ew)}
        wrong = sum(1 for (mm, s, t), v in zip(tri, w)
                    if abs(got.get((int(s), int(t)), -1e9) - float(v)) > 1e-3)
        good = bad == 0 and not errs and wrong == 0
        allok &= good
        print(f"  {dev:4s} neurons {n:3d} metas {m:2d} fanout {f:2d} gs {gs:3d}: "
              f"chains {chains:4d} meta-order-violations {bad}  edges-wrong {wrong}  "
              f"other-validator-errors {errs if errs else 0}  {'OK' if good else 'FAIL'}")
        del sp, chunk
        if dev == "cuda":
            torch.cuda.empty_cache()
print("VERDICT:", "SORT INVARIANT HOLDS ON BOTH BACKENDS" if allok else "VIOLATION")
