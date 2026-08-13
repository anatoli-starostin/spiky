"""Minimal reproducer: cudaErrorIllegalAddress in create_forward_groups at build time.

    PYTHONPATH=<spiky>/native/spiky python repro_crash.py

Crashes 10/10 in ~3 seconds. 64 neurons, 4 synapse metas, 256 synapses -- no STDP, no
episodes, no steady-state loop. Just build the net.

THE TRIGGER is the GROWTH ENGINE's synapse_group_size=128 (from spnet.ipynb's
((M+31)//32)*32 with M=100). Set it back to 2 and this same topology builds fine, which is
the one-line A/B at the bottom of this file.

WHAT LOOKS WRONG. The engine's synapse_group_size and the metas' _forward_group_size mean
different things and are not reconciled anywhere:
  * engine synapse_group_size sizes the INPUT connections buffer -- each block is
    (4 + 2*gs) int32 and holds up to gs (meta, target) pairs, per the ChunkOfConnections
    spec.
  * meta _forward_group_size sizes the OUTPUT forward groups the kernel writes.
`_grow_explicit` sizes the input buffer from the ENGINE value, while create_forward_groups
computes its output capacity per (meta, source) sublist from the META value
(SizeOfMultipleForwardSynapseGroups(..., synapse_meta._forward_group_size, ...)). With
engine=128 and meta=8 those disagree by 16x. Each (meta, source) sublist also gets its own
128-slot block while holding as little as ONE target, so the buffer is mostly padding and
the two sizings drift apart fast.
Consistent with an out-of-bounds write rather than a clean bounds check: the crash is
NON-MONOTONIC in topology size -- 32 neurons/2 metas/fanout 4 crashes 4/4 while 64
neurons/2 metas/fanout 4 (twice the synapses) builds fine. Whether the overflow lands on
mapped memory depends on the allocation layout, which is exactly how a heap overflow behaves.
Note the same kernel ALSO hangs nondeterministically at engine gs=2 (see BUGS.md bug 1);
that is a separate symptom and this script does not reproduce it.
"""
import sys

import numpy as np
import torch
from spiky.spnet.spnet import NeuronMeta, SpikingNet, SynapseMeta
from spiky.util.chunk_of_connections import ChunkOfConnections
from spiky.util.synapse_growth import SynapseGrowthEngine

# ------------------------------------------------------------------ config
ENGINE_GROUP_SIZE = 128     # <-- THE TRIGGER. Set to 2 and the crash disappears.
EXC_META_GROUP_SIZE = 8     # spnet.ipynb excitatory metas
N_NEURONS = 64
N_METAS = 4                 # 4/4 crashes 10/10; the smaller 2/2 (128 synapses) is ~8/10
FANOUT = 4
# --no-weights: build the chunk with weights=None, to separate an overflow in the
# CONNECTION/GROUP structure from one in the weights-buffer sizing or placement.
USE_WEIGHTS = "--no-weights" not in sys.argv

print(f"engine synapse_group_size = {ENGINE_GROUP_SIZE}")
print(f"meta _forward/_backward_group_size = {EXC_META_GROUP_SIZE}")
print(f"topology: {N_NEURONS} neurons, {N_METAS} metas, fanout {FANOUT}")

metas = [SynapseMeta(learning_rate=0.1, min_delay=d + 1, max_delay=d + 1,
                     initial_weight=0.0, min_weight=0.0, max_weight=45.0,
                     initial_noise_level=0.0, weight_decay=0.9, weight_scaling_cf=0.0,
                     _forward_group_size=EXC_META_GROUP_SIZE,
                     _backward_group_size=EXC_META_GROUP_SIZE)
         for d in range(N_METAS)]

spnet = SpikingNet(synapse_metas=metas,
                   neuron_metas=[NeuronMeta(neuron_type=0, a=0.02, d=8.0)],
                   neuron_counts=[N_NEURONS], initial_synapse_capacity=1 << 20,
                   summation_dtype=torch.float32)
spnet.to_device("cuda")
ids = spnet.get_neuron_ids_by_meta(0).cpu().numpy()

# explicit triples (meta_index, source_id, target_id); every source spans every meta,
# and every (src, tgt) pair is unique
triples = np.array([[j % N_METAS, int(ids[s]), int(ids[(s + 1 + j) % N_NEURONS])]
                    for s in range(N_NEURONS) for j in range(FANOUT)
                    if (s + 1 + j) % N_NEURONS != s], dtype=np.int32)
weights = np.full(len(triples), 6.0, dtype=np.float32)
print(f"synapses: {len(triples)}")

engine = SynapseGrowthEngine(device="cuda", synapse_group_size=ENGINE_GROUP_SIZE,
                             max_groups_in_buffer=max(4096, 8 * (len(triples) + N_NEURONS)))
engine.register_neuron_type(max_synapses=8 * N_NEURONS, growth_command_list=[])
nid = torch.tensor(ids, dtype=torch.int32)
engine.add_neurons(neuron_type_index=0, identifiers=nid,
                   coordinates=torch.stack([torch.arange(nid.numel()).float(),
                                            torch.zeros(nid.numel()),
                                            torch.zeros(nid.numel())], dim=1))

tri_t = torch.tensor(triples, dtype=torch.int32, device="cuda")
w_t = torch.tensor(weights, device="cuda")
conn = engine._grow_explicit(tri_t, 1).get_connections()

def group_aligned_weights(conn, triples, weights, gs):
    """Place each explicit edge's weight into the group-aligned buffer matching `conn`.

    Verbatim from es_harness.group_aligned_weights. The owning source of a block must be
    recovered by FOLLOWING THE CHAIN through next_shift (a signed offset to an arbitrary
    block); a cummax/forward-fill over block order is wrong because chained blocks do not
    physically follow their root. Getting this wrong produces a malformed weights buffer
    that crashes the build on its own, in EVERY config -- which masks the bug under test.
    """
    dev = conn.device
    block = 4 + 2 * gs
    n_blocks = conn.numel() // block
    b = conn.view(n_blocks, block).long()
    idx = torch.arange(n_blocks, device=dev)

    cur_src = torch.zeros(n_blocks, dtype=torch.long, device=dev)
    cur = idx[b[:, 0] > 0]
    val = b[cur, 0]
    for _ in range(n_blocks):
        if cur.numel() == 0:
            break
        cur_src[cur] = val
        shift = b[cur, 3]
        alive = shift != 0
        cur, val = ((cur * block + shift) // block)[alive], val[alive]

    srcs, tgts = triples[:, 1].long(), triples[:, 2].long()
    MAXID = int(torch.stack([srcs.max(), tgts.max(), cur_src.max()]).max().item()) + 1
    ekey = srcs * MAXID + tgts
    order = torch.argsort(ekey)
    ekey_s = ekey[order]
    w_s = weights.flatten().to(torch.float32)[order]

    wbuf = torch.zeros(n_blocks * gs, dtype=torch.float32, device=dev)
    zero = torch.zeros(n_blocks, dtype=torch.float32, device=dev)
    for j in range(gs):
        tgt_j = b[:, 4 + 2 * j + 1]
        qkey = cur_src * MAXID + tgt_j
        pos = torch.searchsorted(ekey_s, qkey).clamp(max=ekey_s.numel() - 1)
        hit = (tgt_j > 0) & (ekey_s[pos] == qkey)
        wbuf[idx * gs + j] = torch.where(hit, w_s[pos], zero)
    return wbuf


if not USE_WEIGHTS:
    wbuf = None
elif "--zero-weights" in sys.argv:
    # correctly-sized buffer that group_aligned_weights never touched: if this crashes too,
    # the helper is exonerated and the mere PRESENCE of a weights buffer is the trigger.
    _blk = 4 + 2 * ENGINE_GROUP_SIZE
    wbuf = torch.zeros((conn.numel() // _blk) * ENGINE_GROUP_SIZE,
                       dtype=torch.float32, device="cuda")
else:
    wbuf = group_aligned_weights(conn, tri_t, w_t, ENGINE_GROUP_SIZE)
# --pad-weights N: hand in a buffer N times larger than the layout requires. If the fault is
# an over-READ of the input weights buffer (i.e. the Python sizing is short), slack removes
# it. If it still crashes, the overflow is on the engine's own output-weights allocation and
# no Python-side change can fix it.
if USE_WEIGHTS and "--pad-weights" in sys.argv:
    mult = int(sys.argv[sys.argv.index("--pad-weights") + 1])
    padded = torch.zeros(wbuf.numel() * mult, dtype=wbuf.dtype, device=wbuf.device)
    padded[:wbuf.numel()] = wbuf
    wbuf = padded
    print(f"padded weights buffer x{mult} -> {wbuf.numel()} floats")
print(f"weights buffer: {'chain-following, ' + str(wbuf.numel()) + ' floats' if USE_WEIGHTS else 'NONE'}")

chunk = ChunkOfConnections(conn, ENGINE_GROUP_SIZE, weights=wbuf)
print("calling add_connections -- expect an illegal-address fault in create_forward_groups "
      "(connections_manager.cu:126)")
spnet.add_connections(chunk, 1)                # <-- CRASHES HERE
chunk.recycle()
spnet.compile(shuffle_synapses_random_seed=None)
torch.cuda.synchronize()
print(f"BUILD-OK -- no crash at ENGINE_GROUP_SIZE={ENGINE_GROUP_SIZE} "
      f"(expected only when it is 2)")
