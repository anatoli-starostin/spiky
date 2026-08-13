"""Dump shift_to_next_group values and test the truncating-division hypothesis.

The kernel advances the two pointers by DIFFERENT computations:
    header_ptr    += shift                                          (exact, in ints)
    input_weights += (shift / ConnectionsBlockIntSize(gs)) * gs     (integer division)
If shift is ever not an exact multiple of ConnectionsBlockIntSize(gs) = 4 + 2*gs, the second
truncates and the two pointers desynchronise. Measure it directly.
"""
import argparse

import numpy as np
import torch
from spiky.spnet.spnet import NeuronMeta, SpikingNet, SynapseMeta
from spiky.util.synapse_growth import SynapseGrowthEngine

ap = argparse.ArgumentParser()
ap.add_argument("--gs", type=int, default=128)
ap.add_argument("--metas", type=int, default=6)
ap.add_argument("--fanout", type=int, default=12)
ap.add_argument("--n", type=int, default=64)
a = ap.parse_args()

metas = [SynapseMeta(learning_rate=0.1, min_delay=d + 1, max_delay=d + 1,
                     initial_weight=0.0, min_weight=0.0, max_weight=45.0,
                     initial_noise_level=0.0, weight_decay=0.9, weight_scaling_cf=0.0,
                     _forward_group_size=8, _backward_group_size=8)
         for d in range(a.metas)]
sp = SpikingNet(synapse_metas=metas,
                neuron_metas=[NeuronMeta(neuron_type=0, a=0.02, d=8.0)],
                neuron_counts=[a.n], initial_synapse_capacity=1 << 20,
                summation_dtype=torch.float32)
sp.to_device("cuda")
ids = sp.get_neuron_ids_by_meta(0).cpu().numpy()
tri = np.array([[j % a.metas, int(ids[s]), int(ids[(s + 1 + j) % a.n])]
                for s in range(a.n) for j in range(a.fanout)
                if (s + 1 + j) % a.n != s], np.int32)

ge = SynapseGrowthEngine(device="cuda", synapse_group_size=a.gs,
                         max_groups_in_buffer=max(4096, 8 * (len(tri) + a.n)))
ge.register_neuron_type(max_synapses=8 * a.n, growth_command_list=[])
nid = torch.tensor(ids, dtype=torch.int32)
ge.add_neurons(neuron_type_index=0, identifiers=nid,
               coordinates=torch.stack([torch.arange(nid.numel()).float(),
                                        torch.zeros(nid.numel()),
                                        torch.zeros(nid.numel())], dim=1))
conn = ge._grow_explicit(torch.tensor(tri, dtype=torch.int32, device="cuda"), 1
                         ).get_connections()

BLOCK = 4 + 2 * a.gs                      # == ConnectionsBlockIntSize(gs)
buf = np.asarray(conn.cpu().tolist(), dtype=np.int64).reshape(-1, BLOCK)
n_blocks = buf.shape[0]
shifts = buf[:, 3]
nz = shifts[shifts != 0]
print(f"gs={a.gs}  BLOCK=ConnectionsBlockIntSize={BLOCK} ints  blocks={n_blocks} "
      f"buffer={conn.numel()} ints")
print(f"non-zero shifts: {nz.size}")
if nz.size:
    exact = int((nz % BLOCK == 0).sum())
    print(f"  shift % BLOCK == 0 (exact multiples): {exact}/{nz.size}"
          f"   -> {'ALL EXACT' if exact == nz.size else 'TRUNCATION HAPPENS'}")
    print(f"  distinct shift values (first 10): {sorted(set(int(v) for v in nz))[:10]}")
    print(f"  as block deltas: {sorted(set(int(v) // BLOCK for v in nz))[:10]}")
    rem = sorted(set(int(v) % BLOCK for v in nz))
    print(f"  distinct remainders mod BLOCK: {rem[:10]}")
    # what the two pointers do for the first few chained blocks
    print("  block -> shift, header advance (ints), weight advance (slots), "
          "implied block delta")
    idx = np.nonzero(shifts != 0)[0][:6]
    for b in idx:
        s = int(shifts[b])
        print(f"    b={b:4d} shift={s:8d}  header:{s:+8d} ints  "
              f"weights:{(s // BLOCK) * a.gs:+8d} slots  "
              f"block delta={s / BLOCK:+.3f}")
print(f"roots (source>0): {int((buf[:, 0] > 0).sum())}")

# Do the chains stay inside the buffer? A chain target outside [0, n_blocks) makes the kernel
# read a garbage header, whose garbage shift then sends input_weights somewhere wild.
print("\nCHAIN RANGE CHECK")
oob = cyc = ok = 0
worst = []
for b in range(n_blocks):
    if buf[b, 0] <= 0:
        continue                                    # not a root
    cur, seen, steps = b, {b}, 0
    while True:
        s = int(buf[cur, 3])
        if s == 0:
            ok += 1
            break
        nxt_int = cur * BLOCK + s
        nxt = nxt_int // BLOCK
        if not (0 <= nxt < n_blocks):
            oob += 1
            if len(worst) < 5:
                worst.append((b, cur, s, nxt))
            break
        if nxt in seen:
            cyc += 1
            break
        seen.add(nxt)
        cur = nxt
        steps += 1
        if steps > n_blocks:
            cyc += 1
            break
print(f"  chains terminating cleanly : {ok}")
print(f"  chains leaving the buffer  : {oob}   <-- reads a garbage header if >0")
print(f"  chains cycling             : {cyc}")
for root, cur, s, nxt in worst:
    print(f"    root b={root} at block {cur} shift={s} -> block {nxt} "
          f"(valid range 0..{n_blocks - 1})")
