"""exp012 Stage 1 SCALE-UP: all 136 distinct comparators on the real engine.

The teacher uses every one of C(17,2) = 136 pairs, so one dual-rail comparator is
instantiated per distinct pair and its rails are wired directly to the 192 (table,bit) slots
(median fan-out 1, max 2 -- no fan-out machinery needed).

Layout, all Dale-clean:
    meta 0  17 input neurons + 1 global GATE
    meta 1  272 rails            anti-leaky, cf_1 = +1/20, v_rest = v_reset = 0, thr 1.0
    meta 2  272 interneurons     inhibitory, leaky
    meta 3  272 memory cells     leaky, tau_mem = 600, thr 1.0

The two places the scale-up can diverge from the verified single primitive, both checked
explicitly below: the 17 INPUT NEURONS are shared across 16 comparators each, and the ONE
gate drives all 272 memory cells.
"""
import argparse
import itertools
import json
import os

import numpy as np
import torch

from tiny_lut_order_detect import T_IN, encode

T1, T2, N_TICKS, GATE_TICK = 200, 240, 320, 200
DE, DI, W_EXC, W_INH = 3, 2, 1.5, -10.0
TAU_M, TAU_MEM, W_MEM, W_GATE = 20.0, 600.0, 0.6, 0.6


def pair_list(Z):
    """the 136 distinct unordered pairs, and the map from (table,bit) -> (pair, orientation)"""
    A, B = Z["anchor_a"], Z["anchor_b"]
    pairs, slot = [], []
    index = {}
    for t in range(A.shape[0]):
        for j in range(A.shape[1]):
            a, b = int(A[t, j]), int(B[t, j])
            key = (min(a, b), max(a, b))
            if key not in index:
                index[key] = len(pairs)
                pairs.append(key)
            # orientation: True if the slot's "a" is the pair's first element
            slot.append((index[key], a == key[0]))
    return pairs, slot


def build(pairs, device="cuda"):
    from spiky.spnet.spnet import LIFNeuronMeta, NeuronMeta, SpikingNet, SynapseMeta
    from spiky.util.synapse_growth import SynapseGrowthEngine
    P = len(pairs)
    dmax = max(DE, DI, 1) + 1
    smetas = [SynapseMeta(learning_rate=0.0, min_delay=d, max_delay=d, initial_weight=0.0,
                          min_weight=-1e4, max_weight=1e4, initial_noise_level=0.0,
                          weight_decay=0.9, weight_scaling_cf=0.0,
                          _forward_group_size=32, _backward_group_size=32)
              for d in range(1, dmax + 1)]
    metas = [LIFNeuronMeta(neuron_type=0, tau=TAU_M, threshold=1.0),
             NeuronMeta(neuron_type=1, cf_2=0.0, cf_1=+1.0 / TAU_M, cf_0=0.0, a=0.0, b=0.0,
                        c=0.0, d=0.0, spike_threshold=1.0),
             LIFNeuronMeta(neuron_type=2, tau=TAU_M, threshold=1.0),
             LIFNeuronMeta(neuron_type=3, tau=TAU_MEM, threshold=1.0)]
    net = SpikingNet(synapse_metas=smetas, neuron_metas=metas,
                     neuron_counts=[18, 2 * P, 2 * P, 2 * P],
                     initial_synapse_capacity=1 << 22, summation_dtype=torch.float32)
    net.to_device(device)
    ids = [net.get_neuron_ids_by_meta(i).cpu().numpy() for i in range(4)]
    inp, gate = ids[0][:17], ids[0][17]
    edges = []
    for p, (a, b) in enumerate(pairs):
        r0, r1 = ids[1][2 * p], ids[1][2 * p + 1]      # rail for "a earlier", "b earlier"
        i0, i1 = ids[2][2 * p], ids[2][2 * p + 1]
        m0, m1 = ids[3][2 * p], ids[3][2 * p + 1]
        edges += [(DE, inp[a], r0, W_EXC), (DE, inp[b], r1, W_EXC),
                  (DI, inp[a], i0, W_EXC), (DI, inp[b], i1, W_EXC),
                  (1, i0, r1, W_INH), (1, i1, r0, W_INH),
                  (1, r0, m0, W_MEM), (1, r1, m1, W_MEM),
                  (1, gate, m0, W_GATE), (1, gate, m1, W_GATE)]
    tri = np.array([[d - 1, s, t] for d, s, t, _ in edges], np.int64)
    wts = np.array([w for *_, w in edges], np.float64)
    ge = SynapseGrowthEngine(device=device, synapse_group_size=32,
                             max_groups_in_buffer=max(1 << 16, 8 * len(tri)))
    for i in range(4):
        ge.register_neuron_type(max_synapses=4096, growth_command_list=[])
    for i in range(4):
        t = torch.tensor(ids[i], dtype=torch.int32)
        ge.add_neurons(neuron_type_index=i, identifiers=t,
                       coordinates=torch.stack([torch.arange(t.numel()).float(),
                                                torch.zeros(t.numel()),
                                                torch.full((t.numel(),), float(i))], 1))
    chunk = ge._grow_explicit(torch.tensor(tri, dtype=torch.int32, device=device), 1,
                              weights=torch.tensor(wts, dtype=torch.float32, device=device))
    net.add_connections(chunk, 1)
    chunk.recycle()
    net.compile(shuffle_synapses_random_seed=None)
    return net, ids, len(tri)


def run(net, ids, ticks, device="cuda"):
    """ticks [B,17] -> memory-cell raster [B, 2P, N_TICKS]"""
    from spiky.spnet.spnet import NeuronDataType
    B = ticks.shape[0]
    va = np.zeros((B, N_TICKS, 18), np.float32)
    for j in range(17):
        va[np.arange(B), ticks[:, j], j] = 1e6
    va[:, GATE_TICK, 17] = 1e6
    sid = torch.as_tensor(np.ascontiguousarray(ids[0], dtype=np.int32),
                          device=device).reshape(1, 1, -1).expand(B, N_TICKS, -1).contiguous()
    net.process_ticks(n_ticks_to_process=N_TICKS, batch_size=B, n_input_ticks=N_TICKS,
                      input_values=torch.as_tensor(va, device=device), sparse_input=sid,
                      do_train=False, do_record_voltage=False, do_reset_context=True,
                      _stdp_period=32)
    oid = torch.as_tensor(np.ascontiguousarray(ids[3], dtype=np.int32), device=device)
    R = net.export_neuron_data(oid, B, NeuronDataType.Spike, 0, N_TICKS - 1)
    return R.reshape(B, -1, N_TICKS).ne(0).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--chunk", type=int, default=256)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    Z = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data",
                             "distill_exp19_100k.npz"))
    x = Z["x_norm"].astype(np.float64)
    ntr = len(x) - 4000
    sel = slice(ntr, ntr + a.n)                       # held-out tail
    xs = x[sel]
    ticks = encode(xs)
    A_, B_ = Z["anchor_a"], Z["anchor_b"]
    truth = (xs[:, A_] > xs[:, B_])                   # [n,32,6]
    pairs, slot = pair_list(Z)
    P = len(pairs)
    print(f"{P} distinct comparators, {len(slot)} (table,bit) slots, {a.n} held-out samples")

    # memory constraint over the WHOLE input range, not the single verified pair
    dmax_gap = GATE_TICK - (ticks.min() + DE + 2)
    lhs = W_MEM * np.exp(-dmax_gap / TAU_MEM) + W_GATE
    print(f"memory constraint: largest decision-to-gate gap {dmax_gap} ticks -> "
          f"w_mem*e^(-D/tau)+w_gate = {lhs:.4f} vs threshold 1.0  "
          f"{'OK' if lhs > 1.0 else 'FAILS'}   margin {lhs - 1.0:+.4f}")

    net, ids, nsyn = build(pairs)
    nneur = 18 + 6 * P
    print(f"assembled: {nneur} neurons ({18} in+gate, {2*P} rails, {2*P} interneurons, "
          f"{2*P} memory), {nsyn} synapses\n")

    R = []
    for i in range(0, a.n, a.chunk):
        R.append(run(net, ids, ticks[i:i + a.chunk]))
    R = np.concatenate(R)
    del net
    torch.cuda.empty_cache()

    cnt = R.sum(-1)                                    # [n, 2P]
    first = np.where(cnt > 0, R.argmax(-1), -1)
    # slot -> which rail means "bit == 1" (x[anchor_a] > x[anchor_b] => anchor_a earlier)
    pred = np.zeros_like(truth)
    for s, (p, a_is_first) in enumerate(slot):
        t, j = divmod(s, 6)
        r_a = 2 * p + (0 if a_is_first else 1)
        pred[:, t, j] = cnt[:, r_a] > 0
    acc = float((pred == truth).mean())
    tie = ticks[:, A_] == ticks[:, B_]
    acc_nt = float((pred[~tie] == truth[~tie]).mean())
    # loser false-fire: for each slot, the rail that should NOT fire
    los = np.zeros((len(xs), len(slot)), bool)
    for s, (p, a_is_first) in enumerate(slot):
        t, j = divmod(s, 6)
        r_a = 2 * p + (0 if a_is_first else 1)
        r_b = 2 * p + (1 if a_is_first else 0)
        los[:, s] = np.where(truth[:, t, j], cnt[:, r_b] > 0, cnt[:, r_a] > 0)
    inwin = (first >= T1) & (first < T2)
    fired = cnt > 0
    oow = float((fired & ~inwin).sum() / max(fired.sum(), 1))
    pre = float((R[:, :, :T1].sum() > 0))
    per_table = [float((pred[:, t, :] == truth[:, t, :]).mean()) for t in range(32)]
    six = np.array([[cnt[:, 2 * slot[t * 6 + j][0] + (0 if slot[t * 6 + j][1] else 1)] > 0
                     for j in range(6)] for t in range(32)])
    print(f"1 WINNER CORRECTNESS   overall {100 * acc:.4f}%   non-tied {100 * acc_nt:.4f}%   "
          f"ties {100 * tie.mean():.3f}%")
    print(f"   per-table accuracy: min {100 * min(per_table):.2f}%  "
          f"median {100 * float(np.median(per_table)):.2f}%  max {100 * max(per_table):.2f}%")
    print(f"2 LOSER FALSE-FIRE     {100 * float(los.mean()):.4f}%  over all "
          f"{len(slot)} slots x {a.n} samples")
    print(f"3 OUT-OF-WINDOW        {100 * oow:.4f}% of all emitted spikes; any spike before "
          f"T1: {'YES' if R[:, :, :T1].any() else 'NO'}")
    tickset = np.unique(first[first >= 0])
    print(f"   every emitted spike at tick(s): {tickset.tolist()}")
    print(f"4 rails firing per sample: median {int(np.median(cnt.sum(1)))} "
          f"(expect {P} = one per comparator)")
    if a.out:
        prev = json.load(open(a.out)) if os.path.exists(a.out) else {}
        prev["scale_up"] = dict(n_pairs=P, n_slots=len(slot), n=a.n, neurons=nneur,
                                synapses=nsyn, window=[T1, T2], mem_margin=float(lhs - 1.0),
                                acc_overall=acc, acc_nontied=acc_nt, tie_frac=float(tie.mean()),
                                loser_false_fire=float(los.mean()), out_of_window=oow,
                                spike_ticks=tickset.tolist(),
                                per_table_acc=per_table)
        json.dump(prev, open(a.out, "w"), indent=1)
        print(f"\nwrote {a.out} (step_A/B/C preserved)")


if __name__ == "__main__":
    main()
