"""exp012 Stage 1, STEP B: the LATCH.

Step A's suppression is transient -- inhibition decays with the membrane leak (half-life 13.7
ticks at tau 20), so over a 320-tick episode the loser's veto expires and it fires late.

THE MECHANISM, and it is cheaper than the brief assumes. Make the rails ANTI-LEAKY
(cf_1 = +1/tau, v_rest = 0). Then V = 0 is an UNSTABLE equilibrium, so the runaway is
two-sided:

    winner: first net input POSITIVE -> V grows up   -> fires, decision irreversible
    loser:  first net input NEGATIVE -> V grows DOWN -> permanently silent

So the loser does NOT need renewed inhibition -- the same anti-leaky dynamics that latch the
winner also latch the loser, in the opposite direction. Option (c) of the brief, for free.

The ordering that makes it work is already guaranteed by Step A's geometry: the loser's veto
(from the winner's input, one hop) arrives before the loser's own excitation whenever
dI + lat_i + 1 <= dE, which is the same condition Step A already satisfies.

VERIFIED OVER THE FULL EPISODE, every tick, not just the first few.
"""
import argparse
import itertools
import json
import os

import numpy as np
import torch

from tiny_lut_order_detect import T_IN, encode


def build_latched(dE, dI, w_exc, w_inh, tau_m, anti_leaky, n_ticks, device="cuda"):
    from spiky.spnet.spnet import LIFNeuronMeta, NeuronMeta, SpikingNet, SynapseMeta
    from spiky.util.synapse_growth import SynapseGrowthEngine
    dmax = max(dE, dI, 1) + 1
    smetas = [SynapseMeta(learning_rate=0.0, min_delay=d, max_delay=d, initial_weight=0.0,
                          min_weight=-1e4, max_weight=1e4, initial_noise_level=0.0,
                          weight_decay=0.9, weight_scaling_cf=0.0,
                          _forward_group_size=2, _backward_group_size=2)
              for d in range(1, dmax + 1)]
    rail = (NeuronMeta(neuron_type=1, cf_2=0.0, cf_1=+1.0 / tau_m, cf_0=0.0, a=0.0, b=0.0,
                       c=0.0, d=0.0, spike_threshold=1.0) if anti_leaky
            else LIFNeuronMeta(neuron_type=1, tau=tau_m, threshold=1.0))
    metas = [LIFNeuronMeta(neuron_type=0, tau=tau_m, threshold=1.0), rail,
             LIFNeuronMeta(neuron_type=2, tau=tau_m, threshold=1.0)]
    net = SpikingNet(synapse_metas=smetas, neuron_metas=metas, neuron_counts=[2, 2, 2],
                     initial_synapse_capacity=1 << 20, summation_dtype=torch.float32)
    net.to_device(device)
    ids = [net.get_neuron_ids_by_meta(i).cpu().numpy() for i in range(3)]
    inA, inB = ids[0]
    gt, lt = ids[1]
    iA, iB = ids[2]
    edges = [(dE, inA, gt, w_exc), (dE, inB, lt, w_exc),
             (dI, inA, iA, w_exc), (dI, inB, iB, w_exc),
             (1, iA, lt, w_inh), (1, iB, gt, w_inh)]
    tri = np.array([[d - 1, s, t] for d, s, t, _ in edges], np.int64)
    wts = np.array([w for *_, w in edges], np.float64)
    ge = SynapseGrowthEngine(device=device, synapse_group_size=2, max_groups_in_buffer=8192)
    for i in range(3):
        ge.register_neuron_type(max_synapses=64, growth_command_list=[])
    for i in range(3):
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
    return net, ids


def run_full(net, ids, ta, tb, n_ticks, device="cuda"):
    """-> full spike raster of both rails [B,2,n_ticks] so late firing cannot hide."""
    from spiky.spnet.spnet import NeuronDataType
    B = len(ta)
    va = np.zeros((B, n_ticks, 2), np.float32)
    va[np.arange(B), ta, 0] = 1e6
    va[np.arange(B), tb, 1] = 1e6
    sid = torch.as_tensor(np.ascontiguousarray(ids[0], dtype=np.int32),
                          device=device).reshape(1, 1, -1).expand(B, n_ticks, -1).contiguous()
    net.process_ticks(n_ticks_to_process=n_ticks, batch_size=B, n_input_ticks=n_ticks,
                      input_values=torch.as_tensor(va, device=device), sparse_input=sid,
                      do_train=False, do_record_voltage=False, do_reset_context=True,
                      _stdp_period=32)
    oid = torch.as_tensor(np.ascontiguousarray(ids[1], dtype=np.int32), device=device)
    R = net.export_neuron_data(oid, B, NeuronDataType.Spike, 0, n_ticks - 1)
    return R.reshape(B, 2, n_ticks).ne(0).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--tau-m", type=float, default=20.0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    Z = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data",
                             "distill_exp19_100k.npz"))
    x = Z["x_norm"].astype(np.float64)
    A_, B_ = int(Z["anchor_a"][0, 0]), int(Z["anchor_b"][0, 0])
    ticks = encode(x)
    ta, tb = ticks[:a.n, A_], ticks[:a.n, B_]
    truth = x[:a.n, A_] > x[:a.n, B_]
    tie = ta == tb
    print(f"STEP B latch: pair (x{A_},x{B_}), {a.n} samples, {T_IN} input ticks, "
          f"{100 * tie.mean():.2f}% ties\n")

    rows = []
    for anti, n_ticks in itertools.product((False, True), (160, 320, 640)):
        net, ids = build_latched(3, 2, 1.5, -10.0, a.tau_m, anti, n_ticks)
        R = run_full(net, ids, ta, tb, n_ticks)
        del net
        torch.cuda.empty_cache()
        cnt = R.sum(-1)
        winner_gt = truth                       # ground truth: A earlier => GT should win
        loser_idx = np.where(winner_gt, 1, 0)
        win_idx = 1 - loser_idx
        b = np.arange(len(ta))
        loser_fire = cnt[b, loser_idx] > 0
        win_fire = cnt[b, win_idx] > 0
        win_n = cnt[b, win_idx]
        row = dict(anti_leaky=bool(anti), n_ticks=n_ticks,
                   loser_false_fire=float(loser_fire[~tie].mean()),
                   winner_fires=float(win_fire[~tie].mean()),
                   winner_spikes_median=float(np.median(win_n[~tie])),
                   winner_spikes_max=int(win_n[~tie].max()))
        rows.append(row)
        print(f"  {'ANTI-LEAKY' if anti else 'leaky    '}  episode {n_ticks:4d} ticks -> "
              f"LOSER false-fire {100 * row['loser_false_fire']:7.3f}%   winner fires "
              f"{100 * row['winner_fires']:6.2f}%   winner spikes median "
              f"{row['winner_spikes_median']:.0f} max {row['winner_spikes_max']}")

    print("\nCONTRACT (loser silent for the ENTIRE episode):")
    for r in rows:
        if r["anti_leaky"]:
            ok = r["loser_false_fire"] < 1e-9 and r["winner_fires"] > 0.99
            print(f"  anti-leaky, {r['n_ticks']:4d} ticks: "
                  f"{'PASS' if ok else 'FAIL'}  (loser {100 * r['loser_false_fire']:.4f}%, "
                  f"winner {100 * r['winner_fires']:.2f}%)")
    if a.out:
        prev = json.load(open(a.out)) if os.path.exists(a.out) else {}
        prev["step_B"] = dict(pair=[A_, B_], n=a.n, tau_m=a.tau_m,
                              tie_frac=float(tie.mean()), rows=rows,
                              mechanism="anti-leaky rails: V=0 is an unstable equilibrium, so "
                                        "a net-negative first input runs the loser away "
                                        "DOWNWARD permanently -- no renewed inhibition needed")
        json.dump(prev, open(a.out, "w"), indent=1)
        print(f"\nwrote {a.out} (step_A record preserved)")


if __name__ == "__main__":
    main()
