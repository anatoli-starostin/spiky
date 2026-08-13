"""exp012 Stage 1, STEP C: the gated readout.

THE INTERFACE PROBLEM Step B leaves. The winner rail fires once and resets to V = 0 -- which
is the unstable equilibrium, so it SITS there, indistinguishable by voltage from a rail that
never did anything. The loser is at a large negative V. So after the decision the latch is
readable only from the LOSER's voltage, and the winner's evidence is the spike it already
spent. Gating on rail voltage directly therefore cannot work.

THE FIX: give each rail a MEMORY cell that catches that single spike and holds it
sub-threshold, and let the gate supply the rest of the threshold.

    rail --w_mem--> mem   (leaky, LONG tau, threshold 1.0)      w_mem alone is SUB-threshold
    gate --w_gate--> mem  (global, fires at T1)                 w_gate alone is SUB-threshold
                                                                 together they cross

  winner: mem got w_mem, decays slowly, + w_gate at T1  -> CROSSES, fires once inside the window
  loser:  mem got nothing,               + w_gate at T1  -> stays below, silent

Emission is triggered by the GATE, so an input that arrived at tick 0 and one that arrived at
tick 127 produce the same output tick -- which is the whole point. After firing, mem resets to
0 and receives nothing further, so it fires exactly once.

Dale-clean: gate and rails excite; only the interneurons inhibit.
"""
import argparse
import itertools
import json
import os

import numpy as np
import torch

from tiny_lut_order_detect import T_IN, encode

T1, T2 = 200, 240          # readout window
N_TICKS = 320


def build(dE, dI, w_exc, w_inh, tau_m, w_mem, w_gate, tau_mem, device="cuda"):
    from spiky.spnet.spnet import LIFNeuronMeta, NeuronMeta, SpikingNet, SynapseMeta
    from spiky.util.synapse_growth import SynapseGrowthEngine
    dmax = max(dE, dI, 1) + 1
    smetas = [SynapseMeta(learning_rate=0.0, min_delay=d, max_delay=d, initial_weight=0.0,
                          min_weight=-1e4, max_weight=1e4, initial_noise_level=0.0,
                          weight_decay=0.9, weight_scaling_cf=0.0,
                          _forward_group_size=2, _backward_group_size=2)
              for d in range(1, dmax + 1)]
    metas = [LIFNeuronMeta(neuron_type=0, tau=tau_m, threshold=1.0),          # inA,inB,GATE
             NeuronMeta(neuron_type=1, cf_2=0.0, cf_1=+1.0 / tau_m, cf_0=0.0, a=0.0, b=0.0,
                        c=0.0, d=0.0, spike_threshold=1.0),                   # rails (anti-leaky)
             LIFNeuronMeta(neuron_type=2, tau=tau_m, threshold=1.0),          # interneurons
             LIFNeuronMeta(neuron_type=3, tau=tau_mem, threshold=1.0)]        # memory/readout
    net = SpikingNet(synapse_metas=smetas, neuron_metas=metas, neuron_counts=[3, 2, 2, 2],
                     initial_synapse_capacity=1 << 20, summation_dtype=torch.float32)
    net.to_device(device)
    ids = [net.get_neuron_ids_by_meta(i).cpu().numpy() for i in range(4)]
    inA, inB, gate = ids[0]
    gt, lt = ids[1]
    iA, iB = ids[2]
    mgt, mlt = ids[3]
    edges = [(dE, inA, gt, w_exc), (dE, inB, lt, w_exc),
             (dI, inA, iA, w_exc), (dI, inB, iB, w_exc),
             (1, iA, lt, w_inh), (1, iB, gt, w_inh),
             (1, gt, mgt, w_mem), (1, lt, mlt, w_mem),          # rail -> its memory cell
             (1, gate, mgt, w_gate), (1, gate, mlt, w_gate)]    # global gate -> both
    tri = np.array([[d - 1, s, t] for d, s, t, _ in edges], np.int64)
    wts = np.array([w for *_, w in edges], np.float64)
    ge = SynapseGrowthEngine(device=device, synapse_group_size=2, max_groups_in_buffer=8192)
    for i in range(4):
        ge.register_neuron_type(max_synapses=64, growth_command_list=[])
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
    return net, ids


def run(net, ids, ta, tb, n_ticks=N_TICKS, t_gate=T1, device="cuda"):
    from spiky.spnet.spnet import NeuronDataType
    B = len(ta)
    va = np.zeros((B, n_ticks, 3), np.float32)
    va[np.arange(B), ta, 0] = 1e6
    va[np.arange(B), tb, 1] = 1e6
    va[:, t_gate, 2] = 1e6                                   # the global gate
    sid = torch.as_tensor(np.ascontiguousarray(ids[0], dtype=np.int32),
                          device=device).reshape(1, 1, -1).expand(B, n_ticks, -1).contiguous()
    net.process_ticks(n_ticks_to_process=n_ticks, batch_size=B, n_input_ticks=n_ticks,
                      input_values=torch.as_tensor(va, device=device), sparse_input=sid,
                      do_train=False, do_record_voltage=False, do_reset_context=True,
                      _stdp_period=32)
    out = {}
    for k in (1, 3):
        oid = torch.as_tensor(np.ascontiguousarray(ids[k], dtype=np.int32), device=device)
        R = net.export_neuron_data(oid, B, NeuronDataType.Spike, 0, n_ticks - 1)
        out[k] = R.reshape(B, 2, n_ticks).ne(0).cpu().numpy()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    Z = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data",
                             "distill_exp19_100k.npz"))
    x = Z["x_norm"].astype(np.float64)
    A_, B_ = int(Z["anchor_a"][0, 0]), int(Z["anchor_b"][0, 0])
    ticks = encode(x)
    ta, tb = ticks[:a.n, A_], ticks[:a.n, B_]
    truth = x[:a.n, A_] > x[:a.n, B_]
    tie = ta == tb
    earliest = np.minimum(ta, tb)
    very_early = earliest <= np.percentile(earliest, 5)
    print(f"STEP C: pair (x{A_},x{B_}), {a.n} samples, window [{T1},{T2}), episode {N_TICKS}")
    print(f"  ties {100 * tie.mean():.2f}%   earliest-input tick: min {earliest.min()}, "
          f"5th pct {np.percentile(earliest, 5):.0f}  ({very_early.sum()} 'very early' samples)\n")

    best = None
    for w_mem, w_gate, tau_mem in itertools.product((0.6, 0.8), (0.6, 0.5), (200.0, 600.0)):
        net, ids = build(3, 2, 1.5, -10.0, 20.0, w_mem, w_gate, tau_mem)
        R = run(net, ids, ta, tb)
        del net
        torch.cuda.empty_cache()
        mem = R[3]
        b = np.arange(len(ta))
        win_i = np.where(truth, 0, 1)
        los_i = 1 - win_i
        wc = mem[b, win_i].sum(-1)
        lc = mem[b, los_i].sum(-1)
        firstw = np.where(wc > 0, mem[b, win_i].argmax(-1), -1)
        inwin = (firstw >= T1) & (firstw < T2)
        early_spike = (mem.sum(1)[:, :T1].sum(-1) > 0)
        row = dict(w_mem=w_mem, w_gate=w_gate, tau_mem=tau_mem,
                   winner_fires=float((wc > 0)[~tie].mean()),
                   winner_exactly_once=float((wc == 1)[~tie].mean()),
                   loser_silent=float((lc == 0)[~tie].mean()),
                   in_window=float(inwin[~tie].mean()),
                   any_spike_before_T1=float(early_spike.mean()),
                   early_input_before_T1=float(early_spike[very_early].mean()),
                   spike_tick_min=int(firstw[firstw >= 0].min()) if (firstw >= 0).any() else -1,
                   spike_tick_max=int(firstw.max()))
        print(f"  w_mem {w_mem} w_gate {w_gate} tau_mem {tau_mem:5.0f} -> winner "
              f"{100 * row['winner_fires']:6.2f}%  once {100 * row['winner_exactly_once']:6.2f}%"
              f"  loser-silent {100 * row['loser_silent']:6.2f}%  in-window "
              f"{100 * row['in_window']:6.2f}%  pre-T1 spikes {100 * row['any_spike_before_T1']:.3f}%"
              f"  tick [{row['spike_tick_min']},{row['spike_tick_max']}]")
        score = (row["in_window"] + row["loser_silent"] + row["winner_exactly_once"])
        if best is None or score > best[0]:
            best = (score, row)
    r = best[1]
    print(f"\nSTEP C BEST: w_mem {r['w_mem']} w_gate {r['w_gate']} tau_mem {r['tau_mem']:.0f}")
    for k in ("winner_fires", "winner_exactly_once", "loser_silent", "in_window",
              "any_spike_before_T1", "early_input_before_T1"):
        print(f"   {k:24s} {100 * r[k]:8.4f}%")
    if a.out:
        prev = json.load(open(a.out)) if os.path.exists(a.out) else {}
        prev["step_C"] = dict(pair=[A_, B_], n=a.n, window=[T1, T2], n_ticks=N_TICKS,
                              tie_frac=float(tie.mean()), best=r)
        json.dump(prev, open(a.out, "w"), indent=1)
        print(f"\nwrote {a.out} (step_A / step_B preserved)")


if __name__ == "__main__":
    main()
