"""exp012 Stage 1, STEP A: the dual-rail mutual-inhibition WTA comparator, on the real engine.

    inA --dE--> railGT          inA --dI--> iA --1--> railLT      (A vetoes LT)
    inB --dE--> railLT          inB --dI--> iB --1--> railGT      (B vetoes GT)

Dale-clean: inA/inB are excitatory and only excite; iA/iB are inhibitory and only inhibit;
no neuron does both signs. The veto is ONE hop (input drives the interneuron directly), which
is the variant measured at 97.4% clean WTA at 128 ticks -- the two-hop version has a 4-tick
blind spot that swallows 83% of samples.

railGT fires iff  t_a + dE  <  t_b + dI + lat_i + 1, so the offset dE - dI is swept and the
value that best matches "a fires before b" is chosen empirically rather than derived from an
assumed interneuron latency.

STEP A ONLY. No latch, no gate -- the loser is expected to be suppressed transiently, not for
the whole episode, and emission timing is still tied to input timing. Those are steps B and C.
"""
import argparse
import itertools
import json

import numpy as np
import torch

T_IN, N_TICKS = 128, 320


def encode(x, t_in=T_IN):
    lo, hi = np.percentile(x, 0.5), np.percentile(x, 99.5)
    u = (x - lo) / max(hi - lo, 1e-9)
    return np.clip((1.0 - np.clip(u, 0, 1)) * (t_in - 1), 0, t_in - 1).round().astype(np.int64)


def build(dE, dI, w_exc, w_inh, tau_m, device="cuda"):
    """2 input drivers, 2 excitatory rails, 2 inhibitory interneurons."""
    from spiky.spnet.spnet import LIFNeuronMeta, SpikingNet, SynapseMeta
    from spiky.util.synapse_growth import SynapseGrowthEngine
    dmax = max(dE, dI, 1) + 1
    smetas = [SynapseMeta(learning_rate=0.0, min_delay=d, max_delay=d, initial_weight=0.0,
                          min_weight=-1e4, max_weight=1e4, initial_noise_level=0.0,
                          weight_decay=0.9, weight_scaling_cf=0.0,
                          _forward_group_size=2, _backward_group_size=2)
              for d in range(1, dmax + 1)]
    metas = [LIFNeuronMeta(neuron_type=0, tau=tau_m, threshold=1.0),   # 0,1 = input drivers
             LIFNeuronMeta(neuron_type=1, tau=tau_m, threshold=1.0),   # 0,1 = rails GT, LT
             LIFNeuronMeta(neuron_type=2, tau=tau_m, threshold=1.0)]   # 0,1 = interneurons
    net = SpikingNet(synapse_metas=smetas, neuron_metas=metas, neuron_counts=[2, 2, 2],
                     initial_synapse_capacity=1 << 20, summation_dtype=torch.float32)
    net.to_device(device)
    ids = [net.get_neuron_ids_by_meta(i).cpu().numpy() for i in range(3)]
    inA, inB = ids[0]
    gt, lt = ids[1]
    iA, iB = ids[2]
    edges = [(dE, inA, gt, w_exc), (dE, inB, lt, w_exc),          # drive
             (dI, inA, iA, w_exc), (dI, inB, iB, w_exc),          # input -> interneuron
             (1, iA, lt, w_inh), (1, iB, gt, w_inh)]              # cross veto
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


def run(net, ids, ta, tb, n_ticks=N_TICKS, device="cuda"):
    """-> first-spike tick of railGT and railLT, and their full spike counts."""
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
    R = R.reshape(B, 2, n_ticks).ne(0)
    w = torch.arange(n_ticks, 0, -1, device=R.device, dtype=torch.float32)
    first = (n_ticks - (R.float() * w).amax(-1)).cpu().numpy().astype(np.int64)
    return first, R.sum(-1).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--tau-m", type=float, default=20.0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    import os
    Z = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data",
                             "distill_exp19_100k.npz"))
    x = Z["x_norm"].astype(np.float64)
    A_, B_ = int(Z["anchor_a"][0, 0]), int(Z["anchor_b"][0, 0])
    ticks = encode(x)
    ta, tb = ticks[:a.n, A_], ticks[:a.n, B_]
    truth = (x[:a.n, A_] > x[:a.n, B_])
    tie = ta == tb
    print(f"STEP A: pair (x{A_}, x{B_}), {a.n} samples, {T_IN} input ticks, "
          f"{100 * tie.mean():.2f}% exact ties")
    print(f"ceiling on non-tied = 100%; overall ceiling "
          f"{100 * ((~tie).mean() + tie.mean() * max(truth[tie].mean(), 1 - truth[tie].mean())):.2f}%\n")

    best = None
    for dE, dI, w_inh in itertools.product((3, 4, 5), (1, 2, 3), (-4.0, -10.0)):
        net, ids = build(dE, dI, 1.5, w_inh, a.tau_m)
        first, cnt = run(net, ids, ta, tb)
        del net
        torch.cuda.empty_cache()
        gt_first = first[:, 0] < first[:, 1]
        both = (cnt[:, 0] > 0) & (cnt[:, 1] > 0)
        onlyone = (cnt[:, 0] > 0) ^ (cnt[:, 1] > 0)
        # winner bit: which rail fired (or fired first if both did)
        pred = np.where(cnt[:, 0] > 0, True, False)
        pred = np.where(both, gt_first, pred)
        acc_nt = float((pred[~tie] == truth[~tie]).mean())
        row = dict(dE=dE, dI=dI, w_inh=w_inh, acc_nontied=acc_nt,
                   clean_wta=float(onlyone.mean()), both_fired=float(both.mean()),
                   neither=float(((cnt[:, 0] == 0) & (cnt[:, 1] == 0)).mean()))
        print(f"  dE {dE} dI {dI} w_inh {w_inh:6.1f} -> acc(non-tied) "
              f"{100 * acc_nt:6.2f}%  cleanWTA {100 * row['clean_wta']:6.2f}%  "
              f"both {100 * row['both_fired']:6.2f}%  neither "
              f"{100 * row['neither']:5.2f}%")
        if best is None or acc_nt > best["acc_nontied"]:
            best = row
    print(f"\nSTEP A BEST: dE {best['dE']} dI {best['dI']} w_inh {best['w_inh']}  "
          f"acc(non-tied) {100 * best['acc_nontied']:.2f}%  cleanWTA "
          f"{100 * best['clean_wta']:.2f}%")
    if a.out:
        json.dump(dict(step="A", pair=[A_, B_], n=a.n, t_in=T_IN, tau_m=a.tau_m,
                       tie_frac=float(tie.mean()), best=best), open(a.out, "w"), indent=1)
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
