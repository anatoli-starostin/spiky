"""Why do BOTH rails fire on a tick-tie? Record the membrane traces and find out.

An isolated replica of ONE dual-rail comparator, wired with the exact constants the full
pipeline uses, so the dynamics are the real ones without the 3024-neuron network around it.
Read-only with respect to everything else: nothing else is imported or modified.
"""
import argparse
import json
import os

import numpy as np
import torch

DE, DI, W_EXC, W_INH = 3, 2, 1.5, -10.0
TAU_M_RAIL = 20.0


def build(device="cpu"):
    from spiky.spnet.spnet import LIFNeuronMeta, NeuronMeta, SpikingNet, SynapseMeta
    from spiky.util.synapse_growth import SynapseGrowthEngine
    dmax = max(DE, DI, 1)
    smetas = [SynapseMeta(learning_rate=0.0, min_delay=d, max_delay=d, initial_weight=0.0,
                          min_weight=-1e4, max_weight=1e4, initial_noise_level=0.0,
                          weight_decay=0.9, weight_scaling_cf=0.0,
                          _forward_group_size=2, _backward_group_size=2)
              for d in range(1, dmax + 1)]
    metas = [LIFNeuronMeta(neuron_type=0, tau=TAU_M_RAIL, threshold=1.0),          # 2 inputs
             NeuronMeta(neuron_type=1, cf_2=0.0, cf_1=+1.0 / TAU_M_RAIL, cf_0=0.0,
                        a=0.0, b=0.0, c=0.0, d=0.0, spike_threshold=1.0),          # 2 rails
             LIFNeuronMeta(neuron_type=2, tau=TAU_M_RAIL, threshold=1.0)]          # 2 interneur
    net = SpikingNet(synapse_metas=smetas, neuron_metas=metas, neuron_counts=[2, 2, 2],
                     initial_synapse_capacity=1 << 16, summation_dtype=torch.float32)
    net.to_device(device)
    ids = [net.get_neuron_ids_by_meta(i).cpu().numpy() for i in range(3)]
    (ia, ib), (r0, r1), (i0, i1) = ids
    E = [(DE, ia, r0, W_EXC), (DE, ib, r1, W_EXC),
         (DI, ia, i0, W_EXC), (DI, ib, i1, W_EXC),
         (1, i0, r1, W_INH), (1, i1, r0, W_INH)]
    tri = np.array([[d - 1, s, t] for d, s, t, _ in E], np.int64)
    wts = np.array([w for *_, w in E], np.float64)
    ge = SynapseGrowthEngine(device=device, synapse_group_size=2,
                             max_groups_in_buffer=1 << 12)
    for i in range(3):
        ge.register_neuron_type(max_synapses=1 << 10, growth_command_list=[])
    for i in range(3):
        tt = torch.tensor(ids[i], dtype=torch.int32)
        ge.add_neurons(neuron_type_index=i, identifiers=tt,
                       coordinates=torch.stack([torch.arange(tt.numel()).float(),
                                                torch.zeros(tt.numel()),
                                                torch.full((tt.numel(),), float(i))], 1))
    chunk = ge._grow_explicit(torch.tensor(tri, dtype=torch.int32, device=device), 1,
                              weights=torch.tensor(wts, dtype=torch.float32, device=device))
    net.add_connections(chunk, 1)
    chunk.recycle()
    net.compile(shuffle_synapses_random_seed=None)
    return net, ids


def run(net, ids, ta, tb, NT=24, device="cpu"):
    from spiky.spnet.spnet import NeuronDataType
    va = np.zeros((1, NT, 2), np.float32)
    va[0, ta, 0] = 1e6
    va[0, tb, 1] = 1e6
    sid = torch.as_tensor(np.ascontiguousarray(ids[0], dtype=np.int32),
                          device=device).reshape(1, 1, -1).expand(1, NT, -1).contiguous()
    net.process_ticks(n_ticks_to_process=NT, batch_size=1, n_input_ticks=NT,
                      input_values=torch.as_tensor(va, device=device), sparse_input=sid,
                      do_train=False, do_record_voltage=True, do_reset_context=True,
                      _stdp_period=32)
    out = {}
    for k, lbl in ((1, "rail"), (2, "inh")):
        oid = torch.as_tensor(np.ascontiguousarray(ids[k], dtype=np.int32), device=device)
        V = net.export_neuron_data(oid, 1, NeuronDataType.Voltage, 0, NT - 1)
        S = net.export_neuron_data(oid, 1, NeuronDataType.Spike, 0, NT - 1)
        out[lbl + "_V"] = V.reshape(1, 2, NT).cpu().numpy()[0]
        out[lbl + "_S"] = S.reshape(1, 2, NT).ne(0).cpu().numpy()[0]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    net, ids = build()
    NT = 24
    cases = ([("TIE", 3, 3)] + [("TIE", t, t) for t in (4, 5, 6, 7)]
             + [("1-APART a<b", t, t + 1) for t in (3, 4, 5)]
             + [("1-APART b<a", t + 1, t) for t in (3, 4, 5)])
    rows = []
    for lbl, ta, tb in cases:
        o = run(net, ids, ta, tb, NT)
        rv, rs = o["rail_V"], o["rail_S"]
        iv, isp = o["inh_V"], o["inh_S"]
        f0 = int(np.argmax(rs[0])) if rs[0].any() else -1
        f1 = int(np.argmax(rs[1])) if rs[1].any() else -1
        g0 = int(np.argmax(isp[0])) if isp[0].any() else -1
        g1 = int(np.argmax(isp[1])) if isp[1].any() else -1
        rows.append(dict(case=lbl, t_a=ta, t_b=tb, r0_fire=f0, r1_fire=f1,
                         i0_fire=g0, i1_fire=g1,
                         r0_peak=float(rv[0].max()), r1_peak=float(rv[1].max()),
                         r0_min=float(rv[0].min()), r1_min=float(rv[1].min()),
                         r0_V=[round(float(v), 4) for v in rv[0][:14]],
                         r1_V=[round(float(v), 4) for v in rv[1][:14]]))
        print(f"{lbl:12s} ta={ta} tb={tb} | r0 fires {f0:3d}  r1 fires {f1:3d} | "
              f"i0 {g0:3d} i1 {g1:3d} | r0 peak {rv[0].max():+9.3f} min {rv[0].min():+9.3f}"
              f" | r1 peak {rv[1].max():+9.3f} min {rv[1].min():+9.3f}")
    print("\nrail V traces, ticks 0..13")
    for r in rows:
        print(f"  {r['case']:12s} ta={r['t_a']} tb={r['t_b']}")
        print(f"    r0: {r['r0_V']}")
        print(f"    r1: {r['r1_V']}")
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        json.dump(rows, open(a.out, "w"), indent=1)
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
