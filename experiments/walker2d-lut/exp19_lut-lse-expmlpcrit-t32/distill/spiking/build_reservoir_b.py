"""Full 3600 s STDP reservoir B: train, assay the rhythm, capture edge-exact, persist.

Unlike workbooks/spnet_long (which saved weights WITHOUT their (src,tgt) identity and so
cannot be re-imported), this captures every edge through count_synapses + export_synapses,
both excitatory- and inhibitory-targeting, and saves it in the form es_harness.build()
consumes directly via res_edges.

    python build_reservoir_b.py --seconds 3600
"""
import argparse
import json
import os
import time

import numpy as np
import torch

from es_harness import D_MAX, D_MIN, N_EXC, N_INH, build, fitness, run_episode, verify_round_trip

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")


def make_net(device, seed=1):
    from spiky.spnet.spnet import SpikingNet, SynapseMeta, NeuronMeta
    from spiky.util.synapse_growth import SynapseGrowthEngine, GrowthCommand
    N, M = N_EXC + N_INH, 100
    ge = SynapseGrowthEngine(device=device, synapse_group_size=((M + 31) // 32) * 32,
                             max_groups_in_buffer=((N + 31) // 32) * 32)
    box = dict(x1=-100.0, y1=-100.0, z1=-100.0, x2=100.0, y2=100.0, z2=100.0, p=0.1)
    ge.register_neuron_type(max_synapses=N // 10, growth_command_list=[
        GrowthCommand(target_type=0, synapse_meta_index=0,
                      max_synapses=int((N_EXC / N) * M), **box),
        GrowthCommand(target_type=1, synapse_meta_index=0,
                      max_synapses=int((N_INH / N) * M), **box)])
    ge.register_neuron_type(max_synapses=N_EXC // 10, growth_command_list=[
        GrowthCommand(target_type=0, synapse_meta_index=1, max_synapses=M, **box)])
    sp = SpikingNet(
        synapse_metas=[
            SynapseMeta(learning_rate=0.1, min_delay=D_MIN, max_delay=D_MAX,
                        initial_weight=6.0, _forward_group_size=8, _backward_group_size=8),
            SynapseMeta(learning_rate=0.0, min_delay=D_MIN, max_delay=D_MIN,
                        min_weight=-5.0, max_weight=-5.0, initial_weight=-5.0,
                        _forward_group_size=128, _backward_group_size=128)],
        neuron_metas=[NeuronMeta(neuron_type=0, a=0.02, d=8.0),
                      NeuronMeta(neuron_type=1, a=0.1, d=2.0)],
        neuron_counts=[N_EXC, N_INH], initial_synapse_capacity=M * N,
        summation_dtype=torch.float32)
    sp.to_device(device)
    exc, inh = sp.get_neuron_ids_by_meta(0), sp.get_neuron_ids_by_meta(1)
    r = (torch.arange(N_EXC).float() / N_EXC) * 99.0
    ge.add_neurons(neuron_type_index=0, identifiers=exc,
                   coordinates=torch.stack([torch.zeros_like(r), torch.zeros(N_EXC), r], 1))
    r = (torch.arange(N_INH).float() / N_INH) * 99.0
    ge.add_neurons(neuron_type_index=1, identifiers=inh,
                   coordinates=torch.stack([torch.zeros_like(r), torch.ones(N_INH), r], 1))
    ch = ge.grow(seed)
    sp.add_connections(ch, seed)
    ch.recycle()
    sp.compile(shuffle_synapses_random_seed=seed)
    return sp, exc, inh


def capture(sp, exc, inh, device):
    ids = torch.cat([exc, inh])
    n = sp.count_synapses(ids, True)
    b = [torch.zeros(n, dtype=t, device=device) for t in
         (torch.int32, torch.int32, torch.float32, torch.int32, torch.int32)]
    sp.export_synapses(ids, b[0], b[1], b[2], b[3], b[4], True)
    s, m, w, d, t = (x.cpu().numpy() for x in b)
    e0, i0 = int(exc.min()), int(inh.min())
    s_inh, t_inh = s >= i0, t >= i0
    return (np.where(s_inh, s - i0, s - e0), s_inh,
            np.where(t_inh, t - i0, t - e0), t_inh,
            w.astype(np.float64), np.clip(d, D_MIN, D_MAX))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=int, default=3600)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--w-max", type=float, default=30.0)
    ap.add_argument("--current", type=float, default=200.0)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(OUT, exist_ok=True)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)

    sp, exc, inh = make_net(dev, a.seed)
    print(sp, flush=True)
    N, n_ticks = N_EXC + N_INH, 1000
    spk = torch.randint(N, [1, n_ticks, 1], device=dev, dtype=torch.int32)
    val = torch.ones_like(spk, dtype=torch.float32) * 20.0

    t0 = time.time()
    traj = []
    for s in range(a.seconds):
        nsp = sp.process_ticks(n_ticks_to_process=n_ticks, batch_size=1,
                               n_input_ticks=n_ticks, input_values=val, do_train=True,
                               sparse_input=spk, do_record_voltage=False,
                               do_reset_context=False, _stdp_period=32)
        spk.random_(0, N)
        if (s + 1) in (1, 10, 50, 100, 250, 500, 1000, 2000, 3000) or (s + 1) % 500 == 0:
            traj.append(dict(sim_s=s + 1, hz=nsp / N))
            print(f"  t={s+1:5d}s  {nsp/N:6.2f} Hz/neuron  wall {time.time()-t0:6.0f}s",
                  flush=True)
    wall = time.time() - t0
    print(f"STDP: {a.seconds} simulated seconds in {wall:.0f}s wall "
          f"({wall/a.seconds*1000:.1f} ms per simulated second)", flush=True)

    # ---- rhythm assay: averaged periodogram vs a rate-matched surrogate --------------
    from spiky.spnet.spnet import NeuronDataType
    ids_all = torch.cat([exc, inh])
    K, acc, sur = 40, None, None
    rng = np.random.default_rng(a.seed)
    for _ in range(K):
        sp.process_ticks(n_ticks_to_process=n_ticks, batch_size=1, n_input_ticks=n_ticks,
                         input_values=val, do_train=False, sparse_input=spk,
                         do_record_voltage=False, do_reset_context=False, _stdp_period=32)
        spk.random_(0, N)
        R = sp.export_neuron_data(ids_all, 1, NeuronDataType.Spike, 0,
                                  n_ticks - 1)[0].float().cpu().numpy()
        pop = R.sum(0)
        sh = np.stack([np.roll(R[j], int(rng.integers(R.shape[1])))
                       for j in range(R.shape[0])]).sum(0)
        p1 = np.abs(np.fft.rfft(pop - pop.mean())) ** 2
        p2 = np.abs(np.fft.rfft(sh - sh.mean())) ** 2
        acc = p1 if acc is None else acc + p1
        sur = p2 if sur is None else sur + p2
    acc, sur = acc / K, sur / K
    frq = np.fft.rfftfreq(n_ticks, d=1e-3)
    band = (frq >= 20) & (frq <= 120)
    exc_ratio = acc[band] / sur[band]
    lo = (frq >= 1) & (frq < 20)
    print(f"\nRHYTHM ({K} averaged 1-s windows vs rate-matched surrogate):")
    print(f"  20-120 Hz: peak {exc_ratio.max():.2f}x surrogate at "
          f"{frq[band][int(np.argmax(exc_ratio))]:.0f} Hz, band mean {exc_ratio.mean():.2f}x")
    print(f"  1-20 Hz  : peak {(acc[lo]/sur[lo]).max():.1f}x, mean {(acc[lo]/sur[lo]).mean():.1f}x")

    # ---- capture + persist ------------------------------------------------------------
    edges = capture(sp, exc, inh, dev)
    sl, si, tl, ti, w, d = edges
    n_cap = sl.shape[0]
    print(f"\nCAPTURED {n_cap:,} synapses (ALL, no filtering)")
    print(f"  targeting exc {int((~ti).sum()):,}  targeting inh {int(ti.sum()):,}")
    print(f"  sourced exc  {int((~si).sum()):,}  sourced inh  {int(si.sum()):,}")
    we = w[~si]
    print(f"  excitatory weights: mean {we.mean():.3f} sd {we.std():.3f} "
          f"[{we.min():.2f}, {we.max():.2f}]  at~0 {float((we<=0.1).mean()):.4f} "
          f"at~max {float((we>=9.9).mean()):.4f}")
    print(f"  inhibitory weights: unique {np.unique(w[si])[:3].tolist()}")
    print(f"  delays: exc {d[~si].min()}-{d[~si].max()}, inh {np.unique(d[si]).tolist()}")

    p = os.path.join(OUT, f"reservoir_b_{a.seconds}s.npz")
    np.savez_compressed(p, src_local=sl, src_is_inh=si, tgt_local=tl, tgt_is_inh=ti,
                        weight=w, delay=d, n_exc=N_EXC, n_inh=N_INH,
                        stdp_seconds=a.seconds, seed=a.seed,
                        rhythm_freq=frq, rhythm_psd=acc, rhythm_surrogate=sur,
                        rate_traj=np.array([[t["sim_s"], t["hz"]] for t in traj]),
                        meta=json.dumps(dict(
                            exc_meta="learning_rate 0.1, delay 1-20, w in [0,10], init 6.0",
                            inh_meta="learning_rate 0.0, delay 1, w pinned -5.0",
                            note="edge-exact: src/tgt local ids + type flags, weights, delays")))
    print(f"\nSAVED -> {p}  ({os.path.getsize(p)/1e6:.2f} MB)")

    # ---- round trip on both devices ---------------------------------------------------
    from es_harness import random_genome, reservoir_wiring
    g0 = random_genome(np.random.default_rng(0), a.w_max)
    res = reservoir_wiring(np.random.default_rng(1234))
    print()
    for dd in (["cuda", "cpu"] if dev == "cuda" else ["cpu"]):
        h = build([g0], res, dd, res_edges=edges)
        r = verify_round_trip(h)
        E = r["n_requested"]
        n_res = int(np.isin(h["triples"][:, 1], np.concatenate([h["ids"][0], h["ids"][1]])).sum())
        ok = r["weights_ok"] == E and r["delays_ok"] == E and r["missing"] == 0
        print(f"  {dd:4s}: captured {n_cap:,} | regrown {E:,} | weights {r['weights_ok']}/{E}"
              f" delays {r['delays_ok']}/{E} missing {r['missing']} dropped {n_cap-n_res} -> "
              f"{'EDGE-EXACT' if ok and n_cap == n_res else 'MISMATCH'}")
        if dd == dev:
            hk = h

    # ---- arm-B smoke at the aligned scale ---------------------------------------------
    Z = np.load(os.path.join(HERE, "..", "distill_exp19_100k.npz"))
    from es_harness import LatencyEncoder, N_TICKS
    n_tr = Z["x_norm"].shape[0] - 4000
    idx = np.random.default_rng(0).choice(n_tr, 64, replace=False)
    X, Y = Z["x_norm"][idx].astype(np.float64), Z["y_action_mean"][idx].astype(np.float64)
    enc = LatencyEncoder(Z["x_norm"][:n_tr].astype(np.float64))
    first, _ = run_episode(hk, X, enc, a.current)
    Rexc = hk["spnet"].export_neuron_data(
        torch.tensor(hk["ids"][0], dtype=torch.int32, device=dev), X.shape[0],
        NeuronDataType.Spike, 0, N_TICKS - 1).cpu().numpy()
    of = first < N_TICKS
    tau, _, _ = fitness(hk, X, Y, enc, a.current)
    print(f"\nARM-B SMOKE (w_max {a.w_max}, res_scale 1): reservoir "
          f"{Rexc.sum()/X.shape[0]:.1f} spikes/sample, "
          f"{Rexc.any(-1).sum()/X.shape[0]/N_EXC*100:.1f}% exc recruited, outputs "
          f"{of.mean()*100:.1f}%, ticks {first[of].min():.0f}-{first[of].max():.0f}, "
          f"tau-b {tau[0]:+.4f}")


if __name__ == "__main__":
    main()
