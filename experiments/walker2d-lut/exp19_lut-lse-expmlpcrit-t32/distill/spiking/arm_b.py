"""Arm B: capture an STDP-pretrained reservoir edge-by-edge and regrow it edge-exact.

The 3600 s artifact in workbooks/spnet_long stores weights WITHOUT their (source, target)
identity, so it cannot be mapped back onto edges. The correct path — now that
count_synapses + export_synapses is understood — is to build the reservoir, run STDP
in-process, capture every edge as (src, tgt, weight, delay), and regrow it into the ES net
through the same delay-meta recipe.

`--stdp-seconds` is short here on purpose: this is an import-fidelity gate, not the
science run. The bimodalisation knee is ~250 s (see RESULTS_spnet_long.md); use 3600 for
the real thing.
"""
import numpy as np
import torch

from es_harness import (D_MAX, D_MIN, N_EXC, N_INH, build, fitness, run_episode,
                        verify_round_trip)


def train_reservoir(device, seconds=60, seed=1, n_ticks=1000):
    """Build an 800/200 spnet, run STDP, and export every synapse edge-exact."""
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

    spk = torch.randint(N, [1, n_ticks, 1], device=device, dtype=torch.int32)
    val = torch.ones_like(spk, dtype=torch.float32) * 20.0
    for _ in range(seconds):
        sp.process_ticks(n_ticks_to_process=n_ticks, batch_size=1, n_input_ticks=n_ticks,
                         input_values=val, do_train=True, sparse_input=spk,
                         do_record_voltage=False, do_reset_context=False, _stdp_period=32)
        spk.random_(0, N)

    ids = torch.cat([exc, inh])
    n = sp.count_synapses(ids, True)
    bufs = [torch.zeros(n, dtype=t, device=device) for t in
            (torch.int32, torch.int32, torch.float32, torch.int32, torch.int32)]
    sp.export_synapses(ids, bufs[0], bufs[1], bufs[2], bufs[3], bufs[4], True)
    s, m, w, d, t = (x.cpu().numpy() for x in bufs)
    e0, i0 = int(exc.min()), int(inh.min())
    s_inh, t_inh = s >= i0, t >= i0
    src_local = np.where(s_inh, s - i0, s - e0)
    tgt_local = np.where(t_inh, t - i0, t - e0)
    print(f"  captured {n:,} synapses after {seconds}s STDP — ALL are imported")
    print(f"    targeting excitatory: {int((~t_inh).sum()):,}   "
          f"targeting inhibitory: {int(t_inh.sum()):,}")
    print(f"    sourced from excitatory: {int((~s_inh).sum()):,}   "
          f"from inhibitory: {int(s_inh.sum()):,}")
    print(f"  weights: mean {w.mean():.3f} std {w.std():.3f} "
          f"[{w.min():.2f}, {w.max():.2f}]; delays {d.min()}-{d.max()}")
    return (src_local, s_inh, tgt_local, t_inh, w.astype(np.float64),
            np.clip(d, D_MIN, D_MAX))


def run(device, res, genome, enc, X, Y, args):
    secs = getattr(args, "stdp_seconds", 250)
    edges = train_reservoir(device, seconds=secs)
    n_cap = edges[0].shape[0]
    ok_all = True
    for d in (["cuda", "cpu"] if device == "cuda" else ["cpu"]):
        h = build([genome], res, d, res_edges=edges)
        r = verify_round_trip(h)
        E = r["n_requested"]
        # Count the RESERVOIR edges that actually made it in, by id range, rather than
        # inferring them as (total - 2300 I/O): the I/O genome samples targets WITH
        # REPLACEMENT, so it contributes its own duplicates to the (src,tgt) dedup and
        # subtracting a nominal 2300 misattributes those losses to the reservoir.
        ids = h["ids"]
        res_src = np.concatenate([ids[0], ids[1]])
        res_tgt = np.concatenate([ids[0], ids[1]])
        tri = h["triples"]
        is_res = np.isin(tri[:, 1], res_src) & np.isin(tri[:, 2], res_tgt)
        n_res = int(is_res.sum())
        ok = r["weights_ok"] == E and r["delays_ok"] == E and r["missing"] == 0
        dropped = n_cap - n_res
        print(f"  {d:4s}: captured {n_cap:,} | regrown total {E:,} "
              f"(reservoir {E - 2300:,} + I/O 2,300) | weights {r['weights_ok']}/{E} "
              f"delays {r['delays_ok']}/{E} missing {r['missing']} dropped {dropped} -> "
              f"{'EDGE-EXACT' if ok and dropped == 0 else 'MISMATCH'}")
        ok_all &= ok and dropped == 0
        if d == device:
            hk = h
    # ---- same gates arm A gets, so the two are comparable -------------------------
    import numpy as _np
    from es_harness import (N_EXC, N_OUT, N_TICKS, kendall_tau_b, mutate,
                            random_genome, reservoir_wiring)
    from spiky.spnet.spnet import NeuronDataType

    ids = hk["ids"]
    first, _ = run_episode(hk, X, enc, args.current)
    Rexc = hk["spnet"].export_neuron_data(
        torch.tensor(ids[0], dtype=torch.int32, device=device), X.shape[0],
        NeuronDataType.Spike, 0, N_TICKS - 1).cpu().numpy()
    of = first < N_TICKS
    print(f"  smoke: reservoir {Rexc.sum()/X.shape[0]:.1f} spikes/sample, "
          f"{Rexc.any(-1).sum()/X.shape[0]/N_EXC*100:.1f}% exc recruited, "
          f"outputs fired {of.mean()*100:.1f}%, ticks "
          f"{first[of].min() if of.any() else -1:.0f}-{first[of].max() if of.any() else -1:.0f}")

    tau_real = kendall_tau_b(-first[:, 0, :], Y)
    nulls = _np.array([kendall_tau_b(-first[:, 0, :],
                                     Y[_np.random.default_rng(k).permutation(Y.shape[0])]).mean()
                       for k in range(200)])
    null_mu = float(nulls.mean())
    print(f"  null: random-wiring tau {tau_real.mean():+.4f} (SE "
          f"{tau_real.std()/_np.sqrt(len(tau_real)):.4f})  label-shuffle null "
          f"{null_mu:+.4f} sd {nulls.std():.4f}  => "
          f"{(tau_real.mean()-null_mu)/max(nulls.std(),1e-9):+.2f} sd above null")

    rng = _np.random.default_rng(0)
    gs = [genome] + [random_genome(rng, args.w_max) for _ in range(args.pop - 1)]
    hp = build(gs, res, device, res_edges=edges)
    f0, _, _ = fitness(hp, X, Y, enc, args.current)
    cur, elite = float(f0.max()), gs[int(f0.argmax())]
    print(f"  ES gen 0: best {f0.max():+.4f} mean {f0.mean():+.4f} vs null "
          f"{f0.max()-null_mu:+.4f}")
    for gen in range(1, args.gens + 1):
        kids = [elite] + [mutate(elite, rng, args.w_max) for _ in range(args.pop - 1)]
        hkk = build(kids, res, device, res_edges=edges)
        fk, _, _ = fitness(hkk, X, Y, enc, args.current)
        if fk.max() > cur:
            cur, elite = float(fk.max()), kids[int(fk.argmax())]
        print(f"  ES gen {gen}: best {fk.max():+.4f} mean {fk.mean():+.4f} "
              f"elite {cur:+.4f} vs null {cur-null_mu:+.4f}")
        del hkk
    print(f"  ARM B SUMMARY: random (tau-null) {tau_real.mean()-null_mu:+.4f}   "
          f"best-ES (tau-null) {cur-null_mu:+.4f}")
    return ok_all
