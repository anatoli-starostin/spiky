"""Re-test excitatory backward_group_size=8 against the REBUILT engine (sign-flip fix in).

Real steady-state shape: 40 metas (0-19 plastic exc, 20-39 frozen inh), ~80-105 incoming
plastic synapses per excitatory target. Per spnet.ipynb: exc fwd/bwd=8, inh fwd/bwd=128,
engine synapse_group_size=128. Weights go in via _grow_explicit(weights=) — now that the
stock aligner follows the chain, that path is correct.

Checks build, sanitizer-cleanliness, and — the point — whether the BACKWARD/STDP pass runs
cleanly or overflows when a target has far more incoming plastic synapses than bwd allows.
Silent corruption is checked too, not just crashes.
"""
import argparse

import numpy as np
import torch

import steady_state as S


def build(genomes, exc_bwd, exc_fwd, inh_gs, engine_gs, stdp_lr, w_max=30.0, device="cuda",
          no_weights=False):
    from spiky.spnet.spnet import NeuronMeta, SpikingNet, SynapseMeta
    from spiky.util.synapse_growth import SynapseGrowthEngine
    K = len(genomes)
    exc = [SynapseMeta(learning_rate=stdp_lr, min_delay=d, max_delay=d, initial_weight=0.0,
                       min_weight=0.0, max_weight=1.5 * w_max, initial_noise_level=0.0,
                       weight_decay=0.9, weight_scaling_cf=0.0,
                       _forward_group_size=exc_fwd, _backward_group_size=exc_bwd)
           for d in range(S.D_MIN, S.D_MAX + 1)]
    inh = [SynapseMeta(learning_rate=0.0, min_delay=d, max_delay=d,
                       initial_weight=S.RES_W_INH, min_weight=S.RES_W_INH,
                       max_weight=S.RES_W_INH, initial_noise_level=0.0, weight_decay=0.9,
                       weight_scaling_cf=0.0,
                       _forward_group_size=inh_gs, _backward_group_size=inh_gs)
           for d in range(S.D_MIN, S.D_MAX + 1)]
    metas = exc + inh
    neuron_metas = [NeuronMeta(neuron_type=i, a=0.02 if i != 1 else 0.1,
                               d=8.0 if i != 1 else 2.0) for i in range(4)]
    counts = [K * S.N_EXC, K * S.N_INH, K * S.N_IN, K * S.N_OUT]
    sp = SpikingNet(synapse_metas=metas, neuron_metas=neuron_metas, neuron_counts=counts,
                    initial_synapse_capacity=1 << 23, summation_dtype=torch.float32)
    sp.to_device(device)
    ids = [sp.get_neuron_ids_by_meta(i).cpu().numpy() for i in range(4)]
    base = {S.EXC: S.N_EXC, S.INH: S.N_INH, S.INP: S.N_IN, S.OUTP: S.N_OUT}

    tri, wts = [], []
    for c, g in enumerate(genomes):
        s = np.empty(g["weight"].size, np.int64)
        t = np.empty_like(s)
        for p in (S.EXC, S.INH, S.INP):
            m = g["src_pool"] == p
            if m.any():
                s[m] = ids[p][c * base[p] + g["src_idx"][m]]
        for p in (S.EXC, S.INH, S.OUTP):
            m = g["tgt_pool"] == p
            if m.any():
                t[m] = ids[p][c * base[p] + g["tgt_idx"][m]]
        meta = (g["delay"] - S.D_MIN).copy()
        meta[g["src_pool"] == S.INH] += S.N_DELAY_METAS
        tri.append(np.stack([meta, s, t], 1))
        wts.append(g["weight"])
    triples, weights = np.concatenate(tri, 0), np.concatenate(wts, 0)

    total = sum(counts)
    ge = SynapseGrowthEngine(device=device, synapse_group_size=engine_gs,
                             max_groups_in_buffer=max(4096, 8 * (len(triples) + total)))
    for _ in range(4):
        ge.register_neuron_type(max_synapses=8 * (S.N_EXC + S.N_INH), growth_command_list=[])
    for i in range(4):
        tt = torch.tensor(ids[i], dtype=torch.int32)
        n = tt.numel()
        ge.add_neurons(neuron_type_index=i, identifiers=tt,
                       coordinates=torch.stack([torch.arange(n).float(), torch.zeros(n),
                                                torch.full((n,), float(i))], 1))
    tri_t = torch.tensor(triples, dtype=torch.int32, device=device)
    w_t = torch.tensor(weights, dtype=torch.float32, device=device)
    # weights straight into _grow_explicit, as asked
    chunk = (ge._grow_explicit(tri_t, 1) if no_weights
             else ge._grow_explicit(tri_t, 1, weights=w_t))
    sp.add_connections(chunk, 1)
    chunk.recycle()
    sp.compile(shuffle_synapses_random_seed=None)
    if device == "cuda":
        torch.cuda.synchronize()
    return dict(spnet=sp, ids=ids, P=K, device=device, n_syn=len(triples)), triples, weights


def snapshot(h):
    sp, ids = h["spnet"], h["ids"]
    all_ids = torch.tensor(np.concatenate(ids), dtype=torch.int32, device=h["device"])
    n = sp.count_synapses(all_ids, True)
    b = [torch.zeros(n, dtype=t, device=h["device"]) for t in
         (torch.int32, torch.int32, torch.float32, torch.int32, torch.int32)]
    sp.export_synapses(all_ids, b[0], b[1], b[2], b[3], b[4], True)
    s, m, w, _, t = (x.cpu().numpy() for x in b)
    return s, m, w, t


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bwd", type=int, default=8)
    ap.add_argument("--exc-fwd", type=int, default=8)
    ap.add_argument("--inh-gs", type=int, default=128)
    ap.add_argument("--engine-gs", type=int, default=128)
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--stdp-lr", type=float, default=0.01)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-weights", action="store_true")
    ap.add_argument("--dump", default=None)
    a = ap.parse_args()

    X, Y, Xpool, Ypool, _, _ = S.load(a.batch, 0, 256)
    enc = S.LatencyEncoder(Xpool)
    genomes = [S.seed_genome(np.random.default_rng(i), 30.0) for i in range(a.k)]

    # how many incoming PLASTIC synapses does a target actually carry?
    g0 = genomes[0]
    e = g0["src_pool"] != S.INH
    _, cnt = np.unique(g0["tgt_pool"][e] * 100000 + g0["tgt_idx"][e], return_counts=True)
    print(f"config: exc fwd={a.exc_fwd} bwd={a.bwd}, inh gs={a.inh_gs}, "
          f"engine gs={a.engine_gs}, K={a.k}")
    print(f"incoming plastic per target: max {cnt.max()} mean {cnt.mean():.1f} "
          f"p99 {np.percentile(cnt, 99):.0f}   (bwd={a.bwd})")

    h, triples, weights = build(genomes, a.bwd, a.exc_fwd, a.inh_gs, a.engine_gs,
                                a.stdp_lr, device=a.device, no_weights=a.no_weights)
    print(f"BUILD-OK {h['n_syn']:,} synapses")

    s0, m0, w0, t0 = snapshot(h)
    before = {(int(x), int(y)): float(v) for x, y, v in zip(s0, t0, w0)}
    want = {(int(x), int(y)): float(v) for (_, x, y), v in zip(triples, weights)}
    bad = sum(1 for k in want if k in before and abs(before[k] - want[k]) > 1e-3)
    print(f"PLACEMENT exact={len(want) - bad}/{len(want)} wrong={bad}")

    for i in range(a.rounds):
        Xb, _, _ = S.sample_batch(Xpool, Ypool, a.batch, 0, i)
        S.run_episode(h, Xb, enc, 200.0, train=True)
    if a.device == "cuda":
        torch.cuda.synchronize()
    print(f"STDP-OK {a.rounds} rounds with plasticity on")

    s1, m1, w1, t1 = snapshot(h)
    exc_mask = m1 < S.N_DELAY_METAS
    inh_mask = ~exc_mask
    we, wi = w1[exc_mask], w1[inh_mask]
    ceiling = 1.5 * 30.0
    print(f"AFTER exc: n={we.size:,} min={we.min():.3f} max={we.max():.3f} "
          f"nan={int(np.isnan(we).sum())} inf={int(np.isinf(we).sum())} "
          f"out_of_bounds={int(((we < -1e-3) | (we > ceiling + 1e-3)).sum())}")
    print(f"AFTER inh: n={wi.size:,} all_-5={bool(np.all(np.abs(wi + 5.0) < 1e-6))} "
          f"nan={int(np.isnan(wi).sum())}")
    after = {(int(x), int(y)): float(v) for x, y, v in zip(s1, t1, w1)}
    moved = sum(1 for k in before if k in after and after[k] != before[k])
    print(f"MOVED {moved:,} of {len(before):,} synapses")
    clean = (not np.isnan(we).any() and not np.isinf(we).any()
             and not ((we < -1e-3) | (we > ceiling + 1e-3)).any()
             and bool(np.all(np.abs(wi + 5.0) < 1e-6)) and bad == 0)
    print("VERDICT:", "CLEAN" if clean else "CORRUPTED")
    if a.dump:
        keys = sorted(after)
        np.savez(a.dump, k=np.array(keys, np.int64),
                 w=np.array([after[k] for k in keys], np.float64))
        print(f"DUMPED {len(keys)} weights -> {a.dump}")
