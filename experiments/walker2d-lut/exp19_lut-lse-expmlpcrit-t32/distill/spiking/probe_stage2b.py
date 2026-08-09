"""Which PLASTIC synapse class triggers the stage-two training crash?

Established: with every meta frozen, do_train=True runs fine on the full 97k-synapse net --
so plasticity is required. stage2_metas makes plastic EVERY non-inhibitory synapse, which
means input drive (INP->EXC) and the readout (EXC->OUTP) become plastic too, not just the
reservoir. test_multimeta only ever made plastic a single pool's exc->exc synapses, and it
trains fine, so the difference is which CLASS carries plasticity.

Three meta banks: 0..19 plastic exc, 20..39 FROZEN exc, 40..59 frozen inh. Each variant
routes a different subset into the plastic bank; everything else goes to the frozen exc
bank, so wiring and weights are byte-identical across variants.
"""
import argparse
import subprocess
import sys

import numpy as np
import torch

import steady_state as S

NB = S.N_DELAY_METAS                      # 20


def three_bank_metas(stdp_lr, w_max, gs=S.GROUP_SIZE, bgs=None):
    """bgs: _backward_group_size, INDEPENDENT of the forward size. The build crash forced
    forward to 2; the backward structure is what STDP walks, so it is the natural suspect
    for a per-target capacity limit."""
    from spiky.spnet.spnet import SynapseMeta
    bgs = gs if bgs is None else bgs

    def band(lr, lo, hi, init):
        return [SynapseMeta(learning_rate=lr, min_delay=d, max_delay=d,
                            initial_weight=init, min_weight=lo, max_weight=hi,
                            initial_noise_level=0.0, weight_decay=0.9,
                            weight_scaling_cf=0.0, _forward_group_size=gs,
                            _backward_group_size=bgs)
                for d in range(S.D_MIN, S.D_MAX + 1)]
    return (band(stdp_lr, 0.0, 1.5 * w_max, 0.0)          # 0..19  plastic exc
            + band(0.0, 0.0, 1.5 * w_max, 0.0)            # 20..39 frozen  exc
            + band(0.0, S.RES_W_INH, S.RES_W_INH, S.RES_W_INH))   # 40..59 frozen inh


def plastic_mask(g, variant):
    """Which of this genome's synapses go into the PLASTIC bank."""
    exc_src = g["src_pool"] != S.INH
    from_inp = g["src_pool"] == S.INP
    to_out = g["tgt_pool"] == S.OUTP
    reservoir = exc_src & ~from_inp & ~to_out
    return {"all": exc_src,
            "reservoir": reservoir,
            "input": exc_src & from_inp,
            "readout": exc_src & to_out,
            "none": np.zeros_like(exc_src)}[variant]


def build(genomes, variant, stdp_lr, w_max=30.0, device="cuda", seed=1, bgs=None):
    from spiky.spnet.spnet import SpikingNet, NeuronMeta
    from spiky.util.synapse_growth import SynapseGrowthEngine
    from spiky.util.chunk_of_connections import ChunkOfConnections
    K = len(genomes)
    metas = three_bank_metas(stdp_lr, w_max, bgs=bgs)
    neuron_metas = [NeuronMeta(neuron_type=0, a=0.02, d=8.0),
                    NeuronMeta(neuron_type=1, a=0.1, d=2.0),
                    NeuronMeta(neuron_type=2, a=0.02, d=8.0),
                    NeuronMeta(neuron_type=3, a=0.02, d=8.0)]
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
        pm = plastic_mask(g, variant)
        meta[~pm & (g["src_pool"] != S.INH)] += NB          # frozen exc bank
        meta[g["src_pool"] == S.INH] += 2 * NB              # frozen inh bank
        tri.append(np.stack([meta, s, t], 1))
        wts.append(g["weight"])
    triples, weights = np.concatenate(tri, 0), np.concatenate(wts, 0)
    total = sum(counts)
    ge = SynapseGrowthEngine(device=device, synapse_group_size=S.GROUP_SIZE,
                             max_groups_in_buffer=max(4096, 8 * (len(triples) + total)))
    for i in range(4):
        ge.register_neuron_type(max_synapses=8 * (S.N_EXC + S.N_INH),
                                growth_command_list=[])
    for i in range(4):
        tt = torch.tensor(ids[i], dtype=torch.int32)
        n = tt.numel()
        ge.add_neurons(neuron_type_index=i, identifiers=tt,
                       coordinates=torch.stack([torch.arange(n).float(), torch.zeros(n),
                                                torch.full((n,), float(i))], 1))
    tri_t = torch.tensor(triples, dtype=torch.int32, device=device)
    w_t = torch.tensor(weights, dtype=torch.float32, device=device)
    chunk = ge._grow_explicit(tri_t, seed)
    conn = chunk.get_connections()
    chunk = ChunkOfConnections(conn, S.GROUP_SIZE, weights=S.group_aligned_weights(
        conn, tri_t, w_t, S.GROUP_SIZE))
    sp.add_connections(chunk, seed)
    chunk.recycle()
    sp.compile(shuffle_synapses_random_seed=None)
    n_pl = int(sum(plastic_mask(g, variant).sum() for g in genomes))
    return dict(spnet=sp, ids=ids, P=K, device=device, n_syn=len(triples)), n_pl


VARIANTS = ["none", "reservoir", "input", "readout", "all"]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default=None)
    ap.add_argument("--b", type=int, default=64)
    ap.add_argument("--fanout-e", type=int, default=80,
                    help="exc->exc fanout == INCOMING plastic synapses per exc target")
    ap.add_argument("--sweep-fanout", action="store_true")
    ap.add_argument("--sweep-bgs", action="store_true")
    ap.add_argument("--bgs", type=int, default=None)
    a = ap.parse_args()
    if a.sweep_bgs:
        for bgs in (2, 4, 8, 16, 32, 64, 128):
            r = subprocess.run([sys.executable, __file__, "--variant", "all",
                                "--b", str(a.b), "--bgs", str(bgs)],
                               capture_output=True, text=True)
            txt = r.stdout + r.stderr
            ok = "TRAIN-OK" in txt
            err = [ln.strip() for ln in txt.splitlines() if "Error" in ln]
            print(f"  backward_group_size = {bgs:3d} (forward stays 2) -> "
                  f"{'TRAIN-OK' if ok else 'CRASH ' + (err[-1].split('error')[-1].strip()[:50] if err else '?')}",
                  flush=True)
        sys.exit(0)
    if a.sweep_fanout:
        for fe in (2, 4, 8, 16, 32, 64, 80):
            r = subprocess.run([sys.executable, __file__, "--variant", "reservoir",
                                "--b", str(a.b), "--fanout-e", str(fe)],
                               capture_output=True, text=True)
            txt = r.stdout + r.stderr
            ok = "TRAIN-OK" in txt
            err = [ln.strip() for ln in txt.splitlines() if "Error" in ln]
            print(f"  fanout_e = {fe:3d} (~{fe} incoming plastic/exc target) -> "
                  f"{'TRAIN-OK' if ok else 'CRASH ' + (err[-1].split('error')[-1].strip()[:50] if err else '?')}",
                  flush=True)
        sys.exit(0)
    if a.variant:
        X, Y, Xpool, Ypool, _, _ = S.load(a.b, 0, 256)
        enc = S.LatencyEncoder(Xpool)
        genomes = [S.seed_genome(np.random.default_rng(0), 30.0,
                                 fanout_e=a.fanout_e, fanout_i=2,
                                 fanout_inh=10, fanout_in=10, fanin_out=10)]
        h, n_pl = build(genomes, a.variant, 0.1, bgs=a.bgs)
        print(f"BUILT plastic={n_pl:,}/{h['n_syn']:,}")
        Xb, _, _ = S.sample_batch(Xpool, Ypool, a.b, 0, 0)
        S.run_episode(h, Xb, enc, 200.0, train=True)
        torch.cuda.synchronize()
        print("TRAIN-OK")
    else:
        for v in VARIANTS:
            r = subprocess.run([sys.executable, __file__, "--variant", v,
                                "--b", str(a.b)], capture_output=True, text=True)
            txt = r.stdout + r.stderr
            ok = "TRAIN-OK" in txt
            err = [ln.strip() for ln in txt.splitlines() if "Error" in ln]
            b = [ln for ln in txt.splitlines() if ln.startswith("BUILT")]
            n = b[0].split("plastic=")[1] if b else "?"
            print(f"  plastic = {v:10s} {n:>18s} -> "
                  f"{'TRAIN-OK' if ok else 'CRASH ' + (err[-1].split('error')[-1].strip()[:58] if err else '?')}",
                  flush=True)
