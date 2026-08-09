"""Test spnet.ipynb's group-size configuration against our multi-meta explicit build.

FROM spnet.ipynb (read-only, not modified):
    n_subnets=1000, Ne=800, Ni=200, N=1000, M=100
    SynapseGrowthEngine(synapse_group_size=((M+31)//32)*32,               # -> 128
                        max_groups_in_buffer=((N+31)//32)*32 * n_subnets)  # -> 1024*n
    exc meta: lr=0.1, min_delay=0, max_delay=19, _forward/_backward_group_size=8
    inh meta: lr=0.0, delay 0..0, w=-5,          _forward/_backward_group_size=128

THE KEY DIFFERENCE FROM EVERYTHING WE TRIED: the ENGINE group size (128) and the META
forward group sizes (8 / 128) are DIFFERENT. Every earlier sweep of ours held them equal,
so engine=128 + meta_fwd=8 is untested territory. They control different things -- the
engine size sets how many (meta,target) pairs ride in one INPUT block of the connections
buffer, the meta size sets how many synapses land in one OUTPUT forward group.

CAVEATS, applied honestly rather than silently: the notebook is a TWO-meta SPATIAL-growth
net whose single excitatory meta spans delays 0..19 by positional split. Ours is a 40-meta
EXPLICIT build that encodes delay as meta identity. We keep our meta structure (that is what
we need to test) and adopt only the group sizes. Also note bgs=8 for excitatory contradicts
our earlier measurement that the backward structure overflows past ~8 incoming plastic per
target, and our nets carry ~80-105 -- so the runtime check matters as much as the build one.
"""
import argparse
import subprocess
import sys

import numpy as np
import torch

import steady_state as S

F = ("src_pool", "src_idx", "tgt_pool", "tgt_idx", "delay", "weight")
M_FANIN, N_NEURONS = 100, 1000
ENGINE_GS = ((M_FANIN + 31) // 32) * 32          # 128
GROUPS_BUF_PER_SUBNET = ((N_NEURONS + 31) // 32) * 32   # 1024


def ipynb_metas(stdp_lr, w_max, exc_gs, inh_gs):
    """Our 40-meta bank (20 plastic exc + 20 frozen inh), with the notebook's group sizes."""
    from spiky.spnet.spnet import SynapseMeta
    exc = [SynapseMeta(learning_rate=stdp_lr, min_delay=d, max_delay=d,
                       initial_weight=0.0, min_weight=0.0, max_weight=1.5 * w_max,
                       initial_noise_level=0.0, weight_decay=0.9, weight_scaling_cf=0.0,
                       _forward_group_size=exc_gs, _backward_group_size=exc_gs)
           for d in range(S.D_MIN, S.D_MAX + 1)]
    inh = [SynapseMeta(learning_rate=0.0, min_delay=d, max_delay=d,
                       initial_weight=S.RES_W_INH, min_weight=S.RES_W_INH,
                       max_weight=S.RES_W_INH, initial_noise_level=0.0,
                       weight_decay=0.9, weight_scaling_cf=0.0,
                       _forward_group_size=inh_gs, _backward_group_size=inh_gs)
           for d in range(S.D_MIN, S.D_MAX + 1)]
    return exc + inh


def build(genomes, stdp_lr, w_max, engine_gs, exc_gs, inh_gs, device="cuda", seed=1):
    from spiky.spnet.spnet import SpikingNet, NeuronMeta
    from spiky.util.synapse_growth import SynapseGrowthEngine
    from spiky.util.chunk_of_connections import ChunkOfConnections
    K = len(genomes)
    metas = ipynb_metas(stdp_lr, w_max, exc_gs, inh_gs)
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

    ge = SynapseGrowthEngine(device=device, synapse_group_size=engine_gs,
                             max_groups_in_buffer=GROUPS_BUF_PER_SUBNET * K)
    for _ in range(4):
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
    # group_aligned_weights must use the ENGINE group size -- it indexes the chunk layout.
    chunk = ChunkOfConnections(conn, engine_gs,
                               weights=S.group_aligned_weights(conn, tri_t, w_t, engine_gs))
    sp.add_connections(chunk, seed)
    chunk.recycle()
    sp.compile(shuffle_synapses_random_seed=None)
    return dict(spnet=sp, ids=ids, P=K, device=device, n_syn=int(len(triples)))


def load_genomes(path, K, w_max=30.0):
    if path:
        z = np.load(path, allow_pickle=False)
        n = int(z["n_genomes"][0])
        return [{f: z[f"g{i}_{f}"] for f in F} for i in range(n)]
    return [S.seed_genome(np.random.default_rng(i), w_max) for i in range(K)]


def child(a):
    genomes = load_genomes(a.path, a.k)
    h = build(genomes, a.stdp_lr, 30.0, a.engine_gs, a.exc_gs, a.inh_gs)
    torch.cuda.synchronize()
    print(f"BUILD-OK {h['n_syn']}")
    if not a.runtime:
        return
    # ---- runtime + STDP check, keyed on (src,tgt)
    X, Y, Xpool, Ypool, _, _ = S.load(64, 0, 256)
    enc = S.LatencyEncoder(Xpool)
    all_ids = torch.tensor(np.concatenate(h["ids"]), dtype=torch.int32, device="cuda")

    def snap():
        n = h["spnet"].count_synapses(all_ids, True)
        b = [torch.zeros(n, dtype=t, device="cuda") for t in
             (torch.int32, torch.int32, torch.float32, torch.int32, torch.int32)]
        h["spnet"].export_synapses(all_ids, b[0], b[1], b[2], b[3], b[4], True)
        s, m, w, _, t = (x.cpu().numpy() for x in b)
        return {(int(x), int(y)): (float(v), int(mm)) for x, y, v, mm in zip(s, t, w, m)}

    before = snap()
    for i in range(a.episodes):
        Xm, _, _ = S.sample_batch(Xpool, Ypool, 64, 0, i)
        S.run_episode(h, Xm, enc, 200.0, train=True)
    torch.cuda.synchronize()
    print(f"RUNTIME-OK {a.episodes} episodes")
    after = snap()
    common = [k for k in before if k in after]
    exc = [k for k in common if before[k][1] < S.N_DELAY_METAS]
    inh = [k for k in common if before[k][1] >= S.N_DELAY_METAS]
    dw = max((abs(after[k][0] - before[k][0]) for k in exc), default=0.0)
    moved = sum(1 for k in exc if after[k][0] != before[k][0])
    held = all(after[k][0] == S.RES_W_INH for k in inh)
    print(f"STDP exc={len(exc)} max|dw|={dw:.4f} moved={moved} "
          f"inh={len(inh)} inh_all_-5={held}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=128)
    ap.add_argument("--tries", type=int, default=12)
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--engine-gs", type=int, default=ENGINE_GS)
    ap.add_argument("--exc-gs", type=int, default=8)
    ap.add_argument("--inh-gs", type=int, default=128)
    ap.add_argument("--stdp-lr", type=float, default=0.01)
    ap.add_argument("--path", default=None)
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--runtime", action="store_true")
    ap.add_argument("--child", action="store_true")
    a = ap.parse_args()
    if a.child:
        child(a)
        sys.exit(0)
    print(f"engine_gs={a.engine_gs} exc_gs={a.exc_gs} inh_gs={a.inh_gs} K={a.k} "
          f"genomes={'seed' if not a.path else a.path}")
    tal = {"ok": 0, "err": 0, "hang": 0}
    first_err = ""
    for t in range(a.tries):
        cmd = [sys.executable, __file__, "--child", "--k", str(a.k),
               "--engine-gs", str(a.engine_gs), "--exc-gs", str(a.exc_gs),
               "--inh-gs", str(a.inh_gs), "--stdp-lr", str(a.stdp_lr),
               "--episodes", str(a.episodes)]
        if a.path:
            cmd += ["--path", a.path]
        if a.runtime:
            cmd.append("--runtime")
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=a.timeout)
            txt = r.stdout + r.stderr
            if "BUILD-OK" in txt:
                tal["ok"] += 1
                if a.runtime:
                    for ln in txt.splitlines():
                        if ln.startswith(("RUNTIME-OK", "STDP")):
                            print(f"    {ln.strip()}")
                    if "RUNTIME-OK" not in txt:
                        e = [l.strip() for l in txt.splitlines() if "Error" in l]
                        print(f"    RUNTIME FAILED: {e[-1][:90] if e else '?'}")
            else:
                tal["err"] += 1
                e = [l.strip() for l in txt.splitlines() if "Error" in l]
                if e and not first_err:
                    first_err = e[-1][:80]
        except subprocess.TimeoutExpired:
            tal["hang"] += 1
    print(f"  -> ok {tal['ok']}/{a.tries}  error {tal['err']}/{a.tries}  "
          f"HANG {tal['hang']}/{a.tries}" + (f"   {first_err}" if first_err else ""))
