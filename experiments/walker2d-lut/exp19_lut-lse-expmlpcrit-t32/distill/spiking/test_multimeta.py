"""Per-delay chunks vs one mixed chunk, for the multi-meta build crash.

HYPOTHESIS UNDER TEST (Anatoli): register all metas up front, then grow a SEPARATE
ChunkOfConnections for each delay (each chunk homogeneous in delay / single meta), and add
them all -- instead of growing one mixed chunk spanning many metas.

Each configuration runs in its OWN PROCESS: a CUDA illegal-address poisons the context, so
one process would report every later case as failed regardless of its own merit.

    python test_multimeta.py                       # full table
    python test_multimeta.py --approach per-delay --delays 4 --train 1
"""
import argparse
import subprocess
import sys

import numpy as np
import torch

NE, NI = 200, 50
GS = 8
W_INH = -5.0
FANOUT_E, FANOUT_I = 10, 8


def make_metas(n_delays, lr, gs=GS, homogeneous=False, inh_bank=False):
    """All metas registered UP FRONT: one excitatory meta per delay (min==max==d), plus a
    dedicated frozen inhibitory meta at delay 1.

    homogeneous=True instead mirrors es_harness EXACTLY: every meta identical apart from
    its delay, and NO dedicated inhibitory meta -- inhibitory synapses ride the same delay
    metas and carry their -5 as an explicit per-edge weight. es_harness builds 93,595
    synapses over 20 such metas without crashing, so this is the known-good shape."""
    from spiky.spnet.spnet import SynapseMeta
    exc = [SynapseMeta(learning_rate=lr, min_delay=d, max_delay=d, initial_weight=0.0,
                       min_weight=0.0, max_weight=20.0, initial_noise_level=0.0,
                       weight_decay=0.9, weight_scaling_cf=0.0,
                       _forward_group_size=gs, _backward_group_size=gs)
           for d in range(1, n_delays + 1)]
    if homogeneous:
        return exc
    if inh_bank:
        # mirror steady_state.stage2_metas EXACTLY: a FULL frozen inhibitory bank, one
        # meta per delay, so the net carries 2*n_delays metas rather than n_delays+1.
        return exc + [SynapseMeta(learning_rate=0.0, min_delay=d, max_delay=d,
                                  initial_weight=W_INH, min_weight=W_INH,
                                  max_weight=W_INH, initial_noise_level=0.0,
                                  weight_decay=0.9, weight_scaling_cf=0.0,
                                  _forward_group_size=gs, _backward_group_size=gs)
                      for d in range(1, n_delays + 1)]
    inh = [SynapseMeta(learning_rate=0.0, min_delay=1, max_delay=1,
                       initial_weight=W_INH, min_weight=W_INH, max_weight=W_INH,
                       initial_noise_level=0.0, weight_decay=0.9, weight_scaling_cf=0.0,
                       _forward_group_size=gs, _backward_group_size=gs)]
    return exc + inh


def wiring(n_delays, E, I, rng, homogeneous=False, inh_bank=False):
    """Explicit wiring whose excitatory synapses span ALL n_delays distinct delays.
    Returns {(src, tgt): (meta_index, weight)}. (src,tgt) is unique by construction --
    ChunkOfConnections rule 11 forbids duplicates regardless of meta."""
    # homogeneous: inhibitory rides delay meta 0 (delay 1) like any other synapse
    inh_meta = 0 if homogeneous else n_delays
    keep = {}
    for _ in range(NE * FANOUT_E):
        s, t = int(E[rng.integers(NE)]), int(E[rng.integers(NE)])
        if s != t:
            keep[(s, t)] = (int(rng.integers(0, n_delays)),
                            float(rng.uniform(2.0, 10.0)))
    for _ in range(NI * FANOUT_I):
        s, t = int(I[rng.integers(NI)]), int(E[rng.integers(NE)])
        m = (n_delays + int(rng.integers(0, n_delays))) if inh_bank else inh_meta
        keep[(s, t)] = (m, W_INH)
    return keep, inh_meta


def build(n_delays, per_delay, lr, homogeneous=False, inh_bank=False, dev="cuda"):
    from spiky.spnet.spnet import SpikingNet, NeuronMeta
    from spiky.util.synapse_growth import SynapseGrowthEngine
    from spiky.util.chunk_of_connections import ChunkOfConnections
    from es_harness import group_aligned_weights

    rng = np.random.default_rng(0)
    metas = make_metas(n_delays, lr, homogeneous=homogeneous, inh_bank=inh_bank)
    sp = SpikingNet(synapse_metas=metas,
                    neuron_metas=[NeuronMeta(neuron_type=0, a=0.02, d=8.0),
                                  NeuronMeta(neuron_type=1, a=0.1, d=2.0)],
                    neuron_counts=[NE, NI], initial_synapse_capacity=1 << 22,
                    summation_dtype=torch.float32)
    sp.to_device(dev)
    E, I = sp.get_neuron_ids_by_meta(0), sp.get_neuron_ids_by_meta(1)

    keep, inh_meta = wiring(n_delays, E, I, rng, homogeneous, inh_bank)
    ge = SynapseGrowthEngine(device=dev, synapse_group_size=GS,
                             max_groups_in_buffer=max(4096, 8 * (len(keep) + NE + NI)))
    for _ in range(2):
        ge.register_neuron_type(max_synapses=8 * (NE + NI), growth_command_list=[])
    for ti, ids in ((0, E), (1, I)):
        t = ids.to(torch.int32)
        n = t.numel()
        ge.add_neurons(neuron_type_index=ti, identifiers=t,
                       coordinates=torch.stack([torch.arange(n).float(),
                                                torch.zeros(n),
                                                torch.full((n,), float(ti))], 1))

    def add_chunk(items):
        """items: list of ((s,t),(meta,w)) -> one _grow_explicit + one add_connections."""
        tri = np.array([[m, s, t] for (s, t), (m, _) in items], np.int32)
        w = np.array([w for _, (_, w) in items], np.float32)
        tri_t = torch.tensor(tri, dtype=torch.int32, device=dev)
        w_t = torch.tensor(w, dtype=torch.float32, device=dev)
        conn = ge._grow_explicit(tri_t, 1).get_connections()
        ch = ChunkOfConnections(conn, GS,
                                weights=group_aligned_weights(conn, tri_t, w_t, GS))
        sp.add_connections(ch, 1)
        ch.recycle()

    items = list(keep.items())
    if per_delay:
        # ONE CHUNK PER META: each chunk is homogeneous in delay/meta.
        n_chunks = 0
        for m in range(len(metas)):
            sub = [kv for kv in items if kv[1][0] == m]
            if sub:
                add_chunk(sub)
                n_chunks += 1
    else:
        add_chunk(items)                      # ONE MIXED CHUNK spanning every meta
        n_chunks = 1
    sp.compile(shuffle_synapses_random_seed=None)
    return sp, torch.cat([E, I]), keep, inh_meta, n_chunks, set(int(x) for x in I)


def export(sp, ids):
    """-> {(src,tgt): (weight, delay, meta)}. KEYED, because export order is not stable
    between calls -- an elementwise diff silently compares unrelated synapses."""
    n = sp.count_synapses(ids, True)
    b = [torch.zeros(n, dtype=t, device=ids.device) for t in
         (torch.int32, torch.int32, torch.float32, torch.int32, torch.int32)]
    sp.export_synapses(ids, b[0], b[1], b[2], b[3], b[4], True)   # src, meta, w, delay, tgt
    src, meta, w, dly, tgt = (x.cpu().numpy() for x in b)
    return {(int(s), int(t)): (float(ww), int(dd), int(mm))
            for s, mm, ww, dd, t in zip(src, meta, w, dly, tgt)}


def run_one(approach, n_delays, train, lr=0.1, homogeneous=False, inh_bank=False):
    per_delay = approach == "per-delay"
    torch.manual_seed(1)
    sp, ids, keep, inh_meta, n_chunks, I_ids = build(n_delays, per_delay, lr,
                                                     homogeneous, inh_bank)
    before = export(sp, ids)
    print(f"BUILT chunks={n_chunks} intended={len(keep)} exported={len(before)}")

    # --- weight/delay round trip, keyed on (src,tgt)
    missing = [k for k in keep if k not in before]
    wrong_w = [k for k in keep if k in before
               and abs(before[k][0] - keep[k][1]) > 1e-3]
    def want_delay(m):
        # exc meta m -> delay m+1; any meta in the inhibitory bank -> m - n_delays + 1
        return m + 1 if m < n_delays else m - n_delays + 1
    wrong_d = [k for k in keep if k in before
               and before[k][1] != want_delay(keep[k][0])]
    print(f"ROUNDTRIP missing={len(missing)} wrong_weight={len(wrong_w)} "
          f"wrong_delay={len(wrong_d)}")

    N = NE + NI
    spk = torch.randint(N, [1, 200, 1], device=ids.device, dtype=torch.int32)
    val = torch.ones_like(spk, dtype=torch.float32) * 20.0
    for _ in range(5):
        sp.process_ticks(n_ticks_to_process=200, batch_size=1, n_input_ticks=200,
                         input_values=val, do_train=bool(train), sparse_input=spk,
                         do_record_voltage=False, do_reset_context=False, _stdp_period=32)
        spk.random_(0, N)
    torch.cuda.synchronize()
    after = export(sp, ids)

    common = [k for k in before if k in after]
    # classify by SOURCE, not by meta: under homogeneous metas inhibitory synapses share
    # the excitatory delay metas, so meta index no longer identifies them.
    exc = [k for k in common if k[0] not in I_ids]
    inh = [k for k in common if k[0] in I_ids]
    dw_e = max((abs(after[k][0] - before[k][0]) for k in exc), default=0.0)
    moved_e = sum(1 for k in exc if after[k][0] != before[k][0])
    inh_held = all(abs(after[k][0] - W_INH) < 1e-6 for k in inh)
    print(f"PASS train={train} exc={len(exc)} max|dw|={dw_e:.4f} moved={moved_e} "
          f"inh={len(inh)} inh_all_-5={inh_held}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--approach", default=None, choices=["per-delay", "mixed"])
    ap.add_argument("--delays", type=int, default=4)
    ap.add_argument("--train", type=int, default=0)
    ap.add_argument("--homogeneous", action="store_true")
    ap.add_argument("--gs", type=int, default=GS)
    ap.add_argument("--inh-bank", action="store_true")
    a = ap.parse_args()
    GS = a.gs                       # es_harness, which does not crash, uses 2
    if a.approach:
        run_one(a.approach, a.delays, a.train, homogeneous=a.homogeneous,
                inh_bank=a.inh_bank)
    else:
        print(f"{'approach':22s} {'delays':>6s} {'train':>5s}  result")
        for approach, homo in (("per-delay", False), ("mixed", False),
                               ("per-delay", True), ("mixed", True)):
            for nd in (2, 4, 20):
                for tr in (0, 1):
                    cmd = [sys.executable, __file__, "--approach", approach,
                           "--delays", str(nd), "--train", str(tr)]
                    if homo:
                        cmd.append("--homogeneous")
                    r = subprocess.run(cmd, capture_output=True, text=True)
                    txt = r.stdout + r.stderr
                    keep = [ln.strip() for ln in txt.splitlines()
                            if ln.startswith(("BUILT", "ROUNDTRIP", "PASS"))]
                    if any(ln.startswith("PASS") for ln in keep):
                        res = " | ".join(keep)
                    else:
                        err = [ln.strip() for ln in txt.splitlines() if "Error" in ln]
                        res = ("CRASH " + (err[-1].split("error")[-1].strip()[:70]
                                           if err else f"rc={r.returncode}"))
                        if keep:
                            res = keep[0] + " -> " + res
                    lbl = approach + (" (homog metas)" if homo else " (inh meta)")
                    print(f"{lbl:22s} {nd:6d} {tr:5d}  {res}", flush=True)
