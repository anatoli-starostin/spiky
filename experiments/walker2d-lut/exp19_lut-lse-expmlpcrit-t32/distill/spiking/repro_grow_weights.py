"""Test passing per-edge weights DIRECTLY to _grow_explicit(weights=...), no group_aligned_weights.

Two questions, and the second matters more:
  (a) does it BUILD at engine gs 2/8/32/128?
  (b) do the weights land on the RIGHT edges? Keyed (src,tgt) round-trip against the input.

Why (b) is the crux: _grow_explicit(weights=) does not hand the array to the engine. It builds
a {(src,tgt): w} map and then calls self._build_group_aligned_weights(...), the STOCK helper,
whose own docstring states the assumption "the explicit build lays each source's groups out
contiguously" and recovers the owning source with a forward-fill over block order. That
assumption is false -- next_shift points at an arbitrary block -- which is exactly why
group_aligned_weights was written. So the stock path should be correct ONLY when nothing
chains, i.e. when every (meta,source) sublist fits in one block of gs slots.

Hence two topologies:
  NO-CHAIN : 4 metas x fanout 4 -> 1 synapse per (meta,source) sublist. Fits any gs.
  CHAINING : 2 metas x fanout 8 -> 4 synapses per sublist. Chains whenever gs < 4.
"""
import argparse
import subprocess
import sys

import numpy as np
import torch


USE_OURS = "--ours" in sys.argv


def run(engine_gs, n_neurons, n_metas, fanout, meta_gs, verbose=True):
    from spiky.spnet.spnet import NeuronMeta, SpikingNet, SynapseMeta
    from spiky.util.synapse_growth import SynapseGrowthEngine

    metas = [SynapseMeta(learning_rate=0.1, min_delay=d + 1, max_delay=d + 1,
                         initial_weight=0.0, min_weight=0.0, max_weight=45.0,
                         initial_noise_level=0.0, weight_decay=0.9, weight_scaling_cf=0.0,
                         _forward_group_size=meta_gs, _backward_group_size=meta_gs)
             for d in range(n_metas)]
    sp = SpikingNet(synapse_metas=metas,
                    neuron_metas=[NeuronMeta(neuron_type=0, a=0.02, d=8.0)],
                    neuron_counts=[n_neurons], initial_synapse_capacity=1 << 20,
                    summation_dtype=torch.float32)
    sp.to_device("cuda")
    ids = sp.get_neuron_ids_by_meta(0).cpu().numpy()

    tri, w = [], []
    for s in range(n_neurons):
        for j in range(fanout):
            t = (s + 1 + j) % n_neurons
            if t != s:
                tri.append([j % n_metas, int(ids[s]), int(ids[t])])
                # a weight unique to this edge, so a misplacement cannot hide
                w.append(1.0 + (s * fanout + j) % 40)
    tri = np.array(tri, np.int32)
    w = np.array(w, np.float32)

    ge = SynapseGrowthEngine(device="cuda", synapse_group_size=engine_gs,
                             max_groups_in_buffer=max(4096, 8 * (len(tri) + n_neurons)))
    ge.register_neuron_type(max_synapses=8 * n_neurons, growth_command_list=[])
    nid = torch.tensor(ids, dtype=torch.int32)
    ge.add_neurons(neuron_type_index=0, identifiers=nid,
                   coordinates=torch.stack([torch.arange(nid.numel()).float(),
                                            torch.zeros(nid.numel()),
                                            torch.zeros(nid.numel())], dim=1))

    tri_t = torch.tensor(tri, dtype=torch.int32, device="cuda")
    w_t = torch.tensor(w, dtype=torch.float32, device="cuda")

    if USE_OURS:
        # CONTROL: what we do today -- chain-following helper + explicit weights buffer.
        from es_harness import group_aligned_weights
        from spiky.util.chunk_of_connections import ChunkOfConnections
        conn = ge._grow_explicit(tri_t, 1).get_connections()
        chunk = ChunkOfConnections(conn, engine_gs,
                                   weights=group_aligned_weights(conn, tri_t, w_t, engine_gs))
    else:
        # THE PROPOSED APPROACH: weights straight into _grow_explicit, aligned with triples.
        chunk = ge._grow_explicit(tri_t, 1, weights=w_t)
    sp.add_connections(chunk, 1)
    chunk.recycle()
    sp.compile(shuffle_synapses_random_seed=None)
    torch.cuda.synchronize()
    print(f"BUILD-OK synapses={len(tri)}")

    # ---- keyed (src,tgt) round trip
    all_ids = torch.tensor(ids, dtype=torch.int32, device="cuda")
    n = sp.count_synapses(all_ids, True)
    b = [torch.zeros(n, dtype=t, device="cuda") for t in
         (torch.int32, torch.int32, torch.float32, torch.int32, torch.int32)]
    sp.export_synapses(all_ids, b[0], b[1], b[2], b[3], b[4], True)
    es, _, ew, _, et = (x.cpu().numpy() for x in b)
    got = {(int(a_), int(b_)): float(c_) for a_, b_, c_ in zip(es, et, ew)}
    want = {(int(s), int(t)): float(v) for (_, s, t), v in zip(tri, w)}
    missing = [k for k in want if k not in got]
    wrong = [k for k in want if k in got and abs(got[k] - want[k]) > 1e-3]
    exact = len(want) - len(missing) - len(wrong)
    print(f"ROUNDTRIP exact={exact}/{len(want)} missing={len(missing)} wrong={len(wrong)}")
    if wrong[:3] and verbose:
        for k in wrong[:3]:
            print(f"    edge {k}: want {want[k]:.1f} got {got[k]:.1f}")


TOPOS = {"no-chain": (64, 4, 4), "chaining": (64, 2, 8),
         "meta1": (64, 1, 8), "meta6": (64, 6, 12), "meta40": (64, 40, 40)}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--child", action="store_true")
    ap.add_argument("--engine-gs", type=int, default=2)
    ap.add_argument("--topo", default="no-chain")
    ap.add_argument("--meta-gs", type=int, default=8)
    ap.add_argument("--tries", type=int, default=4)
    ap.add_argument("--ours", action="store_true",
                    help="control: current group_aligned_weights + ChunkOfConnections path")
    a = ap.parse_args()
    if a.child:
        run(a.engine_gs, *TOPOS[a.topo], a.meta_gs)
        sys.exit(0)

    for topo in ("no-chain", "chaining"):
        n, m, f = TOPOS[topo]
        print(f"--- topology {topo}: {n} neurons, {m} metas, fanout {f} "
              f"({f // m} synapses per (meta,source) sublist)")
        for gs in (2, 8, 32, 128):
            ok = 0
            rt = ""
            for _ in range(a.tries):
                r = subprocess.run(
                    [sys.executable, __file__, "--child", "--engine-gs", str(gs),
                     "--topo", topo, "--meta-gs", str(a.meta_gs)],
                    capture_output=True, text=True, timeout=200)
                txt = r.stdout + r.stderr
                if "BUILD-OK" in txt:
                    ok += 1
                    for ln in txt.splitlines():
                        if ln.startswith("ROUNDTRIP"):
                            rt = ln.strip()
            print(f"    engine_gs {gs:3d}: built {ok}/{a.tries}   {rt}", flush=True)
