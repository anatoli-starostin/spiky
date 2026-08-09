"""Does pre-sorting the explicit triples de-interleave the block layout?

NOTE BEFORE MEASURING: _grow_explicit already sorts internally (synapse_growth.py:348-351):
    sort_idx = torch.argsort(explicit_triples[:, 1], stable=True)   # by source
    explicit_triples = explicit_triples[sort_idx]
    sort_idx = torch.argsort(explicit_triples[:, 0], stable=True)   # then by meta
    explicit_triples = explicit_triples[sort_idx]
Both sorts are STABLE, so the function normalises any input order to (meta major, source
minor) by itself. If that is the whole story, caller-side sorting cannot change the emitted
buffer at all. This script checks that rather than assuming it.

    python test_sorting.py                    # layout + placement + crash table
    python test_sorting.py --child --order ms --engine-gs 8 --topo meta6
"""
import argparse
import subprocess
import sys

import numpy as np
import torch


def make(n_neurons, n_metas, fanout):
    tri, w = [], []
    for s in range(n_neurons):
        for j in range(fanout):
            t = (s + 1 + j) % n_neurons
            if t != s:
                tri.append([j % n_metas, s, t])      # 0-based indices, mapped to ids later
                w.append(1.0 + (s * fanout + j) % 40)
    return np.array(tri, np.int32), np.array(w, np.float32)


def reorder(tri, w, order):
    if order == "none":
        return tri, w
    if order == "ms":        # (meta, source)
        k = np.lexsort((tri[:, 1], tri[:, 0]))
    elif order == "sm":      # (source, meta)
        k = np.lexsort((tri[:, 0], tri[:, 1]))
    elif order == "mst":     # (meta, source, target) -- also fixes intra-sublist order
        k = np.lexsort((tri[:, 2], tri[:, 1], tri[:, 0]))
    else:
        raise ValueError(order)
    return tri[k], w[k]


def forward_fill_weights(conn, wmap, gs):
    """THE ORIGINAL (buggy) stock algorithm, kept here so we can ask whether pre-sorting
    would have been an alternative to the chain-following fix."""
    block = 4 + 2 * gs
    buf = conn.cpu().tolist()
    n_blocks = len(buf) // block
    wbuf = torch.zeros([n_blocks * gs], dtype=torch.float32)
    cur_src = 0
    for b in range(n_blocks):
        d = b * block
        if buf[d] != 0:
            cur_src = buf[d]
        for j in range(gs):
            tgt = buf[d + 4 + 2 * j + 1]
            if tgt != 0:
                wbuf[b * gs + j] = wmap.get((cur_src, tgt), 0.0)
    return wbuf.to(device=conn.device)


def layout_stats(conn, gs):
    """How interleaved is the emitted buffer? Count chained blocks that physically follow
    their predecessor (shift == +block) versus jumping elsewhere."""
    block = 4 + 2 * gs
    buf = np.asarray(conn.cpu().tolist(), dtype=np.int64).reshape(-1, block)
    roots = int((buf[:, 0] > 0).sum())
    occupied = int(((buf[:, 0] > 0) | (buf[:, 4 + 1::2] > 0).any(1)).sum())
    chained = adjacent = 0
    for b in range(buf.shape[0]):
        shift = int(buf[b, 3])
        if shift != 0:
            chained += 1
            if shift == block:
                adjacent += 1
    return dict(blocks=buf.shape[0], roots=roots, occupied=occupied,
                chained=chained, adjacent=adjacent)


def build(order, engine_gs, n_neurons, n_metas, fanout, meta_gs=8, do_run=True):
    from spiky.spnet.spnet import NeuronMeta, SpikingNet, SynapseMeta
    from spiky.util.chunk_of_connections import ChunkOfConnections
    from spiky.util.synapse_growth import SynapseGrowthEngine

    tri, w = reorder(*make(n_neurons, n_metas, fanout), order)
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
    # map 0-based neuron indices onto the real ids AFTER reordering, so the sort order under
    # test is the order actually handed to _grow_explicit
    tri = np.stack([tri[:, 0], ids[tri[:, 1]], ids[tri[:, 2]]], 1).astype(np.int32)

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
    conn = ge._grow_explicit(tri_t, 1).get_connections()
    st = layout_stats(conn, engine_gs)
    print(f"LAYOUT order={order} gs={engine_gs} blocks={st['blocks']} roots={st['roots']} "
          f"occupied={st['occupied']} chained={st['chained']} "
          f"adjacent={st['adjacent']}/{st['chained']}")
    print(f"CONNHASH {int(conn.sum().item())} {conn.numel()}")
    if not do_run:
        return

    wmap = {(int(s), int(t)): float(v) for (_, s, t), v in zip(tri, w)}
    wbuf = forward_fill_weights(conn, wmap, engine_gs)
    sp.add_connections(ChunkOfConnections(conn, engine_gs, weights=wbuf), 1)
    sp.compile(shuffle_synapses_random_seed=None)
    torch.cuda.synchronize()
    print(f"BUILD-OK {len(tri)}")

    all_ids = torch.tensor(ids, dtype=torch.int32, device="cuda")
    n = sp.count_synapses(all_ids, True)
    b = [torch.zeros(n, dtype=t, device="cuda") for t in
         (torch.int32, torch.int32, torch.float32, torch.int32, torch.int32)]
    sp.export_synapses(all_ids, b[0], b[1], b[2], b[3], b[4], True)
    es, em, ew, ed, et = (x.cpu().numpy() for x in b)
    got = {(int(s), int(t)): (float(v), int(dd)) for s, t, v, dd in zip(es, et, ew, ed)}
    exact = wrong = missing = dly_bad = 0
    for (m, s, t), v in zip(tri, w):
        k = (int(s), int(t))
        if k not in got:
            missing += 1
            continue
        if abs(got[k][0] - float(v)) > 1e-3:
            wrong += 1
        else:
            exact += 1
        if got[k][1] != int(m) + 1:          # meta m has min==max==m+1
            dly_bad += 1
    print(f"ROUNDTRIP exact={exact}/{len(tri)} wrong={wrong} missing={missing} "
          f"delay_wrong={dly_bad}")


TOPOS = {"meta6": (64, 6, 12), "meta40": (64, 40, 40)}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--child", action="store_true")
    ap.add_argument("--order", default="none", choices=["none", "ms", "sm", "mst"])
    ap.add_argument("--engine-gs", type=int, default=2)
    ap.add_argument("--topo", default="meta6")
    ap.add_argument("--tries", type=int, default=3)
    ap.add_argument("--layout-only", action="store_true")
    a = ap.parse_args()
    if a.child:
        build(a.order, a.engine_gs, *TOPOS[a.topo], do_run=not a.layout_only)
        sys.exit(0)

    def child(order, gs, topo, extra=()):
        r = subprocess.run([sys.executable, __file__, "--child", "--order", order,
                            "--engine-gs", str(gs), "--topo", topo, *extra],
                           capture_output=True, text=True, timeout=200)
        return r.stdout + r.stderr

    print("=== 1. does pre-sorting change the emitted buffer at all? (gs=2, meta6) ===")
    for order in ("none", "ms", "sm", "mst"):
        out = child(order, 2, "meta6", ("--layout-only",))
        for ln in out.splitlines():
            if ln.startswith(("LAYOUT", "CONNHASH")):
                print("   ", ln.strip())

    print("\n=== 2. stock FORWARD-FILL placement with pre-sorted triples (gs=2) ===")
    for topo in ("meta6", "meta40"):
        for order in ("none", "ms", "sm", "mst"):
            out = child(order, 2, topo)
            rt = [l.strip() for l in out.splitlines() if l.startswith("ROUNDTRIP")]
            print(f"    {topo:7s} order={order:4s} -> {rt[0] if rt else 'CRASH/none'}")

    print("\n=== 3. does pre-sorting avoid the multi-meta gs>2 crash? ===")
    for topo in ("meta6", "meta40"):
        for gs in (8, 32, 128):
            row = []
            for order in ("none", "ms", "mst"):
                built = sum(1 for _ in range(a.tries)
                            if "BUILD-OK" in child(order, gs, topo))
                row.append(f"{order}={built}/{a.tries}")
            print(f"    {topo:7s} gs={gs:3d} built: " + "  ".join(row), flush=True)
