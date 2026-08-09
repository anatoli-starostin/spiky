"""GPU build soak for the wider excitatory delay range: D_MAX=48 -> 96 metas.

Default is D_MIN,D_MAX = 1,20 -> 20 exc + 20 inh = 40 metas. Widening to 48 gives 48 + 48 = 96,
i.e. more (meta,source) sublists and more chained blocks per source -- precisely the axis the
create_forward_groups / sort_chains_by_synapse_meta race lived on. The race fix is verified at
40 metas; this gates it at 96 before any training run.

The delay range is patched AT RUNTIME in the child process only, so steady_state.py keeps its
default and a supervisor restart of any live experiment is unaffected.

Each attempt runs in a FRESH PROCESS with faulthandler armed, so a hang self-dumps its Python
stack and exits rather than wedging the GPU. Two independent checks per build:
  1. SORT INVARIANT -- walk every chain from its root; synapse_meta_index must be
     non-decreasing (ChunkOfConnections rule 7, what the sort exists to guarantee).
  2. END-TO-END -- every synapse's DELAY and WEIGHT keyed on (src,tgt). Delay IS meta identity
     here, so a mis-sorted chain shows up immediately as a wrong delay.

    python soak_dmax48.py --tries 100 --k 32 --d-max 48
"""
import argparse
import subprocess
import sys
import time

import numpy as np
import torch


def meta_order_violations(conn, gs):
    """Rule 7: along each chain, the per-sublist synapse_meta_index is non-decreasing."""
    block = 4 + 2 * gs
    buf = conn.cpu().numpy().reshape(-1, block).astype(np.int64)
    n = buf.shape[0]
    bad = chains = 0
    for b in np.nonzero(buf[:, 0] > 0)[0]:
        chains += 1
        cur, prev_meta, steps = int(b), -1, 0
        while True:
            if buf[cur, 2] > 0:
                if buf[cur, 1] < prev_meta:
                    bad += 1
                    break
                prev_meta = buf[cur, 1]
            s = int(buf[cur, 3])
            if s == 0 or steps > n:
                break
            nxt = (cur * block + s) // block
            if not 0 <= nxt < n:
                bad += 1
                break
            cur, steps = nxt, steps + 1
    return chains, bad


def child(k, d_max, hang_timeout):
    import faulthandler
    faulthandler.dump_traceback_later(hang_timeout, exit=True)
    import steady_state as S
    from spiky.spnet.spnet import NeuronMeta, SpikingNet
    from spiky.util.synapse_growth import SynapseGrowthEngine

    S.D_MAX = d_max
    S.N_DELAY_METAS = S.D_MAX - S.D_MIN + 1
    metas = S.stage2_metas(0.01, 30.0)
    print(f"METAS {len(metas)}", flush=True)

    genomes = [S.seed_genome(np.random.default_rng(i), 30.0) for i in range(k)]
    counts = [k * S.N_EXC, k * S.N_INH, k * S.N_IN, k * S.N_OUT]
    sp = SpikingNet(synapse_metas=metas,
                    neuron_metas=[NeuronMeta(neuron_type=i, a=0.02 if i != 1 else 0.1,
                                             d=8.0 if i != 1 else 2.0) for i in range(4)],
                    neuron_counts=counts, initial_synapse_capacity=1 << 23,
                    summation_dtype=torch.float32)
    sp.to_device("cuda")
    ids = [sp.get_neuron_ids_by_meta(i).cpu().numpy() for i in range(4)]
    base = {S.EXC: S.N_EXC, S.INH: S.N_INH, S.INP: S.N_IN, S.OUTP: S.N_OUT}

    tri, wts, dls = [], [], []
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
        wts.append(g["weight"]); dls.append(g["delay"])
    triples = np.concatenate(tri, 0)
    weights = np.concatenate(wts, 0)
    delays = np.concatenate(dls, 0)

    t0 = time.time()
    ge = SynapseGrowthEngine(device="cuda", synapse_group_size=S.ENGINE_GROUP_SIZE,
                             max_groups_in_buffer=max(4096, 8 * (len(triples) + sum(counts))))
    for _ in range(4):
        ge.register_neuron_type(max_synapses=8 * (S.N_EXC + S.N_INH), growth_command_list=[])
    for i in range(4):
        tt = torch.tensor(ids[i], dtype=torch.int32)
        n = tt.numel()
        ge.add_neurons(neuron_type_index=i, identifiers=tt,
                       coordinates=torch.stack([torch.arange(n).float(), torch.zeros(n),
                                                torch.full((n,), float(i))], 1))
    tri_t = torch.tensor(triples, dtype=torch.int32, device="cuda")
    w_t = torch.tensor(weights, dtype=torch.float32, device="cuda")
    chunk = ge._grow_explicit(tri_t, 1, weights=w_t)
    torch.cuda.synchronize()

    chains, bad = meta_order_violations(chunk.get_connections(), S.ENGINE_GROUP_SIZE)
    sp.add_connections(chunk, 1)
    chunk.recycle()
    sp.compile(shuffle_synapses_random_seed=None)
    torch.cuda.synchronize()
    dt = time.time() - t0

    all_ids = torch.tensor(np.concatenate(ids), dtype=torch.int32, device="cuda")
    n = sp.count_synapses(all_ids, True)
    b = [torch.zeros(n, dtype=t, device="cuda") for t in
         (torch.int32, torch.int32, torch.float32, torch.int32, torch.int32)]
    sp.export_synapses(all_ids, b[0], b[1], b[2], b[3], b[4], True)
    es, _, ew, ed, et = (x.cpu().numpy() for x in b)
    MAX = int(max(triples[:, 1].max(), triples[:, 2].max(), es.max(), et.max())) + 1
    gk = triples[:, 1].astype(np.int64) * MAX + triples[:, 2]
    ek = es.astype(np.int64) * MAX + et
    go, eo = np.argsort(gk), np.argsort(ek)
    same = gk.shape == ek.shape and np.array_equal(gk[go], ek[eo])
    wd = int((ed[eo] != delays[go]).sum()) if same else -1
    ww = int((np.abs(ew[eo] - weights[go]) > 1e-3).sum()) if same else -1
    print(f"RESULT syn={len(triples)} build={dt:.2f}s chains={chains} "
          f"meta_order_violations={bad} wrong_delays={wd} wrong_weights={ww}", flush=True)
    if bad == 0 and wd == 0 and ww == 0:
        print("BUILD-OK", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tries", type=int, default=100)
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--d-max", type=int, default=48)
    ap.add_argument("--timeout", type=int, default=90)
    ap.add_argument("--child", action="store_true")
    a = ap.parse_args()
    if a.child:
        child(a.k, a.d_max, a.timeout - 5)
        sys.exit(0)

    ok = hang = err = 0
    times, first = [], None
    for t in range(a.tries):
        try:
            r = subprocess.run([sys.executable, __file__, "--child", "--k", str(a.k),
                                "--d-max", str(a.d_max), "--timeout", str(a.timeout)],
                               capture_output=True, text=True, timeout=a.timeout)
            txt = r.stdout + r.stderr
        except subprocess.TimeoutExpired as e:
            txt = (e.stdout or b"").decode() + (e.stderr or b"").decode()
        res = [l for l in txt.splitlines() if l.startswith("RESULT")]
        if "BUILD-OK" in txt:
            ok += 1
            if res:
                times.append(float(res[0].split("build=")[1].split("s")[0]))
            if t == 0 or t % 25 == 0:
                mt = [l for l in txt.splitlines() if l.startswith("METAS")]
                print(f"  attempt {t}: OK  {mt[0] if mt else ''}  {res[0] if res else ''}",
                      flush=True)
        else:
            hung = "Timeout (" in txt
            hang += hung
            err += (not hung)
            print(f"  attempt {t}: {'HANG' if hung else 'FAIL'}", flush=True)
            if first is None:
                first = txt
    print(f"\nD_MAX={a.d_max} K={a.k} GPU: OK {ok}/{a.tries}  HANG {hang}  FAIL {err}")
    if times:
        tm = np.array(times)
        print(f"build time: mean {tm.mean():.2f}s min {tm.min():.2f}s max {tm.max():.2f}s")
    if first:
        print("--- first failure output ---")
        for l in first.splitlines()[:20]:
            print("  " + l[:160])
