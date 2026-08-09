"""Short, fast probe that localises the K=128 build hang to a construction PHASE.

Fresh process per attempt with a short timeout. The child prints a timestamped line as it
ENTERS each phase, so the last line printed by a hung attempt names the wedged call. It also
arms faulthandler.dump_traceback_later(), so a hung child dumps its OWN Python stack and
exits instead of having to be killed blind — that stack shows exactly which native call it is
sitting in.

Phase -> native work it covers:
    GROW_EXPLICIT   grow_synapses + sort_chains_by_synapse_meta (the CAS kernel) + final_sort
    ADD_CONNECTIONS estimate_forward_groups_capacity + create_forward_groups
    COMPILE         backward stats/groups, fill_backward_groups
    INFER           post-construction, to prove construction is the culprit and not the rest

    python probe_hang_phases.py --tries 40 --timeout 30 --k 128
"""
import argparse
import subprocess
import sys
import time

STAGES = ("SEED", "SPNET", "TO_DEVICE", "ENGINE", "ADD_NEURONS", "GROW_EXPLICIT",
          "ADD_CONNECTIONS", "COMPILE", "INFER", "DONE")


def child(k, hang_timeout):
    import faulthandler
    faulthandler.dump_traceback_later(hang_timeout, exit=True)  # self-dump on hang

    t0 = time.time()

    def phase(name):
        print(f"PHASE {name} t={time.time()-t0:.2f}", flush=True)

    import numpy as np
    import torch
    import steady_state as S
    from spiky.spnet.spnet import NeuronMeta, SpikingNet
    from spiky.util.synapse_growth import SynapseGrowthEngine

    phase("SEED")
    genomes = [S.seed_genome(np.random.default_rng(i), 30.0) for i in range(k)]
    metas = S.stage2_metas(0.01, 30.0)
    counts = [k * S.N_EXC, k * S.N_INH, k * S.N_IN, k * S.N_OUT]

    phase("SPNET")
    sp = SpikingNet(synapse_metas=metas,
                    neuron_metas=[NeuronMeta(neuron_type=i, a=0.02 if i != 1 else 0.1,
                                             d=8.0 if i != 1 else 2.0) for i in range(4)],
                    neuron_counts=counts, initial_synapse_capacity=1 << 23,
                    summation_dtype=torch.float32)
    phase("TO_DEVICE")
    sp.to_device("cuda")
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

    phase("ENGINE")
    ge = SynapseGrowthEngine(device="cuda", synapse_group_size=S.ENGINE_GROUP_SIZE,
                             max_groups_in_buffer=max(4096, 8 * (len(triples) + sum(counts))))
    for _ in range(4):
        ge.register_neuron_type(max_synapses=8 * (S.N_EXC + S.N_INH), growth_command_list=[])

    phase("ADD_NEURONS")
    for i in range(4):
        tt = torch.tensor(ids[i], dtype=torch.int32)
        n = tt.numel()
        ge.add_neurons(neuron_type_index=i, identifiers=tt,
                       coordinates=torch.stack([torch.arange(n).float(), torch.zeros(n),
                                                torch.full((n,), float(i))], 1))
    tri_t = torch.tensor(triples, dtype=torch.int32, device="cuda")
    w_t = torch.tensor(weights, dtype=torch.float32, device="cuda")

    phase("GROW_EXPLICIT")
    chunk = ge._grow_explicit(tri_t, 1, weights=w_t)
    torch.cuda.synchronize()

    phase("ADD_CONNECTIONS")
    sp.add_connections(chunk, 1)
    chunk.recycle()
    torch.cuda.synchronize()

    phase("COMPILE")
    sp.compile(shuffle_synapses_random_seed=None)
    torch.cuda.synchronize()

    phase("INFER")
    N = sum(counts)
    spk = torch.randint(N, [1, 32, 1], device="cuda", dtype=torch.int32)
    val = torch.ones_like(spk, dtype=torch.float32) * 20.0
    sp.process_ticks(n_ticks_to_process=32, batch_size=1, n_input_ticks=32,
                     input_values=val, do_train=False, sparse_input=spk,
                     do_record_voltage=False, do_reset_context=True)
    torch.cuda.synchronize()

    phase("DONE")
    print(f"BUILD-OK synapses={len(triples)}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tries", type=int, default=40)
    ap.add_argument("--timeout", type=int, default=30)
    ap.add_argument("--k", type=int, default=128)
    ap.add_argument("--child", action="store_true")
    ap.add_argument("--stop-on-first", action="store_true",
                    help="exit as soon as one hang/error is caught, dumping its full output")
    a = ap.parse_args()
    if a.child:
        child(a.k, a.timeout - 5)      # self-dump slightly before the parent's hard timeout
        sys.exit(0)

    tally = {"ok": 0, "error": 0, "hang": 0}
    last_phase = {}
    for t in range(a.tries):
        try:
            r = subprocess.run([sys.executable, __file__, "--child", "--k", str(a.k),
                                "--timeout", str(a.timeout)],
                               capture_output=True, text=True, timeout=a.timeout)
            txt = r.stdout + r.stderr
        except subprocess.TimeoutExpired as e:
            txt = (e.stdout or b"").decode() + (e.stderr or b"").decode()
        phases = [l.split()[1] for l in txt.splitlines() if l.startswith("PHASE ")]
        where = phases[-1] if phases else "NO_PHASE"
        if "BUILD-OK" in txt:
            tally["ok"] += 1
            verdict = "ok"
        elif "Timeout (" in txt or "dump_traceback_later" in txt or where != "DONE":
            # faulthandler fired, or the child never reached DONE
            if "Error" in txt and "Timeout (" not in txt:
                tally["error"] += 1
                verdict = "ERROR@" + where
            else:
                tally["hang"] += 1
                verdict = "HANG@" + where
                last_phase[where] = last_phase.get(where, 0) + 1
                stack = [l for l in txt.splitlines()
                         if "File " in l or l.startswith("Thread") or "Timeout (" in l]
                if tally["hang"] <= 2 and stack:
                    print("    --- faulthandler stack from hung child ---", flush=True)
                    for l in stack[:14]:
                        print("    " + l.strip()[:150], flush=True)
        else:
            tally["error"] += 1
            verdict = "ERROR@" + where
        print(f"  attempt {t}: {verdict}", flush=True)
        if a.stop_on_first and verdict.startswith(("HANG", "ERROR")):
            # one hang is enough to diagnose: dump everything the child produced, including
            # the finalize-kernel trace markers, and stop
            print("  --- STOPPING AT FIRST FAILURE; full child output follows ---", flush=True)
            for line in txt.splitlines():
                print("  | " + line[:200], flush=True)
            break
    print(f"\nK={a.k}  ok {tally['ok']}/{a.tries}  error {tally['error']}  hang {tally['hang']}")
    if last_phase:
        print("HANG PHASE DISTRIBUTION: " + ", ".join(f"{k}={v}" for k, v in last_phase.items()))
