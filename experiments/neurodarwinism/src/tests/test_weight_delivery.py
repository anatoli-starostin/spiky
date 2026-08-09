"""REGRESSION: every weight a genome specifies must reach the edge it was written for,
and rebuilding the same genome with the same seed must give the same network.

WHAT THIS CATCHES. `build_pool` used to hand its weights to the engine as
`ge._grow_explicit(tri_t, seed, weights=w_t)`. That path's aligner,
`SynapseGrowthEngine._build_group_aligned_weights` (src/spiky/util/synapse_growth.py:403-425),
recovers each block's owning source by carrying the last non-zero source header FORWARD IN
MEMORY ORDER, on the premise it states in its own docstring: "the explicit build lays each
source's groups out contiguously". It does not.

  * The grow kernel gives each (meta, source) sublist its own thread, and every thread takes
    its next block from one global bump allocator --
    `used = atomicAdd(n_allocated, nIntsToAllocate)`
    (native/spiky/synapse_growth/aux_/synapse_growth_kernels_logic.cu:1324). Block order is
    therefore GPU scheduling order.
  * Worse, `finalize() -> merge_chains` then stitches a source's per-meta chains into one by
    DEMOTING all but one root to source_id 0 and linking the tail with a signed offset to an
    arbitrary block, forwards or backwards (same file, 1400-1418).

Measured on exp002's best genome at group size 128: 38,002 blocks, 19,001 occupied, only
1,017 roots left of ~19,000, ZERO of 17,984 chain links contiguous, about half pointing
backward, median jump 5,082 blocks. The forward-fill named the right source for 5.4 % of
blocks, which delivered 47.8 % of the non-zero weights and 32 % of the total weight; the rest
was silently dropped to 0.0 by a lookup miss. And because block placement is nondeterministic,
a different 32 % survived on every rebuild of the identical genome.

`harness.group_aligned_weights` FOLLOWS the chain via next_shift instead of assuming layout, so
it is exact and reproducible even though the layout underneath it still is not. That is what
`build_pool` uses now, and that is what this test pins.

The two failure modes are independent, so both are asserted separately:
  CORRECTNESS   is a host-side logic bug -- it reproduces on CPU, where the buffer is
                byte-identical between builds and the forward-fill is still 91.7 % wrong.
  REPRODUCIBILITY is the device-side atomic ordering.

Not a pytest file (the chapter's tests are standalone scripts):
    python tests/test_weight_delivery.py [--show-broken-path]
Exits non-zero on failure.
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import steady_state as S                          # noqa: E402
from harness import group_aligned_weights         # noqa: E402

STDP_LR, W_MAX, SEED = 0.01, 30.0, 1
# Small fanouts keep the test to ~10k synapses and a couple of seconds, while leaving every
# source spanning many delay metas -- which is what makes merge_chains stitch chains at all,
# and therefore what the bug needs in order to show up.
FANOUTS = dict(fanout_e=8, fanout_i=2, fanout_inh=10, fanout_in=10, fanin_out=10)


def make_genome(seed=0):
    return S.seed_genome(np.random.default_rng(seed), W_MAX, **FANOUTS)


def requested(g, ids):
    """The (src, tgt, meta) -> weight mapping build_pool asks the engine for."""
    base = {S.EXC: S.N_EXC, S.INH: S.N_INH, S.INP: S.N_IN, S.OUTP: S.N_OUT}
    s = np.empty(g["weight"].size, np.int64)
    t = np.empty_like(s)
    for p in (S.EXC, S.INH, S.INP):
        m = g["src_pool"] == p
        if m.any():
            s[m] = ids[p][g["src_idx"][m]]
    for p in (S.EXC, S.INH, S.OUTP):
        m = g["tgt_pool"] == p
        if m.any():
            t[m] = ids[p][g["tgt_idx"][m]]
    meta = (g["delay"] - S.D_MIN).copy()
    if STDP_LR > 0:
        meta[g["src_pool"] == S.INH] += S.N_DELAY_METAS
    w = np.asarray(g["weight"], np.float32)
    k = np.lexsort((meta, t, s))
    return s[k], meta[k], t[k], w[k]


def exported(H):
    """Every synapse the compiled net holds, in the same canonical order."""
    sp, dev = H["spnet"], H["device"]
    all_ids = torch.tensor(np.concatenate(H["ids"]), dtype=torch.int32, device=dev)
    n = int(sp.count_synapses(all_ids, True))
    bufs = [torch.zeros(n, dtype=t, device=dev) for t in
            (torch.int32, torch.int32, torch.float32, torch.int32, torch.int32)]
    sp.export_synapses(all_ids, bufs[0], bufs[1], bufs[2], bufs[3], bufs[4], True)
    s, m, w, d, t = (x.cpu().numpy() for x in bufs)
    k = np.lexsort((m, t, s))
    return s[k], m[k], w[k], t[k]


def build_broken(g, device="cuda"):
    """build_pool as it was BEFORE the fix, for the A/B report. Not used by the assertions."""
    from spiky.spnet.spnet import SpikingNet, NeuronMeta
    from spiky.util.synapse_growth import SynapseGrowthEngine

    metas = S.stage2_metas(STDP_LR, W_MAX)
    nm = [NeuronMeta(neuron_type=0, a=0.02, d=8.0), NeuronMeta(neuron_type=1, a=0.1, d=2.0),
          NeuronMeta(neuron_type=2, a=0.02, d=8.0), NeuronMeta(neuron_type=3, a=0.02, d=8.0)]
    counts = [S.N_EXC, S.N_INH, S.N_IN, S.N_OUT]
    sp = SpikingNet(synapse_metas=metas, neuron_metas=nm, neuron_counts=counts,
                    initial_synapse_capacity=1 << 23, summation_dtype=torch.float32)
    sp.to_device(device)
    ids = [sp.get_neuron_ids_by_meta(i).cpu().numpy() for i in range(4)]
    s, meta, t, w = requested(g, ids)
    triples = np.stack([meta, s, t], 1)

    ge = SynapseGrowthEngine(device=device, synapse_group_size=S.ENGINE_GROUP_SIZE,
                             max_groups_in_buffer=max(4096, 8 * (len(triples) + sum(counts))))
    for i in range(4):
        ge.register_neuron_type(max_synapses=8 * (S.N_EXC + S.N_INH), growth_command_list=[])
    for i in range(4):
        tt = torch.tensor(ids[i], dtype=torch.int32)
        n = tt.numel()
        ge.add_neurons(neuron_type_index=i, identifiers=tt,
                       coordinates=torch.stack([torch.arange(n).float(), torch.zeros(n),
                                                torch.full((n,), float(i))], 1))
    chunk = ge._grow_explicit(torch.tensor(triples, dtype=torch.int32, device=device), SEED,
                              weights=torch.tensor(w, dtype=torch.float32, device=device))
    sp.add_connections(chunk, SEED)
    chunk.recycle()
    sp.compile(shuffle_synapses_random_seed=None)
    return dict(spnet=sp, ids=ids, P=1, device=device, n_syn=len(triples))


def measure(H, g):
    es, em, ew, et = exported(H)
    rs, rm, rt, rw = requested(g, H["ids"])
    topo = (np.array_equal(es, rs) and np.array_equal(em, rm) and np.array_equal(et, rt))
    ok = np.isclose(ew, rw, atol=1e-4)
    nz = rw != 0
    return dict(topology_ok=bool(topo), n=int(ew.size),
                frac_correct=float(ok.mean()),
                nonzero=int(nz.sum()), nonzero_correct=int(ok[nz].sum()),
                nonzero_frac=float(ok[nz].mean()),
                dropped_to_zero=int(((ew == 0) & nz).sum()),
                weight_requested=float(rw.sum()), weight_delivered=float(ew.sum()),
                weight_frac=float(ew.sum() / rw.sum()) if rw.sum() else 1.0,
                exported_weights=ew)


def report(tag, m):
    print(f"  {tag}: topology_ok={m['topology_ok']}  edges {m['n']:,}  "
          f"all-weights correct {100 * m['frac_correct']:.2f} %")
    print(f"      non-zero {m['nonzero']:,}: correct {m['nonzero_correct']:,} "
          f"({100 * m['nonzero_frac']:.2f} %), dropped to 0 {m['dropped_to_zero']:,}")
    print(f"      weight delivered {m['weight_delivered']:,.0f} / "
          f"{m['weight_requested']:,.0f} ({100 * m['weight_frac']:.2f} %)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--show-broken-path", action="store_true",
                    help="also build via the pre-fix _grow_explicit(weights=) path and print "
                         "its numbers, to show what this test is guarding against")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    if a.device.startswith("cuda") and not torch.cuda.is_available():
        print("SKIP: no CUDA device"), sys.exit(0)

    g = make_genome()
    print(f"synthetic genome: {g['weight'].size:,} synapses, "
          f"ENGINE_GROUP_SIZE={S.ENGINE_GROUP_SIZE}, "
          f"{len(np.unique(g['delay']))} distinct delays, stdp_lr={STDP_LR}\n")

    print("build_pool (fixed, chain-following group_aligned_weights):")
    H1 = S.build_pool([g], a.device, seed=SEED, stdp_lr=STDP_LR, w_max=W_MAX)
    m1 = measure(H1, g)
    report("build 1", m1)
    del H1
    torch.cuda.empty_cache()

    H2 = S.build_pool([g], a.device, seed=SEED, stdp_lr=STDP_LR, w_max=W_MAX)
    m2 = measure(H2, g)
    report("build 2", m2)
    del H2
    torch.cuda.empty_cache()

    identical = bool(np.array_equal(m1["exported_weights"], m2["exported_weights"]))
    print(f"  two builds, same seed: weight vectors byte-identical = {identical}")

    if a.show_broken_path:
        print("\npre-fix path (_grow_explicit(weights=)), for comparison only:")
        for i in range(2):
            Hb = build_broken(g, a.device)
            report(f"build {i + 1}", measure(Hb, g))
            del Hb
            torch.cuda.empty_cache()

    fails = []
    # (a) every non-zero weight lands on the edge it was written for
    if m1["nonzero_correct"] != m1["nonzero"]:
        fails.append(f"only {m1['nonzero_correct']:,}/{m1['nonzero']:,} non-zero weights "
                     f"landed correctly ({100 * m1['nonzero_frac']:.2f} %)")
    if m1["frac_correct"] != 1.0:
        fails.append(f"only {100 * m1['frac_correct']:.2f} % of all weights are correct")
    # (b) the whole requested weight is delivered
    if not np.isclose(m1["weight_delivered"], m1["weight_requested"], rtol=1e-6):
        fails.append(f"weight delivered {m1['weight_delivered']:,.1f} != requested "
                     f"{m1['weight_requested']:,.1f} ({100 * m1['weight_frac']:.2f} %)")
    # (c) same genome + same seed -> same network
    if not identical:
        d = m1["exported_weights"] != m2["exported_weights"]
        fails.append(f"two builds with seed {SEED} differ on {int(d.sum()):,}/{d.size:,} edges")
    if not m1["topology_ok"]:
        fails.append("exported topology does not match the requested triples")

    print()
    if fails:
        for f in fails:
            print(f"FAIL: {f}")
        return 1
    print(f"PASS: {m1['nonzero']:,}/{m1['nonzero']:,} non-zero weights exact, "
          f"100.00 % of requested weight delivered, two builds byte-identical")
    return 0


if __name__ == "__main__":
    sys.exit(main())
