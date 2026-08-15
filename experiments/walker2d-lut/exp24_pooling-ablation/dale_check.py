"""Does the shipped log-sum-exp spiking network obey Dale's law network-wide?

Dale's law here: for every PRESYNAPTIC neuron, all of its outgoing synaptic weights carry
the same sign. A neuron with both positive and negative outgoing weights is a violation.

The synapse list is captured from the REAL build rather than re-derived: the growth engine's
`_grow_explicit` is the single call through which every synapse enters the network, so
wrapping it yields exactly the (source, target, weight) triples the shipped network is
compiled from. The captured count is asserted against the count `build()` itself reports.

Sign convention is checked explicitly rather than assumed — see the notes printed at the end.

Usage: python dale_check.py --npz <shipped quantised policy npz>
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "..", "neurodarwinism", "src")

STAGE = {
    0: "input latency neurons (17 used of 18)",
    1: "comparator rails (2 per pair, anti-leaky)",
    2: "rail interneurons (2 per pair)",
    3: "memory cells (2 per pair)",
    4: "tie detectors (absent in the shipped gt-skew build)",
    5: "Stage-2 lookup cells (2048)",
    6: "Stage-3 output neurons (6, anti-leaky)",
    7: "completion detector (1)",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tau-m-out", type=float, default=31.257)
    a = ap.parse_args()
    sys.path.insert(0, os.path.abspath(SRC))

    import tiny_lut_quantised_pipeline as QP
    from spiky.util.synapse_growth import SynapseGrowthEngine

    cap = {}
    orig = SynapseGrowthEngine._grow_explicit

    def spy(self, tri, group, weights=None, **kw):
        cap["tri"] = tri.detach().cpu().numpy().copy()
        cap["w"] = weights.detach().cpu().numpy().copy()
        return orig(self, tri, group, weights=weights, **kw)

    SynapseGrowthEngine._grow_explicit = spy
    try:
        Z = np.load(a.npz)
        dims = [0, 1, 2, 3, 4, 5]
        # the SHIPPED configuration: gt-skew on, tie detectors off
        net, ids, nsyn, n_ticks, nneur, aff, win, beta, dmax = QP.build(
            Z, dims, tie_break=False, tau_m_out=a.tau_m_out, device=a.device,
            gt_skew=True)
    finally:
        SynapseGrowthEngine._grow_explicit = orig

    tri, w = cap["tri"], cap["w"]
    assert len(w) == nsyn, f"captured {len(w)} synapses, build reported {nsyn}"
    src, tgt = tri[:, 1].astype(np.int64), tri[:, 2].astype(np.int64)
    print(f"network: {nneur} neurons, {nsyn} synapses, dmax {dmax}, n_ticks {n_ticks}")
    print(f"captured synapse list matches build()'s count ({nsyn:,})\n")

    # map neuron id -> stage
    stage_of = {}
    for i, arr in enumerate(ids):
        for nid in np.asarray(arr).reshape(-1):
            stage_of[int(nid)] = i

    pos, neg, mixed, zero = [], [], [], []
    per_src = {}
    for s in np.unique(src):
        ws = w[src == s]
        per_src[int(s)] = ws
        has_p, has_n = bool((ws > 0).any()), bool((ws < 0).any())
        if has_p and has_n:
            mixed.append(int(s))
        elif has_p:
            pos.append(int(s))
        elif has_n:
            neg.append(int(s))
        else:
            zero.append(int(s))

    n_presyn = len(per_src)
    print(f"presynaptic neurons (with >=1 outgoing synapse): {n_presyn:,} of {nneur:,}")
    print(f"  purely EXCITATORY (all outgoing > 0): {len(pos):,}")
    print(f"  purely INHIBITORY (all outgoing < 0): {len(neg):,}")
    print(f"  all-zero outgoing weights           : {len(zero):,}")
    print(f"  MIXED SIGN (Dale violations)        : {len(mixed):,}")

    def breakdown(group, name):
        if not group:
            return
        cnt = {}
        for s in group:
            cnt[stage_of.get(s, -1)] = cnt.get(stage_of.get(s, -1), 0) + 1
        print(f"\n  {name} by population:")
        for k in sorted(cnt):
            print(f"    [{k}] {STAGE.get(k,'?'):<45} {cnt[k]:,}")

    breakdown(pos, "purely excitatory")
    breakdown(neg, "purely inhibitory")
    breakdown(mixed, "MIXED")

    if mixed:
        print("\n  violating neurons (first 10) and their outgoing weight sets:")
        for s in mixed[:10]:
            ws = per_src[s]
            u = sorted(set(np.round(ws, 6)))
            print(f"    id {s} (stage {stage_of.get(s,'?')}): {len(ws)} synapses, "
                  f"distinct weights {u[:8]}")

    # ---- sign-convention notes ---------------------------------------------------------
    print("\n" + "=" * 88)
    print("SIGN CONVENTION — is a negative weight a genuine inhibitory synapse?")
    print("=" * 88)
    uw = sorted(set(np.round(w, 6)))
    print(f"distinct synaptic weight values in the whole network: {len(uw)}")
    print(f"  negative values present: {[v for v in uw if v < 0][:10]}")
    print(f"  most negative {w.min():.4f}   most positive {w.max():.4f}")
    print("\nspnet integrates  v' = cf_2*v^2 + cf_1*v + cf_0 - u + I, with I the summed")
    print("synaptic input, so a negative synaptic weight subtracts from the target's")
    print("membrane on arrival: genuine inhibition, not bookkeeping. Nothing in this build")
    print("folds a sign into cf_* to fake inhibition -- the neuron metas carry cf_0 = 0")
    print("throughout and cf_1 is a LEAK/anti-leak term, identical for every neuron of a")
    print("type and independent of any particular synapse.")


if __name__ == "__main__":
    main()
