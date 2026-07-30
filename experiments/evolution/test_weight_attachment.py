"""test_weight_attachment.py — regression guard for the packed-build weight bug.

The population-packed builder (spnet_harness._build_packed_impl) attaches each edge's
weight to the native SpikingNet via a group-aligned weight buffer. A prior version
re-derived each group's source by "carry the last non-zero source forward" and looked
weights up by a lossy (src, tgt) key — which mis-assigned ~38% of weights (some zeroed,
some stealing another edge's weight) even for a SINGLE genome, and made a genome's
effective weights depend on its packmates.

These tests assert the two invariants that must hold:
  1. the built net's per-edge weights EXACTLY equal the genome's weights;
  2. a genome's weights (and score) are INDEPENDENT of what it is packed with.

Run: pytest test_weight_attachment.py    (or: python test_weight_attachment.py)
"""
import collections

import torch

from neuroevo_lut import build_population, IN_LABELS, OUT_LABELS
import neuroevo_lut as N
from evo_config import fixed_eval_set

# A small, deterministic genome that triggers the bug: several sources with >2 targets
# (forcing chained/ghost groups) across a few hidden neurons. No parallel (src,tgt) edges.
SYN_GENOME = {
    "types": [{"leak": 0.1, "thr": 1.0, "d": 5.0}],
    "hid": {"h0": 0, "h1": 0, "h2": 0},
    "syn": {
        "1": ["i0", "h0", 0.5, 1], "2": ["i0", "h1", -0.7, 2], "3": ["i0", "h2", 1.3, 3],
        "4": ["i1", "h0", 2.1, 1], "5": ["i1", "h1", -1.1, 4], "6": ["h0", "o0", 0.9, 2],
        "7": ["h1", "o0", -0.4, 1], "8": ["h2", "o1", 1.7, 3], "9": ["h0", "o1", -1.9, 5],
        "10": ["h1", "h2", 0.3, 2], "11": ["i2", "h2", -0.8, 1], "12": ["h2", "o0", 2.4, 4],
    },
    "sigma": 0.5,
}

# A different spiking neighbour to pack alongside (its presence must not perturb SYN_GENOME).
NEIGHBOR = {
    "types": [{"leak": 0.2, "thr": 0.8, "d": 12.0}],
    "hid": {"h0": 0, "h1": 0},
    "syn": {
        "1": ["i0", "h0", 1.1, 2], "2": ["i3", "h0", -0.5, 1], "3": ["i0", "h1", 0.7, 3],
        "4": ["h0", "o2", 2.0, 1], "5": ["h1", "o2", -1.3, 4], "6": ["h0", "o0", 0.6, 2],
    },
    "sigma": 0.5,
}


def _n_local(g):
    return len(IN_LABELS) + len(OUT_LABELS) + len(g["hid"])


def _genome_weights(g):
    IN, OUT = IN_LABELS, OUT_LABELS
    loc = {n: i for i, n in enumerate(IN + OUT + list(g["hid"].keys()))}
    return collections.Counter(
        (loc[s], loc[t], int(max(1, d)), round(float(w), 4)) for s, t, w, d in g["syn"].values())


def _built_weights(packed, cand, n_local):
    sp = packed["spnet"]
    ns = sp.n_synapses()
    buf = {k: torch.zeros([ns], dtype=(torch.float32 if k == "w" else torch.int32))
           for k in ["s", "m", "w", "d", "t"]}
    sp.export_synapses(sp.get_all_neuron_ids(), buf["s"], buf["m"], buf["w"], buf["d"], buf["t"],
                       forward_or_backward=True)
    g2l = {packed["gid"](cand, i): i for i in range(n_local)}
    return collections.Counter(
        (g2l[int(buf["s"][i])], g2l[int(buf["t"][i])], int(buf["d"][i]), round(float(buf["w"][i]), 4))
        for i in range(ns) if int(buf["s"][i]) in g2l and int(buf["t"][i]) in g2l)


def test_built_weights_match_genome():
    """Every edge in the built net carries the genome's weight — no zeros, no swaps."""
    g = SYN_GENOME
    got = _built_weights(build_population([g], device="cpu"), 0, _n_local(g))
    want = _genome_weights(g)
    missing = want - got  # genome edges whose (src,tgt,delay,weight) is absent in the build
    assert not missing, "weights mis-assigned; genome edges not reproduced: %s" % list(missing.elements())


def test_weights_independent_of_packing():
    """A genome's built weights must not depend on which neighbours it is packed with."""
    g = SYN_GENOME
    alone = _built_weights(build_population([g], device="cpu"), 0, _n_local(g))
    paired = _built_weights(build_population([g, NEIGHBOR], device="cpu"), 0, _n_local(g))
    assert alone == paired, "candidate-0 weights changed when packed with a neighbour"


def test_score_independent_of_packing():
    """The isolation invariant at the score level: score(g alone) == score(g packed)."""
    g = SYN_GENOME
    xs, tos = fixed_eval_set()
    s_alone = N._score_population(build_population([g], device="cpu"), 1, xs, tos, device="cpu")[0][0]
    s_paired = N._score_population(build_population([g, NEIGHBOR], device="cpu"), 2, xs, tos, device="cpu")[0][0]
    assert abs(s_alone - s_paired) < 1e-9, "score changed with packing: %.6f vs %.6f" % (s_alone, s_paired)


if __name__ == "__main__":
    for fn in [test_built_weights_match_genome, test_weights_independent_of_packing,
               test_score_independent_of_packing]:
        try:
            fn()
            print("PASS", fn.__name__)
        except AssertionError as e:
            print("FAIL", fn.__name__, "->", e)
