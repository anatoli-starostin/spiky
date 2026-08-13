#!/usr/bin/env python3
"""Equivalence check: the VECTORIZED packed scorers (_score_population /
_score_population_stream) must reproduce the loop-based REFERENCE scorers
(_score_population_ref / _score_population_stream_ref) on CPU, per-genome, within
float tolerance (the ranking/strict parts are exact; only the mean division rounds
in float32). Run:  PYTHONPATH=<repo>/src python3 test_vectorized_scoring.py
"""
import random
import torch

import neuroevo_lut as N

TOL = 1e-6


def main():
    torch.set_num_threads(1)
    rng = random.Random(5)
    valid = [g for g in (N.random_genome(rng) for _ in range(40)) if g["syn"]]

    xs = [[rng.uniform(-1, 1) for _ in range(N.D)] for _ in range(24)]
    tos = [N.oracle_order(x) for x in xs]
    packed = N.build_population(valid, device="cpu")
    ref = N._score_population_ref(packed, len(valid), xs, tos, device="cpu")
    vec = N._score_population(packed, len(valid), xs, tos, device="cpu")
    d1 = max(max(abs(a[k] - b[k]) for k in range(3)) for a, b in zip(ref, vec))
    print("single-shot: %d genomes x %d corners | max |ref-vec| = %.3e" % (len(ref), len(xs), d1))

    trials = N.make_stream_trials(rng, 16)
    packed2 = N.build_population(valid, device="cpu")
    refs = N._score_population_stream_ref(packed2, len(valid), trials, 20, device="cpu")
    vecs = N._score_population_stream(packed2, len(valid), trials, 20, device="cpu")
    d2 = max(max(abs(a[k] - b[k]) for k in range(3)) for a, b in zip(refs, vecs))
    print("streamed   : %d genomes x %d trials (K=%d, T=20) | max |ref-vec| = %.3e"
          % (len(refs), len(trials), N.STREAM_K, d2))

    assert d1 < TOL, "single-shot mismatch %.3e" % d1
    assert d2 < TOL, "streamed mismatch %.3e" % d2
    print("OK — both vectorized scorers match the reference (max %.3e < %.0e)" % (max(d1, d2), TOL))
    return 0


if __name__ == "__main__":
    exit(main())
