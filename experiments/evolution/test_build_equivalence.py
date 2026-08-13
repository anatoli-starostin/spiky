#!/usr/bin/env python3
"""Equivalence + speed of the VECTORIZED packed build (build_packed / fast, the default)
vs the REFERENCE build (build_packed_ref). The optimized build vectorizes the per-edge
weight alignment and the neuron-id/triple construction; it must produce a packed net that
scores IDENTICALLY (bit-for-bit here) to the reference build, on CPU and GPU.
Run:  PYTHONPATH=<repo>/src:<repo>/experiments/evolution python3 test_build_equivalence.py [cuda]
"""
import random, sys, time
import torch
import neuroevo_lut as N

TOL = 1e-6


def mk(nhid, rng, nt=4):
    types = [{"leak": 0.03 + 0.02 * k, "thr": 0.5 + 0.15 * k, "d": 8.0 * (k + 1)} for k in range(nt)]
    g = {"types": types, "hid": {}, "syn": {}, "sigma": 0.6}
    hl = [N.new_hidden() for _ in range(nhid)]
    for h in hl:
        g["hid"][h] = rng.randrange(nt)
    pool = N.IN_LABELS + hl
    for h in hl:
        for _ in range(2):
            N.add_syn(g, rng.choice(pool), h, rng.uniform(-1, 3), rng.randint(1, 30))
    for o in N.OUT_LABELS:
        for _ in range(3):
            N.add_syn(g, rng.choice(hl or N.IN_LABELS), o, rng.uniform(-1, 3), rng.randint(1, 30))
    return g


def pop(n, nhid, seed):
    rng = random.Random(seed)
    return [mk(nhid, rng) for _ in range(n)]


def dmax(a, b):
    return max(max(abs(x[k] - y[k]) for k in range(3)) for x, y in zip(a, b))


def main():
    torch.set_num_threads(1)
    dev = "cuda" if (len(sys.argv) > 1 and sys.argv[1] == "cuda" and torch.cuda.is_available()) else "cpu"
    rng = random.Random(1)
    xs = [[rng.uniform(-1, 1) for _ in range(N.D)] for _ in range(8)]
    tos = [N.oracle_order(x) for x in xs]
    trials = N.make_stream_trials(random.Random(7), 12)
    gs = pop(200, 8, seed=5)

    sf = N._score_population(N.build_population(gs, dev, ref=False), len(gs), xs, tos, device=dev)
    sr = N._score_population(N.build_population(gs, dev, ref=True), len(gs), xs, tos, device=dev)
    d1 = dmax(sf, sr)
    vf = N._score_population_stream(N.build_population(gs, dev, ref=False), len(gs), trials, 20, device=dev)
    vr = N._score_population_stream(N.build_population(gs, dev, ref=True), len(gs), trials, 20, device=dev)
    d2 = dmax(vf, vr)
    print("[%s] build equivalence (fast vs ref, 200 genomes): single-shot %.3e | streamed %.3e" % (dev, d1, d2))
    assert d1 < TOL and d2 < TOL, "build mismatch: %.3e / %.3e" % (d1, d2)

    def gps(ref):
        g2 = pop(4096, 8, seed=9)
        N.build_population(pop(128, 8, 1), dev, ref=ref)
        t = time.time()
        for _ in range(3):
            N.build_population(g2, dev, ref=ref)
        return 4096 / ((time.time() - t) / 3)

    print("[%s] build speed chunk=4096: fast %.0f g/s | ref %.0f g/s | %.2fx"
          % (dev, gps(False), gps(True), gps(False) / gps(True)))
    print("OK — vectorized build scores identically to reference (< %.0e)" % TOL)
    return 0


if __name__ == "__main__":
    exit(main())
