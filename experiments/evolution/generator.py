"""Actor 2 — Generator (issue #81). Bootstraps the collection if empty, then claims top
SCORED genomes, breeds NEW_BORN offspring (reusing tournament + crossover + mutate),
records parent_ids and priority = sum of parents' scores, and flips the parents to
PROCESSED. Runs as its own process.

  python generator.py [--once]
"""
import os
import random
import socket
import sys
import time

from evo_config import (SCORED, GENERATOR_PARENTS, GENERATOR_OFFSPRING, SEED_POP,
                        CLAIM_TIMEOUT_S)
from genome_store import GenomeStore
import neuroevo_lut as N


class Generator:
    def __init__(self, store=None, worker_id=None, rng=None):
        self.store = store if store is not None else GenomeStore()
        self.worker_id = worker_id or ("generator-%s-%d" % (socket.gethostname(), os.getpid()))
        seed = os.environ.get("NE_GEN_SEED")
        self.rng = rng if rng is not None else random.Random(int(seed) if seed else None)

    def bootstrap_if_empty(self):
        if self.store.count() == 0:
            ids = self.store.seed_random(SEED_POP, self.rng)
            return len(ids)
        return 0

    def run_once(self):
        self.store.sweep_stale(CLAIM_TIMEOUT_S)
        seeded = self.bootstrap_if_empty()
        if seeded:
            return {"seeded": seeded, "bred": 0}
        parents = self.store.claim_batch(self.worker_id, SCORED, GENERATOR_PARENTS)
        if not parents:
            return {"seeded": 0, "bred": 0}
        scored = [(float(d["score"]), d) for d in parents]     # (score, doc) for tournament
        offspring = []
        for _ in range(GENERATOR_OFFSPRING):
            a = N.tournament(scored, self.rng)
            b = N.tournament(scored, self.rng)
            (sf, df), (sw, dw) = (a, b) if a[0] >= b[0] else (b, a)   # df = fitter parent
            child = N.mutate(N.crossover(df["genome"], dw["genome"], self.rng), self.rng)
            depth = max(df.get("depth", 0), dw.get("depth", 0)) + 1
            offspring.append((child, [df["_id"], dw["_id"]], sf + sw, depth))
        self.store.insert_new_born(offspring)
        self.store.mark_processed([d["_id"] for d in parents])
        return {"seeded": 0, "bred": len(offspring), "parents": len(parents)}

    def loop(self, poll_s=2.0):
        while True:
            r = self.run_once()
            if not r.get("bred") and not r.get("seeded"):
                time.sleep(poll_s)


def main():
    once = "--once" in sys.argv
    gen = Generator()
    if once:
        r = gen.run_once()
        print("generator[%s] %s" % (gen.worker_id, r))
    else:
        print("generator[%s] looping" % gen.worker_id)
        gen.loop()


if __name__ == "__main__":
    main()
