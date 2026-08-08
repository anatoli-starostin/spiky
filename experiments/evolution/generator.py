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
from uuid import uuid4

from evo_config import (NEW_BORN, SCORED, GENERATOR_PARENTS, GENERATOR_OFFSPRING, SEED_POP,
                        CLAIM_TIMEOUT_S, IMMIGRANT_RATIO, GENERATOR_OVERFETCH, LINEAGE_CAP,
                        MAX_BACKLOG)
from genome_store import GenomeStore
import neuroevo_lut as N


def _lineage_filter(cands, cap, want):
    """From priority-sorted candidates, keep at most `cap` per lineage_id, up to `want` total.
    A very large `cap` reproduces plain top-`want` by priority (legacy behavior)."""
    per, chosen = {}, []
    for c in cands:
        lin = c["lineage_id"]
        if per.get(lin, 0) >= cap:
            continue
        per[lin] = per.get(lin, 0) + 1
        chosen.append(c)
        if len(chosen) >= want:
            break
    return chosen


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
        # Backlog throttle: don't out-run the scorers. If the unscored NEW_BORN pile is at/above
        # the ceiling, skip breeding this iteration (the loop then sleeps) so scoring catches up.
        if MAX_BACKLOG and self.store.count(NEW_BORN) >= MAX_BACKLOG:
            return {"seeded": 0, "bred": 0, "throttled": True}
        # Lineage-capped selection via over-fetch-and-filter (LOCAL rule, no global barrier):
        # peek the top-OVERFETCH SCORED by priority, keep <= LINEAGE_CAP per lineage, then claim
        # only the survivors by id. Degrades gracefully — a peek/claim race just yields fewer.
        cands = self.store.peek_top(SCORED, GENERATOR_OVERFETCH)
        if not cands:
            return {"seeded": 0, "bred": 0}
        chosen = _lineage_filter(cands, LINEAGE_CAP, GENERATOR_PARENTS)
        parents = self.store.claim_ids(self.worker_id, [c["_id"] for c in chosen], SCORED)
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
            lineage = df.get("lineage_id") or str(df["_id"])     # child inherits the FITTER parent's lineage
            offspring.append((child, [df["_id"], dw["_id"]], sf + sw, depth, lineage))
        # IMMIGRANTS: steady trickle of fresh random genomes, each rooting its own lineage.
        n_imm = int(round(len(offspring) * IMMIGRANT_RATIO))
        immigrants = [(N.random_genome(self.rng), [], 0.0, 0, uuid4().hex) for _ in range(n_imm)]
        self.store.insert_new_born(offspring + immigrants)
        self.store.mark_processed([d["_id"] for d in parents])
        return {"seeded": 0, "bred": len(offspring), "parents": len(parents), "immigrants": n_imm}

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
