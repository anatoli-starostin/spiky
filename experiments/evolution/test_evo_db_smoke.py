#!/usr/bin/env python3
"""End-to-end smoke test of the two-actor DB neuroevolution loop (issue #81), on CPU with
mongomock: seed random genomes -> Scorer once -> Generator once, asserting the state machine
NEW_BORN -> SCORED -> PROCESSED and that offspring appear as NEW_BORN carrying parent_ids.
Run: PYTHONPATH=<repo>/src:<repo>/experiments/evolution python3 test_evo_db_smoke.py"""
import os
import random

from genome_store import GenomeStore
from scorer import Scorer
from generator import Generator
from evo_config import NEW_BORN, SCORED, PROCESSED


def _client():
    if os.environ.get("NE_REAL_MONGO"):
        import pymongo
        return pymongo.MongoClient(os.environ.get("MONGO_URI", "mongodb://localhost:27017"))
    import mongomock
    return mongomock.MongoClient()


def main():
    c = _client()
    c["neuroevo_test"]["smoke"].drop()
    store = GenomeStore(client=c, db_name="neuroevo_test", collection="smoke")
    rng = random.Random(11)

    # bootstrap path: Generator on an empty collection seeds NEW_BORN
    gen = Generator(store=store, worker_id="gen", rng=rng)
    r0 = gen.run_once()
    assert r0["seeded"] > 0 and store.count(NEW_BORN) == r0["seeded"], "bootstrap seeding"
    print("  seeded %d NEW_BORN" % r0["seeded"])

    # Scorer once: NEW_BORN -> SCORED with real scores (fixed eval set, CPU)
    sc = Scorer(store=store, worker_id="sc", device="cpu")
    n = sc.run_once()
    assert n > 0 and store.count(SCORED) >= 1, "scorer produced SCORED genomes"
    top = store.top(SCORED, 5)
    assert all(t["score"] is not None and t["eval_version"] for t in top), "scores + eval_version stored"
    print("  scored %d genomes; top score=%.4f, eval_version=%s" % (n, top[0]["score"], top[0]["eval_version"]))

    # Generator once (now non-empty): SCORED -> PROCESSED, offspring as NEW_BORN with parent_ids
    r1 = gen.run_once()
    assert r1["bred"] > 0 and store.count(PROCESSED) >= 1, "generator bred + retired parents"
    offspring = store.top(NEW_BORN, 10)
    assert any(len(o["parent_ids"]) == 2 for o in offspring), "offspring carry 2 parent_ids"
    lineage = [o for o in offspring if o["parent_ids"]][0]
    assert all(store.get(pid) is not None for pid in lineage["parent_ids"]), "parent_ids resolve"
    print("  bred %d offspring from %d parents; example priority=%.4f (=sum parent scores)"
          % (r1["bred"], r1["parents"], lineage["priority"]))

    print("states: NEW_BORN=%d SCORED=%d PROCESSED=%d total=%d"
          % (store.count(NEW_BORN), store.count(SCORED), store.count(PROCESSED), store.count()))
    print("ALL smoke assertions PASSED")


if __name__ == "__main__":
    main()
