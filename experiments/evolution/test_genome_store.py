#!/usr/bin/env python3
"""Tests for the Mongo genome-store layer (issue #81), using mongomock (in-memory).
Run: PYTHONPATH=<repo>/src:<repo>/experiments/evolution python3 test_genome_store.py"""
import os
import random
import time

from genome_store import GenomeStore
from evo_config import NEW_BORN, SCORED, PROCESSED, CLAIMED
import neuroevo_lut as N


def _client():
    """Real mongod if NE_REAL_MONGO set (+ MONGO_URI), else in-memory mongomock."""
    if os.environ.get("NE_REAL_MONGO"):
        import pymongo
        return pymongo.MongoClient(os.environ.get("MONGO_URI", "mongodb://localhost:27017"))
    import mongomock
    return mongomock.MongoClient()


def _store():
    c = _client()
    c["neuroevo_test"]["g"].drop()            # isolate each test (real mongo persists)
    return GenomeStore(client=c, db_name="neuroevo_test", collection="g")


def test_roundtrip():
    s = _store(); rng = random.Random(1)
    g = N.random_genome(rng)
    for _ in range(15):
        g = N.mutate(g, rng)              # exercises int-keyed syn, palette, etc.
    (gid,) = s.insert_new_born([(g, [], 0.0)])
    got = s.get(gid)["genome"]
    assert got["syn"] == g["syn"], "syn int-keys must survive round-trip"
    assert got["hid"] == g["hid"] and got["types"] == g["types"] and got["sigma"] == g["sigma"]
    print("  test_roundtrip OK")


def test_no_double_claim():
    s = _store(); s.seed_random(10, random.Random(2))
    a = s.claim_batch("wA", NEW_BORN, 6)
    b = s.claim_batch("wB", NEW_BORN, 6)
    ida = {d["_id"] for d in a}; idb = {d["_id"] for d in b}
    assert not (ida & idb), "no genome claimed by two workers"
    assert len(ida) == 6 and len(idb) == 4, "A takes 6, B takes remaining 4 (=%d,%d)" % (len(ida), len(idb))
    assert s.count(CLAIMED) == 10
    print("  test_no_double_claim OK")


def test_claim_one_atomic():
    s = _store(); s.seed_random(3, random.Random(3))
    d = s.claim_one("w", NEW_BORN)
    assert d and s.count(CLAIMED) == 1 and s.count(NEW_BORN) == 2
    print("  test_claim_one_atomic OK")


def test_sweeper_returns_stale():
    s = _store(); s.seed_random(2, random.Random(4))
    s.claim_batch("w", NEW_BORN, 2)
    assert s.count(CLAIMED) == 2
    n = s.sweep_stale(timeout_s=0, now=time.time() + 10)   # force everything stale
    assert n == 2 and s.count(NEW_BORN) == 2 and s.count(CLAIMED) == 0
    print("  test_sweeper_returns_stale OK")


def test_score_and_process():
    s = _store(); s.seed_random(3, random.Random(5))
    for d in s.claim_batch("w", NEW_BORN, 3):
        s.mark_scored(d["_id"], 0.5, "v1")
    assert s.count(SCORED) == 3 and s.count(CLAIMED) == 0
    top = s.top(SCORED, 3)
    assert all(t["score"] == 0.5 and t["priority"] == 0.5 and t["eval_version"] == "v1" for t in top)
    claimed = s.claim_batch("w2", SCORED, 3)
    s.mark_processed([d["_id"] for d in claimed])
    assert s.count(PROCESSED) == 3
    print("  test_score_and_process OK")


def test_release_on_overshoot():
    s = _store(); s.seed_random(5, random.Random(6))
    c = s.claim_batch("w", NEW_BORN, 5)
    s.release([d["_id"] for d in c[-2:]], NEW_BORN)
    assert s.count(NEW_BORN) == 2 and s.count(CLAIMED) == 3
    print("  test_release_on_overshoot OK")


def main():
    for fn in [test_roundtrip, test_no_double_claim, test_claim_one_atomic,
               test_sweeper_returns_stale, test_score_and_process, test_release_on_overshoot]:
        fn()
    print("ALL genome_store tests PASSED")


if __name__ == "__main__":
    main()
