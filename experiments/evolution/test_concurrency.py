#!/usr/bin/env python3
"""TRUE multi-process concurrency test (issue #81) — REQUIRES a real mongod (mongomock can't
model concurrent processes). Seeds N NEW_BORN, then runs several Scorer processes genuinely in
parallel against the same DB and asserts the two-phase claim held: every genome scored EXACTLY
once (n_scored==1 for all), nothing lost/duplicated, nothing stuck CLAIMED.

  MONGO_URI=mongodb://localhost:27017 PYTHONPATH=<repo>/src:<repo>/experiments/evolution \
      python3 test_concurrency.py
"""
import os
import random
import subprocess
import sys
import time

import pymongo

from genome_store import GenomeStore
from evo_config import NEW_BORN, SCORED, CLAIMED

URI = os.environ.get("MONGO_URI", "mongodb://localhost:27017")
DB = os.environ.get("NE_DB", "neuroevo_conc_test")
COLL = os.environ.get("NE_COLLECTION", "genomes")
N_SEED = int(os.environ.get("NE_CONC_SEED", 500))
N_WORKERS = int(os.environ.get("NE_CONC_WORKERS", 3))
HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    client = pymongo.MongoClient(URI, serverSelectionTimeoutMS=5000)
    client.admin.command("ping")                     # fail fast if mongod is down
    client[DB][COLL].drop()
    store = GenomeStore(client=client, db_name=DB, collection=COLL)
    store.seed_random(N_SEED, random.Random(123))
    assert store.count(NEW_BORN) == N_SEED
    print("seeded %d NEW_BORN; launching %d concurrent Scorer --drain processes on the SAME mongod..."
          % (N_SEED, N_WORKERS), flush=True)

    env = dict(os.environ, MONGO_URI=URI, NE_DB=DB, NE_COLLECTION=COLL, NE_DEVICE="cpu")
    t0 = time.time()
    procs = [subprocess.Popen([sys.executable, os.path.join(HERE, "scorer.py"), "--drain"], env=env)
             for _ in range(N_WORKERS)]
    for p in procs:
        p.wait()
    dt = time.time() - t0

    nb = store.count(NEW_BORN); sc = store.count(SCORED); cl = store.count(CLAIMED); tot = store.count()
    counts = [d.get("n_scored", 0) for d in client[DB][COLL].find({}, {"n_scored": 1})]
    maxn = max(counts, default=0)
    dupes = sum(1 for c in counts if c > 1)
    print("after %d workers in %.1fs: NEW_BORN=%d SCORED=%d CLAIMED=%d total=%d | max n_scored=%d double-scored=%d"
          % (N_WORKERS, dt, nb, sc, cl, tot, maxn, dupes), flush=True)

    assert tot == N_SEED, "no genomes lost or duplicated (%d != %d)" % (tot, N_SEED)
    assert nb == 0 and cl == 0, "all NEW_BORN drained, none stuck CLAIMED (nb=%d cl=%d)" % (nb, cl)
    assert sc == N_SEED, "every genome ended SCORED (%d != %d)" % (sc, N_SEED)
    assert dupes == 0 and maxn == 1, "NO double-claim/double-score: max n_scored=%d, double=%d" % (maxn, dupes)
    print("CONCURRENCY OK — two-phase claim HELD under %d real parallel processes: "
          "%d genomes each scored exactly once, zero double-claims." % (N_WORKERS, N_SEED))
    client[DB][COLL].drop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
