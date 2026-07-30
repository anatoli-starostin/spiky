"""MongoDB persistence layer for the DB-backed neuroevolution framework (issue #81).

One `genomes` collection. Document shape:
  {_id, state, score, priority, parent_ids, genome:{types,hid,syn,sigma}, eval_version,
   claim:{worker_id, token, ts, prev}}

Genome `syn` keys are innovation-number INTEGERS; BSON requires string sub-doc keys, so
they are stringified on write and restored on read.

Claiming (Mongo has no SELECT ... FOR UPDATE SKIP LOCKED):
- single atomic claim: findOneAndUpdate (state -> CLAIMED) sorted by priority.
- batch claim: two-phase stamp — pick top-N candidate ids, update_many only those still in
  the source state (per-document atomicity => no double-claim), then read back by a unique
  token. A sweeper returns stale CLAIMED docs (dead/timed-out workers) to their prior state.
"""
import time
from uuid import uuid4

import pymongo

from evo_config import (MONGO_URI, DB_NAME, COLLECTION, NEW_BORN, SCORED, PROCESSED,
                        CLAIMED, CLAIM_TIMEOUT_S)
import neuroevo_lut as N


def _ser(g):
    return {"types": [dict(t) for t in g["types"]], "hid": dict(g["hid"]),
            "syn": {str(k): list(v) for k, v in g["syn"].items()}, "sigma": float(g["sigma"])}


def _deser(d):
    return {"types": [dict(t) for t in d["types"]], "hid": dict(d["hid"]),
            "syn": {int(k): list(v) for k, v in d["syn"].items()}, "sigma": float(d["sigma"])}


class GenomeStore:
    def __init__(self, uri=None, db_name=None, collection=None, client=None):
        """`client` may be injected (e.g. mongomock.MongoClient) for tests."""
        self._client = client if client is not None else pymongo.MongoClient(uri or MONGO_URI)
        self.col = self._client[db_name or DB_NAME][collection or COLLECTION]
        self.ensure_indexes()

    def ensure_indexes(self):
        self.col.create_index([("state", pymongo.ASCENDING), ("priority", pymongo.DESCENDING)])
        self.col.create_index("claim.ts")

    # ---- writes ----
    def _doc(self, genome, state, priority, parent_ids):
        return {"state": state, "score": None, "priority": float(priority),
                "parent_ids": list(parent_ids), "genome": _ser(genome),
                "eval_version": None, "claim": None}

    def seed_random(self, n, rng):
        """Bootstrap: insert n random NEW_BORN genomes (priority 0, no parents)."""
        docs = [self._doc(N.random_genome(rng), NEW_BORN, 0.0, []) for _ in range(n)]
        return list(self.col.insert_many(docs).inserted_ids) if docs else []

    def insert_new_born(self, items):
        """items: iterable of (genome, parent_ids, priority) -> inserted as NEW_BORN."""
        docs = [self._doc(g, NEW_BORN, pr, pids) for (g, pids, pr) in items]
        return list(self.col.insert_many(docs).inserted_ids) if docs else []

    # ---- claiming ----
    def claim_one(self, worker_id, state_from):
        """Atomic single claim via findOneAndUpdate (highest priority first)."""
        d = self.col.find_one_and_update(
            {"state": state_from},
            {"$set": {"state": CLAIMED,
                      "claim": {"worker_id": worker_id, "token": None, "ts": time.time(), "prev": state_from}}},
            sort=[("priority", pymongo.DESCENDING)],
            return_document=pymongo.ReturnDocument.AFTER)
        return self._out(d) if d else None

    def claim_batch(self, worker_id, state_from, limit):
        """Two-phase stamp claim of up to `limit` top-priority docs in `state_from`."""
        token = "%s:%s" % (worker_id, uuid4().hex)
        ts = time.time()
        ids = [d["_id"] for d in self.col.find({"state": state_from}, {"_id": 1})
               .sort("priority", pymongo.DESCENDING).limit(limit)]
        if not ids:
            return []
        self.col.update_many(
            {"_id": {"$in": ids}, "state": state_from},
            {"$set": {"state": CLAIMED,
                      "claim": {"worker_id": worker_id, "token": token, "ts": ts, "prev": state_from}}})
        docs = list(self.col.find({"state": CLAIMED, "claim.token": token}))
        return [self._out(d) for d in docs]

    def release(self, ids, to_state):
        """Return claimed docs to a state (e.g. NEW_BORN on pack overshoot)."""
        if not ids:
            return 0
        r = self.col.update_many({"_id": {"$in": list(ids)}},
                                 {"$set": {"state": to_state}, "$unset": {"claim": ""}})
        return r.modified_count

    def mark_scored(self, id, score, eval_version):
        """Flip a claimed doc to SCORED; priority := score so top-SCORED sorts by score."""
        self.col.update_one({"_id": id},
                            {"$set": {"state": SCORED, "score": float(score),
                                      "priority": float(score), "eval_version": eval_version},
                             "$unset": {"claim": ""}})

    def mark_processed(self, ids):
        if not ids:
            return 0
        r = self.col.update_many({"_id": {"$in": list(ids)}},
                                 {"$set": {"state": PROCESSED}, "$unset": {"claim": ""}})
        return r.modified_count

    # ---- sweeper ----
    def sweep_stale(self, timeout_s=None, now=None):
        """Return CLAIMED docs whose claim is older than timeout back to their prior state."""
        timeout_s = CLAIM_TIMEOUT_S if timeout_s is None else timeout_s
        cutoff = (time.time() if now is None else now) - timeout_s
        n = 0
        for d in list(self.col.find({"state": CLAIMED, "claim.ts": {"$lt": cutoff}})):
            prev = (d.get("claim") or {}).get("prev", NEW_BORN)
            self.col.update_one({"_id": d["_id"], "state": CLAIMED},
                                {"$set": {"state": prev}, "$unset": {"claim": ""}})
            n += 1
        return n

    # ---- reads ----
    def count(self, state=None):
        return self.col.count_documents({} if state is None else {"state": state})

    def top(self, state, limit=10):
        return [self._out(d) for d in
                self.col.find({"state": state}).sort("priority", pymongo.DESCENDING).limit(limit)]

    def get(self, id):
        d = self.col.find_one({"_id": id})
        return self._out(d) if d else None

    @staticmethod
    def _out(d):
        return {"_id": d["_id"], "state": d["state"], "score": d.get("score"),
                "priority": d.get("priority"), "parent_ids": d.get("parent_ids", []),
                "eval_version": d.get("eval_version"), "genome": _deser(d["genome"])}
