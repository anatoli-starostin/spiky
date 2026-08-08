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
    ot = g.get("out_type") or N.default_out_type()
    st = g.get("stdp")
    return {"types": [dict(t) for t in g["types"]], "hid": dict(g["hid"]),
            "syn": {str(k): list(v) for k, v in g["syn"].items()}, "sigma": float(g["sigma"]),
            "out_type": {k: float(v) for k, v in ot.items()},
            "stdp": ({k: float(v) for k, v in st.items()} if st else None)}


def _deser(d):
    ot = d.get("out_type") or N.default_out_type()
    st = d.get("stdp")
    return {"types": [dict(t) for t in d["types"]], "hid": dict(d["hid"]),
            "syn": {int(k): list(v) for k, v in d["syn"].items()}, "sigma": float(d["sigma"]),
            "out_type": {k: float(v) for k, v in ot.items()},
            "stdp": ({k: float(v) for k, v in st.items()} if st else None)}


class GenomeStore:
    def __init__(self, uri=None, db_name=None, collection=None, client=None):
        """`client` may be injected (e.g. mongomock.MongoClient) for tests."""
        self._client = client if client is not None else pymongo.MongoClient(uri or MONGO_URI)
        self.col = self._client[db_name or DB_NAME][collection or COLLECTION]
        self.ensure_indexes()

    def ensure_indexes(self):
        self.col.create_index([("state", pymongo.ASCENDING), ("priority", pymongo.DESCENDING)])
        self.col.create_index([("state", pymongo.ASCENDING), ("_id", pymongo.ASCENDING)])  # FIFO intake
        self.col.create_index("claim.ts")
        self.col.create_index("lineage_id")

    # ---- writes ----
    def _doc(self, genome, state, priority, parent_ids, depth=0, lineage_id=None):
        return {"state": state, "score": None, "priority": float(priority),
                "parent_ids": list(parent_ids), "depth": int(depth), "genome": _ser(genome),
                "lineage_id": lineage_id, "eval_version": None, "claim": None, "n_scored": 0}

    def seed_random(self, n, rng):
        """Bootstrap: insert n random NEW_BORN genomes (priority 0, no parents). Each is the root
        of its own lineage (fresh lineage_id)."""
        docs = [self._doc(N.random_genome(rng), NEW_BORN, 0.0, [], lineage_id=uuid4().hex)
                for _ in range(n)]
        return list(self.col.insert_many(docs).inserted_ids) if docs else []

    def insert_new_born(self, items):
        """items: iterable of (genome, parent_ids, priority[, depth[, lineage_id]]) -> NEW_BORN.
        lineage_id (5th field) is the root-ancestor id; if omitted a fresh one is assigned so the
        genome roots its own lineage (immigrant semantics)."""
        docs = [self._doc(it[0], NEW_BORN, it[2], it[1],
                          it[3] if len(it) > 3 else 0,
                          it[4] if len(it) > 4 and it[4] else uuid4().hex)
                for it in items]
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

    def claim_batch(self, worker_id, state_from, limit, order="priority"):
        """Two-phase stamp claim of up to `limit` docs in `state_from`.
        order='priority' -> highest priority first (default, legacy); order='fifo' -> insertion
        order (_id ascending), so scorer intake can be decoupled from score."""
        sort_key = ("_id", pymongo.ASCENDING) if order == "fifo" else ("priority", pymongo.DESCENDING)
        token = "%s:%s" % (worker_id, uuid4().hex)
        ts = time.time()
        ids = [d["_id"] for d in self.col.find({"state": state_from}, {"_id": 1})
               .sort([sort_key]).limit(limit)]
        if not ids:
            return []
        self.col.update_many(
            {"_id": {"$in": ids}, "state": state_from},
            {"$set": {"state": CLAIMED,
                      "claim": {"worker_id": worker_id, "token": token, "ts": ts, "prev": state_from}}})
        docs = list(self.col.find({"state": CLAIMED, "claim.token": token}))
        return [self._out(d) for d in docs]

    def peek_top(self, state, limit):
        """READ-ONLY (no claim) top-`limit` docs in `state` by priority DESC, lightweight fields
        only (no genome). Used by the generator to over-fetch, then lineage-filter client-side
        before claiming the survivors by id. lineage_id is resolved to the doc's own id if unset
        (legacy docs from before lineage tracking)."""
        out = []
        for d in (self.col.find({"state": state}, {"priority": 1, "score": 1, "lineage_id": 1, "depth": 1})
                  .sort("priority", pymongo.DESCENDING).limit(limit)):
            out.append({"_id": d["_id"], "priority": d.get("priority"), "score": d.get("score"),
                        "lineage_id": d.get("lineage_id") or str(d["_id"]), "depth": d.get("depth", 0)})
        return out

    def claim_ids(self, worker_id, ids, state_from):
        """Two-phase stamp claim of a SPECIFIC set of ids still in `state_from` (per-document
        atomicity => no double-claim, exactly like claim_batch). Returns the docs actually won."""
        ids = list(ids)
        if not ids:
            return []
        token = "%s:%s" % (worker_id, uuid4().hex)
        ts = time.time()
        self.col.update_many(
            {"_id": {"$in": ids}, "state": state_from},
            {"$set": {"state": CLAIMED,
                      "claim": {"worker_id": worker_id, "token": token, "ts": ts, "prev": state_from}}})
        docs = list(self.col.find({"state": CLAIMED, "claim.token": token}))
        return [self._out(d) for d in docs]

    def backfill_lineage(self):
        """One-time migration: give every doc lacking a lineage_id its own id (str) as the root.
        Idempotent and safe to run on a live collection. Returns the number updated."""
        n = 0
        for d in self.col.find({"lineage_id": None}, {"_id": 1}):
            self.col.update_one({"_id": d["_id"]}, {"$set": {"lineage_id": str(d["_id"])}})
            n += 1
        return n

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
                             "$inc": {"n_scored": 1},          # >1 would mean a double-claim slipped through
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
                "depth": d.get("depth", 0), "lineage_id": d.get("lineage_id") or str(d["_id"]),
                "eval_version": d.get("eval_version"),
                "n_scored": d.get("n_scored", 0), "genome": _deser(d["genome"])}
