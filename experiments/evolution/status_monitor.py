#!/usr/bin/env python3
"""Live progress monitor for a DB-backed neuroevolution run (issue #81). Reads the Mongo
`genomes` collection and writes a rolling status.json + human status.txt + snapshots.jsonl to
EVO_RUN_DIR, so progress is checkable any time and recoverable after a session restart (all
truth lives in Mongo). Read-only; safe to run alongside the Scorer/Generator workers."""
import datetime
import json
import os
import statistics
import time

import pymongo

from evo_config import (MONGO_URI, DB_NAME, COLLECTION, NEW_BORN, SCORED, PROCESSED,
                        CLAIMED, EVAL_VERSION)

RUN_DIR = os.environ.get("EVO_RUN_DIR", os.path.expanduser("~/projects/evo-run"))
INTERVAL = float(os.environ.get("EVO_STATUS_INTERVAL", 10))
SNAP_EVERY = float(os.environ.get("EVO_SNAP_INTERVAL", 120))
WINDOW_S = float(os.environ.get("EVO_WINDOW_S", 5 * 3600))


def _fmt(sec):
    sec = int(sec); h, r = divmod(sec, 3600); m, s = divmod(r, 60)
    return "%dh%02dm%02ds" % (h, m, s)


def _best(col, state):
    d = list(col.find({"state": state}).sort("priority", -1).limit(1))   # priority==score for scored docs
    return d[0]["score"] if d and d[0].get("score") is not None else None


def main():
    os.makedirs(RUN_DIR, exist_ok=True)
    client = pymongo.MongoClient(MONGO_URI)
    col = client[DB_NAME][COLLECTION]
    try:
        col.create_index("depth")
    except Exception:
        pass
    start = time.time()
    with open(os.path.join(RUN_DIR, "start.txt"), "w") as f:
        f.write(str(start))
    prev_scored = None; prev_t = None; last_snap = 0.0
    while True:
        try:
            nb = col.count_documents({"state": NEW_BORN})
            sc = col.count_documents({"state": SCORED})
            pr = col.count_documents({"state": PROCESSED})
            cl = col.count_documents({"state": CLAIMED})
            total = nb + sc + pr + cl
            scored_total = sc + pr
            bests = [x for x in (_best(col, SCORED), _best(col, PROCESSED)) if x is not None]
            best = max(bests) if bests else None
            sample = [d["score"] for d in col.aggregate(
                [{"$match": {"score": {"$ne": None}}}, {"$sample": {"size": 800}}, {"$project": {"score": 1}}])]
            mean = statistics.mean(sample) if sample else None
            median = statistics.median(sample) if sample else None
            dd = list(col.find({}, {"depth": 1}).sort("depth", -1).limit(1))
            maxdepth = dd[0].get("depth", 0) if dd else 0
            now = time.time()
            thr = ((scored_total - prev_scored) / (now - prev_t)) if (prev_scored is not None and now > prev_t) else None
            prev_scored, prev_t = scored_total, now
            elapsed = now - start
            st = {"ts": now, "iso": datetime.datetime.now().isoformat(timespec="seconds"),
                  "elapsed_s": round(elapsed, 1), "window_s": WINDOW_S,
                  "counts": {"NEW_BORN": nb, "SCORED": sc, "PROCESSED": pr, "CLAIMED": cl, "total": total},
                  "genomes_scored": scored_total, "best_score": best, "mean_score": mean,
                  "median_score": median, "max_depth_generation": maxdepth,
                  "throughput_gps": round(thr, 2) if thr is not None else None, "eval_version": EVAL_VERSION}
            with open(os.path.join(RUN_DIR, "status.json"), "w") as f:
                json.dump(st, f, indent=2)
            frac = min(1.0, elapsed / WINDOW_S) if WINDOW_S else 0
            fill = int(frac * 30)
            bar = "#" * fill + "-" * (30 - fill)
            txt = ("NEUROEVO RUN  [%s] %3.0f%%   elapsed %s / %.1fh\n"
                   "  genomes scored : %-10d  throughput : %6.1f g/s\n"
                   "  best score     : %-10.4f  gen (max lineage depth) : %d\n"
                   "  mean %.4f  median %.4f  (sampled)\n"
                   "  pool: NEW_BORN=%d  SCORED=%d  PROCESSED=%d  total=%d\n"
                   "  updated %s   eval=%s\n") % (
                   bar, frac * 100, _fmt(elapsed), WINDOW_S / 3600.0,
                   scored_total, (thr or 0.0), (best if best is not None else 0.0), maxdepth,
                   (mean or 0.0), (median or 0.0), nb, sc, pr, total, st["iso"], EVAL_VERSION)
            with open(os.path.join(RUN_DIR, "status.txt"), "w") as f:
                f.write(txt)
            if now - last_snap >= SNAP_EVERY:
                with open(os.path.join(RUN_DIR, "snapshots.jsonl"), "a") as f:
                    f.write(json.dumps(st) + "\n")
                last_snap = now
        except Exception as e:
            with open(os.path.join(RUN_DIR, "monitor.err"), "a") as f:
                f.write("%f %r\n" % (time.time(), e))
        time.sleep(INTERVAL)


if __name__ == "__main__":
    main()
