"""Actor 1 — Scorer (issue #81). Claims top NEW_BORN genomes, greedily packs them into one
net under the neuron-meta cap, builds + evaluates via the existing packed-build + vectorized
scorer (#30), writes scores back, and flips them to SCORED. Runs as its own process.

  python scorer.py [--once]     # --once: single iteration (for tests/smoke)
"""
import gc
import os
import socket
import sys
import time

from evo_config import (NEW_BORN, DEVICE, PACK_LIMIT, META_SOFT_CAP, DROP_ON_OVERSHOOT,
                        EVAL_VERSION, CLAIM_TIMEOUT_S, fixed_eval_set)
from genome_store import GenomeStore
import neuroevo_lut as N


def _meta_sig(nm):
    """Same signature build_population's meta_index dedups on (neuroevo_lut.py:230)."""
    return (round(nm.cf_2, 6), round(nm.cf_1, 6), round(nm.cf_0, 6), round(nm.a, 6),
            round(nm.b, 6), round(nm.c, 6), round(nm.d, 6), round(nm.spike_threshold, 6),
            int(nm.neuron_type))


class Scorer:
    def __init__(self, store=None, worker_id=None, device=None):
        self.store = store if store is not None else GenomeStore()
        self.worker_id = worker_id or ("scorer-%s-%d" % (socket.gethostname(), os.getpid()))
        self.device = device or DEVICE
        self.xs, self.tos = fixed_eval_set()

    def _greedy_pack(self, claimed):
        """Split claimed docs into (pack, spill): add genomes until the estimated distinct
        neuron-meta count would exceed META_SOFT_CAP or the pack hits PACK_LIMIT."""
        base = {_meta_sig(N.leakfree(N.IN_THR)), _meta_sig(N.leakfree(N.OUT_THR))}
        sigs = set(base)
        pack = []
        for d in claimed:
            if len(pack) >= PACK_LIMIT:
                break
            g = d["genome"]
            gsigs = {_meta_sig(N.type_meta(g, h)) for h in g["hid"]}
            if len(sigs | gsigs) > META_SOFT_CAP:
                break
            sigs |= gsigs
            pack.append(d)
        return pack, claimed[len(pack):]

    def _build_eval_with_retry(self, pack):
        """Build+score the pack; on overshoot (e.g. real 512-meta cap) release the last K
        back to NEW_BORN and retry smaller. Returns [(doc, score)]. Empty-syn genomes -> 0."""
        cur = list(pack)
        empties = [d for d in cur if not d["genome"]["syn"]]
        cur = [d for d in cur if d["genome"]["syn"]]
        results = [(d, 0.0) for d in empties]
        while cur:
            try:
                packed = N.build_population([d["genome"] for d in cur], device=self.device)
                raw = N._score_population(packed, len(cur), self.xs, self.tos, device=self.device)
                results += [(cur[i], float(raw[i][0])) for i in range(len(cur))]
                del packed, raw              # packed nets form ref-cycles holding native memory
                return results
            except Exception:
                k = min(DROP_ON_OVERSHOOT, len(cur))
                drop = cur[-k:]
                self.store.release([d["_id"] for d in drop], NEW_BORN)
                cur = cur[:-k]
        return results

    def run_once(self):
        self.store.sweep_stale(CLAIM_TIMEOUT_S)
        claimed = self.store.claim_batch(self.worker_id, NEW_BORN, PACK_LIMIT)
        if not claimed:
            return 0
        pack, spill = self._greedy_pack(claimed)
        if spill:
            self.store.release([d["_id"] for d in spill], NEW_BORN)
        results = self._build_eval_with_retry(pack)
        for d, score in results:
            self.store.mark_scored(d["_id"], score, EVAL_VERSION)
        gc.collect()                         # reclaim the packed-net ref-cycles (else RSS grows unbounded)
        return len(results)

    def loop(self, poll_s=2.0, max_packs=None):
        max_rss = int(os.environ.get("NE_MAX_RSS_MB", "0")) or None
        done = 0
        while True:
            n = self.run_once()
            if n == 0:
                time.sleep(poll_s)
            else:
                done += 1
                # exit -> supervisor respawns a fresh process, reclaiming the native SpikingNet
                # memory that leaks across builds (the leak scales with genome size, so we cap on
                # actual RSS, not a fixed pack count).
                if max_packs and done >= max_packs:
                    return
                if max_rss and _rss_mb() >= max_rss:
                    return


def _rss_mb():
    try:
        for ln in open("/proc/self/status"):
            if ln.startswith("VmRSS:"):
                return int(ln.split()[1]) / 1024.0
    except Exception:
        return 0.0
    return 0.0


def main():
    sc = Scorer()
    if "--once" in sys.argv:
        n = sc.run_once()
        print("scorer[%s] scored %d genomes (device=%s, eval=%s)" % (sc.worker_id, n, sc.device, EVAL_VERSION))
    elif "--drain" in sys.argv:      # score until the NEW_BORN queue is truly empty, then exit
        total = 0
        empties = 0
        while empties < 5:
            n = sc.run_once()
            total += n
            if n == 0:
                # 0 can mean "queue empty" OR "lost the claim race this round" (another worker
                # grabbed the batch). Only stop once no NEW_BORN actually remain; else back off.
                if sc.store.count(NEW_BORN) == 0:
                    break
                empties += 1
                time.sleep(0.1)
            else:
                empties = 0
        print("scorer[%s] drained %d genomes" % (sc.worker_id, total))
    else:
        mp = int(os.environ.get("NE_MAX_PACKS", "0")) or None
        print("scorer[%s] looping (device=%s, max_packs=%s)" % (sc.worker_id, sc.device, mp))
        sc.loop(max_packs=mp)


if __name__ == "__main__":
    main()
