"""Live Slack progress monitor for the fastlut2 sweep: 3 groups (tph 32/64/128) x 3 seeds,
sequential groups, seeds parallel per group. One bar; current group + its 3 seeds shown live,
with a whole-sweep ETA. Read-only over the run logs. Bound to BODY_TASK 839b2e7e's thread."""
import sys, os, re, time, json, statistics
sys.path.insert(0, "/home/astarostin/work/slack-facade")
import progress

BASE = "/home/astarostin/projects/spiky/experiments/walker2d-lut"
TASK = "839b2e7e"
HANDLE = "0ee56b6b"          # reuse the already-posted Slack progress message
GROUPS = [32, 64, 128]
SEEDS = [0, 1, 2]
TOTAL = 768
# measured per-group walls from exp10-12 (min): drives a sweep-anchored ETA that does not
# depend on when this monitor was (re)started.
WALL_MIN = {32: 20.0, 64: 21.4, 128: 24.4}
DIRS = {32: "exp13_lut-anchor-pair-lutcrit-t32",
        64: "exp14_lut-anchor-pair-lutcrit-t64",
        128: "exp15_lut-anchor-pair-lutcrit-t128"}
PAT = re.compile(r"\[upd\s+([\d,]+)/([\d,]+)\].*?ep_ret\s+(-?[\d.]+).*?([\d,]+)\s*env-steps/s")


def _log(tph, s):
    return f"{BASE}/{DIRS[tph]}/ppo_s{s}.log"


def parse_last(tph, s):
    f = _log(tph, s)
    if not os.path.exists(f):
        return None
    last = None
    for line in open(f):
        m = PAT.search(line)
        if m:
            last = m
    return None if last is None else dict(upd=int(last.group(1).replace(",", "")),
                                          ret=float(last.group(3)))


def done_run(tph, s):
    return os.path.exists(f"{BASE}/{DIRS[tph]}/ppo_s{s}.json")


def eta_minutes(cur):
    """Sweep-anchored ETA: remaining fraction of the current group (by its slowest seed) at
    the measured group wall, plus the full walls of groups not yet started."""
    upds = [TOTAL if done_run(cur, s) else (parse_last(cur, s) or {}).get("upd", 0)
            for s in SEEDS]
    slow = min(upds) if upds else 0
    rem = WALL_MIN[cur] * (1 - slow / TOTAL)
    idx = GROUPS.index(cur)
    rem += sum(WALL_MIN[t] for t in GROUPS[idx + 1:])
    return rem


def main():
    h = HANDLE                 # reuse the existing posted message (no new Slack message)
    t0 = time.time()
    while time.time() - t0 < 3 * 3600:
        done = sum(done_run(t, s) for t in GROUPS for s in SEEDS)
        cur = next((t for t in GROUPS if not all(done_run(t, s) for s in SEEDS)), None)
        if cur is None:
            break
        frac = 0.0
        for t in GROUPS:
            for s in SEEDS:
                if done_run(t, s):
                    frac += 1.0
                elif t == cur:
                    info = parse_last(t, s)
                    if info:
                        frac += info["upd"] / TOTAL
        pct = 100.0 * frac / 9
        eta = eta_minutes(cur)
        eta_s = f" · ETA {eta:.0f}m" if eta else ""
        seg = []
        for s in SEEDS:
            if done_run(cur, s):
                seg.append(f"s{s}✅")
            else:
                info = parse_last(cur, s)
                seg.append(f"s{s} {info['upd']}/{TOTAL} r{info['ret']:.0f}" if info else f"s{s}…")
        gdone = [f"t{t}" for t in GROUPS if all(done_run(t, s) for s in SEEDS)]
        progress.progress_update(h, pct=pct,
                                 stats=f"[t{cur}] " + " · ".join(seg) +
                                       f" · done:{','.join(gdone) or '—'} ({done}/9)" + eta_s)
        time.sleep(20)
    finals = []
    for t in GROUPS:
        fs = [json.load(open(f"{BASE}/{DIRS[t]}/ppo_s{s}.json"))["final_ep_ret"]
              for s in SEEDS if done_run(t, s)]
        if fs:
            finals.append(f"t{t}={statistics.mean(fs):.0f}")
    progress.progress_done(h, ok=True, final_text="9/9 done · " + " · ".join(finals))
    print("finalized", h)


if __name__ == "__main__":
    main()
