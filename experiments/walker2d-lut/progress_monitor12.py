"""Progress monitor for the fastlut sweep: 3 groups (tph 32/64/128) × 3 seeds = 9 runs,
sequential groups. One bar; the current group + its 3 seeds are shown live. Read-only."""
import sys, os, re, time, json, statistics
sys.path.insert(0, "/home/astarostin/work/slack-facade")
import progress

BENCH = "/home/astarostin/projects/walker2d_gpu/bench12"
TASK = "1bca77e5"
GROUPS = [32, 64, 128]
SEEDS = [0, 1, 2]
TOTAL = 768
PAT = re.compile(r"\[upd\s+([\d,]+)/([\d,]+)\].*?ep_ret\s+(-?[\d.]+).*?([\d,]+)\s*env-steps/s")


def parse_last(tph, s):
    f = f"{BENCH}/t{tph}/ppo_s{s}.log"
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
    return os.path.exists(f"{BENCH}/t{tph}/ppo_s{s}.json")


def main():
    h = progress.progress_start("PPO-fastlut sweep · tph 32/64/128 · 3 seeds (9 runs)",
                                task=TASK, width=12, stats="starting…")
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
                                       f" · done:{','.join(gdone) or '—'} ({done}/9)")
        time.sleep(20)
    finals = []
    for t in GROUPS:
        fs = [json.load(open(f"{BENCH}/t{t}/ppo_s{s}.json"))["final_ep_ret"]
              for s in SEEDS if done_run(t, s)]
        if fs:
            finals.append(f"t{t}={statistics.mean(fs):.0f}")
    progress.progress_done(h, ok=True, final_text="9/9 done · " + " · ".join(finals))
    print("finalized", h)


if __name__ == "__main__":
    main()
