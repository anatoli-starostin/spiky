"""Live Slack progress monitor for exp15b (single group, 3 seeds in parallel):
anchor-pair LUT actor tph128 + hyperplane LUT critic tph64. Read-only over the run logs.
Bound to BODY_TASK b6f5c7d3's thread."""
import sys, os, re, time, json, statistics
sys.path.insert(0, "/home/astarostin/work/slack-facade")
import progress

BASE = "/home/astarostin/projects/spiky/experiments/walker2d-lut"
D = f"{BASE}/exp15b_anchor-t128-actor_hyperplane-t64-critic"
TASK = "b6f5c7d3"
SEEDS = [0, 1, 2]
TOTAL = 768
WALL_MIN = 29.0             # anchored on exp15 (anchor t128) ~28.5 min/run
PAT = re.compile(r"\[upd\s+([\d,]+)/([\d,]+)\].*?ep_ret\s+(-?[\d.]+).*?([\d,]+)\s*env-steps/s")


def parse_last(s):
    f = f"{D}/ppo_s{s}.log"
    if not os.path.exists(f):
        return None
    last = None
    for line in open(f):
        m = PAT.search(line)
        if m:
            last = m
    return None if last is None else dict(upd=int(last.group(1).replace(",", "")),
                                          ret=float(last.group(3)))


def done_run(s):
    return os.path.exists(f"{D}/ppo_s{s}.json")


def main():
    h = progress.progress_start("PPO exp15b · anchor-LUT actor t128 + hyperplane-LUT critic t64 (3 seeds)",
                                task=TASK, width=12, stats="starting…")
    t0 = time.time()
    while time.time() - t0 < 2 * 3600:
        if all(done_run(s) for s in SEEDS):
            break
        frac, seg, upds = 0.0, [], []
        for s in SEEDS:
            if done_run(s):
                frac += 1.0; seg.append(f"s{s}✅"); upds.append(TOTAL)
            else:
                info = parse_last(s)
                if info:
                    frac += info["upd"] / TOTAL; upds.append(info["upd"])
                    seg.append(f"s{s} {info['upd']}/{TOTAL} r{info['ret']:.0f}")
                else:
                    upds.append(0); seg.append(f"s{s}…")
        pct = 100.0 * frac / 3
        slow = min(upds) if upds else 0
        eta = WALL_MIN * (1 - slow / TOTAL)
        progress.progress_update(h, pct=pct,
                                 stats=" · ".join(seg) + f" · ETA {eta:.0f}m")
        time.sleep(20)
    fs = [json.load(open(f"{D}/ppo_s{s}.json"))["final_ep_ret"] for s in SEEDS if done_run(s)]
    ft = f"3/3 done · final {statistics.mean(fs):.0f}±{statistics.pstdev(fs):.0f}" if len(fs) > 1 else "done"
    progress.progress_done(h, ok=True, final_text=ft)
    print("finalized", h)


if __name__ == "__main__":
    main()
