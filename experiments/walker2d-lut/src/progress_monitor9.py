"""Progress monitor for the 3-PPO-seeds-in-PARALLEL benchmark. Shows all 3 seeds'
live status at once in ONE Slack message, updated in place every ~20s. Read-only."""
import sys, os, re, time, json
sys.path.insert(0, "/home/astarostin/work/slack-facade")
import progress

BENCH = "/home/astarostin/projects/walker2d_gpu/bench9"
TASK = "d5737586"
SEEDS = [0, 1, 2]
TOTAL = 768
PAT = re.compile(r"\[upd\s+([\d,]+)/([\d,]+)\].*?ep_ret\s+(-?[\d.]+).*?([\d,]+)\s*env-steps/s")


def _int(s):
    return int(s.replace(",", ""))


def parse_last(seed):
    f = os.path.join(BENCH, f"ppo_s{seed}.log")
    if not os.path.exists(f):
        return None
    last = None
    for line in open(f):
        m = PAT.search(line)
        if m:
            last = m
    if not last:
        return None
    return dict(upd=_int(last.group(1)), tot=_int(last.group(2)),
                ret=float(last.group(3)), sps=_int(last.group(4)))


def main():
    h = progress.progress_start("PPO-hyperlut2 · 3 seeds · LUT actor + LUT critic",
                                task=TASK, width=12, stats="starting… (seeds 0/1/2 concurrent, 768 upd each)")
    t0 = time.time()
    while time.time() - t0 < 2 * 3600:
        fracs, parts, done = [], [], 0
        for s in SEEDS:
            if os.path.exists(os.path.join(BENCH, f"ppo_s{s}.json")):
                done += 1; fracs.append(1.0); parts.append(f"s{s} ✅")
                continue
            info = parse_last(s)
            if info:
                fracs.append(info["upd"] / info["tot"])
                parts.append(f"s{s} {info['upd']}/{info['tot']} r{info['ret']:.0f} {info['sps']/1e3:.0f}k")
            else:
                fracs.append(0.0); parts.append(f"s{s} …")
        pct = 100.0 * sum(fracs) / len(SEEDS)
        progress.progress_update(h, pct=pct, stats=" · ".join(parts) + f" · ✅{done}/3")
        if done == len(SEEDS):
            break
        time.sleep(20)

    finals = []
    for s in SEEDS:
        p = os.path.join(BENCH, f"ppo_s{s}.json")
        if os.path.exists(p):
            finals.append(f"s{s}={json.load(open(p)).get('final_ep_ret', float('nan')):.0f}")
    progress.progress_done(h, ok=True, final_text="3/3 done · finals: " + " · ".join(finals))
    print("finalized", h)


if __name__ == "__main__":
    main()
