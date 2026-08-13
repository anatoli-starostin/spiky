"""Lightweight in-cage progress monitor for the PPO-vs-SAC benchmark.
Parses the 6 run logs, updates ONE Slack progress bar (via progress.py green-zone
records; the face reaper posts/edits) every ~20s. Bound to the benchmark BODY_TASK
thread. Read-only w.r.t. the benchmark — never touches the running jobs."""
import sys, os, re, time, json
sys.path.insert(0, "/home/astarostin/work/slack-facade")
import progress

BENCH = "/home/astarostin/projects/walker2d_gpu/bench2"
TASK = "b1b2f203"                       # the (relaunch) benchmark task's thread
RUNS = [("ppo", 0, 384), ("ppo", 1, 384), ("ppo", 2, 384),
        ("sac", 0, 10000), ("sac", 1, 10000), ("sac", 2, 10000)]
PAT = re.compile(r"\[upd\s+([\d,]+)/([\d,]+)\].*?ep_ret\s+(-?[\d.]+).*?([\d,]+)\s*env-steps/s")


def _int(s):
    return int(s.replace(",", ""))


def parse_last(tag):
    f = os.path.join(BENCH, tag + ".log")
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
    h = progress.progress_start("PPO-vs-SAC fair benchmark · N=8192 · physics-graph · 3 seeds ea",
                                task=TASK, width=12,
                                stats="starting… (PPO 384 upd ×3 ≈100M steps, SAC 10000 upd ×3 ≈82M steps)")
    t0 = time.time()
    while time.time() - t0 < 3 * 3600:
        done_tags, cur = [], None
        for algo, seed, tot in RUNS:
            tag = f"{algo}_s{seed}"
            if os.path.exists(os.path.join(BENCH, tag + ".json")):
                done_tags.append(tag)
            else:
                cur = (algo, seed, tot, tag)
                break
        if cur is None:
            break                                   # all 6 finished
        info = parse_last(cur[3])
        frac = (info["upd"] / info["tot"]) if info else 0.0
        pct = 100.0 * (len(done_tags) + frac) / 6
        if info:
            cur_s = (f"▶ {cur[0].upper()} s{cur[1]}: upd {info['upd']:,}/{info['tot']:,} · "
                     f"ret {info['ret']:.0f} · {info['sps']/1e3:.0f}k st/s")
        else:
            cur_s = f"▶ {cur[0].upper()} s{cur[1]}: starting…"
        done_s = f" · ✅{len(done_tags)}/6" + (f" ({','.join(done_tags)})" if done_tags else "")
        progress.progress_update(h, pct=pct, stats=cur_s + done_s)
        time.sleep(20)

    finals = []
    for algo, seed, tot in RUNS:
        p = os.path.join(BENCH, f"{algo}_s{seed}.json")
        if os.path.exists(p):
            d = json.load(open(p))
            finals.append(f"{algo.upper()} s{seed}={d.get('final_ep_ret', float('nan')):.0f}")
    progress.progress_done(h, ok=True, final_text="6/6 done · finals: " + " · ".join(finals))
    print("progress bar finalized:", h)


if __name__ == "__main__":
    main()
