"""exp_c39 diagnosis — phase-driven Slack bar.

Unlike the sweep bars this one is not watching training logs; the analysis is a sequence of
phases, each writing its findings into `diag_status.json`. The bar renders that file, so
progress is reported by the thing making it rather than inferred from a log.

Cage-safe: `progress.py` is a FILE RENDEZVOUS under ~/.cache, not a network call.

There is no warmup-vs-steady-state ETA problem here because there is no warmup -- the ETA
is the sum of the remaining phases' own estimates, which each phase updates as it learns
its real cost. That is the same fix in spirit as the c38/c39 bars' switch to a
most-recent-interval rate: never extrapolate from a segment that is not like the rest.

Usage:
  python slack_bar_diag.py --task <BODY_TASK id> [--interval 60]
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.expanduser("~/work/slack-facade"))
import progress                                            # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
STATUS = os.path.join(HERE, "diag_status.json")

PHASES = [
    ("trajectories", "Trajectory contrast — all 6 MHL seeds (c38 + c39) on shared axes"),
    ("init", "Init forensics — regenerate each seed's exact init, functional stats"),
    ("final", "Final-state forensics — table displacement, detector health, redundancy"),
    ("plots", "Comparison plots"),
]
REF = ("_*exp_c39 failure/success deep-dive.* seed 2 took off (CPU-ref *4217*, 100/100 "
       "full); seeds 0 and 1 stayed flat (*891*, *982*), both flat-never-took-off rather "
       "than collapse-after-takeoff. Config: 1 head × 32 tables × 3 LIF detectors × 4 "
       "buckets, freeze_temperature=True, delay_init_std=4, 28,384 params (101.3% of the "
       "hyperplane baseline). Init is regenerated EXACTLY rather than guessed — the "
       "trainer is deterministic from PRNGKey(seed), so every seed's starting state is "
       "reproducible without having saved it._")


def read():
    if not os.path.exists(STATUS):
        return dict(phase="trajectories", pct=0.0, notes=[], done=False)
    try:
        return json.load(open(STATUS))
    except Exception:
        return dict(phase="trajectories", pct=0.0, notes=[], done=False)


def render(st):
    cur = st.get("phase", "trajectories")
    order = [p for p, _ in PHASES]
    idx = order.index(cur) if cur in order else len(order)
    lines = []
    for i, (key, label) in enumerate(PHASES):
        mark = "✅" if i < idx or st.get("done") else ("⏳" if i == idx else "•")
        lines.append(f"{mark} {label}")
    notes = st.get("notes", [])
    if notes:
        lines += [""] + [f"› {n}" for n in notes[-6:]]
    return st.get("pct", 0.0), "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--handle", default=None)
    ap.add_argument("--interval", type=int, default=60)
    ap.add_argument("--max-hours", type=float, default=3.0)
    a = ap.parse_args()
    h = a.handle or progress.progress_start(
        "exp_c39 — why one seed took off and two did not", task=a.task,
        style="emoji", width=10)
    open(os.path.join(HERE, ".slack_diag.handle"), "w").write(h)
    print(f"bar {h}", flush=True)

    t0 = time.time()
    while True:
        st = read()
        pct, body = render(st)
        progress.progress_update(h, pct=pct, stats=REF + "\n" + body)
        if st.get("done") or (time.time() - t0) > a.max_hours * 3600:
            stamp = time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())
            progress.progress_done(
                h, ok=bool(st.get("done")),
                final_text=(f"exp_c39 diagnosis "
                            f"{'complete' if st.get('done') else 'TIMED OUT'} {stamp}\n"
                            + body))
            print(f"bar closed {stamp}", flush=True)
            return
        time.sleep(a.interval)


if __name__ == "__main__":
    main()
