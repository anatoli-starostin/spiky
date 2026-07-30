"""exp_c22 — live Slack bar for the matched-parameter, n=12 study (#75).

18 cells across two arms, so the bar summarises per arm rather than listing every run: a
20-line bar is not a bar. It shows counts, the current wave, and the running mean of
whichever scores have landed, and it checks failure before progress so a died-at-9k run
cannot show as healthy.
"""
import argparse, json, os, re, statistics, sys, time

sys.path.insert(0, os.path.expanduser("~/work/slack-facade"))
import progress  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "..")
C09 = os.path.join(D, "exp_c09_lut_sac")
C18 = os.path.join(D, "exp_c18_seed_variance")
C19 = os.path.join(D, "exp_c19_mlp_sac_control")
ITER = re.compile(r"\[\s*(\d+)/(\d+)\]", re.M)
DONE = re.compile(r"^done: best MJX ([-\d.]+)", re.M)
FAIL = re.compile(r"Traceback|CUDA out of memory|Killed|RESOURCE_EXHAUSTED")
HIDDEN = 153
NEW_LUT = (6, 7, 8, 9, 10, 11)
MLP_SEEDS = tuple(range(12))
TRAIN_W = 0.9
REF = (f"LUT vs PARAM-MATCHED MLP (2×{HIDDEN} = 28,164 vs the LUT's 28,032) at n=12 · "
       "removes exp_c19's 2.6× capacity confound and its n=6 weakness · 6 new LUT seeds "
       "join exp_c18's 6; 12 fresh MLP seeds · determinism ON · thresholds pre-registered")


def bar(pct, width=14):
    f = int(round(width * max(0.0, min(100.0, pct)) / 100.0))
    return "█" * f + "░" * (width - f)


def started():
    try:
        return "exp_c21 done" in open(os.path.join(HERE, "run_all.log"),
                                      errors="replace").read()
    except OSError:
        return False


def cell(log, ev):
    st = dict(eval_done=os.path.exists(ev), cpu=None, pct=0.0, state="queued")
    if st["eval_done"]:
        try:
            st["cpu"] = json.load(open(ev))["cpu_reference_mean"]
        except Exception:
            st["eval_done"] = False
    try:
        txt = open(log, errors="replace").read()
    except OSError:
        return st
    d = DONE.search(txt)
    if FAIL.search(txt) and not d:
        return dict(st, state="failed")
    if d:
        return dict(st, state="trained", pct=100.0)
    m = ITER.findall(txt)
    if m:
        it, tot = m[-1]
        return dict(st, state="training", pct=100.0 * int(it) / int(tot))
    return dict(st, state="starting")


def arm_state():
    lut = []
    for s in range(12):
        if s in NEW_LUT:
            lut.append(cell(os.path.join(HERE, f"cell_lut_s{s}.log"),
                            os.path.join(C09, f"lut_sac_c22_lut_s{s}_cpueval.json")))
        else:   # already finished in exp_c18
            lut.append(dict(eval_done=True, state="trained", pct=100.0,
                            cpu=json.load(open(os.path.join(
                                C09, f"lut_sac_c18_seed{s}_cpueval.json")))
                            ["cpu_reference_mean"]))
    mlp = [cell(os.path.join(HERE, f"cell_mlp_s{s}.log"),
                os.path.join(C19, f"mlp_sac_c22_mlp{HIDDEN}_s{s}_cpueval.json"))
           for s in MLP_SEEDS]
    return lut, mlp


def summarise(name, cells, reused=0):
    ev = [c for c in cells if c["eval_done"]]
    fail = sum(1 for c in cells if c["state"] == "failed")
    run = sum(1 for c in cells if c["state"] in ("training", "starting"))
    pct = sum(TRAIN_W * c["pct"] / 100.0 + (1 - TRAIN_W) * (1.0 if c["eval_done"] else 0.0)
              for c in cells) / len(cells)
    line = (f"`{name}` {bar(100 * pct)} {100*pct:3.0f}% · {len(ev)}/{len(cells)} scored"
            + (f" ({reused} reused)" if reused else "")
            + (f" · {run} on GPU" if run else "")
            + (f" · *{fail} FAILED*" if fail else ""))
    if len(ev) >= 2:
        v = [c["cpu"] for c in ev]
        line += (f"\n   mean so far *{statistics.mean(v):.0f}* ± "
                 f"{statistics.stdev(v):.0f}  (n={len(v)})")
    return pct, fail, line


def render():
    if not started():
        return 0.0, False, 0, ("*waiting for the 20k run to finish* — queued so nothing "
                               "contends for the GPU.\n`18 runs` " + bar(0) + "  not started")
    lut, mlp = arm_state()
    p1, f1, l1 = summarise("LUT  (hyperplane×hard)", lut, reused=6)
    p2, f2, l2 = summarise(f"MLP  (2×{HIDDEN}, param-matched)", mlp)
    # weight by the work each arm actually represents: 6 new LUT runs against 12 MLP runs
    # (the other 6 LUT seeds are reused from exp_c18 and cost no GPU here)
    frac = (6 * p1 + 12 * p2) / 18
    done = all(c["eval_done"] for c in lut + mlp)
    body = [l1, l2]
    if done:
        a = [c["cpu"] for c in lut]
        b = [c["cpu"] for c in mlp]
        body.append(f"\n*LUT {statistics.mean(a):.1f} ± {statistics.stdev(a):.1f}* vs "
                    f"*MLP {statistics.mean(b):.1f} ± {statistics.stdev(b):.1f}* (n=12 each)"
                    f"\n_variance ratio MLP/LUT = "
                    f"{statistics.variance(b)/statistics.variance(a):.2f}× · "
                    f"F(0.95;11,11) = 2.82 · full test in-thread_")
    return 100.0 * frac, done, f1 + f2, "\n".join(body)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--interval", type=int, default=150)
    a = ap.parse_args()
    h = progress.progress_start("LUT vs param-matched MLP at n=12 (#75)",
                                task=a.task, style="emoji", width=10)
    open(os.path.join(HERE, ".slack_c22.handle"), "w").write(h)
    print(f"bar {h}", flush=True)

    while True:
        pct, done, n_fail, body = render()
        if done or n_fail >= 18:
            stamp = time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())
            progress.progress_done(
                h, ok=(n_fail == 0),
                final_text=(body + "\n_full analysis follows in-thread_\n"
                            f"_finished {stamp} — this bar has stopped on purpose._"))
            print("finished", flush=True)
            return
        progress.progress_update(h, pct=pct, stats=f"_{REF}_\n{body}")
        print(f"[{time.strftime('%H:%M:%S')}] {pct:5.1f}%", flush=True)
        time.sleep(a.interval)


if __name__ == "__main__":
    main()
