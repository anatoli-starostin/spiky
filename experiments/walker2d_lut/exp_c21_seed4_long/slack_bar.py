"""exp_c21 — live Slack bar for the 20k-iteration seed-4 run (#75).

One cell, so the bar carries the running comparison instead of a list: the 10k reference is
5286.6, and the useful thing to see mid-run is where the proxy sits against it. Failure is
checked before progress, so a died-at-18k run cannot display as a healthy 90%.
"""
import argparse, json, os, re, sys, time

sys.path.insert(0, os.path.expanduser("~/work/slack-facade"))
import progress  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
C09 = os.path.join(HERE, "..", "exp_c09_lut_sac")
TAG = "lut_sac_c21_seed4_20k"
ITER = re.compile(r"\[\s*(\d+)/(\d+)\]\s+steps\s+[\d,]+\s+\|\s+MJX ret\s+([-\d.]+)"
                  r"\s+\|\s+row-cov\s+(\S+)\s+\|\s+best\s+([-\d.]+)", re.M)
DONE = re.compile(r"^done: best MJX ([-\d.]+)", re.M)
REF_10K = 5286.6
TRAIN_W = 0.9
REF = ("seed 4 at DOUBLE budget · 20,000 iters vs the 10,000 that scored 5286.6 · every "
       "other knob identical incl. the seed · motivation: at 10k the addressing was still "
       "rewriting 2.5–3.2% of its bits per 500 iters, so 10k was a cut-off not a resting "
       "point · determinism ON")


def bar(pct, width=14):
    f = int(round(width * max(0.0, min(100.0, pct)) / 100.0))
    return "█" * f + "░" * (width - f)


def render():
    ev20 = os.path.join(C09, f"{TAG}_cpueval.json")
    ev10 = os.path.join(C09, f"{TAG}_at10000_cpueval.json")
    done20 = os.path.exists(ev20)
    try:
        txt = open(os.path.join(HERE, "cell_seed4_20k.log"), errors="replace").read()
    except OSError:
        return 0.0, False, 0, "`seed 4 · 20k` " + bar(0) + "  starting…"

    d = DONE.search(txt)
    if re.search(r"Traceback|CUDA out of memory|Killed|RESOURCE_EXHAUSTED", txt) and not d:
        return 0.0, False, 1, "`seed 4 · 20k` " + bar(0) + " ❌ failed"

    rows = ITER.findall(txt)
    lines = []
    if d:
        pct = 100.0
        lines.append(f"`seed 4 · 20k` {bar(100)} trained (best MJX {float(d.group(1)):.0f})"
                     + ("" if done20 else " · eval pending…"))
    elif rows:
        it, tot, ret, _cov, best = rows[-1]
        pct = 100.0 * int(it) / int(tot)
        half = " · past the 10k mark" if int(it) > 10000 else " · first half"
        lines.append(f"`seed 4 · 20k` {bar(pct)} {pct:3.0f}% · {int(it):,}/{int(tot):,}"
                     f"{half}")
        lines.append(f"MJX proxy now *{float(ret):.0f}* · best so far *{float(best):.0f}*")
        lines.append(f"_(the 10k run's own best MJX proxy was 5519; its CPU-reference "
                     f"score was {REF_10K:.1f})_")
    else:
        pct = 0.0
        lines.append(f"`seed 4 · 20k` {bar(0)}  compiling…")

    frac = TRAIN_W * pct / 100.0 + (1 - TRAIN_W) * (1.0 if done20 else 0.0)
    if done20:
        s20 = json.load(open(ev20))["cpu_reference_mean"]
        s10 = (json.load(open(ev10))["cpu_reference_mean"]
               if os.path.exists(ev10) else None)
        lines.append(f"\n✅ *20k CPU-reference {s20:.1f}* vs *10k {REF_10K:.1f}* — "
                     f"gain *{s20 - REF_10K:+.1f}*")
        if s10 is not None:
            lines.append(f"_(this run's own 10k checkpoint re-evaluated: {s10:.1f})_")
    return 100.0 * frac, done20, 0, "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--interval", type=int, default=180)
    a = ap.parse_args()
    h = progress.progress_start("seed 4 at double budget — 20,000 iters (#75)",
                                task=a.task, style="emoji", width=10)
    open(os.path.join(HERE, ".slack_c21.handle"), "w").write(h)
    print(f"bar {h}", flush=True)

    while True:
        pct, done, n_fail, body = render()
        if done or n_fail:
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
