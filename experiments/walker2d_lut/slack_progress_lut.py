"""Live Slack bar for the overnight LUT work (#75): ES runs + distillation reruns.

Same rail as the other bars (`progress.py` green-zone rendezvous, one message edited in
place by the face), its OWN handle so it never touches the SAC or PPO bars.

Renders straight from the job logs, so it needs no cooperation from the jobs:
  * ES:            exp_c05_es/es_<policy>_<algo>.log   -> "gen  N/150  best ..."
  * distillation:  exp_c03_distillation/*.log          -> "epoch i/n" / the eval line

Usage:
  python slack_progress_lut.py --task <BODY_TASK_ID> [--interval 60]
"""
import argparse, glob, os, re, sys, time

sys.path.insert(0, os.path.expanduser("~/work/slack-facade"))
import progress  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ES_DIR = os.path.join(HERE, "exp_c05_es")
DIST_DIR = os.path.join(HERE, "exp_c03_distillation")

GEN_RE = re.compile(r"gen\s+(\d+)/(\d+)\s+best\s+([-\d.]+)\s+mean\s+([-\d.]+)")
DONE_RE = re.compile(r"best MJX fitness\s+([-\d.]+)")
EPOCH_RE = re.compile(r"epoch (\d+)/(\d+)\s+action MSE\s+([\d.]+)")
EVAL_RE = re.compile(r"eval:\s+([-\d.]+)\s+\+/-\s+([-\d.]+)")
FAIL_RE = re.compile(r"Traceback|Error:")

ES_JOBS = [("es_mlp_openai", "MLP · OpenAI-ES"),
           ("es_lut_openai", "LUT · OpenAI-ES"),
           ("es_mlp_sepcma", "MLP · sep-CMA-ES")]


def tail(path, n=60000):
    try:
        with open(path, errors="replace") as f:
            return f.read()[-n:]
    except OSError:
        return ""


def bar(pct, width=14):
    f = int(round(width * max(0.0, min(100.0, pct)) / 100.0))
    return "█" * f + "░" * (width - f)


def es_state(stem):
    txt = tail(os.path.join(ES_DIR, f"{stem}.log"))
    if not txt:
        return dict(state="pending", pct=0.0)
    d = DONE_RE.search(txt)
    if d:
        return dict(state="done", pct=100.0, best=float(d.group(1)))
    gens = GEN_RE.findall(txt)
    if gens:
        g, tot, best, mean = gens[-1]
        return dict(state="running", pct=100.0 * (int(g) + 1) / int(tot),
                    gen=int(g) + 1, total=int(tot), best=float(best),
                    mean=float(mean))
    if FAIL_RE.search(txt):
        return dict(state="failed", pct=0.0)
    return dict(state="starting", pct=0.0)


def dist_state():
    """Any distillation rerun still going (the OOM retry, etc.)."""
    out = []
    for p in sorted(glob.glob(os.path.join(DIST_DIR, "retry_*.log"))):
        txt = tail(p)
        name = os.path.basename(p)[6:-4]
        ev = EVAL_RE.search(txt)
        if ev:
            out.append(f"`{name}` ✅ {float(ev.group(1)):.0f} ± {float(ev.group(2)):.0f}")
            continue
        eps = EPOCH_RE.findall(txt)
        if eps:
            e, tot, mse = eps[-1]
            out.append(f"`{name}` epoch {e}/{tot} · MSE {float(mse):.4f}")
        elif FAIL_RE.search(txt):
            out.append(f"`{name}` ❌ failed")
        else:
            out.append(f"`{name}` starting…")
    return out


def build():
    lines, pcts = [], []
    for stem, label in ES_JOBS:
        s = es_state(stem)
        pcts.append(s["pct"])
        if s["state"] == "done":
            lines.append(f"`{label}` {bar(100)} 100% · ✅ best {s['best']:.0f}")
        elif s["state"] == "running":
            lines.append(f"`{label}` {bar(s['pct'])} {s['pct']:3.0f}% · "
                         f"gen {s['gen']}/{s['total']} · best {s['best']:.0f} · "
                         f"mean {s['mean']:.0f}")
        elif s["state"] == "failed":
            lines.append(f"`{label}` {bar(0)}   — · ❌ failed (see log)")
        else:
            lines.append(f"`{label}` {bar(0)}   — · queued")
    ds = dist_state()
    if ds:
        lines.append("— distillation reruns —")
        lines += ds
    head = "Phase 3 ES (gradient-free ceiling) + LUT reruns · fitness = MJX horizon-400, NOT the CPU reference"
    return sum(pcts) / max(len(pcts), 1), head + "\n" + "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--interval", type=int, default=60)
    ap.add_argument("--handle-file", default=os.path.join(HERE, ".slack_lut.handle"))
    a = ap.parse_args()

    h = None
    if os.path.exists(a.handle_file):
        cand = open(a.handle_file).read().strip()
        if cand and os.path.exists(os.path.expanduser(
                f"~/.cache/slack_facade/progress/{cand}.json")):
            h = cand
            print(f"reusing bar {h}", flush=True)
    if h is None:
        h = progress.progress_start("Overnight LUT work — Phase 3 ES + reruns (#75)",
                                    task=a.task, style="emoji", width=10)
    open(a.handle_file, "w").write(h)
    print(f"lut bar {h} (task {a.task}) interval {a.interval}s", flush=True)

    while True:
        pct, stats = build()
        states = [es_state(s)["state"] for s, _ in ES_JOBS]
        if all(s in ("done", "failed") for s in states):
            n_ok = sum(s == "done" for s in states)
            progress.progress_done(
                h, ok=True,
                final_text=f"{n_ok}/{len(ES_JOBS)} ES runs finished · " +
                           stats.split("\n", 1)[1].replace("\n", " · "))
            print("all ES runs terminal; posted final", flush=True)
            return
        progress.progress_update(h, pct=pct, stats=stats)
        print(f"[{time.strftime('%H:%M:%S')}] {pct:5.1f}% {states}", flush=True)
        time.sleep(a.interval)


if __name__ == "__main__":
    main()
