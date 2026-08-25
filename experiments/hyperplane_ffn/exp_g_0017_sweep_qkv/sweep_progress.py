"""Consolidated Slack progress bar for the exp_g_0017 q/k/v attention sweep (ONE message for all 9
runs, not 12 bars). Reads each per-run subdir's summary.json (done) / metrics.csv (running),
updates a single progress bar with done-count + current best + in-flight steps, and
finalizes with a top-3 ranking when all 9 complete. Launch with BAR_HANDLE set.
"""
import os, sys, time, json, glob
sys.path.insert(0, '/home/astarostin/work/slack-facade')
import progress

D = '/home/astarostin/projects/spiky-fmhl-next/experiments/hyperplane_ffn/exp_g_0017_sweep_qkv'
H = os.environ['BAR_HANDLE']
TOTAL = 9
STEPS_PER = int(os.environ.get('STEPS_PER_RUN', 4000))   # for fractional (smooth) bar fill

for _ in range(1400):                       # ~ up to 11.6h at 30s cadence
    done, running = {}, {}
    for sub in glob.glob(os.path.join(D, 'qkvin*_tph*/')):
        tag = os.path.basename(sub.rstrip('/'))
        sj, mj = os.path.join(sub, 'summary.json'), os.path.join(sub, 'metrics.csv')
        if os.path.exists(sj):
            try:
                done[tag] = float(json.load(open(sj))['final_val_bpb'])
            except Exception:
                pass
        elif os.path.exists(mj):
            try:
                rows = open(mj).read().strip().splitlines()
                running[tag] = int(rows[-1].split(',')[0]) if len(rows) > 1 else 0
            except Exception:
                running[tag] = 0
    ndone = len(done)
    # Fractional fill: count completed runs AND in-flight step progress, so the bar
    # moves smoothly instead of sitting at 0% until the first run completes.
    frac = (ndone * STEPS_PER + sum(min(s, STEPS_PER) for s in running.values())) / (TOTAL * STEPS_PER)
    best = min(done.items(), key=lambda kv: kv[1]) if done else None
    beststr = f"best {best[0]}={best[1]:.4f}" if best else "best —"
    runstr = ", ".join(f"{t}@{s}" for t, s in sorted(running.items()))
    stats = f"{ndone}/{TOTAL} done · {beststr}" + (f" · running: {runstr}" if runstr else "")

    if ndone >= TOTAL:
        ranked = sorted(done.items(), key=lambda kv: kv[1])
        top = " | ".join(f"{t} {b:.4f}" for t, b in ranked[:3])
        progress.progress_done(H, ok=True,
            final_text=f"9/9 done · winner {ranked[0][0]} {ranked[0][1]:.4f} · top3: {top}")
        break
    progress.progress_update(H, pct=100.0 * frac, stats=stats)
    time.sleep(30)
