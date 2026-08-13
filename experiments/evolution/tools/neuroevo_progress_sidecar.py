#!/usr/bin/env python3
"""Sidecar: tail the streamed-neuroevo log and drive a live Slack progress bar.
Read-only w.r.t. the run — only reads /tmp/neuroevo_stream.log and writes progress
records into the green-zone (~/.cache), which the face reaper turns into an in-place
Slack bar. Never touches the training process. Exits when the run finishes."""
import os
import re
import sys
import time

sys.path.insert(0, "/home/astarostin/work/slack-facade")
import progress  # noqa: E402

LOG = "/tmp/neuroevo_stream.log"
TASK = "f573c161"
TOTAL = int(os.environ.get("NE_GEN", 30))

h = progress.progress_start("neuroevo streamed run", task=TASK, style="unicode", width=12)
t0 = time.time()
last_gen = -1
while True:
    try:
        lines = open(LOG).read().splitlines()
    except FileNotFoundError:
        lines = []
    finished = any(("EXIT=" in l) or ("STREAMED RUN DONE" in l) for l in lines)
    genlines = [l for l in lines if l.startswith("gen ")]
    if genlines:
        last = genlines[-1]
        g = int(re.search(r"gen\s+(\d+)", last).group(1))
        ssv = re.search(r"SS-val ([\d.]+)", last)
        stv = re.search(r"STREAM-val@T\d+ ([\d.]+)", last)
        Tt = re.search(r"T=(\d+)", last)
        el = int(time.time() - t0)
        stats = "gen %d/%d · SS %s · STREAM %s · T=%s · +%dm%02ds" % (
            g + 1, TOTAL,
            ssv.group(1) if ssv else "?", stv.group(1) if stv else "?",
            Tt.group(1) if Tt else "?", el // 60, el % 60)
        if g != last_gen:
            progress.progress_update(h, step=g + 1, total=TOTAL, stats=stats)
            last_gen = g
    if finished:
        n = last_gen + 1 if last_gen >= 0 else 0
        progress.progress_update(h, step=TOTAL, total=TOTAL,
                                 stats="finished — %d generations" % n)
        progress.progress_done(h, ok=True, final_text="streamed run complete (%d gens)" % n)
        print("sidecar: run finished at gen %d — bar finalized" % n)
        break
    time.sleep(8)
